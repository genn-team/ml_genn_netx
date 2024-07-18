import numpy as np
import matplotlib.pyplot as plt
import mnist
import lava.lib.dl.netx as netx

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.io.source import RingBuffer as SourceRingBuffer
from lava.proc.monitor.process import Monitor
from ml_genn import Connection, Network, Population
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from tonic.datasets import SHD
from tonic.transforms import ToFrame

from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)
from ml_genn_netx import export
from tqdm import tqdm

from ml_genn.compilers.event_prop_compiler import default_params

NUM_HIDDEN = 256
BATCH_SIZE = 32
NUM_EPOCHS = 300
DT = 1.0
TRAIN = True
KERNEL_PROFILING = True

# Get SHD dataset
dataset = SHD(save_to='../data', train=False, 
              transform=ToFrame(sensor_size=SHD.sensor_size, time_window=1000.0))

# Get number of input and output neurons from dataset 
num_input = int(np.prod(dataset.sensor_size))
num_output = len(dataset.classes)
max_timesteps = 1400

# Preprocess
tensors = []
labels = []
for i in range(len(dataset)):
    tensor, label = dataset[i]
    assert tensor.shape[-1] < max_timesteps
    
    # Transpose tensor and pad time to max
    tensors.append(np.pad(np.reshape(np.transpose(tensor), (num_input, -1)),
                          ((0, 0), (0, max_timesteps - tensor.shape[0]))))
    labels.append(label)

# Stack tensors
tensors = np.hstack(tensors)

serialiser = Numpy("shd_checkpoints")
network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * 1500),
                       num_input, name="Input")
    hidden = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                           tau_refrac=None),
                        NUM_HIDDEN, name="Hidden")
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        num_output, name="Output")

    # Connections
    Connection(input, hidden, Dense(Normal(mean=0.03, sd=0.01)),
               Exponential(5.0))
    Connection(hidden, hidden, Dense(Normal(mean=0.0, sd=0.02)),
               Exponential(5.0))
    Connection(hidden, output, Dense(Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))

network.load((NUM_EPOCHS - 1,), serialiser)
export("shd.net", input, output, dt=DT)

network_lava = netx.hdf5.Network(net_config="shd.net", reset_interval=max_timesteps)

# **TODO** move to recurrent unit test
assert network_lava.input_shape == (num_input,)
assert len(network_lava) == 2
assert type(network_lava.layers[0]) == netx.blocks.process.RecurrentDense
assert type(network_lava.layers[1]) == netx.blocks.process.Dense

# Create source ring buffer to deliver input spike tensors and connect to network input port
input_lava = SourceRingBuffer(data=tensors)
input_lava.s_out.connect(network_lava.inp)

n_samples = 100
# Create monitor to record output voltages (shape is total timesteps)
monitor_output = Monitor()
monitor_output.probe(network_lava.layers[-1].neuron.v, n_samples * max_timesteps)

run_config = Loihi2SimCfg(select_tag="fixed_pt")

for _ in tqdm(range(n_samples)):
    network_lava.run(condition=RunSteps(num_steps=max_timesteps), run_cfg=run_config)

output_v = monitor_output.get_data()
good = 0
for i in range(n_samples):
    out_v = output_v["neuron"]["v"][i*max_timesteps:(i+1)*max_timesteps,:]
    sum_v = np.sum(out_v, axis=0)
    pred = np.argmax(sum_v)
    if pred == labels[i]:
        good += 1


print(f"test accuracy: {good/n_samples*100}")


#output_v = out.data.get()
network_lava.stop()
