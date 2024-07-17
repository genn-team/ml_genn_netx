import numpy as np
import matplotlib.pyplot as plt
import mnist

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from tonic.datasets import SHD

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)

from ml_genn.compilers.event_prop_compiler import default_params

from ml_genn_netx import export

NUM_HIDDEN = 256
BATCH_SIZE = 32
NUM_EPOCHS = 300
DT = 1.0
TRAIN = True
KERNEL_PROFILING = True

# Get SHD dataset
#dataset = SHD(save_to='../data', train=TRAIN)

# Get number of input and output neurons from dataset 
# and round up outputs to power-of-two
num_input = 700#int(np.prod(dataset.sensor_size))
num_output = 20#len(dataset.classes)

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

export("test.hd5", input, output)