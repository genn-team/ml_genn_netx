import numpy as np
import matplotlib.pyplot as plt
import mnist
import lava.lib.dl.netx as netx
import logging

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.io.source import RingBuffer as SourceRingBuffer
from lava.proc.monitor.process import Monitor
from ml_genn import Connection, Network, Population
from ml_genn.callbacks import SpikeRecorder, VarRecorder
from ml_genn.compilers import InferenceCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from tonic.datasets import SHD
from tonic.transforms import ToFrame

from ml_genn.utils.data import preprocess_tonic_spikes
from ml_genn_netx import export
from tqdm import tqdm

from ml_genn.compilers.event_prop_compiler import default_params

NUM_HIDDEN = 256
BATCH_SIZE = 32
NUM_EPOCHS = 300
DT = 1.0
TRAIN = True
PLOT = False
KERNEL_PROFILING = True
NUM_TEST_SAMPLES = 200
MAX_TIMESTEPS = 1400

def get_dataset_num_in_out(dataset):
    # Get number of input and output neurons from dataset 
    num_input = int(np.prod(dataset.sensor_size))
    num_output = len(dataset.classes)

    return num_input, num_output

def build_ml_genn_model(dataset):
    num_input, num_output = get_dataset_num_in_out(dataset)
    serialiser = Numpy("shd_checkpoints")
    network = Network(default_params)
    with network:
        # Populations
        input = Population(SpikeInput(max_spikes=BATCH_SIZE * 15000),
                           num_input, name="Input", record_spikes=True)
        hidden = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                               tau_refrac=None),
                            NUM_HIDDEN, name="Hidden", record_spikes=True)
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
    return network, input, hidden, output

def evaluate_genn(dataset, network, input, hidden, output, plot):
    # Preprocess
    spikes = []
    labels = []
    for i in range(NUM_TEST_SAMPLES):
        events, label = dataset[i]
        spikes.append(preprocess_tonic_spikes(events, dataset.ordering,
                                              dataset.sensor_size))
        labels.append(label)
    
    compiler = InferenceCompiler(evaluate_timesteps=MAX_TIMESTEPS,
                                 reset_in_syn_between_batches=True,
                                 batch_size=BATCH_SIZE)
    compiled_net = compiler.compile(network)

    with compiled_net:
        callbacks = ["batch_progress_bar"]
        if plot:
            callbacks.extend([SpikeRecorder(hidden, key="hidden_spikes"),
                              VarRecorder(output, "v", key="output_v")])

        metrics, cb_data  = compiled_net.evaluate({input: spikes},
                                                  {output: labels},
                                                  callbacks=callbacks)

        print(f"GeNN test accuracy = {100 * metrics[output].result}%")
        
        if plot:
            fig, axes = plt.subplots(2, NUM_TEST_SAMPLES, sharex="col", sharey="row")
            for a in range(NUM_TEST_SAMPLES):
                axes[0, a].scatter(cb_data["hidden_spikes"][0][a], cb_data["hidden_spikes"][1][a], s=1)
                axes[1, a].plot(cb_data["output_v"][a])
            
            axes[0, 0].set_ylabel("Hidden neuron ID")
            axes[1, 0].set_ylabel("Output voltage")

def evaluate_lava(dataset, net_x_filename, plot):
    # Preprocess
    num_input, num_output = get_dataset_num_in_out(dataset)
    transform = ToFrame(sensor_size=SHD.sensor_size, time_window=1000.0)
    tensors = []
    labels = []
    for i in range(NUM_TEST_SAMPLES):
        events, label = dataset[i]

        # Transform events to tensor
        tensor = transform(events)
        assert tensor.shape[-1] < MAX_TIMESTEPS

        # Transpose tensor and pad time to max
        tensors.append(np.pad(np.reshape(np.transpose(tensor), (num_input, -1)),
                              ((0, 0), (0, MAX_TIMESTEPS - tensor.shape[0]))))
        labels.append(label)

    # Stack tensors
    tensors = np.hstack(tensors)

    network_lava = netx.hdf5.Network(net_config="shd.net", reset_interval=MAX_TIMESTEPS)

    # **TODO** move to recurrent unit test
    assert network_lava.input_shape == (num_input,)
    assert len(network_lava) == 2
    assert type(network_lava.layers[0]) == netx.blocks.process.RecurrentDense
    assert type(network_lava.layers[1]) == netx.blocks.process.Dense

    # Create source ring buffer to deliver input spike tensors and connect to network input port
    input_lava = SourceRingBuffer(data=tensors)
    input_lava.s_out.connect(network_lava.inp)

    # Create monitor to record output voltages (shape is total timesteps)
    monitor_output = Monitor()
    monitor_output.probe(network_lava.layers[-1].neuron.v, NUM_TEST_SAMPLES * MAX_TIMESTEPS)

    if plot:
        monitor_hidden = Monitor()
        monitor_hidden.probe(network_lava.layers[0].neuron.s_out, NUM_TEST_SAMPLES * MAX_TIMESTEPS)

    run_config = Loihi2SimCfg(select_tag="fixed_pt")

    # Run model for each test sample
    for _ in tqdm(range(NUM_TEST_SAMPLES)):
        network_lava.run(condition=RunSteps(num_steps=MAX_TIMESTEPS), run_cfg=run_config)

    # Get output and reshape
    output_v = monitor_output.get_data()["neuron"]["v"]
    output_v = np.reshape(output_v, (NUM_TEST_SAMPLES, MAX_TIMESTEPS, num_output))

    # For each example, sum output neuron voltage over time
    sum_v = np.sum(output_v, axis=1)

    # Find maximum output neuron voltage and compare to label
    pred = np.argmax(sum_v, axis=1)
    good = np.sum(pred == labels)

    print(f"Lava test accuracy: {good/NUM_TEST_SAMPLES*100}%")
    if plot:
        hidden_spikes = monitor_hidden.get_data()["neuron"]["s_out"]
        hidden_spikes = np.reshape(hidden_spikes, (NUM_TEST_SAMPLES, MAX_TIMESTEPS, NUM_HIDDEN))
        
        fig, axes = plt.subplots(2, NUM_TEST_SAMPLES, sharex="col", sharey="row")
        for a in range(NUM_TEST_SAMPLES):
            sample_hidden_spikes = np.where(hidden_spikes[a,:,:] > 0.0)
            axes[0, a].scatter(sample_hidden_spikes[0], sample_hidden_spikes[1], s=1)
            axes[1, a].plot(output_v[a,:,:])
        
        axes[0,0].set_ylabel("Hidden neuron ID")
        axes[1,0].set_ylabel("Output voltage")

    network_lava.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Get SHD dataset
    dataset = SHD(save_to='../data', train=False)

    # Build suitable mlGeNN model
    network, input, hidden, output = build_ml_genn_model(dataset)

    # Evaluate in GeNN
    evaluate_genn(dataset, network, input, hidden, output, PLOT)

    # Export to netx
    export("shd.net", input, output, dt=DT)

    # Evaluate in Lava
    evaluate_lava(dataset, "shd.net", PLOT)

    plt.show()







