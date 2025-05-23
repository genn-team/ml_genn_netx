import numpy as np
import matplotlib.pyplot as plt
import mnist
import lava.lib.dl.netx as netx
import logging
import os

from lava.magma.core.run_configs import Loihi2SimCfg, Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.cyclic_buffer.process import CyclicBuffer
from lava.proc.io.source import RingBuffer as SourceRingBuffer
from lava.proc.monitor.process import Monitor
from lava.utils.loihi2_state_probes import StateProbe
from lava.utils.system import Loihi2

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import Callback, Checkpoint, SpikeRecorder, VarRecorder
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential

from tonic.datasets import SHD
from tonic.transforms import CropTime, ToFrame

from copy import copy
from ml_genn.utils.data import preprocess_tonic_spikes
from ml_genn_netx import export
from random import choice
from tqdm import tqdm

from ml_genn.compilers.event_prop_compiler import default_params

NUM_HIDDEN = 1024
BATCH_SIZE = 1
NUM_EPOCHS = 50
DT = 1.0
TRAIN = False
PLOT = True
DEVICE = False
KERNEL_PROFILING = False
NUM_TEST_SAMPLES = 1
MAX_TIMESTEPS = 1024


class EaseInSchedule(Callback):
    def set_params(self, compiled_network, **kwargs):
        self._optimisers = [o for o, _ in compiled_network.optimisers]

    def on_batch_begin(self, batch):
        # Set parameter to return value of function
        for o in self._optimisers:
            if o.alpha < 0.001 :
                o.alpha = (0.001 / 1000.0) * (1.05 ** batch)
            else:
                o.alpha = 0.001


class Shift:
    def __init__(self, f_shift, sensor_size):
        self.f_shift = f_shift
        self.sensor_size = sensor_size

    def __call__(self, events: np.ndarray) -> np.ndarray:
        # Copy events and shift in space by random amount
        events_copy = events.copy()
        events_copy["x"] += np.random.randint(-self.f_shift, self.f_shift)

        # Delete out of bound events
        events_copy = np.delete(
            events_copy,
            np.where(
                (events_copy["x"] < 0) | (events_copy["x"] >= self.sensor_size[0])))
        return events_copy


class Blend:
    def __init__(self, p_blend, sensor_size, n_blend=7644):
        self.p_blend = p_blend
        self.n_blend = n_blend
        self.sensor_size = sensor_size

    def __call__(self, dataset: list, classes: list) -> list:
        # Start with (shallow) copy of original dataset
        blended_dataset = copy(dataset)

        # Loop through number of blends to add
        for i in range(self.n_blend):
            # Pick random example
            idx = np.random.randint(0, len(dataset))
            example_spikes, example_label = dataset[idx]
           
            # Pick another from same class
            idx2 = np.random.randint(0, len(classes[example_label]))
            blend_spikes, blend_label = dataset[classes[example_label][idx2]]
            assert blend_label == example_label
            
            # Blend together to form new dataset
            blended_dataset.append((self.blend(example_spikes, blend_spikes),
                                    example_label))

        return blended_dataset

    def blend(self, X1, X2):
        # Copy spike arrays and align centres of mass in space and time
        X1 = X1.copy()
        X2 = X2.copy()
        mx1 = np.mean(X1["x"])
        mx2 = np.mean(X2["x"])
        mt1 = np.mean(X1["t"])
        mt2 = np.mean(X2["t"])
        X1["x"]+= int((mx2-mx1)/2)
        X2["x"]+= int((mx1-mx2)/2)
        X1["t"]+= int((mt2-mt1)/2)
        X2["t"]+= int((mt1-mt2)/2)

        # Delete any spikes that are out of bounds in space or time
        max_t = MAX_TIMESTEPS * DT * 1000.0
        X1 = np.delete(
            X1, np.where((X1["x"] < 0) | (X1["x"] >= self.sensor_size[0])
                         | (X1["t"] < 0) | (X1["t"] >= max_t)))
        X2 = np.delete(
            X2, np.where((X2["x"] < 0) | (X2["x"] >= self.sensor_size[0]) 
                         | (X2["t"] < 0) | (X2["t"] >= max_t)))

        # Combine random blended subset of spikes
        mask1 = np.random.rand(X1["x"].shape[0]) < self.p_blend
        mask2 = np.random.rand(X2["x"].shape[0]) < (1.0 - self.p_blend)
        X1_X2 = np.concatenate((X1[mask1], X2[mask2]))

        # Resort and return
        idx = np.argsort(X1_X2["t"])
        X1_X2 = X1_X2[idx]
        return X1_X2

def load_data(train, num=None):
    # Get SHD dataset, cropped to maximum timesteps (in us)
    dataset = SHD(save_to="../data", train=train,
                  transform=CropTime(max=MAX_TIMESTEPS * DT * 1000.0))

    # Get raw event data
    raw_data = []
    for i in range(num if num is not None else len(dataset)):
        raw_data.append(dataset[i])

    return raw_data, dataset.sensor_size, dataset.ordering, len(dataset.classes)

def build_ml_genn_model(sensor_size, num_classes):
    network = Network(default_params)
    with network:
        # Populations
        input = Population(SpikeInput(max_spikes=BATCH_SIZE * 15000),
                           int(np.prod(sensor_size)), 
                           name="Input", record_spikes=True)
        hidden = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                               tau_refrac=None),
                            NUM_HIDDEN, name="Hidden", record_spikes=True)
        output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                            num_classes, name="Output")

        # Connections
        input_hidden = Connection(input, hidden, 
                                  Dense(Normal(mean=0.03, sd=0.01)),
                                  Exponential(5.0))
        Connection(hidden, hidden, Dense(Normal(mean=0.0, sd=0.02)),
                   Exponential(5.0))
        Connection(hidden, output, Dense(Normal(mean=0.0, sd=0.03)),
                   Exponential(5.0))

    return network, input, hidden, output, input_hidden

def train_genn(raw_dataset, network, serialiser,
               input, hidden, output, input_hidden,
               sensor_size, ordering):
    # Create EventProp compiler
    compiler = EventPropCompiler(example_timesteps=MAX_TIMESTEPS,
                                 losses="sparse_categorical_crossentropy",
                                 reg_lambda_upper=5e-11, reg_lambda_lower=5e-11, 
                                 reg_nu_upper=14, max_spikes=1500, 
                                 optimiser=Adam(0.001 / 1000.0), batch_size=BATCH_SIZE)
    # Create augmentation objects
    shift = Shift(40.0, sensor_size)
    blend = Blend(0.5, sensor_size)

    # Build classes list
    classes = [[] for _ in range(np.prod(output.shape))]
    for i, (_, label) in enumerate(raw_dataset):
        classes[label].append(i)
    
    # Compile network
    compiled_net = compiler.compile(network)
    input_hidden_sg = compiled_net.connection_populations[input_hidden]
    # Train
    with compiled_net:
        callbacks = [Checkpoint(serialiser), EaseInSchedule(),
                     SpikeRecorder(hidden, key="hidden_spikes",
                                   record_counts=True)]
        # Loop through epochs
        for e in range(NUM_EPOCHS):
            # Apply augmentation to events and preprocess
            spikes_train = []
            labels_train = []
            blended_dataset = blend(raw_dataset, classes)
            for events, label in blended_dataset:
                spikes_train.append(preprocess_tonic_spikes(shift(events), ordering,
                                                            sensor_size, dt=DT,
                                                            histogram_thresh=1))
                labels_train.append(label)
            
            # Train epoch
            metrics, cb_data  = compiled_net.train(
                {input: spikes_train}, {output: labels_train},
                start_epoch=e, num_epochs=1, 
                shuffle=True, callbacks=callbacks)

            # Sum number of hidden spikes in each batch
            hidden_spikes = np.zeros(NUM_HIDDEN)
            for cb_d in cb_data["hidden_spikes"]:
                hidden_spikes += cb_d

            num_silent = np.count_nonzero(hidden_spikes==0)
            print(f"GeNN training epoch: {e}, Silent neurons: {num_silent}, Training accuracy: {100 * metrics[output].result}%")
            
            if num_silent > 0:
                input_hidden_sg.vars["g"].pull_from_device()
                g_view = input_hidden_sg.vars["g"].view.reshape((np.prod(input.shape), NUM_HIDDEN))
                g_view[:,hidden_spikes==0] += 0.002
                input_hidden_sg.vars["g"].push_to_device()

def evaluate_genn(raw_dataset, network, 
                  input, hidden, output, 
                  sensor_size, ordering, plot):
    # Preprocess
    spikes = []
    labels = []
    for events, label in raw_dataset:
        spikes.append(preprocess_tonic_spikes(events, ordering,
                                              sensor_size, dt=DT,
                                              histogram_thresh=1))
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

        print(f"GeNN test accuracy: {100 * metrics[output].result}%")
        
        if plot:
            fig, axes = plt.subplots(2, NUM_TEST_SAMPLES, sharex="col", sharey="row", squeeze=False)
            for a in range(NUM_TEST_SAMPLES):
                if NUM_TEST_SAMPLES > 1:
                    axes[0, a].scatter(cb_data["hidden_spikes"][0][a], cb_data["hidden_spikes"][1][a], s=1)
                else:
                    axes[0,a].scatter(cb_data["hidden_spikes"][0], cb_data["hidden_spikes\
"][1], s=1)
                axes[1, a].plot(cb_data["output_v"][a])
            axes[0, 0].set_ylabel("Hidden neuron ID")
            axes[1, 0].set_ylabel("Output voltage")
                
def evaluate_lava(raw_dataset, net_x_filename, 
                  sensor_size, num_classes, plot, device):
    #logging.basicConfig(level=logging.INFO)
    os.environ["PATH"] += ":/nfs/ncl/bin:"
    os.environ["PARTITION"] = "oheogulch_20m" # _20m _2h (if 2 hours are needed)                              
    os.environ['SLURM'] = '1'
    os.environ['LOIHI_GEN'] = 'N3C1'

    # Preprocess
    num_input = int(np.prod(sensor_size))
    transform = ToFrame(sensor_size=sensor_size, time_window=1000.0*DT) 
    tensors = []
    labels = []
    for events, label in raw_dataset:
        # Transform events to tensor
        tensor = transform(events)
        assert tensor.shape[0] < MAX_TIMESTEPS 

        # Transpose tensor and pad time to max
        print(f"MAX_TIMESTEPS: {MAX_TIMESTEPS}, tensor.shape[0]: {tensor.shape[0]}")
        tensors.append(np.pad(np.reshape(np.transpose(tensor), (num_input, -1)),
                              ((0, 0), (0, MAX_TIMESTEPS - tensor.shape[0]))))
        labels.append(label)

    # Stack tensors
    tensors = np.hstack(tensors).astype(np.int8)

    # **TODO** MAX_TIMESTEPS should be maximum of 256 and P.O.T.
    print(f"MAX_TIMESTEPS: {MAX_TIMESTEPS}")
    if not device:
        network_lava = netx.hdf5.Network(net_config="shd.net", reset_interval=MAX_TIMESTEPS)
    else:
        network_lava = netx.hdf5.Network(net_config="shd.net", input_message_bits=8)
    network_lava._log_config.level = logging.INFO
    # **TODO** move to recurrent unit test
    assert network_lava.input_shape == (num_input,)
    assert len(network_lava) == 2
    assert type(network_lava.layers[0]) == netx.blocks.process.RecurrentDense
    assert type(network_lava.layers[1]) == netx.blocks.process.Dense

    if device:
        first_tensor = tensors[:,0]
        ro_tensors = tensors[:,1:]
        input_lava = CyclicBuffer(first_frame=first_tensor, replay_frames=ro_tensors)
        input_lava.s_out.connect(network_lava.inp)
        #input_lava.s_out.connect(network_lava.layers[0].synapse.s_in)
        probe_output_v = StateProbe(network_lava.layers[-1].neuron.v)
        probe_hidden_v = StateProbe(network_lava.layers[0].neuron.v)
        run_config = Loihi2HwCfg(callback_fxs=[probe_output_v,probe_hidden_v])
        
        if Loihi2.is_loihi2_available:
            print(f'Running on {Loihi2.partition}')
            from lava.utils import loihi2_profiler
        else:
            RuntimeError("Loihi2 compiler is not available in this system. "
                        "This tutorial cannot proceed further.")
        
        # Run model for each test sample
        for _ in tqdm(range(NUM_TEST_SAMPLES)):
            network_lava.run(condition=RunSteps(num_steps=MAX_TIMESTEPS), run_cfg=run_config)   
            # reset the voltage after each trial
            network_lava.layers[0].neuron.v.set(np.zeros((NUM_HIDDEN,), dtype = np.int32))
            network_lava.layers[0].neuron.u.set(np.zeros((NUM_HIDDEN,), dtype = np.int32))
            network_lava.layers[1].neuron.v.set(np.zeros((num_classes,), dtype = np.int32))
            network_lava.layers[1].neuron.u.set(np.zeros((num_classes,), dtype = np.int32))
        
        output_v = probe_output_v.time_series.reshape(num_classes, MAX_TIMESTEPS * NUM_TEST_SAMPLES).T
        hidden_v = probe_hidden_v.time_series.reshape(NUM_HIDDEN, MAX_TIMESTEPS * NUM_TEST_SAMPLES).T
    else:
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

        # Get output
        output_v = monitor_output.get_data()["neuron"]["v"]
    
    
    output_v = np.reshape(output_v, (NUM_TEST_SAMPLES, MAX_TIMESTEPS, num_classes))
    if device:
        hidden_v = np.reshape(hidden_v, (NUM_TEST_SAMPLES, MAX_TIMESTEPS, NUM_HIDDEN))

    # Calculate output weighting
    output_weighting = np.exp(-np.arange(MAX_TIMESTEPS) / MAX_TIMESTEPS)

    # For each example, sum weighted output neuron voltage over time
    sum_v = np.sum(output_v * output_weighting[np.newaxis,:,np.newaxis], axis=1)

    # Find maximum output neuron voltage and compare to label
    pred = np.argmax(sum_v, axis=1)
    good = np.sum(pred == labels)
    print(f"pred: {pred}, labels: {labels}")
    print(f"Lava test accuracy: {good/NUM_TEST_SAMPLES*100}%")
    if plot:
        if not device:
            hidden_spikes = monitor_hidden.get_data()["neuron"]["s_out"]
            hidden_spikes = np.reshape(hidden_spikes, (NUM_TEST_SAMPLES, MAX_TIMESTEPS, NUM_HIDDEN))
        
        fig, axes = plt.subplots(2, NUM_TEST_SAMPLES, sharex="col", sharey="row", squeeze=False)
        for a in range(NUM_TEST_SAMPLES):
            if not device:
                sample_hidden_spikes = np.where(hidden_spikes[a,:,:] > 0.0)
                axes[0, a].scatter(sample_hidden_spikes[0], sample_hidden_spikes[1], s=1)
            else:
                axes[0, a].plot(hidden_v[a,:,:5])
            axes[1, a].plot(output_v[a,:,:])
        axes[0,0].set_ylabel("Hidden neuron ID")
        axes[1,0].set_ylabel("Output voltage")

    network_lava.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Get SHD data
    if TRAIN:
        raw_train_data, sensor_size, ordering, num_classes = load_data(True)
        raw_test_data, _, _, _ = load_data(False, NUM_TEST_SAMPLES)
    else:
        raw_test_data, sensor_size, ordering, num_classes = load_data(False, NUM_TEST_SAMPLES)

    # Build suitable mlGeNN model
    network, input, hidden, output, input_hidden = build_ml_genn_model(sensor_size, num_classes)
    
    serialiser = Numpy("shd_checkpoints_thomas")
    
    if TRAIN:
        train_genn(raw_train_data, network, serialiser,
                   input, hidden, output, input_hidden,
                   sensor_size, ordering)
    
    # Evaluate in GeNN
    network.load((NUM_EPOCHS - 1,), serialiser)
    evaluate_genn(raw_test_data, network, 
                  input, hidden, output, 
                  sensor_size, ordering, PLOT)

    # Export to netx
    export("shd.net", input, output, dt=DT,num_weight_bits=8)

    # Evaluate in Lava
    evaluate_lava(raw_test_data, "shd.net", sensor_size, num_classes, PLOT, DEVICE)

    plt.show()







