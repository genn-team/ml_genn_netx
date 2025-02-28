import h5py
import logging
import math
import numpy as np

from numbers import Number
from typing import Sequence, Tuple
from ml_genn import Network, Population
from ml_genn.connectivity import Dense
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, Neuron
from ml_genn.synapses import Delta, Exponential, Synapse
from ml_genn.utils.value import InitValue

from ml_genn.utils.network import get_underlying_pop
from ml_genn.utils.value import is_value_array, is_value_constant

logger = logging.getLogger(__name__)

# **TODO** upstream back into ml_genn.utils.quantisation
def _find_signed_scale(data, num_bits: int, percentile: float):
    # Calculate desired percentile
    if isinstance(data, Number):
        max_val = data
    else:
        # Split data into positive and negative
        positive_mask = (data > 0)
        positive_data = data[positive_mask]
        negative_data = data[np.logical_not(positive_mask)]

        # Calculate desired percentile
        positive_perc = np.percentile(positive_data, percentile)
        negative_perc = np.percentile(-negative_data, percentile)

        # Calculate the largest of these
        max_val = max(positive_perc, negative_perc)
    
    # Calculate high bit and low bit
    # **NOTE** we floor so max is 2**(high_bit + 1) - 1
    # **NOTE** one bit is used for sign
    high_bit =  math.floor(math.log(max_val, 2))
    low_bit = high_bit - (num_bits - 2)
    
    # We scale to multiples of the low bit
    scale = (2.0 ** low_bit)
    
    # Calculate min and max
    min_quant = (-2.0 ** (high_bit + 1))
    max_quant = (2.0 ** (high_bit + 1)) - scale

    # Return range and scale
    return min_quant, max_quant, scale

def _check_param(value, shape):
    if is_value_constant(value):
        return value
    elif is_value_array(value):
        assert value.shape == shape
        return value
    else:
        raise NotImplementedError("NetX exporter does not support "
                                  "parameters initialised with Initializers")

def _to_fixed_point(data, scale, type):
    # Apply scale and ensure values are within range of integer type
    type_info = np.iinfo(type)
    scaled_data = np.round(data * scale)
    assert np.all((scaled_data >= type_info.min)
                  & (scaled_data <= type_info.max))

    # Round and cast
    return scaled_data.astype(type)

def _get_network_dag(inputs, outputs):
    # Convert inputs and outputs to tuples
    inputs = inputs if isinstance(inputs, Sequence) else (inputs,)
    outputs = outputs if isinstance(outputs, Sequence) else (outputs,)

    # Construct topologically sorted list of layers using Kahn's algorithm as
    # described here: https://en.wikipedia.org/wiki/Topological_sorting)
    dag = []
    recurrent = []
    new_pops = set(get_underlying_pop(i) for i in inputs)
    seen_conns = set()
    while new_pops:
        pop = new_pops.pop()
        dag.append(pop)
        recurrent.append(False)

        # Explore outgoing connections whose
        # upstream connections have all been seen
        for conn in pop.outgoing_connections:
            # If this connection is recurrent, don't recurse through it,
            # Just add it to list of recurrently connected populations`
            target_pop = conn().target()
            if target_pop == pop:
                recurrent[-1] = True
            else:
                # Filter target's list of incoming
                ff_incoming = [c for c in target_pop.incoming_connections
                               if c().source() != target_pop]
                
                seen_conns.add(conn)
                if seen_conns.issuperset(ff_incoming):
                    new_pops.add(conn().target())

    # Check that output layers are in the DAG i.e. reachable from input layers
    if not all(get_underlying_pop(o) in dag
               for o in outputs):
        raise RuntimeError("outputs unreachable from inputs")

    # Zip DAG back together with recurrentness 
    return list(zip(dag, recurrent))

def _get_weight_scale(neuron: Neuron, synapse: Synapse,
                      shape, dt: float):
    # If neuron model has leak
    weight_scale = 1.0
    if isinstance(neuron, (LeakyIntegrateFire, LeakyIntegrate)):
        # If it scales I, multiply weight scale by additional 
        # decay GeNN neuron applies to incoming current
        if neuron.scale_i:
            tau_mem = _check_param(neuron.tau_mem, shape)
            weight_scale *= 1.0 - np.exp(-dt / tau_mem)
    
    # If synapse model is exponential and scales i, multiply weight scale
    # by additional factor GeNN uses to improve match with exact solution
    if isinstance(synapse, Exponential) and synapse.scale_i:
        tau_syn = _check_param(synapse.tau, shape)
        weight_scale *= (tau_syn / dt) * (1.0 - np.exp(-dt / tau_syn))
    
    return weight_scale


def _get_netx_weight(weights: Sequence[Tuple[np.ndarray, int, int]], 
                     num_bits: int, quant_percentile: float,
                     weight_scale: float):
    
    # Reshape weights into (num_src, num_trg) shape and apply weight scale
    # **NOTE** weight scale may be a scalar or num_trg long vector
    weights = [(np.reshape(w, (s, t)) * weight_scale, s, t)
                for w, s, t in weights]

    # Flatten and concatenate weights and find scaling factors
    weights_concat = np.concatenate([w.flatten() for w, _, _ in weights])
    min_quant, max_quant, scale = _find_signed_scale(weights_concat, num_bits,
                                                     quant_percentile)

    # Take transpose of weights, quantise, clip and scale
    quant_weights = [np.clip(scale * np.round(np.transpose(w) / scale),
                             min_quant, max_quant) / scale
                     for w, _, _ in weights]
    logger.info(f"\tWeight scale: {1 / scale}, Min max quant: {min_quant}, {max_quant}")
    
    return scale, *quant_weights

def _get_netx_delays(delay: InitValue, num_src: int, num_trg: int):
    # If delay is constant and zero, do nothing
    is_delay_const = is_value_constant(delay)
    is_delay_array = is_value_array(delay)
    if is_delay_const and delay == 0:
        return None
    # Otherwise, if it's constant or an array
    elif is_delay_array or is_delay_const:
        # Reshape arrays
        if is_delay_array:
            delay = np.reshape(delay, (num_src, num_trg))
        
        # Take transpose and convert to integer
        delay = np.rint(np.transpose(delay)).astype(int)
        if np.any((delay < 0) | (delay >= 62)):
            logger.warn("\tFor Loihi delays must be between 0 and 62")
        return delay
    else:
        raise RuntimeError("Before exporting to NetX, delays "
                           "should be loaded from checkpoints")
    
def _export_neuron(layer_group: h5py.Group, shape, dt: float, 
                   quant_scale: float, neuron: Neuron, synapse: Synapse):
    # Create group
    neuron_group = layer_group.create_group("neuron")
    neuron_group.create_dataset("gradedSpike", dtype="b1")

    # If connections have exponential synapses
    if isinstance(synapse, Exponential):
        # Ensure we have an array of tau values
        tau_syn = _check_param(synapse.tau, shape)
        
        # Calculate decays and convert to fixed point
        i_alpha_fixed_point = _to_fixed_point(1.0 - np.exp(-dt / tau_syn),
                                              2 ** 12, "i4")
    # Otherwise, if synapse is plain delta, add empty iDecay
    elif isinstance(synapse, Delta):
        i_alpha_fixed_point = 2 ** 12
    else:
        raise NotImplementedError(f"NetX doesn't support "
                                  f"{type(synapse).__name__} synapses")

    logger.info(f"\tNeuron synaptic decay {i_alpha_fixed_point}")
    neuron_group.create_dataset("iDecay", data=i_alpha_fixed_point,
                                dtype="i4")

    # If neuron model should be implemented as CUBA
    if isinstance(neuron, (LeakyIntegrateFire, LeakyIntegrate)):
        neuron_group.create_dataset("type", data="CUBA")

        # Get tau
        tau_mem = _check_param(neuron.tau_mem, shape)

        # Calculate decays, convert to fixed point and store in dataset
        v_alpha = 1.0 - np.exp(-dt / tau_mem)
        v_alpha_fixed_point = _to_fixed_point(v_alpha, 2 ** 12,
                                              "i4")

        # If neuron spikes
        if isinstance(neuron, LeakyIntegrateFire):
            # Check reset mechanism is compatible
            if neuron.relative_reset:
                logger.warning("Lava CUBA model does not support relative "
                               "membrane voltage reset")

            # Scale its threshold by quantisation scale
            v_thresh = _check_param(neuron.v_thresh, shape)
            v_thresh_fixed_point = _to_fixed_point(
                v_thresh, 1.0 / quant_scale, "i4")
            
            # If neuron has no refractory time, use zero
            if neuron.tau_refrac is None:
                ref_delay = 0
            # Otherwise, convert to timesteps
            # **TODO** check integrate_during_refrac behaviour
            else:
                ref_delay = np.round(
                    _check_param(neuron.tau_refrac, shape) / dt)

        # Otherwise, set extremely high threshold
        # **YUCK** this also isn't ideal
        else:
            v_thresh_fixed_point = 2**16
            ref_delay = 0

        logger.info(f"\tNeuron membrane decay {v_alpha_fixed_point}")
        logger.info(f"\tNeuron threshold {v_thresh_fixed_point}")
        logger.info(f"\tNeuron refractory timesteps {ref_delay}")

        # Add datasets
        neuron_group.create_dataset("vDecay", data=v_alpha_fixed_point, 
                                    dtype="i4")
        neuron_group.create_dataset("vThMant", data=v_thresh_fixed_point,
                                    dtype="i4")
        neuron_group.create_dataset("refDelay", data=ref_delay, dtype="i4")
    else:
        raise NotImplementedError(f"NetX doesn't support "
                                  f"{type(neuron).__name__} neurons")
    

def _export_feedfoward(layer_group: h5py.Group, pop: Population,
                       dt: float, num_weight_bits: int,
                       quant_percentile: float):
    # Check there's only one incoming connection
    if len(pop.incoming_connections) != 1:
        raise NotImplementedError("NetX does not currently support "
                                  "architectures with 'skip' connections")

    con = pop.incoming_connections[0]()
    logger.info(f"\tFeedforward {con.name}")

    # Check weight is an array
    if not is_value_array(con.connectivity.weight):
        raise RuntimeError("Before exporting to NetX, weights "
                           "should be loaded from checkpoints")

    # If connectivity is dense
    if isinstance(con.connectivity, Dense):
        # Populate layer group
        layer_group.create_dataset("inFeatures", dtype="i4")
        layer_group.create_dataset("outFeatures", dtype="i4")
        layer_group.create_dataset("type", data="dense", dtype="S10")
    else:
        raise NotImplementedError(f"NetX doesn't support "
                                  f"{type(con.connectivity).__name__} "
                                  f"connectivity")

    # Due to implementation details, weights need scaling to 
    # match some mlGeNN neuron types so calculate first
    weight_scale = _get_weight_scale(pop.neuron, con.synapse,
                                     pop.shape, dt)

    # Convert weights to NetX format
    num_src = np.prod(con.source().shape)
    num_trg = np.prod(con.target().shape)
    quant_scale, quant_weights = _get_netx_weight(
        [(con.connectivity.weight, num_src, num_trg)], 
        num_weight_bits, quant_percentile, weight_scale)

    # Create dataset
    layer_group.create_dataset("weight", data=quant_weights, dtype="f4")
    
    # Add delays
    delays = _get_netx_delays(con.connectivity.delay, num_src, num_trg)
    if delays is not None:
        layer_group.create_dataset("delay", data=delays, dtype="i4")
    
    # Export neuron model
    _export_neuron(layer_group, pop.shape, dt, quant_scale,
                   pop.neuron, con.synapse)


def _export_recurrent(layer_group: h5py.Group, pop: Population,
                      dt: float, num_weight_bits: int,
                      quant_percentile: float):
    # Check there's only one incoming connection
    if len(pop.incoming_connections) != 2:
        raise NotImplementedError("NetX does not currently support "
                                  "architectures with 'skip' connections")

    # Determine which incoming connection is 
    # recurrent and which feedforward
    if pop.incoming_connections[0]().source() == pop:
        rec_con = pop.incoming_connections[0]()
        ff_con = pop.incoming_connections[1]()
    else:
        ff_con = pop.incoming_connections[0]()
        rec_con = pop.incoming_connections[1]()

    logger.info(f"\tRecurrent in={ff_con.name} rec={rec_con.name}")

    # Check that both connections have dense connectivity
    if (not isinstance(rec_con.connectivity, Dense) 
        or not isinstance(ff_con.connectivity, Dense)):
            raise NotImplementedError("NetX only supports densely connected "
                                      "recurrent layers")

    # Check weights are all arrays
    if (not is_value_array(rec_con.connectivity.weight)
        or not is_value_array(ff_con.connectivity.weight)):
            raise RuntimeError("Before exporting to NetX, weights "
                               "should be loaded from checkpoints")

    # **TODO** check synapse models match

    # Populate layer group
    layer_group.create_dataset("inFeatures", dtype="i4")
    layer_group.create_dataset("outFeatures", dtype="i4")
    layer_group.create_dataset("type", data="dense_rec", dtype="S10")

    # Due to implementation details, weights need scaling to 
    # match some mlGeNN neuron types so calculate first
    weight_scale = _get_weight_scale(pop.neuron, rec_con.synapse,
                                     pop.shape, dt)

    # Convert weights to NetX format
    num_src = np.prod(ff_con.source().shape)
    num_trg = np.prod(ff_con.target().shape)
    quant_scale, quant_rec_weights, quant_ff_weights = _get_netx_weight(
        [(rec_con.connectivity.weight, num_trg, num_trg),
         (ff_con.connectivity.weight, num_src, num_trg),], 
        num_weight_bits, quant_percentile, weight_scale)

    # Create datasets
    layer_group.create_dataset("weight", data=quant_ff_weights, dtype="f4")
    layer_group.create_dataset("weight_rec", data=quant_rec_weights,
                               dtype="f4")

    
    # Add feedforward delays
    delays = _get_netx_delays(ff_con.connectivity.delay, num_src, num_trg)
    if delays is not None:
        layer_group.create_dataset("delay", data=delays, dtype="i4")
    
    # Add recurrent delays
    delays_rec = _get_netx_delays(rec_con.connectivity.delay, num_trg, num_trg)
    if delays_rec is not None:
        layer_group.create_dataset("delay_rec", data=delays_rec, dtype="i4")
    
    # Export neuron model
    _export_neuron(layer_group, pop.shape, dt, quant_scale,
                   pop.neuron, rec_con.synapse)



def export(path: str, inputs, outputs, dt: float = 1.0, 
           num_weight_bits: int = 8, quant_percentile: float = 99.0):
    """
    Export mlGeNN network to NetX
    
    Args:
        path:               Path to NetX file to generate
        inputs:             Sequence of mlGeNN :class:`ml_genn.Population`
                            or :class:`ml_genn.InputLayer` objects where 
                            input is applied
        outputs:            Sequence of mlGeNN :class:`ml_genn.Population`
                            or :class:`ml_genn.Layer` objects where 
                            output is read
        dt:                 Desired simulation timestep in milliseconds 
                            for Lava simulation
        num_weight_bits:    Number of bits to quantise synaptic weights to
        quant_percentile:   What percentage of weight distribution should
                            quantised format aim to encompass
    """
    # Build Directed Acyclical Graph 
    # from lists of inputs and outputs
    dag = _get_network_dag(inputs, outputs)

    with h5py.File(path, "w") as f:
        # Loop through layers in DAG
        for i, (pop, rec) in enumerate(dag):
            logger.info(f"{i}:")
        
            # Create layer group
            layer_group = f.create_group(f"/layer/{i}")
            
            # Add shape dataset
            layer_group.create_dataset("shape", data=pop.shape, dtype="i8")
            
            # If this is the first population in DAG i.e. an input layer
            # **TODO** non-spiking inputs need handling differently
            if i == 0:
                assert len(pop.incoming_connections) == 0
                logger.info("\tInput")
                # Set layer type to input
                layer_group.create_dataset("type", data="input", dtype="S10")
            # Otherwise, if layer is recurrent
            elif rec:
                _export_recurrent(layer_group, pop, dt, num_weight_bits,
                                  quant_percentile)
            # Otherwise, it must be feedforward
            else:
                _export_feedfoward(layer_group, pop, dt, num_weight_bits,
                                   quant_percentile)


__all__ = ["export"]
