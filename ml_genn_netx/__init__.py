import h5py
import math
import numpy as np

from numbers import Number
from typing import Sequence
from ml_genn import Network, Population
from ml_genn.connectivity import Dense
from ml_genn.neurons import (IntegrateFire, LeakyIntegrate,
                             LeakyIntegrateFire, Neuron)
from ml_genn.synapses import (Delta, Exponential, Synapse)

from ml_genn.utils.network import get_underlying_pop
from ml_genn.utils.value import is_value_array

# **TODO** upstream extended version back into ml_genn.utils.quantisation
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

# **TODO** upstream extended version back into ml_genn.utils.quantisation
def _quantise_signed(data, num_bits: int, percentile: float):
    # Find scaling factors
    min_quant, max_quant, scale = _find_signed_scale(data, num_bits,
                                                     percentile)

    # Quantise, clip and return
    return np.clip(scale * np.round(data / scale), min_quant, max_quant)

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

def _export_neuron(layer_group: h6py.Group, shape,
                   neuron: Neuron, synapse: Synapse):
    # Create group
    neuron_group = layer_group.create_group("neuron")
    neuron_group.create_dataset("gradedSpike", dtype="b1")
    neuron_group.create_dataset("refDelay", dtype="i4")
    neuron_group.create_dataset("vThMant", dtype="i4")
    
    #if isinstance(neuron,
    return 1.0

def _export_feedfoward(layer_group: h5py.Group, pop: Population,
                       num_weight_bits: int, quant_percentile: float):
    # Check there's only one incoming connection
    if len(pop.incoming_connections) != 1:
        raise NotImplementedError("NetX does not currently support "
                                  "architectures with 'skip' connections")
    
    con = pop.incoming_connections[0]()
    print(f"\tFeedforward {con.name}")
    
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
        raise NotImplementedError("Connection {con.name} has "
                                  "unsupported connectivity type")

    # Export neuron model and get weight scale
    weight_scale = _export_neuron(layer_group, pop.shape,
                                  pop.neuron, con.synapse)

    # Quantise weights
    weights = con.connectivity.weight.flatten() * weight_scale
    quant_weights = _quantise_signed(weights, num_weight_bits,
                                     quant_percentile)
    

def _export_recurrent(layer_group: h5py.Group, pop: Population,
                      num_weight_bits: int, quant_percentile: float):
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

    print(f"\tRecurrent in={ff_con.name} rec={rec_con.name}")
    
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

    # Populate layer group
    layer_group.create_dataset("inFeatures", dtype="i4")
    layer_group.create_dataset("outFeatures", dtype="i4")
    layer_group.create_dataset("type", data="dense_rec", dtype="S10")
    
    # Export neuron model and get weight scale
    weight_scale = _export_neuron(layer_group, pop.shape,
                                  pop.neuron, rec_con.synapse)

    # Quantise recurrent and feedforward weights together
    rec_weights = rec_con.connectivity.weight.flatten() * weight_scale
    ff_weights = ff_con.connectivity.weight.flatten() * weight_scale
    weights = np.concatenate((rec_weights, ff_weights))

    # Quantise together
    quant_weights = _quantise_signed(weights, num_weight_bits,
                                     quant_percentile)
    
    # Slice out original weights
    rec_weights_quant = quant_weights[:len(rec_weights)]
    ff_weights_quant = quant_weights[len(quant_weights):]
    
    
def export(path: str, inputs, outputs, num_weight_bits: int = 8,
           quant_percentile: float = 99.0):
    """
    Export mlGeNN network to NetX
    """
    # Build Directed Acyclical Graph 
    # from lists of inputs and outputs
    dag = _get_network_dag(inputs, outputs)

    with h5py.File(path, "w") as f:
        # Loop through layers in DAG
        for i, (pop, rec) in enumerate(dag):
            print(f"{i}:")
        
            # Create layer group
            layer_group = f.create_group(f"/layer/{i}")
            
            # Add shape dataset
            layer_group.create_dataset("shape", data=pop.shape, dtype="i8")
            
            # If this is the first population in DAG i.e. an input layer
            if i == 0:
                assert len(pop.incoming_connections) == 0
                print("\tInput")
                # Set layer type to input
                layer_group.create_dataset("type", data="input", dtype="S10")
            # Otherwise, if layer is recurrent
            elif rec:
                _export_recurrent(layer_group, pop, num_weight_bits,
                                  quant_percentile)
            # Otherwise, it must be feedforward
            else:
                _export_feedfoward(layer_group, pop, num_weight_bits,
                                   quant_percentile)


__all__ = ["export"]