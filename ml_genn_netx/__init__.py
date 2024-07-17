import h5py

from typing import Sequence
from ml_genn import Network
from ml_genn.connectivity import Dense


from ml_genn.utils.network import get_underlying_pop

# **TODO** upstream extended version back into ml_genn.utils.network
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

def _export_feedfoward(layer_group, pop):
    # Check there's only one incoming connection
    if len(pop.incoming_connections) != 1:
        raise NotImplementedError("mlGeNN NetX exporter does not currently "
                                  "support architectures requiring "
                                  "'concatenate' layers")
    
    con = pop.incoming_connections[0]()
    print(f"\tFeedforward {con.name}")

def _export_recurrent(layer_group, pop):
    # Check there's only one incoming connection
    if len(pop.incoming_connections) != 2:
        raise NotImplementedError("mlGeNN NetX exporter does not currently "
                                  "support architectures requiring "
                                  "'concatenate' layers")
    
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
    
def export(path: str, inputs, outputs):
    """
    Export mlGeNN network to NetX
    """
    # Build Directed Acyclical Graph 
    # from lists of inputs and outputs
    dag = _get_network_dag(inputs, outputs)
    

    # "layer" group
    # each layer also a group with string name i.e. "0"
    # shape etc all datasets
    # dense_rec
    # "neuron" is group
    #   
    # "type" is S10 string
    
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
                _export_recurrent(layer_group, pop)
            # Otherwise, it must be feedforward
            else:
                _export_feedfoward(layer_group, pop)
                
        
    

    
    
