import h5py

from ml_genn import Network
from typing import Sequence

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
    
def export(path: str, inputs, outputs):
    """
    Export mlGeNN network to NetX
    """
    dag = _get_network_dag(inputs, outputs)
    
    print("DAG:")
    for d in dag:
        print(d[0].name, d[1])

    # "layer" group
    # each layer also a group with string name i.e. "0"
    # shape etc all datasets
    
    with h5py.File(path, "w") as f:
        pass
        
    

    
    