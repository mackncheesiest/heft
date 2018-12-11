from enum import Enum, auto

import networkx as nx

class MergeMethod(Enum):
    COMMON_ENTRY_EXIT = auto()
    LEVEL_BASED = auto()
    #TODO: Should we implement the Alternating DAGs approach?
    #ALTERNATING_DAGS = auto()
    RANKING_BASED = auto()

def _common_entry_exit_merge(*args):
    # Forces distinct integer node labels (no need to deal with relabeling nodes with a common name across both graphs)
    combined_dag = nx.algorithms.operators.all.disjoint_union_all(args)

    root_nodes = [node for node in combined_dag.nodes() if not any(True for _ in combined_dag.predecessors(node))]
    terminal_nodes = [node for node in combined_dag.nodes() if not any(True for _ in combined_dag.successors(node))]

    num_nodes = combined_dag.number_of_nodes()
    combined_dag.add_node(num_nodes)
    combined_dag.add_node(num_nodes+1)

    for node in root_nodes:
        combined_dag.add_edge(num_nodes, node)
        combined_dag[num_nodes][node]['weight'] = 0
    for node in terminal_nodes:
        combined_dag.add_edge(node, num_nodes+1)
        combined_dag[node][num_nodes+1]['weight'] = 0
    return combined_dag

def _level_based_merge(*args, depth=0):
    return

def _ranking_based_merge(*args):
    return

merge_methods = {
    MergeMethod.COMMON_ENTRY_EXIT: _common_entry_exit_merge,
    MergeMethod.LEVEL_BASED: _level_based_merge,
    MergeMethod.RANKING_BASED: _ranking_based_merge
}
def merge_dags(*args, merge_method=MergeMethod.COMMON_ENTRY_EXIT):
    return merge_methods.get(merge_method)(*args)