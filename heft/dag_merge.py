from enum import Enum, auto
from heft import heft
from math import inf
from types import SimpleNamespace

import networkx as nx

class MergeMethod(Enum):
    COMMON_ENTRY_EXIT = auto()
    RANKING_BASED = auto()
    #TODO: Should we implement the Alternating DAGs or Level-Based approaches?
    #LEVEL_BASED = auto()
    #ALTERNATING_DAGS = auto()

def _common_entry_exit_merge(*args, **kwargs):
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

def _level_based_merge(*args, **kwargs):
    return

def _ranking_based_merge(*args, **kwargs):
    if "computation_matrix" not in kwargs:
        raise RuntimeError(f"A computation matrix is required for ranking-based merge computation")
    if "communication_matrix" not in kwargs:
        raise RuntimeError(f"A communication matrix is required for ranking-based merge computation")

    _self = {
        'computation_matrix': kwargs["computation_matrix"],
        'communication_matrix': kwargs["communication_matrix"],
        'task_schedules': {},
        'proc_schedules': {},
        'numExistingJobs': 0,
        'time_offset': 0,
        'root_node': None
    }
    _self = SimpleNamespace(**_self)

    dag_max_ranks = []
    root_nodes = []

    for dag in args:
        # root_nodes.append(dag.nodes()[next(nx.topological_sort(dag))])
        root_nodes.append(next(nx.topological_sort(dag)))
        heft._compute_ranku(_self, dag)
        dag_max_ranks.append(dag.nodes()[root_nodes[-1]]['ranku'])

    combined_dag = nx.algorithms.operators.all.disjoint_union_all(args)
    # sorted_indices = sorted(range(len(dag_max_ranks)), key=lambda k: dag_max_ranks[k], reverse=True)
    sorted_indices = sorted(range(len(dag_max_ranks)), key=lambda k: dag_max_ranks[k], reverse=False)
    for sort_idx, dag_idx in enumerate(sorted_indices):
        if sort_idx == 0:
            continue
        prev_rank = args[dag_idx-1].nodes()[root_nodes[dag_idx-1]]['ranku']
        curr_rank = args[dag_idx].nodes()[root_nodes[dag_idx]]['ranku']
        
        closest_node = (-1, inf)
        for node in nx.topological_sort(args[dag_idx]):
            if abs((curr_rank - args[dag_idx].nodes()[node]['ranku']) - prev_rank) < closest_node[1]:
                closest_node = (node, abs((curr_rank - args[dag_idx].nodes()[node]['ranku']) - prev_rank))
        target_node = _get_index_with_offset(args, dag_idx, closest_node[0])
        terminal_node = [node for node in args[dag_idx-1].nodes() if not any(True for _ in args[dag_idx-1].successors(node))][0]
        combined_dag.add_edge(terminal_node, target_node)
        combined_dag[terminal_node][target_node]['weight'] = 0

    root_nodes = [node for node in combined_dag.nodes() if not any(True for _ in combined_dag.predecessors(node))]
    num_nodes = combined_dag.number_of_nodes()
    combined_dag.add_node(num_nodes)
    for node in root_nodes:
        combined_dag.add_edge(num_nodes, node)
        combined_dag[num_nodes][node]['weight'] = 0

    return combined_dag

def _get_index_with_offset(dag_list, dag_num, node_label):
    offset = 0
    for i in range(dag_num):
        offset = offset + len(dag_list[i])
    return offset + node_label

merge_methods = {
    MergeMethod.COMMON_ENTRY_EXIT: _common_entry_exit_merge,
    MergeMethod.LEVEL_BASED: _level_based_merge,
    MergeMethod.RANKING_BASED: _ranking_based_merge
}
def merge_dags(*args, merge_method=MergeMethod.COMMON_ENTRY_EXIT, **kwargs):
    return merge_methods.get(merge_method)(*args, **kwargs)