"""Core code to be used for scheduling a given DAG"""

from collections import deque, namedtuple
from math import inf

import argparse
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

logger = logging.getLogger('heft')

ScheduleEvent = namedtuple('ScheduleEvent', 'task start end')

class HEFT_Environment:
    """
    An object representing a HEFT environment. Consists of all features that are relevant for scheduling an arbitrary DAG
    """

    """
    Default computation matrix - taken from Topcuoglu 2002 HEFT paper
    computation matrix: v x q matrix with v tasks and q PEs
    """
    W0 = np.matrix([
        [14, 16, 9],
        [13, 19, 18],
        [11, 13, 19],
        [13, 8, 17],
        [12, 13, 10],
        [13, 16, 9],
        [7, 15, 11],
        [5, 11, 14],
        [18, 12, 20],
        [21, 7, 16]
    ])

    """
    Default communication matrix - not listed in Topcuoglu 2002 HEFT paper
    communication matrix: q x q matrix with q PEs
    """
    C0 = np.matrix([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])

    def __init__(self, computation_matrix=W0, communication_matrix=C0):
        self.computation_matrix = computation_matrix
        self.communication_matrix = communication_matrix
        self.job_schedules = {}
        for i in range(1, len(self.computation_matrix)+1):
            self.job_schedules[i] = None

    def compute_ranku(self, dag, terminal_node):
        nx.set_node_attributes(dag, { terminal_node: np.mean(self.computation_matrix[terminal_node-1]) }, "ranku")
        visit_queue = deque(dag.predecessors(terminal_node))

        while visit_queue:
            node = visit_queue.pop()
            logger.debug("Assigning ranku for node: %s" % node)
            max_successor_ranku = -1
            for succnode in dag.successors(node):
                logger.debug('\tLooking at successor node: %s' % succnode)
                val = float(dag[node][succnode]['weight']) + dag.nodes()[succnode]['ranku']
                if val > max_successor_ranku:
                    max_successor_ranku = val
            assert max_successor_ranku > 0, "Expected maximum successor ranku to be greater than 0 but was %s" % max_successor_ranku
            nx.set_node_attributes(dag, { node: np.mean(self.computation_matrix[node-1]) + max_successor_ranku }, "ranku")

            visit_queue.extendleft([prednode for prednode in dag.predecessors(node) if prednode not in visit_queue])
        
        for node in dag.nodes():
            logger.info("Node: %s, Rank U: %s" % (node, dag.nodes()[node]['ranku']))

    def compute_eft(self, dag, node, proc):
        readytime = 0
        for prednode in dag.predecessors(node):
            assert self.job_schedules[prednode] != None, "Predecessor nodes must be scheduled before their children"
            readytimetemp = self.job_schedules[prednode].end + self.communication_matrix


    def schedule_dag(self, dag):
        # Nodes with no successors cause the any expression to be empty
        terminal_node = [node for node in dag.nodes() if not any(True for _ in dag.successors(node))]
        assert len(terminal_node) == 1, "Expected a single terminal node, found %s" % len(terminal_node)
        terminal_node = terminal_node[0]

        self.compute_ranku(dag, terminal_node)

        sorted_nodes = sorted(dag.nodes(), key=lambda node: dag.nodes()[node]['ranku'], reverse=True)
        for node in sorted_nodes:
            minTaskSchedule = ScheduleEvent(node, inf, inf)
            for proc in range(1, len(self.computation_matrix[node-1])+1):
                taskschedule = self.compute_eft(dag, node, proc)
                if (taskschedule.end < minTaskSchedule.end):
                    minTaskSchedule = taskschedule
            self.job_schedules[proc].append(minTaskSchedule)
            #for p in range(len(self.computation_matrix[node-1])):
            #TODO: Implement EFT(ni, pk) using insertion-based scheduling policy

def readDagMatrix(dag_file, show_dag=False):
    with open(dag_file) as fd:
        contents = fd.read()
        contentsList = contents.split('\n')
        contentsList = list(map(lambda line: line.split(' '), contentsList))
        
        dag = nx.DiGraph(np.matrix(contentsList))
        dag.remove_edges_from(
            # All edges with weight of 0
            [edge for edge in dag.edges() if dag.get_edge_data(*edge)['weight'] == '0']
        )
        # Change 0-based node labels to 1-based
        dag = nx.relabel_nodes(dag, dict(map(lambda node: (node, node+1), list(dag.nodes()))))

        if show_dag:
            nx.draw(dag, with_labels=True)
            plt.show()

        return dag

def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for finding HEFT schedules for given DAG task graphs")
    parser.add_argument("dag_file", help="File to read input DAG from", type=str)
    parser.add_argument("-l", "--loglevel", help="The log level to be used in this module. Default: INFO", type=str, dest="loglevel", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    parser.add_argument("--showGraph", help="Switch used to enable graph display with matplotlib", dest="showGraph", action="store_true")
    return parser

if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()

    logging.basicConfig(format="%(levelname)8s : %(name)16s : %(message)s", level=logging.getLevelName(args.loglevel))

    heftEnv = HEFT_Environment()
    dag = readDagMatrix(args.dag_file, args.showGraph)
    heftEnv.schedule_dag(dag)
