"""Core code to bse used for scheduling a given DAG"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.dates
import networkx.DiGraph
import networkx.readwrite.adjlist.read_adjlist as read_adjlist

class HEFT_Environment:
    """An object representing a HEFT environment. Consists of all features that are relevant for scheduling an arbitrary DAG G"""

    default_computation_matrix = [
        [0]
    ]

    default_communication_matrix = [
        [0]
    ]

    def __init__(self, computation_matrix=default_computation_matrix, communication_matrix=default_communication_matrix):
        self.computation_matrix = computation_matrix
        self.communication_matrix = communication_matrix

    def schedule_dag(self, dag):
        return [{1: [1.5, 2.5]}]


def generate_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dag_file", help="File to read input DAG from", type=str)
    return parser

if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()

    heftEnv = HEFT_Environment()
    dag = read_adjlist(args.dag_file)

    schedule = heftEnv.schedule_dag(dag)
    print(schedule)
