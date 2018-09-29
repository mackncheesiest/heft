"""Core code to be used for scheduling a given DAG"""

from collections import deque, namedtuple
from math import inf
from gantt import showGanttChart

import argparse
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

logger = logging.getLogger('heft')

ScheduleEvent = namedtuple('ScheduleEvent', 'task start end proc')

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
        #Task to ScheduleEvent
        self.task_schedules = {}
        #PE to list of ScheduleEvents
        self.proc_schedules = {}
        for i in range(1, len(self.computation_matrix)+1):
            self.task_schedules[i] = None
        for i in range(1, len(self.communication_matrix)+1):
            self.proc_schedules[i] = []

    def schedule_dag(self, dag):
        # Nodes with no successors cause the any expression to be empty
        terminal_node = [node for node in dag.nodes() if not any(True for _ in dag.successors(node))]
        assert len(terminal_node) == 1, f"Expected a single terminal node, found {len(terminal_node)}"
        terminal_node = terminal_node[0]

        logger.debug(""); logger.debug("====================== Performing Rank-U Computation ======================\n"); logger.debug("")
        self._compute_ranku(dag, terminal_node)

        logger.debug(""); logger.debug("====================== Computing EFT for each (task, processor) pair and scheduling in order of decreasing Rank-U ======================"); logger.debug("")
        sorted_nodes = sorted(dag.nodes(), key=lambda node: dag.nodes()[node]['ranku'], reverse=True)
        for node in sorted_nodes:
            minTaskSchedule = ScheduleEvent(node, inf, inf, -1)
            for proc in range(1, len(self.communication_matrix)+1):
                taskschedule = self._compute_eft(dag, node, proc)
                if (taskschedule.end < minTaskSchedule.end):
                    minTaskSchedule = taskschedule
            self.task_schedules[node] = minTaskSchedule
            self.proc_schedules[minTaskSchedule.proc].append(minTaskSchedule)
            self.proc_schedules[minTaskSchedule.proc] = sorted(self.proc_schedules[minTaskSchedule.proc], key=lambda schedule_event: schedule_event.start)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('\n')
                for proc, jobs in self.proc_schedules.items():
                    logger.debug(f"Processor {proc} has the following jobs:")
                    logger.debug(f"\t{jobs}")
                logger.debug('\n')
            for proc in range(1, len(self.proc_schedules) + 1):
                for job in range(len(self.proc_schedules[proc])-1):
                    first_end = self.proc_schedules[proc][job].end
                    second_start = self.proc_schedules[proc][job+1].start
                    assert first_end <= second_start, \
                    f"Jobs on a particular processor must finish before the next can begin, but one job ends at {first_end} and its successor starts at {second_start}"
        
        return self.proc_schedules, self.task_schedules
    
    def _compute_ranku(self, dag, terminal_node):
        """
        Uses a basic BFS approach to traverse upwards through the graph assigning ranku along the way
        """
        nx.set_node_attributes(dag, { terminal_node: np.mean(self.computation_matrix[terminal_node-1]) }, "ranku")
        visit_queue = deque(dag.predecessors(terminal_node))

        while visit_queue:
            node = visit_queue.pop()
            while self._node_can_be_processed(dag, node) is not True:
                node2 = visit_queue.pop() or None
                assert node2 != None, f"Node {node} cannot be processed, and there are no other nodes in the queue to process instead!"
                visit_queue.appendleft(node)
                node = node2

            logger.debug(f"Assigning ranku for node: {node}")
            max_successor_ranku = -1
            for succnode in dag.successors(node):
                logger.debug(f"\tLooking at successor node: {succnode}")
                val = float(dag[node][succnode]['weight']) + dag.nodes()[succnode]['ranku']
                if val > max_successor_ranku:
                    max_successor_ranku = val
            assert max_successor_ranku > 0, f"Expected maximum successor ranku to be greater than 0 but was {max_successor_ranku}"
            nx.set_node_attributes(dag, { node: np.mean(self.computation_matrix[node-1]) + max_successor_ranku }, "ranku")

            visit_queue.extendleft([prednode for prednode in dag.predecessors(node) if prednode not in visit_queue])
        
        logger.debug("")
        for node in dag.nodes():
            logger.debug(f"Node: {node}, Rank U: {dag.nodes()[node]['ranku']}")

    def _node_can_be_processed(self, dag, node):
        """
        Validates that a node is able to be processed in Rank U calculations. Namely, that all of its successors have their Rank U values properly assigned
        """
        for succnode in dag.successors(node):
            if 'ranku' not in dag.nodes()[succnode]:
                return False
        return True

    def _compute_eft(self, dag, node, proc):
        """
        Computes the EFT of a particular node if it were scheduled on a particular processor
        It does this by first looking at all predecessor tasks of a particular node and determining the earliest time a task would be ready for execution (ready_time)
        It then looks at the list of tasks scheduled on this particular processor and determines the earliest time (after ready_time) a given node can be inserted into this processor's queue
        """
        ready_time = 0
        logger.debug(f"Computing EFT for node {node} on processor {proc}")
        for prednode in list(dag.predecessors(node)):
            predjob = self.task_schedules[prednode]
            assert predjob != None, f"Predecessor nodes must be scheduled before their children, but node {node} has an unscheduled predecessor of {predjob}"
            logger.debug(f"\tLooking at predecessor node {prednode} with job {predjob} to determine ready time")
            ready_time_t = predjob.end + self.communication_matrix[predjob.proc-1, proc-1] * float(dag[predjob.task][node]['weight'])
            logger.debug(f"\tNode {prednode} can have its data routed to processor {proc} by time {ready_time_t}")
            if ready_time_t > ready_time:
                ready_time = ready_time_t
        if ready_time == 0:
            assert len(list(dag.predecessors(node))) is 0, f"Only nodes without predecessors should have a ready time of 0, but node {node} has predecessors {list(dag.predecessors(node))}"
        logger.debug(f"\tReady time determined to be {ready_time}")

        computation_time = self.computation_matrix[node-1, proc-1]
        job_list = self.proc_schedules[proc]
        for idx in range(len(job_list)):
            prev_job = job_list[idx]
            if idx == len(job_list)-1:
                job_start = max(ready_time, prev_job.end)
                min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
                break
            next_job = job_list[idx+1]
            #Start of next job - computation time == latest we can start in this window
            #Max(ready_time, previous job's end) == earliest we can start in this window
            #If there's space in there, schedule in it
            logger.debug(f"\tLooking to fit a job of length {computation_time} into a slot of size {next_job.start - max(ready_time, prev_job.end)}")
            if (next_job.start - computation_time) - max(ready_time, prev_job.end) >= 0:
                logger.debug("\tIt looks like we can pull it off!")
                job_start = max(ready_time, prev_job.end)
                min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
                break
        else:
            #For-else loop: the else executes if the for loop exits without break-ing, which in this case means the number of jobs on this processor are 0
            min_schedule = ScheduleEvent(node, ready_time, ready_time + computation_time, proc)
        logger.debug(f"\tFor node {node} on processor {proc}, the EFT is {min_schedule}")
        return min_schedule    

def readCsvToNumpyMatrix(csv_file):
    """
    Given an input file consisting of a comma separated list of numeric values with a single header row and header column, 
    this function reads that data into a numpy matrix and strips the top row and leftmost column
    """
    with open(csv_file) as fd:
        logger.debug(f"Reading the contents of {csv_file} into a matrix")
        contents = fd.read()
        contentsList = contents.split('\n')
        contentsList = list(map(lambda line: line.split(','), contentsList))
        contentsList = contentsList[0:len(contentsList)-1] if contentsList[len(contentsList)-1] == [''] else contentsList
        
        matrix = np.matrix(contentsList)
        matrix = np.delete(matrix, 0, 0) # delete the first row (entry 0 along axis 0)
        matrix = np.delete(matrix, 0, 1) # delete the first column (entry 0 along axis 1)
        matrix = matrix.astype(float)
        logger.debug(f"After deleting the first row and column of input data, we are left with this matrix:\n{matrix}")
        return matrix

def readDagMatrix(dag_file, show_dag=False):
    """
    Given an input file consisting of a connectivity matrix, reads and parses it into a networkx Directional Graph (DiGraph)
    """
    #TODO: Migrate this to use our newer CSV format
    matrix = readCsvToNumpyMatrix(dag_file)

    dag = nx.DiGraph(matrix)
    dag.remove_edges_from(
        # Remove all edges with weight of 0 since we have no placeholder for "this edge doesn't exist" in the input file
        [edge for edge in dag.edges() if dag.get_edge_data(*edge)['weight'] == '0.0']
    )
    # Change 0-based node labels to 1-based
    dag = nx.relabel_nodes(dag, dict(map(lambda node: (node, node+1), list(dag.nodes()))))

    if show_dag:
        nx.draw(dag, with_labels=True)
        plt.show()

    return dag

def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for finding HEFT schedules for given DAG task graphs")
    parser.add_argument("dag_file", help="File containing input DAG to be scheduled", type=str)
    parser.add_argument("-p", "--pe_connectivity_file", help="File containing connectivity/bandwidth information about PEs. Uses a default 3x3 matrix from Topcuoglu 2002 if none given.", type=str, default=None)
    parser.add_argument("-t", "--task_execution_file", help="File containing execution times of each task on each particular PE. Uses a default 10x3 matrix from Topcuoglu 2002 if none given.", type=str, default=None)
    parser.add_argument("-l", "--loglevel", help="The log level to be used in this module. Default: INFO", type=str, dest="loglevel", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    parser.add_argument("--showDAG", help="Switch used to enable display of the incoming task DAG", dest="showDAG", action="store_true")
    parser.add_argument("--showGantt", help="Switch used to enable display of the final scheduled Gantt chart", dest="showGantt", action="store_true")
    return parser

if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()

    logger.setLevel(logging.getLevelName(args.loglevel))
    consolehandler = logging.StreamHandler()
    consolehandler.setLevel(logging.getLevelName(args.loglevel))
    consolehandler.setFormatter(logging.Formatter("%(levelname)8s : %(name)16s : %(message)s"))
    logger.addHandler(consolehandler)

    heftArgs = {}
    if args.pe_connectivity_file:
        communication_matrix = readCsvToNumpyMatrix(args.pe_connectivity_file)
        heftArgs['communication_matrix'] = communication_matrix
    if args.task_execution_file:
        computation_matrix = readCsvToNumpyMatrix(args.task_execution_file)
        heftArgs['computation_matrix'] = computation_matrix
    heftEnv = HEFT_Environment(**heftArgs)

    dag = readDagMatrix(args.dag_file, args.showDAG)
    processor_schedules, _ = heftEnv.schedule_dag(dag)
    for proc, jobs in processor_schedules.items():
        logger.info(f"Processor {proc} has the following jobs:")
        logger.info(f"\t{jobs}")
    if args.showGantt:
        showGanttChart(processor_schedules)
