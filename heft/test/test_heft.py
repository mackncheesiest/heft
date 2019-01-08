from heft import heft
from pytest import approx
from types import SimpleNamespace
import numpy as np


def test_canonical_graph():
    expected_proc_sched = {
        0: [heft.ScheduleEvent(task=1, start=27.0, end=40.0, proc=0), 
            heft.ScheduleEvent(task=7, start=57.0, end=62.0, proc=0)],
        1: [heft.ScheduleEvent(task=3, start=18.0, end=26.0, proc=1), 
            heft.ScheduleEvent(task=5, start=26.0, end=42.0, proc=1), 
            heft.ScheduleEvent(task=8, start=56.0, end=68.0, proc=1), 
            heft.ScheduleEvent(task=9, start=73.0, end=80.0, proc=1)],
        2: [heft.ScheduleEvent(task=0, start=0, end=9.0, proc=2), 
            heft.ScheduleEvent(task=2, start=9.0, end=28.0, proc=2), 
            heft.ScheduleEvent(task=4, start=28.0, end=38.0, proc=2), 
            heft.ScheduleEvent(task=6, start=38.0, end=49.0, proc=2)]
    }
    expected_task_sched = {
        0: heft.ScheduleEvent(task=0, start=0, end=9.0, proc=2),
        1: heft.ScheduleEvent(task=1, start=27.0, end=40.0, proc=0),
        2: heft.ScheduleEvent(task=2, start=9.0, end=28.0, proc=2),
        3: heft.ScheduleEvent(task=3, start=18.0, end=26.0, proc=1),
        4: heft.ScheduleEvent(task=4, start=28.0, end=38.0, proc=2),
        5: heft.ScheduleEvent(task=5, start=26.0, end=42.0, proc=1),
        6: heft.ScheduleEvent(task=6, start=38.0, end=49.0, proc=2),
        7: heft.ScheduleEvent(task=7, start=57.0, end=62.0, proc=0),
        8: heft.ScheduleEvent(task=8, start=56.0, end=68.0, proc=1),
        9: heft.ScheduleEvent(task=9, start=73.0, end=80.0, proc=1)
    }
    expected_matrix_sched = np.array([
        [2, 0],
        [0, 0],
        [2, 1],
        [1, 0],
        [2, 2],
        [1, 1],
        [2, 3],
        [0, 1],
        [1, 2],
        [1, 3]
    ])

    dag = heft.readDagMatrix('test/canonicalgraph_task_connectivity.csv')
    comm = heft.readCsvToNumpyMatrix('test/canonicalgraph_resource_BW.csv')
    comp = heft.readCsvToNumpyMatrix('test/canonicalgraph_task_exe_time.csv')
    proc_sched, task_sched, matrix_sched = heft.schedule_dag(dag, communication_matrix=comm, computation_matrix=comp, proc_schedules=None, time_offset=0, relabel_nodes=True)

    assert proc_sched == expected_proc_sched
    assert task_sched == expected_task_sched
    assert np.array_equal(matrix_sched, expected_matrix_sched)

def test_canonical_graph_twice():
    expected_proc_sched = {
        0: [heft.ScheduleEvent(task=10, start=10.0, end=24.0, proc=0),
            heft.ScheduleEvent(task=1, start=27.0, end=40.0, proc=0),
            heft.ScheduleEvent(task=12, start=40.0, end=51.0, proc=0),
            heft.ScheduleEvent(task=7, start=57.0, end=62.0, proc=0),
            heft.ScheduleEvent(task=14, start=62.0, end=74.0, proc=0),
            heft.ScheduleEvent(task=16, start=74.0, end=81.0, proc=0)],
        1: [heft.ScheduleEvent(task=3, start=18.0, end=26.0, proc=1), 
            heft.ScheduleEvent(task=5, start=26.0, end=42.0, proc=1), 
            heft.ScheduleEvent(task=13, start=42.0, end=50.0, proc=1),
            heft.ScheduleEvent(task=8, start=56.0, end=68.0, proc=1), 
            heft.ScheduleEvent(task=9, start=73.0, end=80.0, proc=1),
            heft.ScheduleEvent(task=18, start=87.0, end=99.0, proc=1),
            heft.ScheduleEvent(task=19, start=102.0, end=109.0, proc=1)],
        2: [heft.ScheduleEvent(task=0, start=0, end=9.0, proc=2), 
            heft.ScheduleEvent(task=2, start=9.0, end=28.0, proc=2), 
            heft.ScheduleEvent(task=4, start=28.0, end=38.0, proc=2), 
            heft.ScheduleEvent(task=6, start=38.0, end=49.0, proc=2),
            heft.ScheduleEvent(task=11, start=49.0, end=67.0, proc=2),
            heft.ScheduleEvent(task=15, start=67.0, end=76.0, proc=2),
            heft.ScheduleEvent(task=17, start=77.0, end=91.0, proc=2)]
    }
    expected_task_sched = {
        0: None,
        1: None,
        2: None,
        3: None,
        4: None,
        5: None,
        6: None,
        7: None,
        8: None,
        9: None,
        10: heft.ScheduleEvent(task=10, start=10.0, end=24.0, proc=0),
        11: heft.ScheduleEvent(task=11, start=49.0, end=67.0, proc=2),
        12: heft.ScheduleEvent(task=12, start=40.0, end=51.0, proc=0),
        13: heft.ScheduleEvent(task=13, start=42.0, end=50.0, proc=1),
        14: heft.ScheduleEvent(task=14, start=62.0, end=74.0, proc=0),
        15: heft.ScheduleEvent(task=15, start=67.0, end=76.0, proc=2),
        16: heft.ScheduleEvent(task=16, start=74.0, end=81.0, proc=0),
        17: heft.ScheduleEvent(task=17, start=77.0, end=91.0, proc=2),
        18: heft.ScheduleEvent(task=18, start=87.0, end=99.0, proc=1),
        19: heft.ScheduleEvent(task=19, start=102.0, end=109.0, proc=1)
    }
    expected_matrix_sched = np.array([
        [2, 0],
        [0, 1],
        [2, 1],
        [1, 0],
        [2, 2],
        [1, 1],
        [2, 3],
        [0, 3],
        [1, 3],
        [1, 4],
        [0, 0],
        [2, 4],
        [0, 2],
        [1, 2],
        [0, 4],
        [2, 5],
        [0, 5],
        [2, 6],
        [1, 5],
        [1, 6]
    ])

    dag = heft.readDagMatrix('test/canonicalgraph_task_connectivity.csv')
    comm = heft.readCsvToNumpyMatrix('test/canonicalgraph_resource_BW.csv')
    comp = heft.readCsvToNumpyMatrix('test/canonicalgraph_task_exe_time.csv')
    proc_sched, task_sched, matrix_sched = heft.schedule_dag(dag, communication_matrix=comm, computation_matrix=comp, proc_schedules=None, time_offset=0, relabel_nodes=True)
    proc_sched, task_sched, matrix_sched = heft.schedule_dag(dag, communication_matrix=comm, computation_matrix=comp, proc_schedules=proc_sched, time_offset=10, relabel_nodes=True)

    assert proc_sched == expected_proc_sched
    assert task_sched == expected_task_sched
    assert np.array_equal(matrix_sched, expected_matrix_sched)

def test_random_graph():
    expected_proc_sched = {
        0: [heft.ScheduleEvent(task=1, start=approx(65.9566, 0.001), end=approx(95.881, 0.001), proc=0), 
            heft.ScheduleEvent(task=3, start=approx(95.881, 0.001), end=approx(107.4534, 0.001), proc=0), 
            heft.ScheduleEvent(task=7, start=approx(109.8242, 0.001), end=approx(130.6393, 0.001), proc=0)],
        1: [heft.ScheduleEvent(task=2, start=approx(61.2566, 0.001), end=approx(103.0477, 0.001), proc=1), 
            heft.ScheduleEvent(task=8, start=approx(107.0242, 0.001), end=approx(124.2442, 0.001), proc=1), 
            heft.ScheduleEvent(task=6, start=approx(124.2442, 0.001), end=approx(136.3771, 0.001), proc=1)],
        2: [heft.ScheduleEvent(task=0, start=0, end=approx(56.9566, 0.001), proc=2), 
            heft.ScheduleEvent(task=4, start=approx(56.9566, 0.001), end=approx(105.0242, 0.001), proc=2), 
            heft.ScheduleEvent(task=5, start=approx(105.0242, 0.001), end=approx(168.2705, 0.001), proc=2), 
            heft.ScheduleEvent(task=9, start=approx(168.2705, 0.001), end=approx(184.1923, 0.001), proc=2)]
    }
    expected_task_sched = {
        0: heft.ScheduleEvent(task=0, start=0, end=approx(56.9566, 0.001), proc=2),
        1: heft.ScheduleEvent(task=1, start=approx(65.9566, 0.001), end=approx(95.881, 0.001), proc=0),
        2: heft.ScheduleEvent(task=2, start=approx(61.2566, 0.001), end=approx(103.0477, 0.001), proc=1),
        3: heft.ScheduleEvent(task=3, start=approx(95.881, 0.001), end=approx(107.4534, 0.001), proc=0),
        4: heft.ScheduleEvent(task=4, start=approx(56.9566, 0.001), end=approx(105.0242, 0.001), proc=2),
        5: heft.ScheduleEvent(task=5, start=approx(105.0242, 0.001), end=approx(168.2705, 0.001), proc=2),
        6: heft.ScheduleEvent(task=6, start=approx(124.2442, 0.001), end=approx(136.3771, 0.001), proc=1),
        7: heft.ScheduleEvent(task=7, start=approx(109.8242, 0.001), end=approx(130.6393, 0.001), proc=0),
        8: heft.ScheduleEvent(task=8, start=approx(107.0242, 0.001), end=approx(124.2442, 0.001), proc=1),
        9: heft.ScheduleEvent(task=9, start=approx(168.2705, 0.001), end=approx(184.1923, 0.001), proc=2)
    }
    expected_matrix_sched = np.array([
        [2, 0],
        [0, 0],
        [1, 0],
        [0, 1],
        [2, 1],
        [2, 2],
        [1, 2],
        [0, 2],
        [1, 1],
        [2, 3]
    ])
    
    dag = heft.readDagMatrix('test/randomgraph_task_connectivity.csv')
    comm = heft.readCsvToNumpyMatrix('test/randomgraph_resource_BW.csv')
    comp = heft.readCsvToNumpyMatrix('test/randomgraph_task_exe_time.csv')
    proc_sched, task_sched, matrix_sched = heft.schedule_dag(dag, communication_matrix=comm, computation_matrix=comp, proc_schedules=None, time_offset=0, relabel_nodes=True)

    assert proc_sched == expected_proc_sched
    assert task_sched == expected_task_sched
    assert np.array_equal(matrix_sched, expected_matrix_sched)

def test_mean_ranku():
    expected_ranku = [
        108,
        77,
        80,
        80,
        69,
        63.333,
        42.667,
        35.667,
        44.333,
        14.667
    ]

    dag = heft.readDagMatrix('test/canonicalgraph_task_connectivity.csv')
    comm = heft.readCsvToNumpyMatrix('test/canonicalgraph_resource_BW.csv')
    comp = heft.readCsvToNumpyMatrix('test/canonicalgraph_task_exe_time.csv')

    _self = {
        'computation_matrix': comp,
        'communication_matrix': comm,
        'task_schedules': {},
        'proc_schedules': {},
        'numExistingJobs': 0,
        'time_offset': 0,
        'root_node': None
    }
    _self = SimpleNamespace(**_self)

    heft._compute_ranku(_self, dag)

    for node in dag.nodes():
        print(dag.nodes()[node]['ranku'])

    for rankidx in range(len(expected_ranku)):
        assert dag.nodes()[rankidx]['ranku'] == approx(expected_ranku[rankidx], 0.001)