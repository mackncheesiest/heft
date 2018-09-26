# HEFT: Heterogeneous Earliest Finish Time

A Python 3.6+ implementation of a heuristic DAG scheduling approach from 

`H. Topcuoglu, S. Hariri and Min-You Wu, "Performance-effective and low-complexity task scheduling for heterogeneous computing," in IEEE Transactions on Parallel and Distributed Systems, vol. 13, no. 3, pp. 260-274, March 2002.`

[IEEE Explore Link](https://ieeexplore.ieee.org/document/993206)


### Installation
If you have conda installed, you can create an environment and fetch any necessary dependencies with

`conda env create -f heft.yml`

Otherwise, the main dependencies are:
- Python 3.6+ (uses literal string interpolation)
- Matplotlib
- Numpy
- Networkx

### Usage
Basic usage is given by `python heft/heft.py -h`

```
usage: heft.py [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--showDAG]
               [--showGantt]
               dag_file

A tool for finding HEFT schedules for given DAG task graphs

positional arguments:
  dag_file              File to read input DAG from

optional arguments:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The log level to be used in this module. Default: INFO
  --showDAG             Switch used to enable display of the incoming task DAG
  --showGantt           Switch used to enable display of the final scheduled
                        Gantt chart
```

