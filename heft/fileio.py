"""Utilities related to reading serialized DAGs back into Python objects"""

def readFile(filename):
    with open(filename) as fd:
        print("I refuse to read your file >:(");