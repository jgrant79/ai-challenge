#!/usr/bin/env python3

import argparse
import csv
import numpy as np
import os
import random
import sys

maxVertices = 64
permutations = 10000

LABELS = [
        'unknown',
        'and',
        'or',
        'xor',
        'adder',
        ]
LABEL_CLASSES = dict(zip(LABELS, range(len(LABELS))))

def parseArgs():
    """
    Parse command line arguments
    """
    #Main argument parser
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('csv', metavar='CSV file', type=str, help='A CSV file describing a Verilog circuit')
    parser.add_argument('-o', '--output', metavar='<FILE>', dest='outputFile', type=str, required=True, help='Path to output file to create')
    parser.add_argument('-l', '--label', metavar='<Label>', dest='label', type=str, choices=LABELS, required=True, help='Label of the circuit represented')
    parser.add_argument('-p', '--perm', metavar='<# of Iterations>', type=int, dest='permutations', default=permutations, help='Number of graph permutations to generate')
    parser.add_argument('-v', '--vertices', metavar='<# of Vertices>', type=int, dest='numVertices', default=maxVertices, help='Number of vertices in matrix')

    return parser.parse_args()

class CSVFilter(object):
    def __init__(self, fp):
        self.fp = fp

    def __iter__(self):
        return self

    def __next__(self):
        line = self.fp.__next__().strip()
        if not line or line.startswith('#'):
            return next(self)
        return line

NODE_TYPES = [
        'IN',
        'OUT',
        'and',
        'mux2',
        'nand',
        'nor',
        'not',
        'or',
        'xnor',
        'xor',
        ]
NODE_CLASSES = dict(zip(NODE_TYPES, range(len(NODE_TYPES))))

class Vertex(object):
    def __init__(self, type, name):
        self.type = type
        if type not in NODE_TYPES:
            raise Exception(f"ERROR: Unknown Node type {type}")
        self.name = name
        self.include = 0
        self.inputs = []
        self.output = None
        self.id = None

    def setOutput(self, edge):
        if self.output:
            print(f'WARNING: Vertex {self.name} already has an output set. Previous output = {self.output.name}; New output = {edge.name}')
        self.output = edge
        edge.setDriver(self)

    def addInput(self, edge):
        if edge in self.inputs or edge.name in [i.name for i in self.inputs]:
            print(f'WARNING: Vertex {self.name} already has {edge.name} in input list. Not adding parallel edge.')
            return

        self.inputs.append(edge)
        edge.addLoad(self)

    def print(self):
        print(f'  Vertex {self.name} ({self.include}):')
        for i in self.inputs:
            print(f'    {i.name} ->')
        if self.output:
            print(f'    -> {self.output.name}')

class Edge(object):
    def __init__(self, name):
        self.name = name
        self.driver = None
        self.loads = []

    def setDriver(self, vertex):
        self.driver = vertex

    def addLoad(self, vertex):
        self.loads.append(vertex)

class Graph(object):
    def __init__(self):
        self.vertices = {}
        self.edges = {}

    def addVertex(self, type, name):
        if name in self.vertices:
            raise Exception(f"Already added vertex {name}.")
        v = Vertex(type, name)
        self.vertices[name] = v
        return v

    def getEdge(self, name):
        if name in self.edges:
            return self.edges[name]
        e = Edge(name)
        self.edges[name] = e
        return e

    def printGraph(self):
        print('Graph:')
        for vName, v in self.vertices.items():
            v.print()

    def generateMatrix(self, numVertices=64):
        ids = list(range(numVertices))
        random.shuffle(ids)
        
        for i, vName in enumerate(self.vertices):
            v = self.vertices[vName]
            id = ids[i]
            v.id = id

        matrix = [0]*numVertices
        for vName in self.vertices:
            v = self.vertices[vName]
            adj = 0
            if not v.output:
                continue
            for load in v.output.loads:
                adj |= (0x1 << load.id)
            matrix[v.id] = adj
        return matrix

    def printMatrix(self, matrix=None, numVertices=64):
        print('Matrix:')
        if not matrix:
            matrix = self.generateMatrix(numVertices)
        for i, vName in enumerate(self.vertices):
            v = self.vertices[vName]
            print(f'  Vertex {vName}: Id #{v.id}')
        for i, adj in enumerate(matrix):
            print(f'Id #{i:02}: {adj:016x}')

def addRow(row, g):
    type = row[1].strip()
    name = row[2].strip()
    v = g.addVertex(type, name)
    v.include = int(row[0])
    wireName = row[3].strip()
    if wireName:
        edge = g.getEdge(row[3].strip())
        v.setOutput(edge)
    for wireName in row[4:]:
        wireName = wireName.strip()
        if not wireName:
            continue
        edge = g.getEdge(wireName)
        v.addInput(edge)

def makeTrainingEntry(size, label, graph, matrix):
    entry = np.zeros(size)
    features = entry[:len(matrix)*2]
    labels = entry[len(matrix)*2:]

    #Features
    vertexTypes = features[:len(matrix)]
    for v in graph.vertices:
        vertexTypes[graph.vertices[v].id] = NODE_CLASSES[graph.vertices[v].type]
    features[len(matrix):] = matrix

    #Labels
    labels[0] = LABEL_CLASSES[label]
    for v in graph.vertices:
        if graph.vertices[v].include:
            labels[1+graph.vertices[v].id] = 1

    return entry

def main():
    args = parseArgs()

    if not os.path.exists(args.csv):
        print(f'ERROR: File {csvFile} does not exist.')
        return 1

    if not args.outputFile:
        print(f'ERROR: Output file (-o, --output) must be specified.')
        return 1

    g = Graph()
    with open(args.csv, 'r') as f:
        reader = csv.reader(CSVFilter(f))
        for row in reader:
            addRow(row, g)

    numModelInputs = args.numVertices*2
    numModelOutputs = 1+args.numVertices
    entrySize = numModelInputs+numModelOutputs
    with open(args.outputFile, 'wb') as f:
        data = np.zeros((args.permutations, entrySize))
        for i in range(args.permutations):
            m = g.generateMatrix(args.numVertices)
            data[i] = makeTrainingEntry(entrySize, args.label, g, m)
        np.save(f, data)

    return 0

if __name__ == '__main__':
    sys.exit(main())
