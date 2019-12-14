#!/usr/bin/env python3

import argparse
import csv
import datetime
import makegraph as mg
import numpy as np
import os
import random
import sys
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import activations
from tensorflow.python.framework.tensor_shape import Dimension

print("TensorFlow version: {}".format(tf.__version__))
tf.enable_eager_execution()

def parseArgs():
    """
    Parse command line arguments
    """
    #Main argument parser
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('npys', metavar='NumPy training file', type=str, nargs='*', help='')
    parser.add_argument('-i', '--inputs', metavar='<# Inputs>', dest='numInputs', type=int, required=True, help='Number of model inputs')
    parser.add_argument('-v', '--vertices', metavar='<# Vertices>', dest='numVertices', type=int, required=True, help='Number of graph vertices')
    parser.add_argument('-t', '--tests', metavar='<# Tests>', dest='numTests', type=int, required=True, help='Number of records to reserve for testing the model')
    parser.add_argument('-e', '--epochs', metavar='<# Epochs>', dest='numEpochs', type=int, default=1000, help='Number of epochs to train the model for')
    parser.add_argument('-c', '--checkpoint', metavar='<Directory>', dest='checkpointDir', type=str, help='Directory to save model checkpoints')

    return parser.parse_args()

def buildGraph(features, numVertices):
    """
    Convert dense graph to adjacency matrix
    """
    g = np.zeros((numVertices, numVertices))
    graphData,types = np.hsplit(features, (numVertices,))
    for i, v in enumerate(graphData):
        for j in range(numVertices):
            g[(i,j)] = (v >> j) & 0x1
    return g

class SpectralLayer(Layer):
    """
    Class to perform spectral graph convolution
    Reference: https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-62acf5b143d0

    Weights can be this layer disabled to make this layer behave like a pure input layer
    """
    def __init__(self, A, units, activation, useWeights=True, **kwargs):
        super(SpectralLayer, self).__init__(**kwargs)
        self.A = A
        self.units = units
        self.activation = activations.get(activation)
        self.useWeights = useWeights

    def build(self, input_shape):
        self.I = np.eye(*self.A.shape)
        A_hat = self.A + self.I
        D = np.sum(A_hat, axis=0)
        D_inv = D**-0.5
        D_inv = np.diag(D_inv)
        A_hat = D_inv * A_hat * D_inv
        z = K.zeros(A_hat.shape)
        self.A_hat = z + A_hat

        if self.useWeights:
            self.kernel = self.add_weight(
                'kernel',
                shape=(input_shape[2], self.units),
                trainable=True)
            print("kernel shape = ", self.kernel.shape)

        super(SpectralLayer, self).build(input_shape)

    def call(self, X):
        #When building the model, a three dimensional tensor is passed through
        # If so, grab the second dimension for dot product
        if X.shape.rank > 2:
            X_sub = X[1]
            X = X_sub

        #Apply the feature vector to the normalized graph
        aggregate = K.dot(self.A_hat, X)

        #Apply the weights if necessary and expand the batch dimension
        if self.useWeights:
            dot = K.dot(aggregate, self.kernel)
        else:
            dot = aggregate

        dot = tf.expand_dims(dot,0)
        return dot

def convertTraining(training, numVertices, numInputs):
    """
    Convert training data to categorical forms
    """
    converted = []
    for entry in training:
        inputs, labels = np.hsplit(entry, (numInputs,))
        labels = np.matrix(labels)
        vertexTypes, denseGraph = np.hsplit(inputs, (numVertices,))
        #Convert node types into categories
        vertexCategories = np.zeros((numVertices, len(mg.NODE_TYPES)))
        for i, t in enumerate(vertexTypes):
            vertexCategories[(i,t)] = 1

        #Separate labels into circuit types and node inclusion labels
        classLabels, inclusionLabels = np.hsplit(labels, (1,))
        inclusionLabels = inclusionLabels.reshape(inclusionLabels.shape[1], inclusionLabels.shape[0])

        #Convert dense graph into adjacency matrix
        graph = buildGraph(denseGraph, denseGraph.shape[0])

        #Add identity matrix to graph
        I = K.eye(graph.shape[0])
        features = K.concatenate((I, vertexCategories))
        spectral = SpectralLayer(graph, features.shape[1], activation=tf.nn.relu, input_shape=features.shape, useWeights=False)
        converted.append((graph, spectral, features, classLabels))
    return converted
    

def train(training, numFeatures, numVertices, numEpochs, checkpointDir=None):
    """
    Train the model
    """
    data = convertTraining(training, numVertices, numFeatures)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    scc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    bce = tf.keras.losses.BinaryCrossentropy()

    def loss(model, features, labels):
        preds = model(features)
        lossValue = scc(y_true=labels, y_pred=preds)
        return lossValue

    def grad(model, features, labels):
        with tf.GradientTape() as tape:
            lossValue = loss(model, features, labels)
        return lossValue, tape.gradient(lossValue, model.trainable_variables)

    #Core model to be trained by all graphs in training set
    coreModel = tf.keras.Sequential([
        tf.keras.layers.Reshape((numVertices*(numVertices+len(mg.NODE_TYPES)),), input_shape=(1, numVertices, numVertices+len(mg.NODE_TYPES))),
        tf.keras.layers.Dense(numVertices, activation=tf.nn.relu),
        tf.keras.layers.Dense(numVertices//2, activation=tf.nn.relu),
        tf.keras.layers.Dense(len(mg.LABELS))
    ])

    for epochNum in range(numEpochs):
        epochStart = datetime.datetime.today()
        print("Starting Epoch {:03d} at {:s}:".format(epochNum, epochStart.strftime("%H:%M:%S")))
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop
        for (i, (graph, spectral, features, labels)) in enumerate(data):
            if i != 0 and i % 500 == 0:
                print("    Graph #{:d}".format(i))

            model = tf.keras.Sequential()
            model.add(spectral)
            model.add(coreModel)
            model.compile(optimizer, scc, metrics=['accuracy'])

            loss_value, grads = grad(model, features, labels)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            #Add current loss
            epoch_loss_avg(loss_value)

            # Compare predicted label to actual label
            preds = model(features)
            epoch_accuracy(labels, preds)

        if epochNum % 1 == 0:
            epochEnd = datetime.datetime.today()
            epochDuration = epochEnd - epochStart
            print("  Loss: {:.3f}, Accuracy: {:.3%}, Duration: {:d}s".format(epoch_loss_avg.result(), epoch_accuracy.result(), int(epochDuration.total_seconds())))
            if checkpointDir:
                checkpointName = "check.{:d}".format(epochNum)
                checkpointPath = os.path.join(checkpointDir, checkpointName)
                coreModel.save_weights(checkpointPath)


def main():
    args = parseArgs()
    #print(args.npys)

    if not args.npys:
        print(f'ERROR: At least one training file is required')
        return 1

    for npy in args.npys:
        if not os.path.exists(npy):
            print(f'ERROR: File {npy} does not exist.')
            return 1

    graphData = []
    print("INFO: Training from {} files...".format(len(args.npys)))
    for npy in args.npys:
        graphData.append(np.load(npy))

    if len(graphData) == 0:
        print(f'ERROR: No training data read from input files.')
        return 1

    training = np.concatenate(graphData, axis=0).astype(np.int64)
    print("INFO: Training set shape = {}".format(training.shape))
    np.random.shuffle(training)
    training, test = np.vsplit(training, (training.shape[0]-args.numTests,))

    numFeatures = args.numInputs
    numVertices = args.numVertices
    numClasses = len(mg.LABELS)
    numInclusionLabels = training.shape[1]-args.numInputs-1

    train(training, numFeatures, numVertices, args.numEpochs, args.checkpointDir)
    return 0

    test_accuracy = tf.keras.metrics.Accuracy()

    for (x, y) in test_dataset:
        logits = model(x)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

    tf.stack([y,prediction],axis=1)

    return 0

if __name__ == '__main__':
    sys.exit(main())
