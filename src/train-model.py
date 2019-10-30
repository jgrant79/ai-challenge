#!/usr/bin/env python3

import argparse
import csv
import numpy as np
import os
import random
import sys
import tensorflow as tf

tf.enable_eager_execution()

def parseArgs():
    """
    Parse command line arguments
    """
    #Main argument parser
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('npys', metavar='NumPy training file', type=str, nargs='*', help='')
    parser.add_argument('-i', '--inputs', metavar='<# Inputs>', dest='numInputs', type=int, required=True, help='Number of model inputs')
    parser.add_argument('-c', '--classes', metavar='<# Classes>', dest='numClasses', type=int, required=True, help='Number of classes to classify as')
    parser.add_argument('-t', '--tests', metavar='<# Tests>', dest='numTests', type=int, required=True, help='Number of records to reserve for testing the model')
    parser.add_argument('-e', '--epochs', metavar='<# Epochs>', dest='numEpochs', type=int, default=1000, help='Number of epochs to train the model for')
    parser.add_argument('-b', '--batches', metavar='<Batch size>', dest='batchSize', type=int, default=32, help='Size of batch for each epoch')

    return parser.parse_args()

def main():
    args = parseArgs()
    print(args.npys)

    if not args.npys:
        print(f'ERROR: At least one training file is required')
        return 1

    for npy in args.npys:
        if not os.path.exists(npy):
            print(f'ERROR: File {npy} does not exist.')
            return 1

    graphData = []
    for npy in args.npys:
        graphData.append(np.load(npy))
        #print(graphData[-1].shape)

    if len(graphData) == 0:
        print(f'ERROR: No training data read from input files.')
        return 1

    training = np.concatenate(graphData, axis=0)
    print(training.shape)
    np.random.shuffle(training)
    training, test = np.vsplit(training, (training.shape[0]-args.numTests,))

    numFeatures = args.numInputs
    numClasses = args.numClasses
    numInclusionLabels = training.shape[1]-args.numInputs-1
    #features, classLabels, inclusionLabels = np.hsplit(training, (numFeatures,numFeatures+1))
    #print(features.shape)
    #print(classLabels.shape)
    #print(inclusionLabels.shape)

    classModel = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(numFeatures,)),  # input shape required
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(numClasses)
    ])

    inclusionModel = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(numFeatures,)),  # input shape required
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(numInclusionLabels)
    ])

    #predictions = model(features)
    #predictions[:5]
    #tf.nn.softmax(predictions[:5])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    scc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    bce = tf.keras.losses.BinaryCrossentropy()

    def loss(lossObj, model, x, y):
        y_ = model(x)
        return lossObj(y_true=y, y_pred=y_)

    def grad(lossObj, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(lossObj, model, inputs, targets)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    #numEpochs = min(args.numEpochs, training.shape[0]//args.batchSize)
    numBatches = training.shape[0]//args.batchSize
    batches = np.vsplit(training, [i*args.batchSize for i in range(1,numBatches)])
    #batches = np.vsplit(training, training.shape[0]//args.batchSize)

    #features, classLabels, inclusionLabels = np.hsplit(training, (numFeatures,numFeatures+1))

    for epochNum in range(args.numEpochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop
        for batchNum, batch in enumerate(batches):
            features, classLabels, inclusionLabels = np.hsplit(batch, (numFeatures,numFeatures+1))
            # Optimize the model
            loss_value, grads = grad(scc, classModel, features, classLabels)
            optimizer.apply_gradients(zip(grads, classModel.trainable_variables))

            # Track progress
            epoch_loss_avg(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            epoch_accuracy(classLabels, classModel(features))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epochNum % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epochNum, epoch_loss_avg.result(), epoch_accuracy.result()))

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
