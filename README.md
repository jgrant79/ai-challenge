# Alion 2019 AI Challenge

This project is an effort to train a machine learning model to recognize higher-order digital logic functions in clouds of discrete logic components. Specifically, the logic components are expressed in terms of an adjacency matrix which allows for the machine learning model to consume it as a numerical model.

## Usage
The project is broken down into three main areas. 
1. Psuedo-Verilog patterns
  - In the `verilog` directory, there are subfolders for each higher-order function the model is being trained for. Each subdirectory contains multiple psuedo-verilog files which contain an implementation of that function in primitive gates.
2. ./generate-data.sh & ./src/makegraph.py
  - This python script (and bash script for invoking the python script) serve to read the patterns described above and generate a graph representation. Furthermore, it permutes the order of vertices in the graph to inflate the number of training samples. These graph representations are then saved in the `generated` folder as NumPy files.
3. ./src/trainmodel.py
  - This is the python script where model training occurs. This script accepts multiple graph files from the `generated` folder and trains a classifier to recognize the structure of the various digital logic functions. It currently saves a checkpoint of the model weights in the `generated/checkpoints` folder for each epoch, but use of these files has not been tested yet.
