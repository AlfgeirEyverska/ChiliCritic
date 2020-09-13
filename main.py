from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import csv
from os import path


# Load data from csv file
def load_data(filename):
    inputs = list()
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)

        for r in csv_reader:
            inputs.append(r)
    return inputs


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 1)
        # self.fc4 = nn.Linear(12, 1)
        # self.fc5 = nn.Linear(4, 1)

    def forward(self, x):
        # x = self.fc5(torch.tanh(self.fc4(torch.tanh(self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(x)))))))))
        # x = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
        x = self.fc3(torch.sigmoid(self.fc2(torch.sigmoid(self.fc1(x)))))
        # x = self.fc4(torch.sigmoid(self.fc3(torch.sigmoid(self.fc2(torch.sigmoid(self.fc1(x)))))))
        return x


# Load data for training, validating, and predicting
input_data = load_data('input_data.csv')
raw_validation_data = load_data('validation_data.csv')
raw_training_data = load_data('training_data.csv')

validation_data = list()
for row in raw_validation_data:
    validation_data.append((row[0:-1], row[-1]))

training_data = list()
for row in raw_training_data:
    training_data.append((row[0:-1], row[-1]))

# Setting up the neural net
net = Net()
loss_fn = nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.01)  # , momentum=0.5

# Naming scheme
# ChiliCritic_LossFN_ActivationFN_HiddenLayers_Size
nn_path = 'NeuralNets/ChiliCritic_L1_Sigmoid_1_12_725'


if path.exists(nn_path):
    net = torch.load(nn_path)
    net.eval()
else:

    # Training the neural net
    for epoch in range(725):
        # i is a counter, dat represents the row in the data
        for i, dat in enumerate(training_data):
            # X represents the input data, Y represents the actual output
            X, Y = iter(dat)
            X, Y = Variable(torch.FloatTensor(X), requires_grad=True), Variable(torch.FloatTensor([Y]), requires_grad=False)

            optimizer.zero_grad()

            outputs = net(X)

            loss = loss_fn(outputs, Y)
            loss.backward()

            optimizer.step()

            # if i % 5 == 0:
            print(f'    Epoch {epoch} --- Loss: {loss.data.item()}')
            print(f'prediction: {outputs[0]}  actual: {Y[0]}')
    # Save NN so it doesn't need to be recomputed
    torch.save(net, nn_path)


print(nn_path.split('/')[-1])
# Testing the neural net
for i, dat in enumerate(validation_data):
    # X represents the input data, Y represents the actual output
    X, Y = iter(dat)
    X, Y = Variable(torch.FloatTensor(X), requires_grad=True), Variable(torch.FloatTensor([Y]), requires_grad=False)

    outputs = net(X)

    loss = loss_fn(outputs, Y)

    prediction = outputs[0]
    actual = Y[0]
    difference = round(float(Y[0] - outputs[0]), 1)
    pd = round(float(difference/actual*100), 1)
    print(f'prediction: {prediction}  actual: {actual}  difference: {difference}  percent difference: {pd}%')

# Guesses for the unknowns
guesses = list()
for i, dat in enumerate(input_data):
    # X represents the input data, Y represents the actual output
    X = dat
    X = Variable(torch.FloatTensor(X), requires_grad=False)

    outputs = net(X)

    prediction = round(float(outputs[0]))
    guesses.append(prediction)
    print(f'prediction: {prediction}')

print(sum(guesses)/len(guesses))
