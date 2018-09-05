"""pytorch_lenet.py
PyTorch implementation of LeNet MNIST classifier from
http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
"""

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data/', one_hot=True)

class LeNet(nn.Module):
    """
    LeNet implementation from 
    http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
    
    in: n/a
    out: n/a
    """

    def __init__(self):
        super(LeNet, self).__init__()
        #Conv part
        self.pool = nn.MaxPool2d(kernel_size=2, 
                                   stride=2)
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(in_channels=1, 
                                out_channels=6,
                                kernel_size=5)

        self.conv_2 = nn.Conv2d(in_channels=6,
                                out_channels=16,
                                kernel_size=5)
        # Linear part
        self.linear_1 = nn.Linear(400,
                                  120)
        self.linear_2 = nn.Linear(120,
                                  84)
        self.out = nn.Linear(84, 
                             10)
        

    def forward(self, x):
        # conv 1
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.pool(x)

        # conv 2
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Linear part 1
        x = self.linear_1(x)
        x = self.relu(x)

        # Linear part 2
        x = self.linear_2(x)
        x = self.relu(x)

        # output layer
        x = self.out(x)

        return F.log_softmax(x)


"""
Step 0: Read and pre-process the data
"""
mnist = input_data.read_data_sets('../data/', 
                                  one_hot=True,
                                  reshape=False)

train_data = mnist.train.images
train_labels = mnist.train.labels
test_data = mnist.test.images
test_labels = mnist.test.labels
print(test_data.shape)

train_data = np.pad(train_data, ((0, 0),
                                 (2, 2),
                                 (2, 2),
                                 (0, 0)),
                                 'constant')
test_data = np.pad(test_data, ((0, 0),
                               (2, 2),
                               (2, 2),
                               (0, 0)),
                               'constant')

# Convert the data into tensors
train_data = torch.tensor(train_data,
                          device='cpu')\
                          .view(-1, 1, 32, 32)
train_labels = torch.tensor(train_labels,
                            device='cpu',
                            dtype=torch.long)
test_data = torch.tensor(test_data,
                         device='cpu')\
                         .view(-1, 1, 32, 32)
test_labels = torch.tensor(test_labels,
                           device='cpu',
                           dtype=torch.long)

print(train_data.size())

"""
Step 1: Define all the parameters
"""
lr = 0.001                  # irrelevant for adam
n_epochs = 10                # Number of epochs
batch_size = 100            # batch size

rows, cols = (32, 32)
n_classes = 10              # 10 digits. 

"""
Step 2: Pre-define the network structure and the input format with placeholders and 
layers.
"""

lenet = LeNet()

"""
Step 3: Define loss function, optimizer and the initialiser
object.
"""
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
# Note that the optimizer keeps track of the 
# graph of the network (mlp.parameters())
optimizer = torch.optim.Adam(lenet.parameters())


""" 
Step 4: Initialize the variables and commence training. 
"""
n_batches = mnist.train.num_examples // batch_size

for epoch in range(n_epochs):
    epoch_loss = 0
    for i in range(n_batches):
        # Have to resize the batches to meet torch specifications
        train_batch = train_data[i * batch_size : (i + 1) * batch_size,
                                        :, :, :]
        train_label_batch = train_labels[i * batch_size : (i + 1) * batch_size,
                                          :]
        # reset the gradient calculations each time it is used
        optimizer.zero_grad()

        # to get the output of the network we simply pass the batch 
        # through the net with a function call
        out = lenet(train_batch)
        loss = criterion(out, train_label_batch.argmax(1))

        # Backprop step
        loss.backward()
        optimizer.step()

        epoch_loss += loss
        # compute the average loss for all batches. 
    
    epoch_loss /= n_batches
    print("Epoch {} - Average batch loss: {}".format(epoch + 1, epoch_loss))

with torch.no_grad():
    out = lenet(test_data).argmax(1)
    acc = sum((out == test_labels.argmax(1)).numpy()) / 10000
    print(acc)