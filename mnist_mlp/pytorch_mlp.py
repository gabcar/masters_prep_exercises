import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data/', one_hot=True)


class Mlp(nn.Module):
    """
    Multi-layer perceptron implementation with pytorch
    
    Initialize the network with

    in: 

    input_dim [int]
    hidden_dim [array-like with ints]
    output_dim [int]
    """
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 output_dim):
        super(Mlp, self).__init__()
        # For efficient storage of hidden layers:
        self.hidden_layers = dict()

        # Define the input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        # Define the hidden layers
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers['h_{}'.format(i)] = nn.Linear(hidden_dims[i], hidden_dims[i + 1])

        # Define the output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        #ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Required method by pytorch
        """
        out = self.input_layer(x)
        out = self.relu(out)
        for i in range(len(self.hidden_layers)):
            out = self.hidden_layers['h_{}'.format(i)](out)
            out = self.relu(out)
        return F.log_softmax(self.output_layer(out))

"""
Step 0: Read and pre-process the data
"""
mnist = input_data.read_data_sets('../data/', one_hot=True)

train_data = mnist.train.images
train_labels = mnist.train.labels
test_data = mnist.test.images
test_labels = mnist.test.labels

# convert the data into tensors
train_data = torch.tensor(train_data, device='cpu')
train_labels = torch.tensor(train_labels, device='cpu', dtype=torch.long)
test_data = torch.tensor(test_data, device='cpu')
test_labels = torch.tensor(test_labels, device='cpu', dtype=torch.long)

print(train_labels.size())

"""
Step 1: Define all the parameters
"""
lr = 0.001                  # irrelevant for adam
n_epochs = 15               # Number of epochs
batch_size = 100            # batch size
# calculate the number of batches necessary
n_batches = mnist.train.num_examples // batch_size

n_features = 28 * 28        # mnist images are 28x28 but flattened
n_classes = 10              # 10 digits. 

hidden_dims = [64,          # Dimensions of hidden layers
               32,
               16]

"""
Step 2: Pre-define the network structure and the input format with placeholders and 
layers.
"""
# Follow definition for network structure
mlp = Mlp(input_dim=n_features,
          hidden_dims=hidden_dims,
          output_dim=n_classes)

"""
Step 3: Define loss function, optimizer and the initialiser
object.
"""
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
# Note that the optimizer keeps track of the 
# graph of the network (mlp.parameters())
optimizer = torch.optim.Adam(mlp.parameters())  

""" 
Step 4: Initialize the variables and commence training. 
"""
for epoch in range(n_epochs):
    epoch_loss = 0
    for i in range(n_batches):
        # Have to resize the batches to meet torch specifications
        train_batch = train_data[i * batch_size : (i + 1) * batch_size, :]\
                                .view(batch_size, -1)
        train_label_batch = train_labels[i * batch_size : (i + 1) * batch_size, :]\
                                        .view(batch_size, -1)

        # reset the gradient calculations each time it is used
        optimizer.zero_grad()

        # to get the output of the network we simply pass the batch 
        # through the net with a function call
        out = mlp(train_batch)
        loss = criterion(out, train_label_batch.argmax(1))

        # Backprop step
        loss.backward()
        optimizer.step()

        epoch_loss += loss
        # compute the average loss for all batches. 
    
    epoch_loss /= n_batches
    print("Epoch {} - Average batch loss: {}".format(epoch + 1, epoch_loss))

"""
Step 5: evaluate the model
"""
# We use this context manager to calculate the 
# test accuracy without logging it on the graph.
with torch.no_grad():
    out = mlp(test_data).argmax(1)
    acc = sum((out == test_labels.argmax(1)).numpy()) / 10000
    print(acc)






