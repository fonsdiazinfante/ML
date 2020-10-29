import torch
import torchvision
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


train = datasets.MNIST("", train=True, download=True,
                        transform = transforms.Compose([transforms.ToTensor()])) # We want data to go locally, that's why empty brackets

test = datasets.MNIST("", train=False, download=True,
                        transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

"""

total = 0
counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in trainset:
    Xs, ys = data
    for y in ys:
        counter_dict[int(y)] +=1
        total +=1
print(counter_dict)

for i in counter_dict:
    print (f"{i}: {counter_dict[i]/total*100}")

x,y = data[0][0], data[1][0]"""

# plt.imshow(data[0][0].view(28,28)) #28 is the size of the image
# plt.show()

class Net(nn.Module):
    def __init__(self):
        super().__init__() # Super corresponds to nn module and runs inicialization for nn.Module, inherits de methods
        self.fc1 = nn.Linear(784, 64) # fc is fully connected, 784 comes from 28*28 (because the image is 28*28), 3 layers of 64 neurons for our hidden layers
        self.fc2 = nn.Linear(64, 64) # Previous layer outputs 64 so the input for this layer is 64
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10) # Output 10 comes from having 10 classe

    def forward(self, x): # Method that defines how the data is going to flow through the network
        x = F.relu(self.fc1(x)) # F.relu (rectified linear) is activation function over the entire layer, activation is wether or not the neuron is "firing"
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) # X is going to pass through all the layers
        return F.log_softmax(x, dim=1)

X = torch.rand((28,28))
X = X.view(-1,28*28) # -1 is that we dont know the size of the input


net = Net()
print(net)
output = net(X)
print(output)

optimizer = optim.Adam(net.parameters(), lr=0.001) # lr is learning rate

EPOCHS = 3 # Epoch is a full pass through our dataset


for epoch in range(EPOCHS):
    for data in trainset:
        X, y = data
        net.zero_grad() # you need to zero de gradient unless it will continue to add itself
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y) #
        loss.backward() # Backpropagates the loss
        optimizer.step() # adjusts the weights
    print(loss)

correct = 0
total = 0

with torch.no_grad(): # without gradient, we want to know how good the network is
    for data in trainset:
        X, y = data
        output = net(X.view(-1,784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct +=1
            total +=1

print("Accuracy: ", round(correct/total, 3))
