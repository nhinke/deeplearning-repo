# Nick Hinke
# 520.638 Deep Learning
# PyTorch Tutorials
#
# Adapted from PyTorch tutorials done in class 
#

import psutil
import GPUtil
import humanize
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as functional

import numpy as np
import matplotlib.pyplot as plt

# function to list all available GPUs and report their memory utilization
def mem_report():
    print("CPU RAM free: " + humanize.naturalsize( psutil.virtual_memory().available ))
    GPUs = GPUtil.getGPUs()
    for i,gpu in enumerate(GPUs):
        print('GPU {:d} ... Mem free: {:.0f}MB / {:.0f}MB | Utilization:{:3.0f}%'.format(i,gpu.memoryFree,gpu.memoryTotal,gpu.memoryUtil*100.0))

# function to generate dummy training and test data for linear regression example
def generate_data(num_train,num_test):
    x_train_values = np.random.uniform(1,100,num_train)
    y_train_values = [2.0*x+1.0 for x in x_train_values] # modeling y=2*x+1
    x_test_values = np.random.uniform(1,100,num_test)
    y_test_values = [2.0*x+1.0 for x in x_test_values] # modeling y=2*x+1
    x_train = np.array(x_train_values,dtype=np.float32).reshape(-1,1)
    y_train = np.array(y_train_values,dtype=np.float32).reshape(-1,1)
    x_test = np.array(x_test_values,dtype=np.float32).reshape(-1,1)
    y_test = np.array(y_test_values,dtype=np.float32).reshape(-1,1)
    return x_train, y_train, x_test, y_test

# function to display an image
def imshow(img):
    img = img/2.0 + 0.5 # unnormalize image
    npimg = img.numpy()
    plt.imshow( np.transpose(npimg,(1,2,0)) )
    plt.show()

# function to test network performance
def compute_model_accuracy(network,data_loader,device):
    
    total = 0
    correct = 0
    with torch.no_grad():
        for data in data_loader:

            # get labeled test data (where 'data' is list of [inputs,labels]) and move to appropriate device (e.g. GPU)
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # make predictions on test data
            outputs = network(inputs) # compute outputs by passing inputs through network
            _, predicted = torch.max(outputs.data,1) # class with highest energy is chosen as prediction
        
            # update network performance statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total, correct

# torch.nn.Module which is the basic Neural Network module containing all required functions
class linearRegression(torch.nn.Module):

    def __init__(self, inputSize, outputSize):
        super(linearRegression,self).__init__()
        self.linear = torch.nn.Linear(inputSize,outputSize) # y=W*x+b

    def forward(self, x):
        out = self.linear(x)
        return out

class CNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,6,5) # in_channels = 3, out_channels = 6, kernel_size = 5
        self.pool = torch.nn.MaxPool2d(2,2) # kernel_size = 2, stride = 2
        self.conv2 = torch.nn.Conv2d(6,16,5)
        self.fc1 = torch.nn.Linear(16*5*5,120) # in-features = 16*5*5, out_features = 120
        self.fc2 = torch.nn.Linear(120,84)
        self.fc3 = torch.nn.Linear(84,10) # out_features = # of classes = 10
        # self.fc4 = torch.nn.Sequential( torch.nn.Linear(120,84),torch.nn.ReLU() ) # then 'x = self.fc4(x)' equivalent to: 'x = functional.relu(self.fc2(x))' used below

    def forward(self, x):
        # x.shape = [B,3,32,32]
        x = self.pool(functional.relu(self.conv1(x)))
        # x.shape = [B,6,14,14]
        x = self.pool(functional.relu(self.conv2(x)))
        # x.shape = [B,16,5,5]
        x = torch.flatten(x,1) # flatten all dimensions except batch
        # x.shape = [B,16*5*5]
        x = functional.relu(self.fc1(x))
        # x.shape = [B,120]
        x = functional.relu(self.fc2(x))
        # x.shape = [B,84]
        x = self.fc3(x)
        # x.shape = [B,10]
        return x


def main():

    # two examples (1=simple linear regression, 2=CIFAR10 CNN classification)
    NUM_EXAMPLE_TO_RUN = 2

    if (NUM_EXAMPLE_TO_RUN == 1):
    
        mem_report()

        # define training and test data
        num_train = 50
        num_test = 25
        x_train, y_train, x_test, y_test = generate_data(num_train,num_test)
        print(x_train.shape)
        print(y_train.shape)

        # define model
        inputDim = 1
        outputDim = 1
        learningRate = 0.0001
        epochs = 100
        model = linearRegression(inputDim,outputDim)
        if (torch.cuda.is_available()):
            model.cuda()
            print('GPU available, mode model to cuda: {}'.format(torch.cuda.current_device()))
            mem_report()
        else:
            print('GPU not available, keep model on CPU')

        # intialize loss function (Mean Squared Error) and optimizer (Stochastic Gradient Descent)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(),lr=learningRate)
        
        # train model
        for epoch in range(epochs):
            if (torch.cuda.is_available()):
                inputs = torch.from_numpy(x_train).cuda()
                labels = torch.from_numpy(y_train).cuda()
            else:
                inputs = torch.from_numpy(x_train)
                labels = torch.from_numpy(y_train)

            outputs = model(inputs) # get outputs from model given inputs
            loss = criterion(outputs,labels) # get loss for predicted output

            optimizer.zero_grad() # clear gradient buffers since we don'tn want any gradients from previous epoch to carry forward
            loss.backward() # get gradients with respect to parameters
            optimizer.step() # update parameters

            print('epoch {}, loss {}'.format(epoch,loss.item()))

        # test model
        with torch.no_grad(): # no gradients needed when inferencing
            if (torch.cuda.is_available()):
                y_predicted = model(torch.from_numpy(x_test).cuda()).cpu().data.numpy()
            else:
                y_predicted = model(torch.from_numpy(x_test)).data.numpy()

        # plot results
        plt.clf()
        plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
        plt.plot(x_test, y_predicted, '--', label='Predictions', alpha=0.5)
        plt.legend(loc='best')
        plt.show()

    elif (NUM_EXAMPLE_TO_RUN == 2):

        FORCE_RETRAIN_NETWORK = False
        network_param_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Homework/hw3/data/cifar_network.pth'
        
        # torchvision.transforms contains many common image transformations 

        # torchvision datasets provide PILImage images on range [0,1]
        # we want to transform those images to tensors on normalized range [-1,1]
        transform = transforms.Compose( [transforms.ToTensor(),transforms.Normalize( (0.5,0.5,0.5),(0.5,0.5,0.5) )] )

        epochs = 20
        momentum = 0.9
        batch_size = 16 # must be even for printing labels for image grid below
        learningRate = 0.001

        # load datasets
        data_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Homework/hw3/data'
        total_train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        total_test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

        # split total training set into training and validation sets
        percent_validation = 0.15
        num_valid = round(percent_validation*total_train_set.__len__())
        num_train = total_train_set.__len__() - num_valid
        split_train_set, split_valid_set = torch.utils.data.random_split(total_train_set, [num_train,num_valid])

        # create dataloaders
        valid_loader = torch.utils.data.DataLoader(split_valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
        train_loader = torch.utils.data.DataLoader(split_train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(total_test_set, batch_size=batch_size, shuffle=False, num_workers=4)

        # define list of all classes in dataset
        classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

        # randomly get one batch of training data, then show images and print labels
        show_batch_of_images = False
        if (show_batch_of_images):
            dataiter = iter(train_loader)
            images,labels = dataiter.next()
            print( ' '.join(f'{classes[labels[j]]:5s}' for j in range(int(batch_size/2))) )
            print( ' '.join(f'{classes[labels[int(batch_size/2+j)]]:5s}' for j in range(int(batch_size/2))) )
            imshow(torchvision.utils.make_grid(images))

        # check if GPU if available
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {device}\n')

        # initialize network and move to GPU if available
        cnn = CNN().to(device)

        # intialize loss function and optimizer (Stochastic Gradient Descent)
        criterion = torch.nn.CrossEntropyLoss() # for classification
        optimizer = torch.optim.SGD(cnn.parameters(),lr=learningRate,momentum=momentum)

        # train network if network parameter file does not exist yet
        if (FORCE_RETRAIN_NETWORK or not Path(network_param_path).is_file()):
            for epoch in range(epochs):

                running_loss = 0.0
                for i,data in enumerate(train_loader):
                    
                    # get labeled training data and move to GPU (data is list of [inputs,labels])
                    inputs,labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # perform backpropagation
                    optimizer.zero_grad()
                    outputs = cnn(inputs)
                    loss = criterion(outputs,labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    # print and reset running_loss statistic every 1000 mini-batches
                    if (i % 1000 == 999):   
                        print(f'[{epoch+1},{i+1:5d}] loss: {running_loss/1000.0:.3f}')
                        running_loss = 0.0

                # compute and print validation error
                total,correct = compute_model_accuracy(network=cnn, data_loader=valid_loader, device=device)        
                print(f'EPOCH {epoch+1} COMPLETE -- network validation accuracy: {100.0*correct/(1.0*total):.1f}%\n')

            # save trained model parameters
            print(f'Finished training!\nSaving model parameters to: {network_param_path}\n')
            torch.save(cnn.state_dict(),network_param_path)

        # load the saved network parameters
        cnn.load_state_dict(torch.load(network_param_path))

        # test network performance
        total,correct = compute_model_accuracy(network=cnn, data_loader=test_loader, device=device)
        print(f'Network accuracy on the {total} test images: {100.0*correct/(1.0*total):.1f}%\n')

    else:
        print("Error: please set 'NUM_EXAMPLE_TO_RUN' variable to appropriate value at top of main() function")


if __name__ == "__main__":
    main()

