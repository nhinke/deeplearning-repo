# Nick Hinke
# 520.638 Deep Learning
# Homework 3 - Problem 2
#
#
#

# import numpy as np

import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable as Variable

def main():

    USE_VGG16 = False
    USE_ALEXNET = True

    USE_LFWPAIRS_DATASET = True
    USE_LFWPEOPLE_DATASET = False

    DATASET_DIRECTORY = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Homework/hw3/data'
    NETWORK_PARAM_DIRECTORY = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Homework/hw3/data'

    epochs = 10
    momentum = 0.9
    batch_size = 22
    learningRate = 0.0001
    percent_validation = 0.01

    # all params trainable, learningRate .0001, epochs = 10 --> auc = .8093

    # check if GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}\n')

    # check to make sure network type selected appropriately
    if (not USE_ALEXNET and not USE_VGG16):
        print("Error: please set either 'USE_VGG16' or 'USE_ALEXNET' booleans to 'True' at top of main()")
        quit()
    elif (USE_ALEXNET and USE_VGG16):
        print("Error: please set only one of 'USE_VGG16' or 'USE_ALEXNET' booleans to 'True' at top of main()")
        quit()

    # check to make sure dataset type selected appropriately
    if (not USE_LFWPAIRS_DATASET and not USE_LFWPEOPLE_DATASET):
        print("Error: please set either 'USE_LFWPAIRS_DATASET' or 'USE_LFWPEOPLE_DATASET' booleans to 'True' at top of main()")
        quit()
    elif (USE_LFWPAIRS_DATASET and USE_LFWPEOPLE_DATASET):
        print("Error: please set only one of 'USE_LFWPAIRS_DATASET' or 'USE_LFWPEOPLE_DATASET' booleans to 'True' at top of main()")
        quit()

    if (USE_ALEXNET):
        model = models.alexnet(pretrained=True)

    elif (USE_VGG16):
        model = models.vgg16(pretrained=True)

    # define proper image transformations
    transform = transforms.Compose([transforms.Resize(256), 
                                    transforms.CenterCrop(224),   
                                    transforms.ToTensor(),  
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    # move selected model to GPU if available, and set to training mode
    model.to(device)
    model.train(True)

    pdist = torch.nn.PairwiseDistance(p=2.0)

    # intialize loss function (Binary Cross Entropy) and optimizer (Stochastic Gradient Descent)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    if (USE_LFWPAIRS_DATASET):

        # load total training set
        total_train_set = datasets.LFWPairs(root=DATASET_DIRECTORY, download=True, split='train', image_set='original', transform=transform)

        # split total training set into training and validation sets
        num_valid = round(percent_validation*total_train_set.__len__())
        num_train = total_train_set.__len__() - num_valid
        split_train_set, split_valid_set = torch.utils.data.random_split(total_train_set, [num_train,num_valid])
        train_loader = torch.utils.data.DataLoader(split_train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        # valid_loader = torch.utils.data.DataLoader(split_valid_set, batch_size=batch_size, shuffle=False, num_workers=4)

        for param in model.features.parameters():
            param.requires_grad = True 

        # train model
        max_dissimilarity = -1.0
        one_tensor = Variable(torch.tensor([1.0]).to(device))
        zero_tensor = Variable(torch.tensor([0.0]).to(device))
        for epoch in range(epochs):
            running_loss = 0.0
            for id,data in enumerate(train_loader):
                # load batch of image pairs and labels and move to GPU as appropriate
                input1, input2, labels = data
                input1 = Variable(input1.to(device))
                input2 = Variable(input2.to(device))
                labels = Variable((1.0-labels).to(device,dtype=torch.float))
                # zero gradients and compute outputs for batch of image pairs
                optimizer.zero_grad()
                output1 = model(input1)
                output2 = model(input2)
                # compute dissimilarity measure for each image pair within batch and "normalize" it such that identical images have a dissimilarity of 0.0 and the most dissimilar images have a dissimilarity of 1.0
                batch_dissimilarity = pdist(output1,output2)
                torch_max = torch.max(batch_dissimilarity).cpu().data.numpy().reshape(-1)[0]
                if (max_dissimilarity < torch_max):
                    max_dissimilarity = torch_max
                batch_dissimilarity_normalized = torch.div(batch_dissimilarity,max_dissimilarity)
                batch_dissimilarity_normalized = torch.maximum(torch.minimum(batch_dissimilarity_normalized,one_tensor),zero_tensor)

                # batch_dissimilarity = np.linalg.norm(output1.cpu().data.numpy() - output2.cpu().data.numpy(), axis=1)
                # if (max_dissimilarity < np.max(batch_dissimilarity)):
                #     max_dissimilarity = np.max(batch_dissimilarity)
                # if (min_dissimilarity > np.min(batch_dissimilarity)):
                #     min_dissimilarity = np.min(batch_dissimilarity)
                # batch_dissimilarity = 1.0*batch_dissimilarity/max_dissimilarity
                # batch_dissimilarity = 1.0*(batch_dissimilarity-min_dissimilarity)/(max_dissimilarity-min_dissimilarity)
                # batch_dissimilarity = torch.from_numpy(batch_dissimilarity).to(device)
                # batch_dissimilarity = Variable(batch_dissimilarity,requires_grad=True)
                
                # perform backpropagation using specified loss function 
                loss = criterion(batch_dissimilarity_normalized,labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # print and reset running_loss statistic every 10 batches
                if ((id+1) % 10 == 0):   
                    print(f'[{epoch+1},{id+1:3d}] loss: {running_loss/10.0:.3f}')
                    running_loss = 0.0
            print(f'EPOCH {epoch+1}/{epochs} COMPLETE...\n')

    elif (USE_LFWPEOPLE_DATASET):

        # load total training set
        total_train_set = datasets.LFWPeople(root=DATASET_DIRECTORY, download=True, split='train', image_set='original', transform=transform)

    # save trained model parameters
    if (USE_ALEXNET):
        network_param_path = NETWORK_PARAM_DIRECTORY + '/alexnet-finetuned.pth'
    elif (USE_VGG16):
        network_param_path = NETWORK_PARAM_DIRECTORY + '/vgg16-finetuned.pth'
    print(f'Finished training!\nSaving model parameters to: {network_param_path}\n')
    torch.save(model.state_dict(),network_param_path)

if __name__ == "__main__":
    main()

