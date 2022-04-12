# Nick Hinke
# 520.638 Deep Learning
# Homework 3 - Problem 2
#
# Script used to fine-tune AlexNet or VGG16 pre-trained model for verification task on LFW dataset
# Script will save resulting model state_dict to location specified by 'NETWORK_PARAM_DIRECTORY'
#

import os
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable as Variable

def main():

    USE_CPU = False

    USE_VGG16 = True
    USE_ALEXNET = not USE_VGG16

    USE_LFWPAIRS_DATASET = True
    USE_LFWPEOPLE_DATASET = not USE_LFWPAIRS_DATASET

    DATASET_DIRECTORY = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Homework/hw3/data'
    NETWORK_PARAM_DIRECTORY = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Homework/hw3/data'

    percent_validation = 0.01 
    USE_VALIDATION_SET = False
       
    if (USE_VGG16):
        epochs = 5
        momentum = 0.9
        batch_size = 4
        learningRate = 0.0001
        num_epochs_to_update_conv_layers = 2
    elif (USE_ALEXNET):
        epochs = 20
        momentum = 0.9
        batch_size = 22
        learningRate = 0.0001
        num_epochs_to_update_conv_layers = 15


    # check if GPU if available
    if (not USE_CPU):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if (device.type == 'cuda'):
            torch.cuda.empty_cache()
            os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:128'
    else:
        device = torch.device('cpu')
    print(f'\nDevice: {device}')

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
        model_str = 'AlexNet'
    elif (USE_VGG16):
        model = models.vgg16(pretrained=True)
        model_str = 'VGG16'

    # define proper image transformations
    transform = transforms.Compose([transforms.Resize(256), 
                                    transforms.CenterCrop(224),   
                                    transforms.ToTensor(),  
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    # move selected model to GPU if available, and set to training mode
    model.to(device)
    model.train(True)

    # define function to compute euclidean distance between two tensors (using pdist(tensor1,tensor2))
    pdist = torch.nn.PairwiseDistance(p=2.0)

    # intialize loss function (Binary Cross Entropy) and optimizer (Stochastic Gradient Descent)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)

    if (USE_LFWPAIRS_DATASET):

        print(f'Fine-tuning {model_str} pre-trained model on LFWPairs dataset...\n')

        # load total training set
        total_train_set = datasets.LFWPairs(root=DATASET_DIRECTORY, download=True, split='train', image_set='original', transform=transform)

        # split total training set into training and validation sets
        if (USE_VALIDATION_SET):
            num_valid = round(percent_validation*total_train_set.__len__())
            num_train = total_train_set.__len__() - num_valid
            split_train_set, split_valid_set = torch.utils.data.random_split(total_train_set, [num_train,num_valid])
            train_loader = torch.utils.data.DataLoader(split_train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            valid_loader = torch.utils.data.DataLoader(split_valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
            num_train_samples = split_train_set.__len__()
        else:
            train_loader = torch.utils.data.DataLoader(total_train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            num_train_samples = total_train_set.__len__()
        
        # freeze parameters within convolutional layers (model.features.parameters()) if 'num_epochs_to_update_conv_layers' set to less than total number of epochs
        if (num_epochs_to_update_conv_layers < epochs):
            conv_param_flag = False
            for param in model.features.parameters():
                param.requires_grad = False 
            for param in model.classifier.parameters():
                param.requires_grad = True 
        else:
            conv_param_flag = True
            for param in model.features.parameters():
                param.requires_grad = True 
            for param in model.classifier.parameters():
                param.requires_grad = True 

        # train model
        eps = 0.01
        max_dissimilarity = -1.0
        num_mini_batches = round(1.0*num_train_samples/batch_size/10.0)
        for epoch in range(epochs):
            # unfreeze parameters in convolutional layers as specified by 'num_epochs_to_update_conv_layers' 
            if (not conv_param_flag and epoch >= (epochs-num_epochs_to_update_conv_layers)):
                conv_param_flag = True
                for param in model.features.parameters():
                    param.requires_grad = True
            # reset running_loss statistic to zero
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
                batch_dissimilarity = pdist(output1,output2) + eps
                torch_max = torch.max(batch_dissimilarity).cpu().data.numpy().reshape(-1)[0]
                if (max_dissimilarity < torch_max):
                    max_dissimilarity = torch_max
                batch_dissimilarity_normalized = torch.div(batch_dissimilarity,max_dissimilarity+eps)
                # perform backpropagation using specified loss function and gradient clipping to prevent exploding gradients
                loss = criterion(batch_dissimilarity_normalized,labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=100.0)
                optimizer.step()
                running_loss += loss.item()
                # print and reset running_loss statistic every num_mini_batches batches
                if ((id+1) % num_mini_batches == 0):   
                    print(f'[{epoch+1},{id+1:3d}] loss: {1.0*running_loss/num_mini_batches:.3f}')
                    running_loss = 0.0
            print(f'EPOCH {epoch+1}/{epochs} COMPLETE...\n')

    elif (USE_LFWPEOPLE_DATASET):

        # todo
        print("Error: please set 'USE_LFWPAIRS_DATASET' boolean to 'True' at top of main()")
        quit()

        print(f'Fine-tuning {model_str} pre-trained model on LFWPeople dataset...\n')

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

