import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import models

def main():

    # define testing parameters
    test_batch_size = 10
    visualize_first_batch = True

    # define paths to save model parameters
    model_ae_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Homework/hw4/model-params/autoencoder.pth'
    model_denoising_ae_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Homework/hw4/model-params/denoising-autoencoder.pth'

    # construct models to be trained
    model_ae = models.AutoEncoder(model_path=model_ae_path)
    model_denoising_ae = models.AutoEncoder(model_path=model_denoising_ae_path)
    # print(model_ae)
    # print(model_denoising_ae)

    # determine device on which to train network (GPU if possible)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice to be used during testing: {device}')

    # define input and output transformations
    im_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float)
    im_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float)
    input_transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=im_mean.tolist(), std=im_std.tolist())] )
    output_transform = torchvision.transforms.Compose( [torchvision.transforms.Normalize(mean=(-im_mean/im_std).tolist(), std=(1.0/im_std).tolist()), torchvision.transforms.ToPILImage()] )

    # load test dataset and apply appropriate transformation
    data_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Homework/hw4/data'
    test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=input_transform, download=True)

    # create dataloader for test dataset
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=4)

    # define list of all classes in dataset
    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

    # load previously saved model parameters
    model_ae.load_saved_model()
    model_denoising_ae.load_saved_model()

    # move models to appropriate device
    model_ae.to(device)
    model_denoising_ae.to(device)

    # put models in evaluation mode
    model_ae.eval()
    model_denoising_ae.eval()

    test_loss_ae = list()
    test_loss_denoising_ae = list()
    running_loss_ae = 0.0 
    running_loss_denoising_ae = 0.0
    for id,data in enumerate(test_loader):

        inputs,labels = data
        noisy_inputs = torch.clamp(inputs + torch.randn(size=inputs.shape)*0.1, min=-1.0, max=1.0)
        inputs = inputs.to(device)
        noisy_inputs = noisy_inputs.to(device)
        
        outputs_ae,loss_ae = model_ae.testing_step(inputs, true_inputs=inputs)
        outputs_denoising_ae,loss_denoising_ae = model_denoising_ae.testing_step(noisy_inputs, true_inputs=inputs)

        running_loss_ae += loss_ae.item()
        running_loss_denoising_ae += loss_denoising_ae.item()

        if (id == 0 and visualize_first_batch):
            print('TODO')

    MSEloss_ae = running_loss_ae/(1.0*test_set.__len__()/test_batch_size)
    MSEloss_denoising_ae = running_loss_denoising_ae/(1.0*test_set.__len__()/test_batch_size)

    print(f'AE Testing:  ( MSE loss = {MSEloss_ae:.3f} )')
    print(f'DAE Testing: ( MSE loss = {MSEloss_denoising_ae:.3f} )')



if __name__ == "__main__":
    main()