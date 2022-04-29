import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import models

def main():
    
    # define training parameters
    num_epochs = 3
    valid_batch_size = 16
    train_batch_size = 16
    perform_validation = True

    # define paths to save model parameters
    model_ae_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Homework/hw4/model-params/autoencoder.pth'
    model_denoising_ae_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Homework/hw4/model-params/denoising-autoencoder.pth'

    # construct models to be trained
    model_ae = models.AutoEncoder(sgd_optim=False, learning_rate=1e-4, model_path=model_ae_path)
    model_denoising_ae = models.AutoEncoder(sgd_optim=False, learning_rate=1e-4, model_path=model_denoising_ae_path)
    # print(model_ae)
    # print(model_denoising_ae)

    # determine device on which to train network (GPU if possible)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice to be used during training: {device}')

    # define input and output transformations
    im_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float)
    im_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float)
    input_transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=im_mean.tolist(), std=im_std.tolist())] )
    output_transform = torchvision.transforms.Compose( [torchvision.transforms.Normalize(mean=(-im_mean/im_std).tolist(), std=(1.0/im_std).tolist()), torchvision.transforms.ToPILImage()] )

    # load training dataset and apply appropriate transformation
    data_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Homework/hw4/data'
    total_train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=input_transform, download=True)
    
    # split total training set into training and validation sets
    percent_validation = 0.10
    num_valid_samples = round(percent_validation*total_train_set.__len__())
    num_train_samples = total_train_set.__len__() - num_valid_samples
    train_set, valid_set = torch.utils.data.random_split(total_train_set, [num_train_samples, num_valid_samples])

    # create dataloaders for each dataset
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=valid_batch_size, shuffle=False, num_workers=4)
        
    # define list of all classes in dataset
    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

    # load previously saved model parameters
    # model_ae.load_saved_model()
    # model_denoising_ae.load_saved_model()

    # move models to appropriate device
    model_ae.to(device)
    model_denoising_ae.to(device)

    # put models in training mode
    model_ae.train()
    model_denoising_ae.train()

    # define lists to store loss function values
    iterations = [0]
    train_loss_ae = list()
    valid_loss_ae = list()
    train_loss_denoising_ae = list()
    valid_loss_denoising_ae = list()

    epoch_times = list()
    start_time = time.time()
    for epoch in range(num_epochs):

        running_loss_ae = 0.0
        running_loss_denoising_ae = 0.0
        start_ep_time = time.time()

        for id,data in enumerate(train_loader):

            inputs,_ = data
            noisy_inputs = torch.clamp(inputs + torch.randn(size=inputs.shape)*0.1, min=-1.0, max=1.0)
            inputs = inputs.to(device)
            noisy_inputs = noisy_inputs.to(device)

            _,loss_ae = model_ae.training_step(inputs, true_inputs=inputs)
            _,loss_denoising_ae = model_denoising_ae.training_step(noisy_inputs, true_inputs=inputs)

            running_loss_ae += loss_ae.item()
            running_loss_denoising_ae += loss_denoising_ae.item()

            if (id % 100 == 99): 
                iterations.append(id+iterations[-1])
                MSEloss_ae = running_loss_ae/100.0
                MSEloss_denoising_ae = running_loss_denoising_ae/100.0
                train_loss_ae.append(MSEloss_ae)
                train_loss_denoising_ae.append(MSEloss_denoising_ae)

                print('\n')
                print(f'[{epoch+1},{id+1}] AE Training:    ( MSE loss = {MSEloss_ae:.3f} )')
                print(f'[{epoch+1},{id+1}] DAE Training:   ( MSE loss = {MSEloss_denoising_ae:.3f} )')
                running_loss_ae = 0.0 
                running_loss_denoising_ae = 0.0

                if (perform_validation):
                    model_ae.eval()
                    model_denoising_ae.eval()
                    # running_loss_ae = 0.0
                    # running_loss_denoising_ae = 0.0
                    for data in valid_loader: 

                        inputs,_ = data
                        noisy_inputs = torch.clamp(inputs + torch.randn(size=inputs.shape)*0.1, min=-1.0, max=1.0)
                        inputs = inputs.to(device)
                        noisy_inputs = noisy_inputs.to(device)
                        
                        _,loss_ae = model_ae.validation_step(inputs, true_inputs=inputs)
                        _,loss_denoising_ae = model_denoising_ae.validation_step(noisy_inputs, true_inputs=inputs)

                        running_loss_ae += loss_ae.item()
                        running_loss_denoising_ae += loss_denoising_ae.item()

                    MSEloss_ae = running_loss_ae/(1.0*valid_set.__len__()/valid_batch_size)
                    MSEloss_denoising_ae = running_loss_denoising_ae/(1.0*valid_set.__len__()/valid_batch_size)

                    valid_loss_ae.append(MSEloss_ae)
                    valid_loss_denoising_ae.append(MSEloss_denoising_ae)

                    print(f'[{epoch+1},{id+1}] AE Validation:  ( MSE loss = {MSEloss_ae:.3f} )')
                    print(f'[{epoch+1},{id+1}] DAE Validation: ( MSE loss = {MSEloss_denoising_ae:.3f} )')
                    running_loss_ae = 0.0 
                    running_loss_denoising_ae = 0.0

                    model_ae.train()
                    model_denoising_ae.train()

        epoch_times.append(time.time()-start_ep_time)
        print(f'EPOCH {epoch+1}/{num_epochs} COMPLETE (took {epoch_times[-1]:.1f} seconds)...\n')

    iterations.pop(0)
    print(f'Total elapsed time: {time.time()-start_time:.1f} seconds')

    model_ae.save_model()
    model_denoising_ae.save_model()

    # plot results for regular autoencoder
    plt.figure()
    plt.plot(iterations, train_loss_ae, iterations, valid_loss_ae)
    plt.legend(['training','validation'])
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('MSE Loss of Regular Autoencoder')

    # plot results for denoising autoencoder
    plt.figure()
    plt.plot(iterations, train_loss_denoising_ae, iterations, valid_loss_denoising_ae)
    plt.legend(['training','validation'])
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('MSE Loss of Denoising Autoencoder')

    # display plots
    plt.show()


if __name__ == "__main__":
    main()