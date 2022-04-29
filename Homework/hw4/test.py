# Nick Hinke
# 520.638 Deep Learning
# Homework 4
#
# Script to evaluate and visualize the performance of both the regular and denoising autoencoders on the CIFAR10 test dataset (10k images)
#

# import relevant libraries
import torch
import models
import numpy as np
import torchvision
import skimage.metrics
import matplotlib.pyplot as plt

# function to plot autoencoder output with corresponding input
def visualize_autoencoder_outputs(label, clean_im, noisy_im, output, denoising_ae):
    if (denoising_ae):
        _,(ax1,ax2,ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
        ax1.imshow(clean_im)
        ax1.set_title(f'Clean Input (class={label})')
        ax1.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        ax1.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
        ax2.imshow(noisy_im)
        ax2.set_title(f'Noisy Input (class={label})')
        ax2.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        ax2.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
        ax3.imshow(output)
        ax3.set_title(f'Denoising Autoencoder Output (class={label})')
        ax3.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        ax3.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    else:
        _,(ax1,ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        ax1.imshow(clean_im)
        ax1.set_title(f'Clean Input (class={label})')
        ax1.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        ax1.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
        ax2.imshow(output)
        ax2.set_title(f'Regular Autoencoder Output (class={label})')
        ax2.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        ax2.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)

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

    # define list to store error metric values
    dae_psnr = list()
    dae_ssim = list()

    # evaluate and visualize model performance
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

        for input, out_dae in zip(inputs, outputs_denoising_ae):
            np_input = np.array(output_transform(input), dtype=np.float64)
            np_out_dae = np.array(output_transform(out_dae), dtype=np.float64)
            # mse = np.mean((np_input-np_out_dae)**2, dtype=np.float64)
            # psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))
            psnr = skimage.metrics.peak_signal_noise_ratio(image_true=np_input, image_test=np_out_dae, data_range=255)
            ssim = skimage.metrics.structural_similarity(im1=np_input, im2=np_out_dae, data_range=255.0, multichannel=True, channel_axis=2)
            dae_psnr.append(psnr)
            dae_ssim.append(ssim)

        if (id == 0 and visualize_first_batch):
            for label, input, noisy_input, out_ae, out_dae in zip(labels, inputs, noisy_inputs, outputs_ae, outputs_denoising_ae):
                label_str = classes[label.item()]
                clean_img_in = output_transform(input)
                noisy_img_in = output_transform(noisy_input)
                ae_img_out = output_transform(out_ae)
                dae_img_out = output_transform(out_dae)
                visualize_autoencoder_outputs(label=label_str, clean_im=clean_img_in, noisy_im=noisy_img_in, output=ae_img_out, denoising_ae=False)
                visualize_autoencoder_outputs(label=label_str, clean_im=clean_img_in, noisy_im=noisy_img_in, output=dae_img_out, denoising_ae=True)      

    # compute and print MSE loss values on whole test set
    MSEloss_ae = running_loss_ae/(1.0*test_set.__len__()/test_batch_size)
    MSEloss_denoising_ae = running_loss_denoising_ae/(1.0*test_set.__len__()/test_batch_size)
    print(f'AE Testing:  ( MSE loss = {MSEloss_ae:.3f} )')
    print(f'DAE Testing: ( MSE loss = {MSEloss_denoising_ae:.3f} )')

    # compute and print average error metrics for denoising autoencoder
    dae_psnr_avg = np.mean(dae_psnr)
    dae_ssim_avg = np.mean(dae_ssim)
    print(f'\nDAE average PSNR on test set: {dae_psnr_avg:.2f}')
    print(f'DAE average SSIM on test set: {dae_ssim_avg:.3f}\n')

    # display all figures
    plt.show()



if __name__ == "__main__":
    main()