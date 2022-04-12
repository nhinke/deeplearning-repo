# Nick Hinke
# 520.638 Deep Learning
# Homework 3 - Problem 2
#
#
#

import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

import torch
# import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# function to display an image
def imshow(img):
    img = img/2.0 + 0.5 # unnormalize image
    npimg = img.numpy()
    plt.imshow( np.transpose(npimg,(1,2,0)) )
    plt.show()

# function to get true positive rate (TPR) and false positive rate (FPR) from true and predicted Numpy arrays
def getROC(y_true,y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for yt,yp in zip(y_true,y_pred):
        if (yt == 1 and yp == 1):
            tp += 1
        elif (yt == 1 and yp == 0):
            fn += 1
        elif (yt == 0 and yp == 1):
            fp += 1
        elif (yt ==0 and yp == 0):
            tn += 1
    tpr = (1.0*tp)/(tp+fn)
    fpr = (1.0*fp)/(fp+tn)
    return tpr,fpr

def main():

    WITH_FINE_TUNING = True

    USE_VGG16 = False
    USE_ALEXNET = not USE_VGG16

    USE_LFWPAIRS_DATASET = True
    USE_LFWPEOPLE_DATASET = not USE_LFWPAIRS_DATASET

    DATASET_DIRECTORY = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Homework/hw3/data'
    NETWORK_PARAM_DIRECTORY = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Homework/hw3/data'


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

    # retreive selected model and set ROC plot title
    if (USE_ALEXNET):
        if (WITH_FINE_TUNING):
            try:
                model = models.alexnet(pretrained=False)
                model.load_state_dict(torch.load(NETWORK_PARAM_DIRECTORY+'/alexnet-finetuned.pth'))
            except:
                print(f"Error: please ensure finetuned AlexNet model saved at '{NETWORK_PARAM_DIRECTORY}/alexnet-finetuned.pth'")
                quit()
            plot_title = 'AlexNet (with Fine-Tuning)'
        else:
            model = models.alexnet(pretrained=True)
            plot_title = 'AlexNet (without Fine-Tuning)'
    elif (USE_VGG16):
        if (WITH_FINE_TUNING):
            try:
                model = models.vgg16(pretrained=False)
                model.load_state_dict(torch.load(NETWORK_PARAM_DIRECTORY+'/vgg16-finetuned.pth'))
            except:
                print(f"Error: please ensure finetuned VGG16 model saved at '{NETWORK_PARAM_DIRECTORY}/vgg16-finetuned.pth'")
                quit()
            plot_title = 'VGG16 (with Fine-Tuning)'
        else:
            model = models.vgg16(pretrained=True)
            plot_title = 'VGG16 (without Fine-Tuning)'

    # define proper image transformations
    transform = transforms.Compose([transforms.Resize(256), 
                                    transforms.CenterCrop(224),   
                                    transforms.ToTensor(),  
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    # move selected model to GPU if available, and set to evaluation mode
    model.to(device)
    model.eval()

    if (USE_LFWPAIRS_DATASET):

        # set dataset name
        dataset_name = 'LFWPairs'

        # load dataset and create dataloader
        test_dataset = datasets.LFWPairs(root=DATASET_DIRECTORY, download=True, split='test', image_set='original', transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=36, shuffle=False, num_workers=4)

        # initialize lists to store output labels and dissimilarity measures for each pair of images
        output_labels = list()
        dissimilarity_vector = list()

        # compute dissimilarity measure for each pair in pairs dataset
        with torch.no_grad():
            for data in test_loader:
                # load batch of image pairs and labels and move to GPU as appropriate
                input1, input2, labels = data
                input1 = input1.to(device)
                input2 = input2.to(device)
                # compute outputs for batch of image pairs and move to CPU
                output1 = model(input1).cpu().data.numpy()
                output2 = model(input2).cpu().data.numpy()
                # compute dissimilarity measure for each image pair within batch
                for out1,out2,label in zip(output1,output2,labels):
                    output_labels.append(int(label.data.numpy()))
                    dissimilarity_vector.append(np.linalg.norm(out1-out2))
        
        # convert lists to numpy arrays, and "normalize" dissimilarity measures such that identical images have a dissimilarity of 0.0, and the most dissimilar images have a dissimilarity of 1.0
        output_labels = np.array(output_labels)
        dissimilarity_vector = np.array(dissimilarity_vector)
        dissimilarity_vector = dissimilarity_vector/np.max(dissimilarity_vector)

    elif (USE_LFWPEOPLE_DATASET):

        # set dataset name
        dataset_name = 'LFWPeople'
        
        # load dataset
        test_dataset = datasets.LFWPeople(root=DATASET_DIRECTORY, download=True, split='test', image_set='original', transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=36, shuffle=False, num_workers=4)

        # initialize lists to store output labels (individual identities) and output feature vectors (dim=1000x1) for each image
        output_features = list()
        output_identities = list()

        # compute model outputs and store image identities for each image in dataset
        with torch.no_grad():
            print('Computing model outputs for each image in dataset...')
            for i,data in enumerate(test_loader):
                # load batch of images and identity labels and move to GPU as appropriate
                inputs, labels = data
                inputs = inputs.to(device)
                # compute outputs for batch of image pairs and move to CPU
                outputs = model(inputs).cpu().data.numpy()
                # store each output feature vector and identity label
                for out,identity in zip(outputs,labels.data.numpy()):
                    output_features.append(out)
                    output_identities.append(identity)

        # initialize list dissimilarity measures between every pair of images
        dissimilarity_vector = list()

        # convert lists to numpy arrays
        output_features = np.array(output_features)
        output_identities = np.array(output_identities)
        
        # initialize lists to store output labels and dissimilarity measures for each pair of images
        output_labels = list()
        dissimilarity_vector = list()

        # loop over every possible pair of images in dataset
        num_images = output_features.shape[0]
        num_comparisons = math.comb(num_images,2)
        pairs = 0
        percent = 0
        ten_percent = round(num_comparisons/10.0)
        print(f'Computing dissimilarity measure for all {num_comparisons} pairs of images in dataset...')
        for id1,output1 in enumerate(output_features):
            label1 = output_identities[id1]
            for id2 in range(id1+1, num_images):
                label2 = output_identities[id2]
                output2 = output_features[id2]
                # compare image identities for image pair and store as label, and compute and store dissimilarity measure for image pair
                output_labels.append(int(label1 == label2))
                dissimilarity_vector.append(np.linalg.norm(output1-output2))
                # print progress
                pairs += 1
                if (pairs % ten_percent == 0):   
                    percent += 10
                    print(f'  {percent}%')
        print(f'  100%')

        # "normalize" dissimilarity measures such that identical images have a dissimilarity of 0.0, and the most dissimilar images have a dissimilarity of 1.0
        dissimilarity_vector = np.array(dissimilarity_vector)
        dissimilarity_vector = dissimilarity_vector/np.max(dissimilarity_vector)

    # get ROC curve metrics including false positive rate (fpr), true positive rate (tpr), thresholds (thr), and area under curve (auc)
    fpr,tpr,thr = metrics.roc_curve(output_labels, 1-dissimilarity_vector, pos_label=1)
    roc_auc = metrics.auc(fpr,tpr)

    # plot ROC curve
    plt.figure()
    plt.plot(fpr,tpr,label=f'auc = {roc_auc:0.4f}')
    plt.legend(loc='best')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC for {plot_title} on {dataset_name} Dataset')
    plt.show()


    
    # load datasets
    # data_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Homework/hw3/data'
    # total_test_set = datasets.LFWPeople(root=data_path, download=True, split='test', image_set='original', transform=transform)
    # total_train_set = datasets.LFWPeople(root=data_path, download=True, split='train', image_set='original', transform=transform)


    # dataiter = iter(train_loader)
    # images,labels = dataiter.next()
    # print(labels)
    # imshow(torchvision.utils.make_grid(images))


    # plt.figure()
    # thresholds = [0.20,0.25,0.30,0.35]
    # y_predicted = list()
    # for threshold in thresholds:
    #     y_pred = [1 if diff < threshold else 0 for diff in dissimilarity_vector]
    #     y_predicted.append(y_pred)
    #     tpr,fpr = getROC(y_true,y_pred)
    #     print(f'\nTPR:  {tpr}')
    #     print(f'FPR:  {fpr}')
    #     fpr, tpr, thr = metrics.roc_curve(y_true, y_pred, drop_intermediate=False)
    #     roc_auc = metrics.auc(fpr, tpr)
    #     print(f'TPR:  {tpr}')
    #     print(f'FPR:  {fpr}')
    #     print(f'THR:  {thr}')
    #     plt.plot(fpr,tpr,label="th="+str(threshold)+", auc="+str(roc_auc))


    

if __name__ == "__main__":
    main()

