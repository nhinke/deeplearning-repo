# Nick Hinke
# 520.638 Deep Learning
# Homework 1 - Problem 3
#
# Train and test the performance of four different algorithms {Eigenfaces (PCA), Fisherfaces (LDA), SVM, SRC}
# for face recognition on the extended YaleB dataset (32x32) using a varying number of training samples per object class
#

# import relevant libraries
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.decomposition import SparseCoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# function to randomly split all samples from one class into training and test sets using a defined number for the training set
def get_random_samples_of_one_class_for_datasets(x_samples, y_samples, num_samples, num_training):
    id_total = np.linspace(0, num_samples-1, num_samples, dtype=int)
    id_train = np.sort(np.random.choice(id_total, size=num_training, replace=False))
    id_test = [id for id in id_total if id not in id_train]
    x_samples_train = np.array([list(x_samples[id]) for id in id_train])
    y_samples_train = np.array([list(y_samples[id]) for id in id_train])
    x_samples_test = np.array([list(x_samples[id]) for id in id_test])
    y_samples_test = np.array([list(y_samples[id]) for id in id_test])
    return x_samples_train, y_samples_train, x_samples_test, y_samples_test

# function to randomly split the whole dataset into training and test sets using a defined number of training samples per object class
def get_training_and_test_sets(x_samples, y_samples, num_samples, num_training):
    x_samples_train = np.zeros(shape=(1,len(x_samples[0])))
    y_samples_train = np.zeros(shape=(1,len(y_samples[0])))
    x_samples_test = np.zeros(shape=(1,len(x_samples[0])))
    y_samples_test = np.zeros(shape=(1,len(y_samples[0])))
    for i in range(np.min(y_samples),np.max(y_samples)+1):
        i_id = [id for id,val in enumerate(y_samples) if (val==i)]
        i_x_samples = [x_samples[id] for id in i_id]
        i_y_samples = [y_samples[id] for id in i_id]
        i_num_samples = len(i_y_samples)
        i_x_train, i_y_train, i_x_test, i_y_test = get_random_samples_of_one_class_for_datasets(i_x_samples, i_y_samples, i_num_samples, num_training)
        x_samples_train = np.vstack((x_samples_train,i_x_train))
        y_samples_train = np.vstack((y_samples_train,i_y_train))
        x_samples_test = np.vstack((x_samples_test,i_x_test))
        y_samples_test = np.vstack((y_samples_test,i_y_test))
    x_samples_train = np.delete(x_samples_train, 0, axis=0)
    y_samples_train = np.delete(y_samples_train, 0, axis=0)
    x_samples_test = np.delete(x_samples_test, 0, axis=0)
    y_samples_test = np.delete(y_samples_test, 0, axis=0)
    return x_samples_train, y_samples_train, x_samples_test, y_samples_test


def main():

    # load YaleB-32x32 dataset from .mat file
    try:
        dataset = scipy.io.loadmat('dataset.mat')
        x_total = list(dataset.get('fea'))
        y_total = list(dataset.get('gnd'))
        num_samples = len(y_total)
    except:
        print("Please ensure that the YaleB dataset is saved as 'dataset.mat' in the correct directory.")
        quit()


    # set to any number (1-4) to determine which part of problem 3 will be executed by the program
    # note that numbers 1-4 correspond to {PCA,LDA,SVM,SRC} in order
    NUM_PROBLEM_TO_COMPLETE = 4


    if (NUM_PROBLEM_TO_COMPLETE == 1):
        print('Completing problem 3 using Eigenfaces algorithm (PCA)...')

        # define number of training samples per class to test
        M = [10,20,30,40,50]

        classification_errors = list()
        for m in M:

            # split dataset into training and test sets
            num_training_samples = m
            x_train, y_train, x_test, y_test = get_training_and_test_sets(x_total, y_total, num_samples, num_training_samples)  
            num_test_samples = y_test.shape[0]

            # construct PCA, fit it to training data, and transform both training and test data
            # note that keeping 97% of input variance found experimentally to reduce dimension of feature space by ~factor of 10 (i.e. 1024 --> ~100)
            pca = PCA(n_components=0.97) # 0.97 corresponds to 97% of the variance being preserved
            x_train_reduced = pca.fit_transform(x_train)
            x_test_reduced = pca.transform(x_test)

            # drop the first few eigenvectors from W and updated reduced training and test data accordingly
            drop_first_num_eigenvectors = 2
            W = np.transpose(pca.components_)
            W = W[:,drop_first_num_eigenvectors:W.shape[1]]
            x_train_reduced = x_train_reduced[:,drop_first_num_eigenvectors:x_train_reduced.shape[1]]
            x_test_reduced = x_test_reduced[:,drop_first_num_eigenvectors:x_test_reduced.shape[1]]
           
            # reconstruct training and test images from reduced dim. face space (still with first few eigenvectors disregarded)
            x_train_recovered = np.matmul(x_train_reduced,np.transpose(W))
            x_test_recovered = np.matmul(x_test_reduced,np.transpose(W))

            # make predictions by assigning test image the label of the closest training image in full dim. space (using eucliden distance metric)
            # note that using images in full dim. (recovered) space is necessary to get benefit of removing first few eigenvectors from W
            y_pred = list()
            for test_im in x_test_recovered:
                min_err = 1e9
                for id,train_im in enumerate(x_train_recovered):
                    im_err = np.linalg.norm(test_im-train_im)
                    if (im_err < min_err):
                        min_err = im_err
                        test_im_pred = y_train[id]
                y_pred.append(test_im_pred)
            # line below is equivalent to above method of finding y_pred but in one line (too unreadable to use, but I was too happy with it to just delete):
            # y_pred = [y_train[np.argmin([np.linalg.norm(test_im-train_im) for train_im in x_train_recovered])] for test_im in x_test_recovered]

            # compute classification error rate
            err = sum([1 for id,y_val in enumerate(y_test) if y_pred[id] != y_val[0]])*100.0/(1.0*num_test_samples) # error percentage
            classification_errors.append(err)
            print("Done with m = " + str(m) + "!")

        # plot computed classification error rates
        plt.figure()
        plt.plot(M, classification_errors, marker='o', linestyle='dashed', markersize='11')
        plt.title("(3.1) Error Rate vs Number of Training Samples using Eigenfaces (PCA)")
        plt.xlabel("# of training samples per class")
        plt.ylabel("error rate (%)")
        plt.show()            

    elif (NUM_PROBLEM_TO_COMPLETE == 2):
        print('Completing problem 3 using Fisherfaces algorithm (LDA)...')

        # define number of training samples per class to test
        M = [10,20,30,40,50]

        classification_errors = list()
        for m in M:

            # split dataset into training and test sets
            num_training_samples = m
            x_train, y_train, x_test, y_test = get_training_and_test_sets(x_total, y_total, num_samples, num_training_samples)  
            num_test_samples = y_test.shape[0]

            # construct LDA classifier, fit it to the training data, and transform both the training data and test data
            lda = LDA() 
            x_train_reduced = lda.fit_transform(x_train,y_train.ravel())
            x_test_reduced = lda.transform(x_test)

            # use my own method of making predictions (rather than built-in method on line below) to ensure euclidean distance metric being used
            # y_pred = lda.predict(x_test)

            # make predictions by assigning test image the label of the closest training image in reduced dim. space (using eucliden distance metric)
            y_pred = list()
            for test_im in x_test_reduced:
                min_err = 1e9
                for id,train_im in enumerate(x_train_reduced):
                    im_err = np.linalg.norm(test_im-train_im)
                    if (im_err < min_err):
                        min_err = im_err
                        test_im_pred = y_train[id]
                y_pred.append(test_im_pred)
            # line below is equivalent to above method of finding y_pred but in one line (too unreadable to use, but I was too happy with it to just delete):
            # y_pred = [y_train[np.argmin([np.linalg.norm(test_im-train_im) for train_im in x_train_recovered])] for test_im in x_test_recovered]
           
            # compute classification error rate
            err = sum([1 for id,y_val in enumerate(y_test) if y_pred[id] != y_val[0]])*100.0/(1.0*num_test_samples) # error percentage
            classification_errors.append(err)
            print("Done with m = " + str(m) + "!")

        # plot computed classification error rates
        plt.figure()
        plt.plot(M, classification_errors, marker='o', linestyle='dashed', markersize='11')
        plt.title("(3.2) Error Rate vs Number of Training Samples using Fisherfaces (LDA)")
        plt.xlabel("# of training samples per class")
        plt.ylabel("error rate (%)")
        plt.show()            

    elif (NUM_PROBLEM_TO_COMPLETE == 3):
        print('Completing problem 3 using Support Vector Machine (SVM)...')

        # define number of training samples per class to test
        M = [10,20,30,40,50]

        classification_errors = list()
        for m in M:

            # split dataset into training and test sets
            num_training_samples = m
            x_train, y_train, x_test, y_test = get_training_and_test_sets(x_total, y_total, num_samples, num_training_samples)  
            num_test_samples = y_test.shape[0]

            # set hyperparameters to be chosen via cross validation using GridSearchCV
            params = [
                {'C': [0.1, 1, 10], 'kernel': ['rbf']},
                {'C': [0.1, 1, 10], 'kernel': ['linear']},
                {'C': [0.1, 1, 10], 'kernel': ['poly']}
            ]

            # construct SVM classifier and fit it to the training data using GridSearchCV to select hyperparameters
            svc = GSCV(SVC(),param_grid=params)
            svc.fit(x_train,y_train.ravel())
            # print(svc.best_params_) # it was found experimentally that {'C':1,'kernel':'linear} performed best almost every time

            # make predictions on test data
            y_pred = svc.predict(x_test)

            # compute classification error rate
            err = sum([1 for id,y_val in enumerate(y_test) if y_pred[id] != y_val[0]])*100.0/(1.0*num_test_samples) # error percentage
            classification_errors.append(err)
            print("Done with m = " + str(m) + "!")

        # plot computed classification error rates
        plt.figure()
        plt.plot(M, classification_errors, marker='o', linestyle='dashed', markersize='11')
        plt.title("(3.3) Error Rate vs Number of Training Samples using Support Vector Machine (SVM)")
        plt.xlabel("# of training samples per class")
        plt.ylabel("error rate (%)")
        plt.show()           

    elif (NUM_PROBLEM_TO_COMPLETE == 4):
        print('Completing problem 3 using Sparse Representation-based Classification (SRC)...')
        
        # define number of training samples per class to test
        M = [10,20,30,40,50]

        classification_errors = list()
        for m in M:

            # split dataset into training and test sets
            num_training_samples = m
            x_train, y_train, x_test, y_test = get_training_and_test_sets(x_total, y_total, num_samples, num_training_samples)  
            num_test_samples = y_test.shape[0]

            # create dictionary using training data
            D = x_train

            # construct sparse coder and transform test data into sparse representation 
            src = SparseCoder(D)
            x_test_sparse = src.transform(x_test)

            # get list of indices of training and test data that correspond to each unique class
            train_class_indices_list = list()
            # test_class_indices_list = list()
            for i in range(int(np.min(y_test)),int(np.max(y_test))+1):
                train_class_indices = [id for id,val in enumerate(y_train) if (val==i)]
                train_class_indices_list.append(train_class_indices)
                # test_class_indices = [id for id,val in enumerate(y_test) if (val==i)]
                # test_class_indices_list.append(test_class_indices)

            # construct list of class dictionaries where each element is a dictionary of images of one class and zero otherwise
            # note that each class dictionary is of the same size as the original dictionary, so you can perform classwise image reconstruction simply by multiplying an image's sparse representation by the appropriate class dictionary
            D_class_list = list()
            D_id_total = np.linspace(0,D.shape[0]-1,D.shape[0],dtype=int).tolist()
            for id_list in train_class_indices_list:
                D_class = np.array([D[index] if index in id_list else [0]*D.shape[1] for index in D_id_total])
                D_class_list.append(D_class)

            # make predictions on the test data
            y_pred = list()
            for id_test,test_im in enumerate(x_test): # for every test image
                test_im_sparse = x_test_sparse[id_test]
                min_err = 1e9
                class_label = 0.
                for class_id,id_list in enumerate(train_class_indices_list): # for every class
                    test_im_reconstructed = np.matmul(test_im_sparse,D_class_list[class_id])
                    class_reconstruction_err = np.linalg.norm(test_im-test_im_reconstructed)
                    if (class_reconstruction_err < min_err):
                        min_err = class_reconstruction_err
                        class_label = y_train[id_list[0]]
                y_pred.append(class_label)

            # compute classification error rate
            err = sum([1 for id,y_val in enumerate(y_test) if y_pred[id] != y_val[0]])*100.0/(1.0*num_test_samples) # error percentage
            classification_errors.append(err)
            print("Done with m = " + str(m) + "!")

        # plot computed classification error rates
        plt.figure()
        plt.plot(M, classification_errors, marker='o', linestyle='dashed', markersize='11')
        plt.title("(3.4) Error Rate vs Number of Training Samples using Sparse Representation-based Classification (SRC)")
        plt.xlabel("# of training samples per class")
        plt.ylabel("error rate (%)")
        plt.show()  

    else:
        print("Please set 'NUM_PROBLEM_TO_COMPLETE' variable at top of main() function to a number on {1,2,3,4} and run again.")



if __name__ == "__main__":
    main()
