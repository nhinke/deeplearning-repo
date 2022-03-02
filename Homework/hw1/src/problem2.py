# Nick Hinke
# 520.638 Deep Learning
# Homework 1 - Problem 2
#
# Perform a variety of experiments by constructing, training, and testing a k-NN classifier for face recognition on the extended YaleB dataset (32x32)
# Experiments include varying 'k', varying the number of training samples per object class, varying 'p' in the p-norm of the distance metric, etc.
#

# import relevant libraries
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.feature import local_binary_pattern as lbp
from sklearn.neighbors import KNeighborsClassifier as KNC

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

# function to split a training set into a training set and validation set using a defined percentage of the data for the validation set
def split_training_set_to_get_validation_set(x_train, y_train, percent_validation):
    num_training = y_train.shape[0]
    num_valid = int(np.round(1.0*percent_validation*num_training/100.0))
    id_total = np.linspace(0, num_training-1, num_training, dtype=int)
    id_valid = np.sort(np.random.choice(id_total, size=num_valid, replace=False))
    id_train = [id for id in id_total if id not in id_valid]
    x_samples_train = np.array([list(x_train[id]) for id in id_train])
    y_samples_train = np.array([list(y_train[id]) for id in id_train])
    x_samples_valid = np.array([list(x_train[id]) for id in id_valid])
    y_samples_valid = np.array([list(y_train[id]) for id in id_valid])
    return x_samples_train, y_samples_train, x_samples_valid, y_samples_valid

# function to get training, test, and validation sets using a defined number of test samples per object class and a defined percentage of the remaining training set for the validation set
# def get_training_and_test_and_validation_sets(x_samples, y_samples, num_samples, num_test, percent_validation):
#     x_samples_test, y_samples_test, x_samples_train, y_samples_train = get_training_and_test_sets(x_samples, y_samples, num_samples, num_test)
#     x_samples_train, y_samples_train, x_samples_valid, y_samples_valid = split_training_set_to_get_validation_set(x_samples_train, y_samples_train, percent_validation)
#     return x_samples_train, y_samples_train, x_samples_test, y_samples_test, x_samples_valid, y_samples_valid


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


    # set to any number (1-4,6) to determine which part of problem 2 will be executed by the program
    # note that numbers 1-4 correspond to 2.1.# (e.g. 2.1.3) whereas number 6 corresponds to problem 2.2
    NUM_PROBLEM_TO_COMPLETE = 1


    if (NUM_PROBLEM_TO_COMPLETE == 1):
        print("Completing problem (2.1.1)...")

        # define number of neighbors (K) and number of training samples per class (M) to test
        K = [1,2,3,5,10]
        M = [10,20,30,40,50]
        
        classification_errors = list()
        for k in K:
            errors_for_given_k = list()
            for m in M:
                # split dataset into training and test sets
                num_training_samples = m
                x_train, y_train, x_test, y_test = get_training_and_test_sets(x_total, y_total, num_samples, num_training_samples)  
                num_test_samples = y_test.shape[0]

                # train k-NN classifier 
                clf = KNC(n_neighbors=k,p=2)
                clf.fit(x_train,y_train.ravel())

                # make predictions and compute error
                y_pred = clf.predict(x_test)
                err = sum([1 for id,y_val in enumerate(y_test) if y_pred[id] != y_val[0]])*100.0/(1.0*num_test_samples) # error percentage
                errors_for_given_k.append(err)
            print("Done with k = " + str(k) + "!")
            classification_errors.append(errors_for_given_k)

        # plot computed classification error rates
        for id,k in enumerate(K):
            plt.figure()
            plt.plot(M, classification_errors[id], marker='o', linestyle='dashed', markersize='11')
            plt.title("(2.1.1) Error Rate vs Number of Training Samples with k = " + str(k))
            plt.xlabel("# of training samples per class")
            plt.ylabel("error rate (%)")
        plt.show()

    elif (NUM_PROBLEM_TO_COMPLETE == 2):
        print("Completing problem (2.1.2)...")

        # define number of neighbors (K) and number of training samples per class (M) to test
        K = [1,2,3,5,10]
        M = [10,20,30,40,50]
        
        classification_errors = list()
        for m in M:
            # split dataset into training and test sets
            num_training_samples = m
            x_train, y_train, x_test, y_test = get_training_and_test_sets(x_total, y_total, num_samples, num_training_samples)  
            num_test_samples = y_test.shape[0]
            errors_for_given_m = list()
            for k in K:
                # train k-NN classifier 
                clf = KNC(n_neighbors=k,p=2)
                clf.fit(x_train,y_train.ravel())

                # make predictions and compute error
                y_pred = clf.predict(x_test)
                err = sum([1 for id,y_val in enumerate(y_test) if y_pred[id] != y_val[0]])*100.0/(1.0*num_test_samples) # error percentage
                errors_for_given_m.append(err)
            print("Done with m = " + str(m) + "!")
            classification_errors.append(errors_for_given_m)

        # plot computed classification error rates
        for id,m in enumerate(M):
            plt.figure()
            plt.plot(K, classification_errors[id], marker='o', linestyle='dashed', markersize='11')
            plt.title("(2.1.2) Error Rate vs Number of Neighbors in k-NN with m = " + str(m))
            plt.xlabel("# of neighbors (k)")
            plt.ylabel("error rate (%)")
        plt.show()

    elif(NUM_PROBLEM_TO_COMPLETE == 3):
        print("Completing problem (2.1.3)...")

        # split dataset into training and test sets
        num_training_samples = 30
        x_train, y_train, x_test, y_test = get_training_and_test_sets(x_total, y_total, num_samples, num_training_samples)     
        num_test_samples = y_test.shape[0]
            
        # define number of neighbors (k) and p values (P) to use in p-norm of distance metric
        k = 3
        P = [1,2,3,5,10]

        classification_errors = list()
        for p in P:
            # train k-NN classifier using p value in distance metric
            clf = KNC(n_neighbors=k,p=p)
            clf.fit(x_train,y_train.ravel())

            # make predictions and compute error
            y_pred = clf.predict(x_test)
            err = sum([1 for id,y_val in enumerate(y_test) if y_pred[id] != y_val[0]])*100.0/(1.0*num_test_samples) # error percentage
            classification_errors.append(err)
            print("Done with p = " + str(p) + "!")

        # plot computed classification error rates
        plt.figure()
        plt.plot(P, classification_errors, marker='o', linestyle='dashed', markersize='11')
        plt.title("(2.1.3) Error Rate vs Distance Metric Parameter (k = " + str(k) + ")")
        plt.xlabel("'p' in distance metric p-norm")
        plt.ylabel("error rate (%)")
        plt.show()

    elif (NUM_PROBLEM_TO_COMPLETE == 4):
        print("Completing problem (2.1.4)...")

        # split dataset into training and test sets
        num_training_samples = 30
        x_train, y_train, x_test, y_test = get_training_and_test_sets(x_total, y_total, num_samples, num_training_samples)
        num_test_samples = y_test.shape[0]

        # reshape image representations from one image = one 1x1024 row of 2D array to one image = one 32x32 matrix of 3D array
        x_train_3d = x_train.swapaxes(0,1).reshape((32,32,-1),order='F')
        x_test_3d = x_test.swapaxes(0,1).reshape((32,32,-1),order='F')

        # represent images using lbp feature descriptors where one image descriptor = one 32x32 matrix of 3D array
        x_train_lbp = np.array([lbp(x_train_3d[:,:,id],P=16,R=2) for id in range(x_train_3d.shape[2])])
        x_test_lbp = np.array([lbp(x_test_3d[:,:,id],P=16,R=2) for id in range(x_test_3d.shape[2])])
        x_train_lbp_3d = x_train_lbp.swapaxes(1,2).swapaxes(0,2).reshape((32,32,-1),order='F')
        x_test_lbp_3d = x_test_lbp.swapaxes(1,2).swapaxes(0,2).reshape((32,32,-1),order='F')

        # represent images using lbp feature descriptors where one image descriptor = one 1x1024 row of 2D array
        x_train_lbp = x_train_lbp_3d.swapaxes(0,2).reshape((-1,1024),order='C')
        x_test_lbp = x_test_lbp_3d.swapaxes(0,2).reshape((-1,1024),order='C')

        # represent images using hog feature descriptors where one image descriptor = one 1x324 row of 2D array
        x_train_hog = np.array([hog(x_train_3d[:,:,id],orientations=18,pixels_per_cell=(4,4),cells_per_block=(2,2)) for id in range(x_train_3d.shape[2])])
        x_test_hog = np.array([hog(x_test_3d[:,:,id],orientations=18,pixels_per_cell=(4,4),cells_per_block=(2,2)) for id in range(x_test_3d.shape[2])])
         
        # define number of neighbors (k) and p values (P) to use in p-norm of distance metric
        k = 3
        P = [1,2]

        # define different types of image features to be used
        F = ["Pixel Intensities","HOG","LBP"]

        classification_errors = list()
        for f in F:
            errors_for_given_p = list()
            for p in P:
                # construct k-NN classifier using p value in distance metric
                clf = KNC(n_neighbors=k,p=p)

                # train k-NN classifier using appropriate feature choice, then make predictions
                if (f == "Pixel Intensities"):
                    clf.fit(x_train,y_train.ravel())
                    y_pred = clf.predict(x_test)
                elif (f == "HOG"):
                    clf.fit(x_train_hog,y_train.ravel())
                    y_pred = clf.predict(x_test_hog)
                elif (f == "LBP"):
                    clf.fit(x_train_lbp,y_train.ravel())
                    y_pred = clf.predict(x_test_lbp)

                # compute error in predictions
                err = sum([1 for id,y_val in enumerate(y_test) if y_pred[id] != y_val[0]])*100.0/(1.0*num_test_samples) # error percentage
                errors_for_given_p.append(err)
            print("Done with feature = '" + f + "'!")
            classification_errors.append(errors_for_given_p)
            
        # plot computed classification error rates
        for id,f in enumerate(F):
            plt.figure()
            plt.plot(P, classification_errors[id], marker='o', linestyle='dashed', markersize='11')
            plt.title("(2.1.4) Error Rate vs Distance Metric Parameter using Feature = " + f + " (k = " + str(k) + ")")
            plt.xlabel("'p' in distance metric p-norm")
            plt.ylabel("error rate (%)")
        plt.show()

    elif (NUM_PROBLEM_TO_COMPLETE == 6):
        print("Completing problem (2.1.6)...")

        # define number of neighbors (K) and p values (P) to use in p-norm of distance metric
        K = [1,2,3,5,10]
        P = [1,2,3,5]

        # define number of test samples per class and percent of training data to use for validation
        num_test_samples = 20
        percent_validation = 25

        # define number of folds for cross validation
        num_folds_cv = 5

        # split dataset into total training set and test set
        x_test, y_test, x_train_tot, y_train_tot = get_training_and_test_sets(x_total, y_total, num_samples, num_test_samples)

        min_params_list = list()
        for id in range(num_folds_cv):
            # split total training set into training and validation sets
            min_params = {'k':0,'p':0,'validation_error':100.0}
            x_train, y_train, x_valid, y_valid = split_training_set_to_get_validation_set(x_train_tot, y_train_tot, percent_validation)
            num_training_samples = y_train.shape[0]
            num_valid_samples = y_test.shape[0]

            # for each pair of k and p
            for k in K:
                for p in P:
                    # construct k-NN classifier using p value in distance metric
                    clf = KNC(n_neighbors=k,p=p)

                    # train classifier and make predictions on validation set
                    clf.fit(x_train,y_train.ravel())
                    y_pred = clf.predict(x_valid)

                    # compute classification error on validation set and update minimum parameters if applicable
                    err = sum([1 for id,y_val in enumerate(y_valid) if y_pred[id] != y_val[0]])*100.0/(1.0*num_valid_samples) # error percentage
                    if (err < min_params['validation_error']):
                        min_params['k'] = k
                        min_params['p'] = p
                        min_params['validation_error'] = err
            min_params_list.append(min_params)
            print("Done with fold " + str(id+1) + "!")

        # find the parameters that performed best on average over all folds
        max_count_instances_kp_pair = 0
        avg_min_params = {'k':0,'p':0,'validation_error':100.0,'test_error':100.0}
        for min in min_params_list:
            count_instances_kp_pair = sum([1 for m in min_params_list if (m['k'] == min['k'] and m['p'] == min['p'])])
            if (count_instances_kp_pair > max_count_instances_kp_pair):
                max_count_instances_kp_pair = count_instances_kp_pair
                avg_min_params['k'] = min['k']
                avg_min_params['p'] = min['p']
                avg_min_params['validation_error'] = sum([m['validation_error'] for m in min_params_list if (m['k'] == min['k'] and m['p'] == min['p'])])/(1.0*count_instances_kp_pair)
        
        # compute test error using best parameters
        clf = KNC(n_neighbors=avg_min_params['k'],p=avg_min_params['p'])
        clf.fit(x_train,y_train.ravel())
        y_pred = clf.predict(x_test)
        avg_min_params['test_error'] = sum([1 for id,y_val in enumerate(y_test) if y_pred[id] != y_val[0]])*100.0/(1.0*(y_test.shape[0])) # error percentage

        # print results
        print("\nBest parameters found during cross validation:")
        print(avg_min_params)     
        print("\n")

    else:
        print("Please set 'NUM_PROBLEM_TO_COMPLETE' variable at top of main() function to a number on {1,2,3,4,6} and run again.")



if __name__ == "__main__":
    main()
