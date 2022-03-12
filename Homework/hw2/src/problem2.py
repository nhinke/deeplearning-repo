# Nick Hinke
# 520.638 Deep Learning
# Homework 2 - Problem 2
#
# Train and test 10 perceptrons on the MNIST handwritten digit dataset (using the first 20k training samples and 2k test samples)
# The 'i-th' percecptron in the model will perform binary classification to determine whether or not the sample is an instance of digit 'i'
# The 10 perceptrons will then be combined into a single 10-way model that outputs the class of whichever sub-binary-classifier is most confident
#

# import relevant libraries
import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
# from sklearn.linear_model import SGDClassifier

# function to extract datasets and labels as numpy arrays from the four .gz folders available at http://yann.lecun.com/exdb/mnist/
def get_datasets_from_gz_folders(im_size, num_test, num_train):

    # try to load datasets from .gz folders
    try:
        training_images = gzip.open('train-images-idx3-ubyte.gz','rb')
        training_labels = gzip.open('train-labels-idx1-ubyte.gz','rb')
        test_images = gzip.open('t10k-images-idx3-ubyte.gz','rb')
        test_labels = gzip.open('t10k-labels-idx1-ubyte.gz','rb')
    except:
        print("Please ensure that the four MNIST .gz folders are saved in the correct directory.")
        quit()
    
    # convert training images to numpy array of shape (num_train,im_size*im_size)
    training_images.read(16)
    buf = training_images.read(im_size*im_size*num_train)
    training_images = np.frombuffer(buf,dtype=np.uint8).astype(np.float32)
    training_images = training_images.reshape(-1,im_size*im_size)
    
    # convert training labels to numpy array of shape (num_train,)
    training_labels.read(8)
    buf = training_labels.read(num_train)
    training_labels = np.frombuffer(buf,dtype=np.uint8)

    # convert test images to numpy array of shape (num_test,im_size*im_size)
    test_images.read(16)
    buf = test_images.read(im_size*im_size*num_test)
    test_images = np.frombuffer(buf,dtype=np.uint8).astype(np.float32)
    test_images = test_images.reshape(-1,im_size*im_size)

    # convert test labels to numpy array of shape (num_test,)
    test_labels.read(8)
    buf = test_labels.read(num_test)
    test_labels = np.frombuffer(buf,dtype=np.uint8)

    return training_images, training_labels, test_images, test_labels


def main():

    # define image size and number of images to keep from each dataset
    image_size = 28
    num_test_samples = 2000
    num_training_samples = 20000

    # get datasets as numpy arrays with the following shapes: {x_train,y_train} = {(num_training_samples,784),(num_training_samples,)} and {x_test,y_test} = {(num_test_samples,784),(num_test_samples,)}
    x_train, y_train, x_test, y_test = get_datasets_from_gz_folders(image_size, num_test_samples, num_training_samples)
    
    # reshape images from row vectors to 28x28 matrices for visualization purposes (e.g. plt.imshow(x_test_images[image_index]))
    x_test_images = x_test.reshape(num_test_samples, image_size, image_size, 1)
    x_train_images = x_train.reshape(num_training_samples, image_size, image_size, 1)
    
    # append a column of ones to the matrices of images corresponding to the bias term
    x_test = np.hstack((np.ones(shape=(num_test_samples,1)),x_test))
    x_train = np.hstack((np.ones(shape=(num_training_samples,1)),x_train))

    # train 10 binary classifiers (using perceptrons) where the i-th classifier predicts whether or not the digit in the image = i
    num_digits = 10
    perceptron_list = list()
    for id in range(num_digits):
        y_train_id = [1 if yi == id else 0 for yi in y_train]
        perceptron_list.append(Perceptron().fit(x_train,y_train_id))
        # perceptron_list.append(SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None).fit(x_train,y_train_id)) // exactly equivalent to Perceptron class used above
        print("Done with training classifier for digit " + str(id) + "...")

    # compute classification errors on test dataset for each binary classifier
    test_error_list = list()
    print("Computing test error for each binary classifer individually...")
    for id,clf in enumerate(perceptron_list):
        y_test_id = [1 if yi == id else 0 for yi in y_test]
        y_pred = clf.predict(x_test)
        misclass_rate = sum([1 for id,y_val in enumerate(y_test_id) if y_pred[id] != y_val])*100.0/(1.0*num_test_samples) # error percentage
        test_error_list.append(misclass_rate)
    
    # plot test errors for each binary classifier
    plt.figure()
    plt.bar(range(num_digits),test_error_list)
    plt.title("(2.1) Misclassification Rate of each Digit's Binary Classifier")
    plt.xlabel('classifier digit')
    plt.ylabel('error rate (%)')
    plt.gca().set_xticks(range(num_digits))

    # print results
    print("\n(2.1) Misclassification Rate (%) of each Binary Classifier:")
    for id in range(num_digits):
        print("Digit %d:  %.2f" % (id,test_error_list[id]))
    print()
    
    # compute classification errors on test dataset for combined 10-way model (model predicition = class of sub-classifier (one for each digit) with highest confidence)
    y_pred = list()
    print("Computing test error of combined 10-way model...")
    for id,test_im in enumerate(x_test):
        decision_fn_vals = [clf.decision_function(test_im.reshape(1,-1))[0] for clf in perceptron_list]
        y_pred_id = np.argmax(decision_fn_vals)
        y_pred.append(y_pred_id)
    num_error = sum([1 for id,y_val in enumerate(y_test) if y_pred[id] != y_val])
    num_success = sum([1 for id,y_val in enumerate(y_test) if y_pred[id] == y_val])
    misclass_rate = num_error*100.0/(1.0*num_test_samples) # error percentage
    # success_rate = num_success*100.0/(1.0*num_test_samples)
    
    # print results
    print("\n(2.2) 10-Way Model Performance on Test Set:")
    print("Erroneous classifications:   %d" % num_error)
    print("Successful classifications:  %d" % num_success)
    print("Misclassification rate (%%):  %.1f\n" % misclass_rate)

    # display plot
    plt.show()



if __name__ == "__main__":
	main()
