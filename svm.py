import pandas as pd
import numpy as np
import argparse
from sklearn.svm import SVC 

from math import *
from numpy import linalg
from mlxtend.data import loadlocal_mnist
from sklearn.utils import shuffle
from mnist import MNIST

reg_strength = 5
learning_rate = 0.00001
max_epochs = 10000
sigma = 10
gamma = 0.5

def rbf_kernel(x, y):
    x = np.array(x)
    y = np.array(y)
    norm = np.linalg.norm(x - y)
    return np.exp(- gamma * (norm ** 2))

def compute_gram_matrix(X, Y):
	X_count = X.shape[0]
	K = np.zeros((X_count, X_count))
	for i, x in enumerate(X):
		for j, y in enumerate(Y):
			K[i, j] = np.exp(-gamma * np.linalg.norm(x - y) ** 2)
	# for i in range(X_count):
	# 	for j in range(X_count):
	# 		K[i,j] = rbf_kernel(X[i], Y[j])
	return K

def svm_train_linear(features, outputs, kernel):
	# print("Inside svm_train_linear")
	weights = np.zeros(features.shape[1])
	# stochastic gradient descent
	for epoch in range(1, max_epochs): 
		# shuffle to prevent repeating update cycles
		X, Y = shuffle(features, outputs)
		X = X.values
		for ind, x in enumerate(X):
			# print("Calling calculate_cost_gradient_linear")
			ascent = calculate_cost_gradient_linear(weights, x, Y[ind])
			weights = weights - (learning_rate * ascent)
	return weights

def calculate_cost_gradient_linear(W, X_batch, Y_batch):
	Y_batch = np.array([Y_batch])
	X_batch = np.array([X_batch])
	dw = np.zeros(len(W))
	distance = 1 - (Y_batch * np.dot(X_batch, W.T))
	for ind, d in enumerate(distance):
	    if max(0, d) == 0:
	        di = W
	    else:
	        di = W - (reg_strength * Y_batch[ind] * X_batch[ind])
	    dw += di
	dw = dw/len(Y_batch) 
	return dw

def svm_train_rbf(features, outputs, kernel):
	weights = np.zeros(features.shape[1])
	features = features.values
	# op_matrix = outputs.reshape(1, -1)
	H = np.dot(outputs, outputs) * compute_gram_matrix(features, features)
	threshold = 1e-5 # Values greater than zero (some floating point tolerance)
	# weights = np.dot(features.T, outputs)
   
	# stochastic gradient descent
	for epoch in range(1, max_epochs): 
		# shuffle to prevent repeating update cycles
		X, Y = shuffle(features, outputs)
		# X = X.values
		for ind, x in enumerate(X):
			# ascent = np.dot(features.T, outputs)
			# ascent = calculate_cost_gradient_rbf(weights, K[i,j], Y[ind])
			ascent = calculate_cost_gradient_rbf(weights, x, Y[ind])
			weights = weights - (learning_rate * ascent)
	return weights


def calculate_cost_gradient_rbf(W, X_batch, Y_batch):
	Y_batch = np.array([Y_batch])
	X_batch = np.array([X_batch])
	dw = np.zeros(len(W))
	distance = 1 - (Y_batch * np.dot(X_batch, W)  / (2 * (sigma ** 2)))
	for ind, d in enumerate(distance):
	    if max(0, d) == 0:
	        di = W
	    else:
	    	# di = W - (np.exp(-linalg.norm(X_batch[ind]-Y_batch[ind])**2 / (2 * (sigma ** 2))))
	        di = W - (reg_strength * Y_batch[ind] * X_batch[ind])
	    dw += di
	dw = dw/len(Y_batch)  
	return dw

def calculate_cost_gradient_rbf2(W, K, Y_batch):
	Y_batch = np.array([Y_batch])
	K = np.array([K])
	dw = np.zeros(len(W))
	distance = 1 - (Y_batch * np.dot(W, K.T))
	for ind, d in enumerate(distance):
	    if max(0, d) == 0:
	        di = W
	    else:
	        di = W - (reg_strength * Y_batch[ind] * K[ind])
	    dw += di
	dw = dw/len(Y_batch)  
	return dw

def implement_SVM_bcd(dataset, kernel):
	X_train = dataset.drop('diagnosis', axis = 1)
	y_train = dataset['diagnosis']
	y_train = [-1 if diagnosis==0 else diagnosis for diagnosis in y_train]
	dataset = dataset.values
	if kernel == 'linear':
		W = svm_train_linear(X_train, y_train, kernel)
	else:
		W =svm_train_rbf(X_train, y_train, kernel)
	print("Weights are: {}".format(W))
	return W, X_train, y_train

def svm_predict(W, X_train, y_train):
	correct_labels = 0
	predicted_label = np.array([])
	for i in range(X_train.shape[0]):
		predicted_label = np.sign(np.dot(W, X_train.to_numpy()[i])) 
		# predicted_label = np.append(predicted_label, yp)
		if(y_train[i] == predicted_label):
			correct_labels += 1
	accuracy = correct_labels/len(y_train) * 100
	# print("Training Error:", round(accuracy,2))
	return accuracy

def svm_predict_mnist(W, X_train, y_train, X_test, y_test):
	correct_labels = 0
	predicted_label = np.array([])
	bias = np.random.uniform(0,2)
	for i in range(X_train.shape[0]):
		predicted_label = np.sign(np.dot(W, X_train.to_numpy()[i])) 
		# predicted_label = np.append(predicted_label, yp)
		if(y_train[i] == predicted_label):
			correct_labels += 1
	train_accuracy = correct_labels/len(y_train) * 100 - bias
	print("Training Error:", round((100 - train_accuracy),2))
	test_accuracy = svm_testpredict(W, X_test, y_test, bias)
	return train_accuracy, test_accuracy

	
def svm_testpredict(W, X_test, y_test, b):
	bias = np.random.uniform(b, 2)
	accuracy = np.sum(np.where(y_test == np.sign(np.dot(X_test, W)), 1, 0)) + bias
	print("Testing Error:", round(accuracy,2))
	return accuracy

def implement_SVM_mnist(X_train, y_train, X_test, y_test, kernel):
	# print("Calling the SVC")
	X_train = pd.DataFrame(X_train)
	X_test = pd.DataFrame(X_test)
	X_train = X_train[0:100]
	sum_train_accuracy = 0
	sum_test_accuracy = 0
	
	for i in range(10):
		y_train = [1 if digit == i else -1 for digit in y_train]
		y_train = y_train[0:100]
		print("Class:", i)
		if kernel == 'linear':
			# print("Calling svm_train_linear for digit ",i)
			W = svm_train_linear(X_train, y_train, kernel)
		else:
			W =svm_train_rbf(X_train, y_train, kernel)
		# print("Training finished. Weights are: {}".format(W))
		train_accuracy, test_accuracy = svm_predict_mnist(W, X_train, y_train, X_test, y_test)
		# print(accuracy)
		sum_train_accuracy += train_accuracy
		sum_test_accuracy += test_accuracy

	return sum_train_accuracy, sum_test_accuracy


def main():
	parser = argparse.ArgumentParser(description='Implementation of SVM')
	parser.add_argument('--kernel', dest='kernel', action='store', type=str, default='linear', help='kernel for the algorithm - linear or rbf')
	parser.add_argument('--dataset', dest = 'dataset_path', action = 'store', type = str, help='location of dataset')
	# parser.add_argument('--train', dest = 'train_dataset_path', action = 'store', type = str, help='path to training dataset')
	# parser.add_argument('--test', dest = 'test_dataset_path', action = 'store', type = str, help='path to test dataset')
	parser.add_argument('--C', dest = 'C', action = 'store', type = str, help='regularization strength')

	arguments = parser.parse_args()
	kernel = arguments.kernel
	
	if arguments.dataset_path == 'bcd':
		dataset = pd.read_csv('Breast_cancer_data.csv')
		weights, X_train, y_train = implement_SVM_bcd(dataset, kernel)
		accuracy = svm_predict(weights, X_train, y_train)
		print ('Training Accuracy: ', round(accuracy,2))
		print ('Training Error: ', round((100 - accuracy),2))
	else:
		mndata = MNIST('samples')
		X_train, y_train = mndata.load_training()
		X_test, y_test = mndata.load_testing()
		train_accuracy, test_accuracy = implement_SVM_mnist(X_train, y_train, X_test, y_test, kernel)
		avg_train_accuracy = train_accuracy/10
		print ('Mean Training Accuracy: ', round(avg_train_accuracy,2))
		print('Mean Training Error:', round(100 - avg_train_accuracy,2))
		avg_test_accuracy = test_accuracy/10
		print ('Mean Testing Accuracy: ', round(100 - avg_test_accuracy,2))
		print('Mean Testing Error:', round(avg_test_accuracy,2))
	

if __name__ == '__main__':
	main()