import os
#import cv2
import pickle
import numpy as np
import pdb
import requests
from collections import defaultdict
import random 
import time
import sys
import math
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import FastICA as ICA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier #sas CART
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import *
from sklearn.neural_network import MLPClassifier
from functools import wraps 
from time import time as _timenow 
from sys import stderr
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
#from sklearn.model_selection import GridSearchCV




def load_cifar():
    
    trn_data, trn_labels, tst_data, tst_labels = [], [], [], []
    def unpickle(file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
            print(np.shape(data))
        return data
    
    

    for i in trange(1):
        #print('Kai')
        #print(i)
        #batchName = './data/data_batch_{0}'.format(i + 1)
        batchName = './data/data_batch_{0}'.format(i + 1)
        #print(batchName)
        unpickled = unpickle(batchName)
        #print(unpickled)
        trn_data.extend(unpickled['data'])
        trn_labels.extend(unpickled['labels'])
    unpickled = unpickle('./data/test_batch')
    tst_data.extend(unpickled['data'])
    tst_labels.extend(unpickled['labels'])
    return trn_data, trn_labels, tst_data, tst_labels


def image_prep(image):
	

	#scaler = 




	#return tf.image.per_image_standardization(tf.image.rgb_to_grayscale(np.reshape(image,(32,32,3))))
	nc = len(image)

    #mu = 0
    #sigma = 0

    #processed_image = np.zeros(nc)



    #for i in trange(nc):
    #	mu = mu + image[i]

    #for i in trange(nc):
    #	processed_image[i] = image[i] - mu


    #for i in trange(nc):
    #	sigma = sigma + processed_image[i]*processed_image[i]


    #sigma = math.sqrt(sigma)

    #for i in trange(nc):
    #	processed_image[i] = processed_image[i]/sigma

    #return processed_image


def mu_sigma(matrix,red):
	#noOfsamplesXnoOffeatures
	scaler = StandardScaler()
	#scaler.fit(matrix)

	scaler.fit(matrix)
	matrix = scaler.transform(matrix)
	red = scaler.transform(red)
	transformer = Normalizer()
	transformer.fit(matrix)
	transformer.fit(red)

	matrix = transformer.transform(matrix)
	red = transformer.transform(red)


	return matrix , red



	nr , nc = np.shape(matrix)
	


	mu = np.zeros( nc, dtype = int) 
	sigma = np.zeros(nc, dtype = int)


	print(nr)
	nmatrix = np.zeros([nr,nc], dtype = int)




	for j in trange(nc):
		for i in trange(nr):
			#print(i,j)
			mu[j] = mu[j] + matrix[i][j]

		mu[j] = mu[j]*(1/nr)

		for i in trange(nr):
			nmatrix[i][j] = matrix[i][j] - mu[j]

	for j in trange(nc):
		for i in trange(nr):
			sigma[j] = sigma[j] + nmatrix[i][j]*nmatrix[i][j]

		sigma[j] = math.sqrt(sigma[j]*(1/nr))

		for i in trange(nr):
			nmatrix[i][j] = nmatrix[i][j]/sigma[j]


	return mu, sigma, nmatrix



#kwargs contains method-PCA LDA or ICA, data - the samples , y - labels , n_components - No of componenets  (hyperparameter)
def reduce_dim(data,labels,n_components,**kwargs):
    ''' performs dimensionality reduction'''
    if kwargs['method'] == 'pca':
        

        matrix = data
        #transformer = Normalizer()
        #transformer.fit(matrix)



        pca = PCA(n_components=n_components,svd_solver = 'full')
        pca.fit(matrix)
        #return pca.fit_transform(matrix)
        #pass
        return pca.transform(matrix)
        
    if kwargs['method'] == 'lda':
    	transformer = Normalizer()


    	label = labels
    	matrix = data
    	transformer.fit(matrix)
    	lda = LDA(n_components = n_components)
    	lda.fit(transformer.transform(matrix),label)
    	return lda.transform(matrix)
        #pass    


    if kwargs['method'] == 'ica':
    	
    	matrix = data
    	ica = ICA(n_components = n_components,random_state = 0)
    	return ica.fit_transform(matrix)


    	
#train the classifier - X is noOfsamplesXnoOffeatures ; y is noOfsamples ,method - type of classifier
def classify(data,labels,**kwargs):
    ''' trains a classifier by taking input features
        and their respective targets and returns the trained model'''
    
    if kwargs['method'] == 'LR':

    	clf = LogisticRegression(random_state=0, solver='sag',multi_class='multinomial',C = 1.4,max_iter = 200)
    	matrix = data
    	label = labels
    	return clf.fit(matrix,label)




    if kwargs['method'] == 'SVM':

        clf = LinearSVC(random_state=0, tol=1e-5,C=2)
        matrix = data
        label = labels
        return clf.fit(matrix,label)

        pass


    if kwargs['method'] == 'RF':
    	clf = RandomForestClassifier( criterion = 'gini',oob_score = 'True',min_weight_fraction_leaf=0.0,bootstrap = 'True',n_estimators=100, max_depth=20,random_state=0, min_samples_leaf=2, min_samples_split=3)
    	matrix = data
    	label = labels


    	return clf.fit(matrix, label)



    if kwargs['method'] == 'CART':
    	clf = DecisionTreeClassifier(random_state=0)
    	matrix = data
    	label = labels
    	return clf.fit(matrix,label)


    if kwargs['method']	== 'KSVM':
    	matrix = data
    	label = labels
    	#parameters = {'kernel':'rbf', 'C':10}


    	#svc = SVC(gamma = 'scale')
    	#clf = GridSearchCV(svc, parameters, cv=5)

    	clf = SVC(kernel = 'rbf',C = 100,gamma = 0.01)

    	return clf.fit(matrix,label)

    if kwargs['method'] == 'MLP':
    	matrix = data
    	label = labels
    	clf = MLPClassifier(activation= 'relu',solver='sgd', alpha=0.00001,hidden_layer_sizes=(100,40),momentum = 0.9,power_t = 0.7, random_state=1,max_iter=1000,nesterovs_momentum = True,learning_rate_init = 0.001,learning_rate = 'adaptive')
    	return clf.fit(matrix,label)

    	
def evaluate(target, predicted):
    f1 = f1_score(target, predicted, average='micro')
    acc = accuracy_score(target, predicted)
    return f1, acc

def test(data,labels,clf,**kwargs):

	#classifier = kwargs['clf']
	classifier = clf

	target = labels

	test = data

	predicted = classifier.predict(test)

	return evaluate(target, predicted)
	'''takes test data and trained classifier model,
    performs classification and prints accuracy and f1-score'''
    

def main():
    trn_data, trn_labels, tst_data, tst_labels = load_cifar()
    #trn_datas = list(map(lambda x : tf.image.rgb_to_grayscale(np.reshape(x, (32,32,3))),trn_data))
    trn_data, tst_data = mu_sigma(trn_data,tst_data)



    #trn_data, tst_data = list(map(image_prep, trn_data)), list(map(image_prep, tst_data))
    X_train, X_val, y_train, y_val = train_test_split(trn_data, trn_labels,test_size = 0.20) 
    ''' perform dimesioality reduction/feature extraction and classify the features into one of 10 classses
        print accuracy and f1-score.
        '''
    #mu, sigma, nmatr3072= mu_sigma(trn_data)

    print(np.shape(X_train))
    print(np.shape(X_val))
    print(np.shape(y_train))
    print(np.shape(y_val))

    #sys.exit()






    rPCA = reduce_dim(X_train,y_train,800, method ='pca')    
    #rLDA = reduce_dim(X_train,y_train,3072,method ='lda')
    #rICA = reduce_dim(X_train,y_train,256,method='ica')
    #rRAW = X_train

    rLabel = y_train

    #print(np.shape(rPCA))
    #print(np.shape(rLDA))
    #print(np.shape(rRAW))
    #print(np.shape(rICA))

    oPCA = reduce_dim(tst_data,tst_labels,800,method = 'pca')
    #oLDA = reduce_dim(tst_data,tst_labels,256,method = 'lda')
    #oRAW = tst_data

    oLabel = tst_labels

    #vPCA = reduce_dim(y_train,y_val,100,method = 'pca')

    #vLabel = y_val



    #sys.exit()

    #cLRp = classify(rPCA,rLabel,method = 'LR')
    #cLRl = classify(rLDA,rLabel,method = 'LR')
    #cLRi = classify(rICA,rLabel,method = 'LR')
    #cLRr = classify(rRAW,rLabel,method = 'LR')





    #cSVMp = classify(rPCA,rLabel,method = 'SVM')
    #cSVMl = classify(rLDA,rLabel,method = 'SVM')
    #cSVMi = classify(data=rICA,labels = rLabel,method = 'SVM')
    #cSVMr = classify(rRAW,rLabel,method = 'SVM')


    #cMLPp = classify(rPCA,rLabel,method = 'MLP')
    #cMLPl = classify(rLDA,rLabel,method = 'MLP')
    #cMLPi = classify(data=rICA,labels = rLabel,method = 'MLP')
    #cMLPr = classify(rRAW, rLabel,method = 'MLP')

    cKSVMp = classify(rPCA, rLabel,method = 'KSVM')
    #cKSVMl = classify(rRAW, rLabel,method = 'KSVM')
    #cKSVMi = classify(data=rICA,labels = rLabel,method = 'KSVM')
    #cKSVMr = classify(rRAW, rLabel,method = 'KSVM')

    #cCARTp = classify(rPCA, rLabel,method = 'CART')
    #cCARTl = classify(rLDA, rLabel,method = 'CART')
    #cCARTi = classify(data=rICA,labels = rLabel,method = 'CART')
    #cCARTr = classify(rRAW, rLabel,method = 'CART')

    #cRFp = classify(rPCA, rLabel,method = 'RF')
    #cRFl = classify(rLDA, rLabel,method = 'RF')
    #cRFi = classify(rICA, rLabel,method = 'RF')
    #cRFr = classify(rRAW, rLabel,method = 'RF')


    F, A = test(oPCA,oLabel,cKSVMp)

    print(F)
    print(A)

    sys.exit()



    #print(np.shape(X_train))
    #print(np.shape(X_val))
    #print(np.shape(y_train))
    #print(np.shape(y_val))





    #print(X_val)

    #print(trn_data)
    #print(trn_labels)



    


    print('Val - F1 score: {}\n Accuracy: {}'.format(f_score, accuracy_))    



if __name__ == '__main__':
    main()





