import numpy as np
from PIL import Image
import sys
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def samplearray(path2train):
	
	X = []
	lbarr = []
	contents = open(path2train,"r")
	for line in contents:
		temp = line.split(" ")
		img = Image.open(temp[0]).convert('L')
		#img.resize((36,36),Image.ANTIALIAS)
		img.load()
		lbarr.append(temp[1])
		arr = []
		data = np.asarray( img, dtype="int32")
		numrows, numcols = np.shape(data)
		for k in range(numrows):
			for l in range(numcols):
				arr.append(data[k][l])

		X.append(arr)
	

	#print(lbarr)
	#sys.exit() 
	return  X, lbarr




def pca(arr):
	arr_mm = []
	numrows, numcols = np.shape(arr)

	for i in range(numcols):
		mean = 0
		for j in range(numrows):
			mean += arr[j][i]
		mean = mean/numrows
		
		arr_mm.append(mean)
		for k in range(numrows):
			arr[k][i] -= mean
	
	#print(arr)
	#sys.exit()

	arrt = np.transpose(arr)#dxN 
	kernel = np.matmul(arr,arrt)
	#print(kernel)
	#sys.exit()

	eigv,eigvec = np.linalg.eig(kernel)
	#print(eigvec)
	#sys.exit()
	eigvec = np.transpose(eigvec)

	E = []
	
	for y in range(len(eigv)):
		E.append((eigv[y],eigvec[y]))
	  

	basis = []
	#print(E)
	#sys.exit()
	E.sort(key=lambda x: x[0], reverse=True)
	#print(E)
	#sys.exit()


	for y in E:
		basis.append(y[1])
	
	#print(basis)
	#sys.exit()
	basis = np.transpose(basis)  
	pcs = basis[:,:32]
	#print(pcs)
	#sys.exit()

	actualbasis = np.matmul(arrt,pcs) #dxk
	#print(actualbasis)
	#sys.exit()
	actualbasis = np.transpose(actualbasis) 
	#now we normalize actualbasis
	#print(actualbasis)
	#sys.exit()

	dimvec, numvec = np.shape(actualbasis)
	for y in range(dimvec):
		norm = 0
		for p in range(numvec):
		    norm += actualbasis[y][p]*actualbasis[y][p]

		norm = np.sqrt(norm)	
		for p in range(numvec):
			actualbasis[y][p] = actualbasis[y][p]/norm

#this arr is samplea rowwise i.e., each sample is a row and the actual basis is column wise so arrred is also row wise
	actualbasis = np.transpose(actualbasis)
	#print(actualbasis)
	#sys.exit()


	arrred = np.matmul(arr,actualbasis)
	#print(arrred)
	#sys.exit()
	#print(arr_mm)
	#sys.exit()

	return arrred , actualbasis, arr_mm



def group_classes(arrred,labelsarray):
	#print(arrred)
	#sys.exit()
	#print(labelsarray)
	#sys.exit()

	classes = dict()
	
   
	numrows, numcols = np.shape(arrred)
	#print(numrows)
	#sys.exit()

	for i in range(numrows):
		label = labelsarray[i]
		#print(label)
		if  label in classes:
			#classes[label] = []
			classes[label].append(arrred[i])
			#print(arrred[i])
			#print(classes)
			#sys.exit()

		else:
			#print(classes)
			#sys.exit()
			#print(arrred[i])
			classes[label] = []
			classes[label].append(arrred[i])
			#print(classes)
			#sys.exit()
		#print(classes)
		#sys.exit()

	#sys.exit()	
	#for l in classes:
	#	print(l)

	#	print(np.shape(classes[l]))

	
	#sys.exit()	

    #classes = dict()
	#nsamples, d = np.shape(nx) #Nxk
	#print(nsamples)
	#sys.exit()
	  
	return classes


def param_est(classes,samnum):

	#print(classes)
	#sys.exit()



	params = dict()
	for label in classes:
		u = []
		numexs, numfeatures = np.shape(classes[label])
		#print(numexs)
		#print(numfeatures)
		#sys.exit()

		#feature mean and covariance:
		for i in range(numfeatures):
			mean = 0
			for j in range(numexs):
				mean = mean + classes[label][j][i]

			mean = mean/numexs
			
			for j in range(numexs):
				classes[label][j][i] = classes[label][j][i] - mean

			u.append(mean)
			

		sigma = np.matmul(np.transpose(classes[label]),classes[label])
		numexs = numexs/samnum[0]
		params[label] = (u, sigma, numexs)

	#print(params)
	#sys.exit()	


	return params    




def classify2(arr_mm,path2test,actualbasis):
	#shape arr_mm = shape features

	contents = open(path2test, "r")
	R = []
	countr = 0
	for line in contents:
		countr += 1
		temp = line.split('\n')
		img = Image.open(temp[0]).convert('L') 
		#img.resize((36,36),Image.ANTIALIAS)
		img.load()
		features = []
		data = np.asarray( img, dtype = "int32")
		numrows, numcols = np.shape(data)
		for i in range(numrows):
			for j in range(numcols):
				features.append(data[i][j])

		features = np.array(features)

		arr_mm = np.array(arr_mm)

		features = features - arr_mm

		reducedfeatures = np.matmul(features,actualbasis)
		reducedfeatures = np.array(reducedfeatures)

		#kat = np.transpose(reducedfeatures)

		shape = np.shape(reducedfeatures)

		#print(shape)
		#sys.exit()

		redfp = np.zeros((1,33))
		#print(np.shape(redfp))
		#sys.exit()
		i = 0

		while (i < 33):
			if (i == 0):
				redfp[0][i] = 1

			else:
				redfp[0][i] =reducedfeatures[i-1]
			i += 1	

		




		R.append(redfp)

	nR = np.zeros((countr,33))
	p = 0
	while (p<countr):
		q = 0
		while (q<33):
			nR[p][q] = R[p][0][q]
			q += 1

		p += 1	


	#print(np.shape(nR))	



	return nR			

        


















def classify(arr_mm,path2test,actualbasis,dictofclasses):
	#shape arr_mm = shape features

	contents = open(path2test, "r")
	labels = []
	for line in contents:
		temp = line.split('\n')
		img = Image.open(temp[0]).convert('L') 
		#img.resize((64,64),Image.ANTIALIAS)
		img.load()
		features = []
		data = np.asarray( img, dtype = "int32")
		numrows, numcols = np.shape(data)
		for i in range(numrows):
			for j in range(numcols):
				features.append(data[i][j])

		features = np.array(features)

		arr_mm = np.array(arr_mm)

		features = features - arr_mm

		reducedfeatures = np.matmul(features,actualbasis)

		xlass = 0
		xval = -1*(10e+20)
		i = 0

		for i in dictofclasses:
			redfea = np.zeros(32)
			#j = 0

			for j in range(32):
				redfea[j] = reducedfeatures[j]
				
			g = 0
			g = g + -0.5*math.log(abs(np.linalg.det(dictofclasses[i][1]))) 
			for k in range(32):
				redfea[k] = redfea[k] - dictofclasses[i][0][k]

			g = g + -0.5*np.matmul(np.matmul(redfea, np.linalg.inv(dictofclasses[i][1])),np.transpose(redfea))  + math.log(dictofclasses[i][2])  
			if(g>xval):
				xval = g
				xlass = i


		labels.append(xlass)

	return labels




def accuracy(labels,path2test):
	truths = 0
	contents = open(path2test,"r")
	numtest = 0
	for line in contents:
		names = line.split('/')
		cname = names[2].split(" ")

		if(str(cname[0]) == "000"  and labels[numtest].strip() == "alice"):
			truths += 1
		elif(str(cname[0]) == "002"  and labels[numtest].strip() == "bob"):
			truths += 1
		elif(str(cname[0]) == "005"  and labels[numtest].strip() == "abc"):
			truths += 1

		numtest += 1
			
	return truths, numtest


def totallabels(labelsarray):
	ulabs = np.unique(labelsarray)
	numuniqs = len(ulabs)

	return ulabs , numuniqs

def assignclass(labelsarray,uniqs):
	i =0
	aiglabs = []
	while i<len(labelsarray):
		k = 0
		while k < len(uniqs):
			if (uniqs[k] == labelsarray[i]):
				aiglabs.append(k)
			k += 1
		i += 1

	return aiglabs			



    
def sigmoid(weights,arrreda,numun,nsams):
	k = 0
	i = 0
	H = np.zeros((numun,nsams))


	while (k < nsams):
		x = arrreda[k,:]
		x = np.transpose(x)
		#print(x)
		expsum = 0
		i = 0
		while (i < numun):
			
			w = np.array(weights[i,:])
			dot = np.matmul(w,x)
			#print(dot)
			#sys.exit()
			#print(k)
			dote = math.exp(dot)
			#print(dote)
			H[i][k] = dote
			#print(dote)
			expsum += dote
			i += 1

		#print(expsum)
		
		#sys.exit()
		#if (k == 33):
		#	sys.exit()
		#sys.exit()


		i = 0
		while(i<numun):
			#print("a")
			#print(expsum)
			#sys.exit()
			H[i][k] = H[i][k]/expsum
			i += 1
			#print (H[i][k])

		k += 1
	
	#print("a")

	#print(H)
	
   
	#sys.exit()	
	
	return H		

    #H is column wis]



def gradient(H,aiglabs,arrreda,ylabs):
	grad = np.zeros((numun,33))
	#gradient is rowwise

	i = 0
	while (i<numun):
		j =0
		sumo = np.zeros((1,33))
		sumy = np.zeros((1,33))
		#print(sumo)
		


		while (j<nsams):
			x = arrreda[j,:]
			sumo += (H[i][j]- ylabs[aiglabs[j]][i])*x
			sumy += (0.2 - ylabs[aiglabs[j]][i])*x
			#print(sumy)
			#if (j == 3):
			#	sys.exit()



			j += 1

		#i += 1

		k = 0
		while (k < 33):
			grad[i][k] = sumo[0][k]/nsams
			#print(sumo[0][k])
			k += 1

		i += 1	

	#print(grad)	
	#print("hi")
	


	#sys.exit()	


	return grad		
		



def softmax(weights,eta,arrreda,aiglabs,ylabs,niter,numun,nsams):

	i = 0
	while (i < niter):
		i += 1
		H = sigmoid(weights,arrreda,numun,nsams)
		grad = gradient(H,aiglabs,arrreda,ylabs)
		#print(grad)
		#sys.exit()
		#print(grad)
		#print(weights)

		#print(i)


		weights -= eta*grad
		tester = np.matmul(weights[0,:],np.transpose(arrreda[0,:]))
		#print(weights[0,:])
		#print(arrreda[0,:])
		#print(tester)
		#sys.exit()

		#print(weights)
		#sys.exit()

	#print(weights)
	#sys.exit()


	return weights	



def classifyf(weights,arrreda,numun,nsams2):
	H = sigmoid(weights,arrreda,numun,nsams2) 
	H = np.matrix(H)
	classified = H.argmax(axis=0)
	return classified	



def NormMatrixcols(matrix):
	##########################I think axis+0 is for columns###################
	#matrix = np.matrix(matrix)
	shape = np.shape(matrix)
	#mean = np.sum(matrix,axis = 0)
	#matrix_N = matrix
	#matrixbefore = matrix



	i = 0
	while (i < shape[0]):
		j = 0
		sums = 0
		while (j<shape[1]):
			sums += matrix[i][j]*matrix[i][j]
			j += 1

		#sums /= shape[0]
		#print(sums)
		sums = math.sqrt(sums)


		j = 0
		while(j < shape[1]):
			#print(matrix[i][j]) 
			matrix[i][j] /= sums
			#print(matrix[i][j])
			j += 1

		i += 1	

		




	#print (matrix_N)
	#print(mean)
	#sys.exit()

	#matrix_N = matrix - mean

	#print(matrix_N - matrix)

	#sys.exit()


	return matrix







    	
    	




path2train = sys.argv[1]
path2test = sys.argv[2]

#print(path2test)
#sys.exit()
arr, labelsarray = samplearray(path2train)
#print (labelsarray)
#sys.exit()


arrred, actualbasis, arr_mm = pca(arr)
#print(arrred)
#sys.exit()

arrred = NormMatrixcols(arrred)

#print(arrred)
#print(arrred.shape)
#sys.exit()

#################arrred is 33 32) i.e., rowwise#############3





samnum = np.shape(arrred)

#print(param_est(group_classes(arrred,labelsarray),samnum))
#sys.exit()




uniqs , dummy= totallabels(labelsarray)

numun = len(uniqs)

ylabs = np.zeros((numun,numun))

i =0
while (i<numun):
	ylabs[i][i]=1
	i += 1

nsams ,dummy2 = np.shape(arrred)

arrreda = np.zeros((nsams,(dummy2+1)))

count = 0
i = 0
j = 0
while (i<nsams):
	j = 0
	while (j<(dummy2+1)):
		if (j == 0):
			count += 1
			arrreda[i][j] = 1
		else:
			arrreda[i][j] = arrred[i][j-1]
		j += 1
	
	i += 1

#print (arrreda)
#sys.exit()



aiglabs = assignclass(labelsarray,uniqs)
#print(aiglabs)

#arred is rowwise samples
#weights are also row wise

weights = 0.00001*np.ones((numun,33))
#print(ylabs)
#print(weights)
#sys.exit()

weights = softmax(weights,0.1,arrreda,aiglabs,ylabs,70,numun,nsams)


#print(weights)
#sys.exit()


R = classify2(arr_mm,path2test,actualbasis)


nsams2, dummy3 = np.shape(np.matrix(R))


dummy4 = dummy3 - 1

dR = np.zeros((nsams2,dummy4))


i = 0
while (i<nsams2):
	j = 0
	while (j<dummy4):
		dR[i][j] = R[i][j+1]
		j += 1
	i += 1
	

newR = NormMatrixcols(dR)

i = 0
while (i<nsams2):
	j = 0
	while(j<dummy4):
		R[i][j+1] = newR[i][j]
		j += 1
	i += 1
	
#print(R)
#sys.exit()		


#print(nsams2)
#sys.exit()

classified = classifyf(weights,R,numun,nsams2)
classified = np.array(classified)
#print(classified[0][1])
#print(nsams2)

#print(uniqs[4])

i = 0
while (i<nsams2):
	print(uniqs[classified[0][i]])
	i += 1

#print(np.shape(classified))

#print(classified)


#y = arrred[:,1]
#z = arrred[:,2]

#plt.scatter(arrred[:,0],y,z)
#plt.title('Scatter plot pythonspot.com')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()

#fig = plt.figure()
#ax = Axes3D(fig)


#ax.scatter(arrred[:,0], y, z)
#plt.show()


#labels = classify(arr_mm,path2test,actualbasis,param_est(group_classes(arrred,labelsarray),samnum))
#print(labels)
#print(matches, count)          




	 













