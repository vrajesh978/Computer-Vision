import cv2
import numpy as np
import math
import os
import sys
import HOG_FEATURE
import random
from PrewittOperator import prewitt,compute_gradient_magnitude_angle
import neural_network

STATIC_PATH = "Gradient Magnitude Test Images"
TEXT_FILE_PATH = "HOG descriptor"

def calculateFeatureVectorImgName(img_path):
	"""
	@param 1: img_path, full path of the image
	@return feature_vector, contains features which is used as an input to our neural network. dimension [7524 x 1]
	"""
	img_c = cv2.imread(img_path)  #reading the image.
	img_gray_scale = np.round(0.299*img_c[:,:,2] + 0.587*img_c[:,:,1] + 0.114*img_c[:,:,0]) # converting image into grayscale.
	gx,gy = prewitt(img_gray_scale) # finding horizontal gradient and vertical gradient.
	gradient_magnitude,gradient_angle = compute_gradient_magnitude_angle(gx,gy) # finding gradient magnitude and gradient angle.
	
	img_path = img_path.split('/')
	
	# save gradient magnitude files for test images. 
	if("Test_" in img_path[1]):
		if not os.path.exists(STATIC_PATH):
			os.makedirs(STATIC_PATH)
		cv2.imwrite(STATIC_PATH+"/"+str(img_path[2]),gradient_magnitude)
	
	feature_vector = HOG_FEATURE.calculateHOG(img_gray_scale,gradient_magnitude,gradient_angle)  #calculate hog descriptior
	
	feature_vector = feature_vector.reshape(feature_vector.shape[0],1) # reshaping our vector. making dimension [7524 x 1]

	# this below code is used to store the feature vector of crop001278a.bmp and crop001278a.bmp into txt file.
	if(img_path[2] in ["crop001278a.bmp","crop001045b.bmp"]):
		if not os.path.exists(TEXT_FILE_PATH):
			os.makedirs(TEXT_FILE_PATH)
		# saving hog descriptor value. Here,%10.14f will store upto 14 decimal of value
		np.savetxt(TEXT_FILE_PATH+"/"+str(img_path[2][:-3])+"txt",feature_vector,fmt="%10.14f")

	return feature_vector


#Preprocessing. Getting the folders where the images are stored.
TRAIN_PATH = ["Images/Train_Negative","Images/Train_Positive"]
TEST_PATH = ["Images/Test_Positive","Images/Test_Neg"]
y_train = [] #contains training samples label.
y_test = [] # contains testing samples label.

train_images_feature_vector_list = [] #contrains training samples feature vector.
test_images_feature_vector_list = [] #contrains testing samples feature vector.

print("---------Start finding feature vector for training samples-------------")
i = 0
for path in TRAIN_PATH:
	for root,dirs,files in os.walk(path):
		for name in files:
			#calculating hog descriptor of the all train images and store it into train_images_feature_vector_list.
			train_images_feature_vector_list.append(calculateFeatureVectorImgName(path+"/"+str(name)))
			y_train.append(np.array([[i]])) # if human is present in the image we label as 1 otherwise 0.
	i = 1
print("---------Finished finding feature vector for training samples-----------")

test_img_path = [] #storing path of the test images

print("---------Start finding feature vector for testing samples-------------")
i = 1
for path in TEST_PATH:
	for root,dirs,files in os.walk(path):
		for name in files:
			#storing path of the test images.
			test_img_path.append(path+'/'+str(name))
			#calculating hog descriptor of the all train images and store it into train_images_feature_vector_list.
			test_images_feature_vector_list.append(calculateFeatureVectorImgName(path+"/"+str(name)))
			y_test.append(np.array([[i]])) # if human is present in the image we label as 1 otherwise 0.
	i = 0
print("---------Finished finding feature vector for testing samples-----------")

#Shuffle the data. It's a good thing to shuffle our data.
combine = list(zip(train_images_feature_vector_list,y_train))
random.shuffle(combine)
train_images_feature_vector_list, y_train = zip(*combine) 

"""Let's train our neural network."""
for no_hidden_neurons in [250,500,1000]:
	print("-------------------Start training where ",no_hidden_neurons," hidden neurons----------------")
	model = neural_network.trainNeuralNetwork(train_images_feature_vector_list,y_train,no_hidden_neurons)
	print("Saving model in data",str(no_hidden_neurons),".npy file")
	neural_network.saveModelFile(model,"data"+str(no_hidden_neurons)) # save model file. we can use it later for prediction.
	print("successfully trained our neural network containing ",no_hidden_neurons," hidden neurons.")
	print("--------------------------------------------------------------------------------")

""" Let's test our trained neural network."""
for no_hidden_neurons in [250,500,1000]:
	neural_network_output = []  #storing predicted value of the test image
	model = neural_network.loadModelFile("data"+str(no_hidden_neurons))   # load model file for getting weights and bias.
	
	print("Predicted value of the test images where number of neurons = ",no_hidden_neurons)

	#getting all images from the list of test images and print output value of the neural network.
	for test_img,test_img_name in zip(test_images_feature_vector_list,test_img_path):
		neural_network_output.append(neural_network.predict(test_img,model)) 
		print(test_img_name,"	Predicted value = ",neural_network_output[-1][0][0])
	
	print("-------------------------------------------------------------------------------")
	print("Accuracy = ",neural_network.accuracy(neural_network_output,y_test)) #print accuracy of our neural network.
	print("Finished prediction of the neural network where number of neurons in hidden layers = ",no_hidden_neurons)