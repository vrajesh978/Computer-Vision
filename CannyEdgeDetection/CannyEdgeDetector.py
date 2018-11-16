import os
import numpy as np
import cv2
import HelperModule
import GaussianFilter
import PrewittOperator


STATIC_PATH = "Images/" #static path for storting images

def CannyEdgeDetector(image_path,src,edges):
	"""
	The function finds edges in the input image using canny edge detection algorithm.
	Paramaters:
		src : Input Image
		edges : save output edge map into a list which has 3 size. 
	"""
	#creating folders if not created previously
	if not os.path.exists(STATIC_PATH):
		os.makedirs(STATIC_PATH)
	if not os.path.exists(STATIC_PATH+image_path):
		os.makedirs(STATIC_PATH+image_path)

	filtered_img,height,width = GaussianFilter.gaussian_filtering(src) #compute gaussian filter
	cv2.imwrite(STATIC_PATH+image_path+"/gaussian_blur.bmp",filtered_img) #save gaussian image.
	
	gx,gy = PrewittOperator.prewitt(filtered_img,height,width) # find horizontal gradient
	cv2.imwrite(STATIC_PATH+image_path+"/horizontal_gradient.bmp",gx) #save horizontal gradient
	cv2.imwrite(STATIC_PATH+image_path+"/vertical_gradient.bmp",gy) #save vertical gradient
	
	#compute gradient magnitude and angle
	gradient_magnitude,gradient_angle = HelperModule.compute_gradient_magnitude_angle(gx,gy) 
	cv2.imwrite(STATIC_PATH+image_path+"/gradient_magnitude.bmp",gradient_magnitude) #save gradient_magnitude

	#non maximuma supression on gradient magnitude.
	cedges = HelperModule.non_maxima_suppression(gradient_magnitude,gradient_angle)
	cv2.imwrite(STATIC_PATH+image_path+"/non_maxima_suppression.bmp",cedges)
	
	#storing egde value to edges list 
	edges.append(HelperModule.thresholding(cedges,10))
	cv2.imwrite(STATIC_PATH+image_path+"/p-tile-10.bmp",edges[0][2])
	
	edges.append(HelperModule.thresholding(cedges,30))
	cv2.imwrite(STATIC_PATH+image_path+"/p-tile-30.bmp",edges[1][2])
	
	edges.append(HelperModule.thresholding(cedges,50))
	cv2.imwrite(STATIC_PATH+image_path	+"/p-tile-50.bmp",edges[2][2])