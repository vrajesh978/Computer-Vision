import numpy as np
import math

def compute_gradient_magnitude_angle(gx,gy):
	"""
	Parameters: 
	param1 : gx, horizontal gradient
	param2 : gy, vertical gradient
	return: new matrix which has gradient magnitude of every pixel.
	"""
	gradient_magnitude=np.zeros((gx.shape[0],gx.shape[1]))
	gradient_angle=np.zeros((gx.shape[0],gx.shape[1]))
	
	for i in range(gx.shape[0]):
		for j in range(gx.shape[1]):
			gradient_magnitude[i,j] = math.sqrt((gx[i,j]*gx[i,j]) + (gy[i,j]*gy[i,j]))
			if(gx[i,j]==0):
				if(gy[i,j]>0):
					gradient_angle[i,j] = 90.0
				else:
					gradient_angle[i,j] = -90.0
			else:
				gradient_angle[i,j] = math.degrees(math.atan(gy[i,j]/gx[i,j]))
	return gradient_magnitude/1.4142,gradient_angle

def check_sector(angle):
	"""
		parameters :
		param1 : angle, gradient angle.
		return : sector, gives sector according to gradient angle of the center pixel.
	"""
	if( angle < 0):
		angle = angle + 360
	if( 0 <= angle <= 22.5 or 157.5 < angle <= 202.5 or 337.5 < angle <= 360):
		return 0
	elif ( 22.5 < angle <= 67.5 or 202.5 < angle <= 247.5):
		return 1
	elif ( 67.5 < angle <= 112.5 or 247.5 < angle <= 292.5):
		return 2
	elif ( 112.5 < angle <= 157.5 or 292.5 < angle <= 337.5):
		return 3

def non_maxima_suppression(gradient_magnitude,gradient_angle):
	"""
	Perform non maxima suppression on matrix
	Parameters:
	param 1: gradient_magnitude. It holds gradient magnitude of every pixel.
	param 2: gradient_angle holds gradient angle of every pixel.  
	return : image matrix which has gradient magnitude of those pixels which contributes in forming edge.
	"""
	gradient_magnitude_height=gradient_magnitude.shape[0]
	gradient_magnitude_width=gradient_magnitude.shape[1]
	nms = np.zeros((gradient_magnitude_height,gradient_magnitude_width))
	for i in range(1,gradient_magnitude_height-1):		
		for j in range(1,gradient_magnitude_width-1):
			sector = check_sector(gradient_angle[i,j])
			if sector == 0:
				maximum = max(gradient_magnitude[i,j-1],gradient_magnitude[i,j+1])
				if gradient_magnitude[i,j] > maximum:
					nms[i,j] = gradient_magnitude[i,j]
			elif sector == 1:
				maximum = max(gradient_magnitude[i-1,j+1],gradient_magnitude[i+1,j-1])
				if gradient_magnitude[i,j] > maximum:
					nms[i,j] = gradient_magnitude[i,j]
			elif sector == 2:
				maximum = max(gradient_magnitude[i-1,j],gradient_magnitude[i+1,j])
				if gradient_magnitude[i,j] > maximum:
					nms[i,j] = gradient_magnitude[i,j]
			elif sector == 3:
				maximum = max(gradient_magnitude[i-1,j-1],gradient_magnitude[i+1,j+1])
				if gradient_magnitude[i,j] > maximum:
					nms[i,j] = gradient_magnitude[i,j]
	return nms

def thresholding(cedges,th):
	"""
	thresholding is done using p-tile method
	param1: cedges, matrix after performing non maxima supression.
	param2: th, foreground pixel's percentage in image.
	return : 
		threshold, threshold value for separating foreground pixels and background pixel.
		no_of_edge_points, returns number of edge points in image.
		pixel_matrix, matrix containg edges.
	"""
	objects=0
	height,width = cedges.shape[0],cedges.shape[1]
	pixel_matrix = np.zeros((height,width))
	sum = 0 
	no_of_edge_points = 0
	threshold=0
	hist_dict = {}
	
	#generate histogram using dictionary
	for i in range(height):
		for j in range(width):
			if(cedges[i,j] >0):
				objects += 1
			if(cedges[i,j] in hist_dict):
				hist_dict[(cedges[i,j])]+=1
			else:
				hist_dict[(cedges[i,j])]=1

	foreground_pixel = ((objects*th)/100) # calculating foreground pixel

	#finding threshold value from back side because foreground is brighter than background
	greay_value = []
	for key,value in hist_dict.items():
		greay_value.append(key)
	greayvalue= sorted(greay_value,reverse=True)
	for key in greayvalue:
		if(sum < foreground_pixel):
			sum+=hist_dict[key]
		else:
			threshold = round(key)
			break
	"""
		setting pixel value according threshold value.
		if pixel value lower than threshold value it becomes zero.
		otherwise it is an edge.
	"""
	for i in range(height):
		for j in range(width):
			if(cedges[i,j]>threshold):
				pixel_matrix[i,j] = 255
				no_of_edge_points += 1

	return (threshold,no_of_edge_points,pixel_matrix)

