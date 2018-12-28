import numpy as np
import math

PREWIT_OPERATOR_GX = np.array([[-1,0,1],
								[-1,0,1],
								[-1,0,1]])

PREWIT_OPERATOR_GY = np.array([[1,1,1],
								[0,0,0],
								[-1,-1,-1]])


def convolution(img,g):
	"""
		parameters:
			@param 1 : img, image matrix
			@param 2 : g, kernel 3x3 prewitt operator  
		@return : img_conv, matrix performing convolution img*g
	"""
	rows,cols = img.shape
	heightG,widthG = g.shape[0]//2,g.shape[1]//2
	img_conv = np.zeros(img.shape)
	for i in range(1,rows-1):
		for j in range(1,cols-1):
			img_conv[i,j] = 0
			for k in range(-heightG, heightG + 1):
				for m in range(-widthG , widthG + 1):
					img_conv[i,j] = img_conv[i,j] + (g[heightG+k,widthG+m] * img[i+k,j+m])
			# if(img_conv[i,j] < 0):
			# 	img_conv[i,j] = abs(img_conv[i,j]) # taking absolute value
			img_conv[i,j] = img_conv[i,j] / 3 # normalizing gradients
	return img_conv

def prewitt(img):
	"""
		parameters:
			@param1 : img, numpy array of the image.
		@return : prewittGx, image matrix after img*PREWIT_OPERATOR_GX
		@return : prewittGy, image matrix after img*PREWIT_OPERATOR_GY
	"""
	prewittGx = convolution(img,PREWIT_OPERATOR_GX)
	prewittGy = convolution(img,PREWIT_OPERATOR_GY)
	return prewittGx,prewittGy

def compute_gradient_magnitude_angle(gx,gy):
	"""
	Parameters: 
		@param1 : gx, horizontal gradient
		@param2 : gy, vertical gradient
	@return: gradient magnitude and gradient angle.
	"""
	gradient_magnitude=np.zeros((gx.shape[0],gx.shape[1]))
	gradient_angle=np.zeros((gx.shape[0],gx.shape[1]))
	
	for i in range(gx.shape[0]):
		for j in range(gx.shape[1]):
			gradient_magnitude[i,j] = math.sqrt((gx[i,j]*gx[i,j]) + (gy[i,j]*gy[i,j]))
			gradient_magnitude[i,j] = gradient_magnitude[i,j]/np.sqrt(2)
			
			if(gx[i,j]==0) and (gy[i,j]==0):
				gradient_angle[i,j] = 0
			elif(gx[i,j]==0):
				if(gy[i,j]>0):
					gradient_angle[i,j] = 90
				else:
					gradient_angle[i,j] = -90
			else:
				gradient_angle[i,j] = math.degrees(np.arctan(gy[i,j]/gx[i,j]))
			
			if(gradient_angle[i,j]<0):
				gradient_angle[i,j]=180+gradient_angle[i,j]

			if(gradient_angle[i,j]==-0):
				gradient_angle[i,j]=0

	return gradient_magnitude,gradient_angle
