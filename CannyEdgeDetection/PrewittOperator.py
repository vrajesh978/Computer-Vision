import numpy as np

PREWIT_OPERATOR_GX = np.array([[-1,0,1],
								[-1,0,1],
								[-1,0,1]])

PREWIT_OPERATOR_GY = np.array([[1,1,1],
								[0,0,0],
								[-1,-1,-1]])


def convolution(img,g,height,width):
	"""
		parameters:
			param 1 : img, image matrix
			param 2 : g, kernel 3x3 prewitt operator  
			param 3 : height, gaussian image height
			param 4 : width, gaussian image width
		return : img_conv, matrix performing convolution img*g
	"""
	rows,cols = img.shape
	heightG,widthG = g.shape[0]//2,g.shape[1]//2
	img_conv = np.zeros(img.shape)
	for i in range(height + 1,rows - (height + 1)):
		for j in range(width + 1,cols - (width + 1)):
			img_conv[i,j] = 0
			for k in range(-heightG, heightG + 1):
				for m in range(-widthG , widthG + 1):
					img_conv[i,j] = img_conv[i,j] + g[heightG+k,widthG+m] * img[i+k,j+m]
			if(img_conv[i,j] < 0):
				img_conv[i,j] = abs(img_conv[i,j]) # taking absolute value
			img_conv[i,j] = img_conv[i,j] / 3.0 # normalizing gradients
	return img_conv

def prewitt(img,height,width):
	"""
		parameters:
			param1 : img, gaussian image
			param2 : height, gaussian image height
			param3 : width, gaussian image width
		return : prewittGx, image matrix after img*PREWIT_OPERATOR_GX
		return : prewittGy, image matrix after img*PREWIT_OPERATOR_GY
	"""
	prewittGx = convolution(img,PREWIT_OPERATOR_GX,height,width)
	prewittGy = convolution(img,PREWIT_OPERATOR_GY,height,width)
	return prewittGx,prewittGy

