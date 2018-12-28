import cv2
import numpy as np
import math

def calculateHOG(img,gradient_magnitude,gradient_angle):
	"""
	@param1: img, input image
	@param2: gradient_magnitude, image's gradient magnitude.
	@param3: gradient_angle, image's gradient angle.
	@return: feature_vector, hog descriptor of the img.
	@return: row, height of the image
	@return: col, width of the image
	"""
	cellHistogram,row,col = calculateCellHisto(img,gradient_magnitude,gradient_angle)
	feature_vector = calculateFeatureVector(cellHistogram,row,col)
	return feature_vector

def calculateCellHisto(img,gradient_magnitude,gradient_angle):
	"""
	@param1: img, input image
	@param2: gradient_magnitude, image's gradient magnitude.
	@param3: gradient_angle, image's gradient angle.
	@return: cellHistogram, it contains histogram of every cell 8x8.
	"""
	height,width = img.shape
	row=math.floor(height/8)
	col=math.floor(width/8)
	row_hist = 0
	col_hist = 0
	cellHistogram=np.zeros((row,col,9))
	for r in range(0,height,8):
		for c in range(0,width,8):
			i_row = r
			limit_i_row = i_row + 8
			histogram = [0]*9    #initially all values of histogram bins are zero.
			for i in range(i_row,limit_i_row):
				j_col = c
				limit_j_col = j_col + 8
				
				for j in range(j_col,limit_j_col):
					if(gradient_angle[i,j] == 0 or gradient_angle[i,j] == 180):
						histogram[0] += gradient_magnitude[i,j]
					elif(gradient_angle[i,j] > 0 and gradient_angle[i,j] < 20):
						histogram[0] += ((20 - gradient_angle[i,j]) / 20) * gradient_magnitude[i,j]
						histogram[1] += ((gradient_angle[i,j] - 0) / 20) * gradient_magnitude[i,j]
					elif(gradient_angle[i,j] == 20):
						histogram[1] += gradient_magnitude[i,j]
					elif(gradient_angle[i,j] > 20 and gradient_angle[i,j] < 40):
						histogram[1] += ((40 - gradient_angle[i,j]) / 20) * gradient_magnitude[i,j]
						histogram[2] += ((gradient_angle[i,j] - 20) / 20) * gradient_magnitude[i,j]
					elif(gradient_angle[i,j] == 40):
						histogram[2] += gradient_magnitude[i,j]
					elif(gradient_angle[i,j] > 40 and gradient_angle[i,j] < 60):
						histogram[2] += ((60 - gradient_angle[i,j]) / 20) * gradient_magnitude[i,j]
						histogram[3] += ((gradient_angle[i,j] - 40) / 20) * gradient_magnitude[i,j]
					elif(gradient_angle[i,j] == 60):
						histogram[3] += gradient_magnitude[i,j]
					elif(gradient_angle[i,j] > 60 and gradient_angle[i,j] < 80):
						histogram[3] += ((80 - gradient_angle[i,j]) / 20) * gradient_magnitude[i,j]
						histogram[4] += ((gradient_angle[i,j] - 60) / 20) * gradient_magnitude[i,j]
					elif(gradient_angle[i,j] == 80):
						histogram[4] += gradient_magnitude[i,j]
					elif(gradient_angle[i,j] > 80 and gradient_angle[i,j] < 100):
						histogram[4] += ((100 - gradient_angle[i,j]) / 20) * gradient_magnitude[i,j]
						histogram[5] += ((gradient_angle[i,j] - 80) / 20) * gradient_magnitude[i,j]
					elif(gradient_angle[i,j] == 100):
						histogram[5] += gradient_magnitude[i,j]
					elif(gradient_angle[i,j] > 100 and gradient_angle[i,j] < 120):
						histogram[5] += ((120 - gradient_angle[i,j]) / 20) * gradient_magnitude[i,j]
						histogram[6] += ((gradient_angle[i,j] - 100) / 20) * gradient_magnitude[i,j]
					elif(gradient_angle[i,j] == 120):
						histogram[6] += gradient_magnitude[i,j]
					elif(gradient_angle[i,j] > 120 and gradient_angle[i,j] < 140):
						histogram[6] += ((140 - gradient_angle[i,j]) / 20) * gradient_magnitude[i,j]
						histogram[7] += ((gradient_angle[i,j]- 120) / 20) * gradient_magnitude[i,j]
					elif(gradient_angle[i,j] == 140):
						histogram[7] += gradient_magnitude[i,j]
					elif(gradient_angle[i,j] > 140 and gradient_angle[i,j] < 160):
						histogram[7] += ((160 - gradient_angle[i,j]) / 20) * gradient_magnitude[i,j]
						histogram[8] += ((gradient_angle[i,j] - 140) / 20) * gradient_magnitude[i,j]
					elif(gradient_angle[i,j] == 160):
						histogram[8] += gradient_magnitude[i,j]
					elif(gradient_angle[i,j] > 160):
						histogram[8] += ((180 - gradient_angle[i,j]) / 20) * gradient_magnitude[i,j]
						histogram[0] += ((gradient_angle[i,j] - 160) / 20) * gradient_magnitude[i,j]
			
			cellHistogram[row_hist,col_hist] = histogram
			col_hist = col_hist + 1
		
		row_hist = row_hist + 1
		col_hist = 0
	return cellHistogram,row,col


def calculateFeatureVector(cellHistogram,row,col):
	"""
	@param1: cellHistogram, it contains histogram of every cell 8x8.
	@param2: row, height of the image.
	@param3: col, width of the image.
	@return: feature vector, it contains hog descriptor of the image.  
	"""
	sum_=0.0
	feature_vector=np.zeros(1)
	for i in range(0,row-1):
		for j in range(0,col-1):
			sum_ = 0.0
			# creating a temporary block size 36
			temp_block=np.zeros(1)
			temp_block=np.append(temp_block,cellHistogram[i,j])
			temp_block=np.append(temp_block,cellHistogram[i,j+1])
			temp_block=np.append(temp_block,cellHistogram[i+1,j])
			temp_block=np.append(temp_block,cellHistogram[i+1,j+1])
			temp_block=temp_block[1:]
			#line 115-121 is process of l2-normalization.
			for k in range(0,36):
				sum_=sum_ + np.square(temp_block[k])
			l2_norm_factor=np.sqrt(sum_) # getting l2 norm factor.
			for k in range (0,36):
				if l2_norm_factor == 0:
					continue
				temp_block[k]=temp_block[k]/l2_norm_factor # l2 normalization.
			feature_vector = np.append(feature_vector,temp_block)
	feature_vector=feature_vector[1:]
	return feature_vector
