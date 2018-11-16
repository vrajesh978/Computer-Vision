import numpy as np

GAUSS_FILTER = np.array([[1,1,2,2,2,1,1],[1,2,2,4,2,2,1],[2,2,4,8,4,2,2],[2,4,8,16,8,4,2],
[2,2,4,8,4,2,2],[1,2,2,4,2,2,1],[1,1,2,2,2,1,1]])

def gaussian_filtering(img):
	"""
	parameters:
		param 1 : img, img matrix containg pixel value
	return : img_conv,filtterd and normalized image matrix after img*GAUSS_FILTER.
			mid_h, height of gaussian
			mid_w, width of gaussian
	"""
	imgh,imgw,kernelh,kernelw=img.shape[0],img.shape[1],GAUSS_FILTER.shape[0],GAUSS_FILTER.shape[1]
	mid_h = kernelh//2
	mid_w = kernelw//2
	img_conv = np.zeros((imgh,imgw))
	
	#simple convolution operation
	for i in range(mid_h,img_conv.shape[0]-mid_h):		
		for j in range(mid_w,img_conv.shape[1]-mid_w):
			k=0
			while(k<kernelh):
				m=0
				while(m<kernelw):
					img_conv[i,j] = img_conv[i,j]+\
					(img[i-mid_h+(k),j-mid_w+(m)]*GAUSS_FILTER[k,m])
					m+=1
				k+=1
	return img_conv/140,mid_h,mid_w #normalized gaussian smoothing