from CannyEdgeDetector import CannyEdgeDetector
import numpy as np
import cv2
import sys

try:
	image_path = str(sys.argv[1]) #getting image path
	img = cv2.imread(image_path)	# read image 
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert into grayscale image.
	edges=[] #empty edges.
	CannyEdgeDetector(image_path,gray_image,edges) #getting edges 
	L=[10,30,50]
	j=0
	for i in edges:
		print("--------For P = "+str(L[j])+"%--------")
		print("Threshold :"+str(i[0]))
		print("Edge points :"+str(i[1]))
		j=j+1
except:
	print("Please provide image path")


