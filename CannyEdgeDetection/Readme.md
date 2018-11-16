# Steps in Canny Edge Detection  
1. Gaussian Smoothing.
2. Finding Gradient Magnitude and gradient angle. Here, I have used prewitt Operator.
3. Non Maxima Suppression.
4. Thresholding. Here, I have used P-tile method for thresholding.

# Source Files
1. ```GaussianFilter.py```  <-- for creating Gaussian Smoothed image
2. ```PrewittOperator.py``` <-- for horizontal gradient & vertical gradient.
3. ```HelperModule.py``` <-- for utilies function for finding gradient magnitude, gradient angle, non maxima suppression and thresholding.
4. ```CannyEdgeDetector.py```
5. ```DriverProgram.py``` <-- Main Function to run whole Canny Edge Detection.

# Running Program
```python DriverProgram.py imagepath```

# Images
Below images are stored in ```Images/imagepath``` path. If path is not available, program will create the path. 
1. Gaussian Blur
2. Horizontal Gradient
3. Vertical Gradient
4. Gradient Magnitude
5. Non Maxima Suppresion
6. P-tile threshoidng where P = 10%
7. P-tile threshoidng where P = 30%
8. P-tile threshoidng where P = 50%
