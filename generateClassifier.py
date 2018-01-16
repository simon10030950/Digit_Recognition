# Import the modules
from sklearn.externals import joblib 						# To save the classifier
from sklearn import datasets 										# Be used to download the MNIST database
from skimage.feature import hog									# To calculate the HOG features
from sklearn.svm import LinearSVC								# To perform prediction after training the claasifier
import numpy as np 															# Store HOG features and labels in numpy arrays
from collections import Counter
import cv2

# Deskew the image
# def deskew(img):
#     SZ = 28
#     affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
#     m = cv2.moments(img)
#     if abs(m['mu02']) < 1e-2:
#         return img.copy()
#     skew = m['mu11']/m['mu02']
#     M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
#     img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
#     return img

# Rotate the image
# def rotateImage(image, angle):
#   image_center = tuple(np.array(image.shape)/2)
#   print (image.shape)
#   rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
#   result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
#   return result

# Download the dataset
dataset = datasets.fetch_mldata("MNIST Original") 

# Extract the features and labels
features = np.array(dataset.data, 'int16') 
labels = np.array(dataset.target, 'int')

#features_deskewed = list(map(deskew, features))

# Extract the HOG features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations = 9, pixels_per_cell = (14, 14), cells_per_block = (1, 1), visualise = False)
    list_hog_fd.append(fd)
    # #random rotate
    # feature = imutils.rotate(feature, 30)
    # fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    # list_hog_fd.append(fd)
    # #random rotate
    # feature = imutils.rotate(feature, 330)
    # fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    # list_hog_fd.append(fd)

hog_features = np.array(list_hog_fd, 'float64')

print "Count of digits in dataset", Counter(labels)

# Create an linear SVM object
clf = LinearSVC()

# Perform the training
clf.fit(hog_features, labels)

# Save the classifier
joblib.dump(clf, "digits_cls.pkl", compress=3)