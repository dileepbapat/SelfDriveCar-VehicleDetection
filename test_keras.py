import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from lesson_functions import *
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import grid_search
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from keras_util import *


# Read in car and non-car images
vehicle_images = glob.glob('data/vehicles/*/*.png')
non_vehicle_images = glob.glob('data/non-vehicles/*/*.png')
cars = []
notcars = []
for image in vehicle_images:
    cars.append(image)
for image in non_vehicle_images:
    notcars.append(image)


color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 3  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = False  # Histogram features on or off
hog_feat = True  # HOG features on or off


car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
y = np.vstack((np.array([[1, 0] for i in range(len(car_features))]), np.array([[0, 1] for i in range(len(notcar_features))])))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Feature vector length:', len(X_train[0]))
# Use a linear SVC

sequential = Sequential([
    Dense(20, input_shape=(X_train.shape[1],)),
    Dense(10),
    Dropout(.5),
    Dense(5),
    Dense(2,activation='sigmoid')
])
sequential.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
sequential.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=100)
# mdl = KerasClassifier(sequential)
# # Check the training time for the SVC
# t=time.time()
# #svc.fit(X_train, y_train)
# parameters = {'kernel':('linear', 'rbf'), 'C':[.1, 1, 2, 3, 5, 7, 10], 'gamma':[0.1, 1, 10]}
# clf = grid_search.GridSearchCV(mdl, parameters)
# clf.fit(X_train, y_train)
# t2 = time.time()
# print(round(t2-t, 2), 'Seconds to train SVC...')
# print clf.best_params_
# # Check the score of the SVC
# print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# # Check the prediction time for a single sample
n_predict = 20
t=time.time()
prediction = sequential.predict_classes(X_test[0:n_predict])
t2 = time.time()
print('My nn predicts: %s' % prediction)
print('        actual: %s'% y_test[0:n_predict][:,1])
print('Seconds to predict : %s , %s labels with nn'%(round(t2-t, 5), n_predict))