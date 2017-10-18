# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hog_1]: ./images/hog_1.png
[hog_2]: ./images/hog_2.png
[hog_3]: ./images/hog_3.png
[hog_4]: ./images/hog_4.png
[hog_hue]: ./images/hog_hue.png
[hog_sat]: ./images/hog_sat.png
[hog_value]: ./images/hog_value.png
[sliding_window]: ./images/sliding_window.png
[detected_image]: ./images/detected_image.png
[heatmap_images]: ./images/heatmap_images.png
[label_image]: ./images/label_image.png
[final_car_bound_box]: ./images/final_car_bound_box.png
[pipeline_steps]: ./images/pipeline_steps.png
[image_vehicle1]: ./images/image_vehicle1.png
[image_vehicle2]: ./images/image_vehicle2.png
[image_non_vehicle1]: ./images/image_non_vehicle1.png
[image_non_vehicle2]: ./images/image_non_vehicle2.png
[video1]: ./project_video.mp4

---

### Writeup / README

This document is walk through of solution approach and findings in implementing the solution. 

### Histogram of Oriented Gradients (HOG)

1. If you consider a small segment of image and see the gradient among adjacent pixels in different directions we can observe variation according to the direction of line in that segment. Here general idea is to club them into histogram so that we can represent the direction in fixed number of bins.  Once we get the magnitude of each bin, highest value will represent the edge of object in this segment. We are going to use this as feature as this encodes shapes well, As it is histogram of gradients in different directions they are called HOG (histogram of oriented gradients)

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Image of vehicle][image_vehicle1]  ![Image of vehicle][image_vehicle2]

![Image of non vehicle][image_non_vehicle1]  ![Image of non vehicle][image_non_vehicle2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Trying different parameters for best hog features.

I tried variations in `number of bins` and `pixels per cell` to see which combination give a good shape
 representation of the image. Below are some of the examples for sample image in hog visualization. This was using grayscale version of the image.

![HOG visualization of image with 9 bin 8 pixel per cell][hog_1]

Then I tried with lesser number of bins so that only horizontal and vertical lines can be captured.

![HOG visualization of image with less number of bins][hog_2]

Then tried to see if more number of bins and lower cell size can capture any better. While running this I notices this operation of
 finding hog features was taking too long. By looking at below image, there is no significant visual difference however it is
 computationally intensive. So decided to use 9 bins with 8 pixel cell.
 
![HOG visualization of image 16 bins and smaller cell size][hog_3] 

However, we can look at different color spaces to see if they give better insights into shape information.

![HOG visualization Hue channel of HSV color space][hog_hue]

and 
![HOG visualization Saturation channel of HSV color space][hog_sat]
![HOG visualization Value channel of HSV color space][hog_value]

After comparison of various combinations, I found hog on all channel would yield a better result. 

#### 3. Training the classifier.

I trained a linear SVM using scikit framework, first loaded all images of cars and non-car (roughly similar sample size). 
Then split them into test and train set using `train_test_split` with different parameters compared the accuracy of the model 


### Sliding Window Search

#### 1. Sliding window search.
As the car can appear anywhere inside or front, we need to search the image a tile by tile. This technique is referred as 
sliding window. However while doing that a tile may cover the half car which may lead to miss that car detection. So use
 of overlapping sliding window would resolve the problem. 

Another problem in such search is car image size becomes smaller and smaller as it goes far. However, our sliding window is of fixed size so we need to do multiple scale sliding window search in order to find the car effectively.

Visualization of the sliding window.

![Sliding window example][sliding_window]


After experimenting with multiple window sizes I decided to use scale factor of 1.5 (96x96) and 1.2 (76x76)


#### 2. Optimizing the classifier

I did an exhaustive search for optimal parameters in feature selection. Below is various options searched for:

```python
    for color_space in ['RGB', 'HSV', 'LUV','HLS', 'YUV', 'YCrCb']:
        for orient in [2,4,5,9,13]:
            for pix_per_cell in [4,8,16]:
                for cell_per_block in [1,2,3]:
                    for hog_channel in [0,1,2,'ALL']:
```

And below is the excerpt from the run:
```commandline
Acc: 0.926 Using color RGB orient:2 pix_per_cell:4 cell_per_block:1 hog_channel:0 time: 21.62
Acc: 0.9333 Using color RGB orient:2 pix_per_cell:4 cell_per_block:1 hog_channel:1 time: 20.16
Acc: 0.9282 Using color RGB orient:2 pix_per_cell:4 cell_per_block:1 hog_channel:2 time: 22.79
Acc: 0.9248 Using color RGB orient:2 pix_per_cell:4 cell_per_block:1 hog_channel:ALL time: 24.64
Acc: 0.9234 Using color RGB orient:2 pix_per_cell:4 cell_per_block:2 hog_channel:0 time: 19.02
Acc: 0.942 Using color RGB orient:2 pix_per_cell:4 cell_per_block:2 hog_channel:1 time: 18.13
Acc: 0.9389 Using color RGB orient:2 pix_per_cell:4 cell_per_block:2 hog_channel:2 time: 19.6
Acc: 0.9386 Using color RGB orient:2 pix_per_cell:4 cell_per_block:2 hog_channel:ALL time: 47.07
Acc: 0.9496 Using color RGB orient:2 pix_per_cell:4 cell_per_block:3 hog_channel:0 time: 31.82
Acc: 0.9502 Using color RGB orient:2 pix_per_cell:4 cell_per_block:3 hog_channel:1 time: 29.11
Acc: 0.9454 Using color RGB orient:2 pix_per_cell:4 cell_per_block:3 hog_channel:2 time: 29.04
Acc: 0.9443 Using color RGB orient:2 pix_per_cell:4 cell_per_block:3 hog_channel:ALL time: 91.47
.....
Acc: 0.9913 Using color YCrCb orient:9 pix_per_cell:8 cell_per_block:3 hog_channel:ALL time: 32.07
.....
```

Last run was to find best combination of features for the classifier, from the run we can say leaving out histogram was not having any impact on results. So it was removed in final pipeline. 
Trained model with optimal settings is saved in modelv4.pkl (pickle dump of configuration and model itself).

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color in the feature vector, which provided a nice result.  Here are some example images:


![Detected image with single pass window slide technique][detected_image]

---

### Video Implementation

#### 1. Video output
Here's a [link to Youtube video](https://www.youtube.com/watch?v=75pMA4K6c6c)


#### 2. Pipeline
* In order to avoid false positives I had to use multi window scan and use threshold for actual car detection. There are some dark area 
detected as car however after taking heatmap of multiple window detection I could avoid detecting wrong area as car.
* Also I used confidence level for the car as a function of past detections and how many overlapping windows detected.

I recorded the positions of positive detections in each frame of the video.  From the positive detections, I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
All these information is encapsulated in State class and Car. `Car` holds information about latest car detected with attributes such as position, bounding box and confidence.

```python
class Car:
    def __init__(self):
        self.pos = None
        self.dir = None
        self.box = None
        self.boxes = []
        self.age = 0
        self.confidence = 0
        
    def set_pos(self, pos, box):
        last_pos = self.pos
        self.pos = pos
        self.box = box
        self.boxes.append(box)
        self.age += 1
        self.confidence += 1.0
        if (last_pos is not None):
            self.dir = (pos[0]-last_pos[0], pos[1]-last_pos[1])
    def __str__(self):
        return "car %s confidence %s"%(self.pos,self.confidence)
    def __repr__(self):
        return self.__str__()
    def get_bounding_box(self):
        last_n_frames = 3
        p1x = p2x = p1y = p2y = 0
        c=0
        for b in self.boxes[-last_n_frames:]:
            p1x += b[0][1]
            p1y += b[0][0]
            p2x += b[1][1]
            p2y += b[1][0]
            c+=1
        avg_box = ((p1y/c, p1x/c),(p2y/c, p2x/c))
        return avg_box
    def frame(self):
        self.confidence *= .6 if self.age <10 else .9
        
```

Bounding box was considered as average of last 3 frames to smooth the detection boundary.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the images:

##### Here are few frames and their corresponding heatmaps:

![Images showing heatmap of car detection][heatmap_images]

##### Here is the example output of `scipy.ndimage.measurements.label()` on heatmap:
![Visualization of labels detected out of heatmap][label_image]

So here is the image with annotated intermediate findings and final car detected:
![Visualization of intermediate steps in the pipeline][pipeline_steps]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![Final car detected in bound box][final_car_bound_box]



---

### Discussion

#### Challenges and learnings:
* One of the problems I faced while detecting cars in the video was in one of the frame car was not being detected with high confidence. I had to use the concept of history with confidence score attached to the car detected.
* Another issue was some of the dark area on road divider was getting detected as a car. This was avoided by using a minimum threshold for accurate detection. Also, I am waiting for ther car to get detected in at-least 2 consecutive frames to mark it as car.
#### Where it might fail
* Current training set doesnt have the front of the car so it takes a while to detect the car, once the complete car is visible it gets detected.
* Other cases like the bad weather might affect the detection very badly. A possible solution would be to add more blurry images to training set.
* Colored lighting, night condition or weather conditions like the sunset (red tint) will impact the current pipeline significantly.

