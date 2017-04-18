# Behavioral Cloning Project
## Writeup 
Author: Roman Stanchak (rstanchak@gmail.com)

Date: 2017-Apr-17

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 containing a video of the final model driving around track 1
* README.md containing a summary of the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model-09.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

```sh
python model.py <path to driving_log.csv>
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network 5x1 and 1x5 filter with depth 8 (model lines 103-105), a dense fully-connected layer of 128 nodes (code line 110) and a single output node (code line 112).

Each convolution and fully-connected layer use a RELU activation function to introduce nonlinearity. The data is normalized in the model using a Keras lambda layer (code line 96) and croped using a Keras cropping lyaer (code line 98).

#### 2. Attempts to reduce overfitting in the model

In order to prevent overfitting, the model contains dropout layers: one between the conv layer and the fully-connected layer, and one between the hidden layer and the output layer (code lines 109, 111) in order to reduce overfitting (model.py lines TODO).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 52). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

The provided training data was used.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple one-unit network, then iteratively apply preprocessing, data augmentation, and network architecture adjustments and observe the performance on the training and validation sets.  In addition to these iterative adjustments, I implemented a few reference networks from the literature to experiment with their performance. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

At the end of the process, I tested the models the vehicle is able to drive autonomously around the track without leaving the road.

The iterative steps, MSE loss, notes are recorded in the table below:

| # | Description | Train Loss | Val Loss  | Notes |
|---|-------------|------------|-----------|-------|
| 1 | linear regression using data as provided | 3340.0979 | 1675.4634
| 2 | same as (1) with images flipped left-right |   369119.3010 | 106462.8109
| 3 | normalize input|   2.9456 | 1.6964
| 4 | multiple cameras with adjustment factor of 0.12 (stddev of data) |   4.0997 | 3.1326
| 5 | crop top by 64 px and bottom by 22 px|   0.4917 | 0.3236 |
| 6 | add 8@5x1 conv layer + relu | 0.0116 | 0.0125 | |
| 7 | add 8@1x5 conv layer + relu | 0.0101 | 0.0108 | |
| 8 | Add 128-node fully-connected layer + relu + p=0.25 dropout | 0.0103 | 0.0107 | track 1 OK |
| 9 | add p=5 dropout between conv and fully-connected layer | 0.0090 | 0.0109 ||

The other networks implemented are as follows:

A simplification of All-CNN-C model from the paper Striving for Simplicity: The All Convolutional Net by Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, Martin Riedmiller, accepted as a workshop contribution at ICLR 2015.  The architecture used:
- 3x3x24 with RELU activation
- 3x3x24 with 2x2 downsampling and RELU activation
- 3x3x48 with RELU activation
- 3x3*48 with 2x2 downsampling and RELU activation
- 3x3x48 with RELU activation
- 1x1x48 with RELU activation
- 1x1x5 with RELU activation
- output node

LeNet architecture with the following structure:
- 5x5x8 convolution layer with RELU activation
- 2x2 max pooling layer
- 5x5x16 convolution layer with RELU activation
- 2x2 max pooling layer
- 128 node fully-connected layer
- dropout layer (p=0.25)
- output node

An interpretation of the network described in the paper End to End Learning for Self-Driving Cars by M. Bojarski, et al.
- 5x5x24 conv with 2x2 stride and RELU activation
- 5x5x36 conv with 2x2 stride and RELU activation
- 5x5x48 conv with 2x2 stride and RELU activation
- 3x3x64 conv with RELU activation
- 3x3x64 conv with RELU activation
- 100 unit fully-connected layer with RELU activation
- Dropout layer (p=0.25)
- 50 unit fully-connected layer with RELU activation
- Dropout layer (p=0.25)
- 10 unit fully-connected layer (no activation function, oops?)
- output node

The performance of these models on the training a validation sets was promising, but the test result in the simulator were disappointing.  The table below summarizes the results of these 3 networks.

| # | Description | Train Loss | Val Loss  | Notes |
|---|-------------|------------|-----------|-------|
| A | model-all-cnn-c-simple |   0.0088 | 0.0090 | track 1 NOT OK |
| B | Lenet | 0.0081 | 0.0102 | track 1 OK, track 2 NOT OK |
| C | Nvidia | 0.0098 | 0.0091 | track 1 NOT OK |


#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:
1. A 5x1 convolution layer with 8 features and 2x1 dimension reduction
2. A RELU activation layer
3. A 1x5 convolution layer with 8 features and 1x2 dimension reduction
4. A RELU activation layer
5. A 128-node fully-connected layer
6. A RELU activation layer
7. A drop-out layer with p=0.25
8. The single node output layer

#### 3. Creation of the Training Set & Training Process

I did not have access to a PC capable of running the simulator until 1 day before the project due date, so I utilized the provided training data.

To augment the dataset, I flipped the images and angles in order to take advantage of the horizontal symmetry of the control problem.

To further augment the dataset, and encode some recovery data, I utilized the right and left camera views with an adjustment factor added to the measurement. I used the standard deviation of the steering measurements as the adjustment factor.  I observed that the training data consisted of mostly straight line driving with some small corrections made by a human driver, so it seemed reasonable to use the average variation in the steering angles as a correction factor.   After I made this adjustment, the MSE nearly doubled which perhaps suggests the adjustment factor needs to be optimized.

After the collection process, I had roughly 30,000 data points. I then preprocessed this data as follows:
1. Normalize and center
2. Crop the top of the image by 64 pixels and the bottom by 22 pixels

Then finally, I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I trained the model for 7 epochs, but did not have an opportunity to tune this parameter.  I used an adam optimizer so that manually training the learning rate wasn't necessary.
