# Traffic-Sign-Classifier-using-Deep-Learning-LeNet-5
This project uses LeNet-5, a type of Convolutional Neural Network, to classify German traffic signs

**The goals / steps of this project are the following:**
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## How does the algorithm work?

Let me explain the whole code, part by part.

### Step 1: Dataset summary

* The dataset comes from [Ruhr University Bochum in Germany](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

* After loading the dataset, I just explored and printed out some details of the dataset.

* The number of training examples = 34799 (meaning there are 34,799 images of traffic signs in this dataset)
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3, meaning they are low-res RGB images.
* The number of unique classes/labels in the data set is 43

#### Step 1.1: Visualizing the dataset

* This part deals with visualizing the images in the dataset.
* It displays random images from the training set

![Visualizing dataset](https://github.com/vikasnataraja/Traffic-Sign-Classifier-using-Deep-Learning-LeNet-5/blob/master/output_images/data_images_original.png)

### Step 2: Design and Test Model Architecture

In this step, I am designing and implementing a deep learning model that learns to recognize traffic signs. Training and testing the model is done using the German Traffic Sign Dataset.

LeNet-5, a type of CNN, is being used here. LeNet is a deep-learning model developed by Yann LeCun, more information can be found [here](http://yann.lecun.com/exdb/lenet/).

There are various aspects to consider when thinking about this problem:

    - Neural network architecture (is the network over or underfitting?)
    - Preprocessing techniques (normalization, rgb to grayscale, etc)
    - Number of examples per label (some have more than others).


#### Step 2.1: Pre-process the Dataset (normalization, grayscale, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data and can be used in this project.

Here, I am using a simple normalization technique -> (x-min)/(max-min) for all the pixels in the image.

What this does is it essentially normalizes the pixel values to between 0 and 1, thereby chaning contrast. Images before and after normalization are shown below:

In terms of contrast, after normalization, the black has become blacker, white has become whiter. The color separation is more clear now.

![effect of normalization](https://github.com/vikasnataraja/Traffic-Sign-Classifier-using-Deep-Learning-LeNet-5/blob/master/output_images/after_norm.png)

While there are no major changes in color, normalizing the image helps the model to limit its range to between 0 and 1, i.e. instead of searching all pixel values from 0 to 255, it can limit its search to numbers between 0 and 1.

Further, I decided not to use grayscale in this case because traffic signs are also color-dependent, meaning red colored signs are usually stop or yield signs, yellow ones are usually cautionary and so on.

While I am aware converting it to grayscale would very likely have yielded a higher accuracy and maybe even made the model run faster, as the authors themselves have said in the paper, I chose not to for the simple reason that color has information in this dataset.

#### Step 2.2 Model architecture

Here is an image that shows the overall architecture of LeNet-5, taken from a [paper published by Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf):

![Architecture of LeNet](https://cdn-images-1.medium.com/max/1200/1*1TI1aGBZ4dybR6__DI9dzA.png)

|Layer            | Description/Comments                               |  
|---              |---                                                 |
|Input            | 32x32x3 RGB (color) images                         |   
|Convolution (5x5)| 2 stride (or 2x2), valid padding, outputs 28x28x6  |  
|ReLU             | Activation function                                |
|Max pooling      | 2x2 stride, outputs 14x14x6                        |
|ReLU             | Activation function                                |
|Convolution (5x5)| 	2x2 stride, valid padding, outputs 10x10x16      |
|ReLU             | Activation function                                |
|Max pooling      | 2x2 stride, outputs 5x5x16                         |
|Convolution (1x1)| 2x2 stride, valid padding, outputs 1x1x412         |
|ReLU             | Activation function                                |
|Fully connected  | in 412, out 122                                    |
|ReLU             | Activation function                                |
|Dropout          | Keep 0.5 or 50%                                    |
|Fully connected  | in 122, out 84                                     |
|ReLU             | Activation function                                |
|Dropout          | Keep 0.5 or 50%                                    |
|Fully connected  | in 84, out 43(=unique classes)                     |


## Step 3: Training the model and tuning hyperparameters

I trained the model using LeNet with an additional convolutional layer towards the end. I used the Adam Optimizer, the learning rate was 0.004,
for 25 epochs with a batch size of 156. These hyperparameters could possibly be even further tuned for better accuracies, but under the time constraints,
this was the best I could do. 

## Step 4: Solution and the approach

I chose an iterative approach to arrive at my solution.
  * First off, I did not try the exact approach as outlined in the paper in the context of pre-processing and converting the colorspaces.
  * As mentioned earlier, I felt color had an important role to play in this dataset, so I retained the colorspace.
  * Dropout significantly improved accuracies. Pooling made it faster. I also added an additional layer of convolution
    which gave higher accuracies beyond 96.8% which is what I was seeing before doing so.
  * I tuned the epochs, batch size, learning rate and the dropout parameters. 
    * Increasing the epochs significantly increased the accuracies initially but ultimately started saturating and overfitting.
    * Learning rate was a fiddly parameter. There was no single value that resulted in extremely high values.
    * Batch size did not seem to affect accuracies beyond 156.
   
I plotted a figure to track the variations in accuracy for each epoch. This allowed me to decide a stopping point for the epochs beyond which the accuracies were not improving.
I did this for both validation and training accuracies. 

![Plot of training accuracy vs epochs](https://github.com/vikasnataraja/Traffic-Sign-Classifier-using-Deep-Learning-LeNet-5/blob/master/output_images/accuracies.png)

![Plot of validation accuracy vs epochs](https://github.com/vikasnataraja/Traffic-Sign-Classifier-using-Deep-Learning-LeNet-5/blob/master/output_images/accuracies_valid.png)


My final accuracies were:
  * Training accuracy = 99.7%
  * Validation accuracy = 94.2%
  * Test accuracy = 92.7%

## Step 5: Test model on new images

For this part, I downloaded 6 images from the internet by just googling it. The images are again 32x32x3. Here are the images I chose:

![Test images from the web](https://github.com/vikasnataraja/Traffic-Sign-Classifier-using-Deep-Learning-LeNet-5/blob/master/output_images/new_test_images_mysigns.png)

* Then I normalized these images, just as I had done with the original dataset.
* I then ran the images through the neural net.

### Step 5.1: Results for the test images

* The algorithm worked really well on the new images. In fact, they were predicted with 100% accuracy on the first guess. 
* I initially had lower accuracies but after tuning my neural network and re-training my model with variants of hyperparameters, obtaining a better training accuracy, I was able to get 100% here.

The results look like this:

![Results for the test images from the web](https://github.com/vikasnataraja/Traffic-Sign-Classifier-using-Deep-Learning-LeNet-5/blob/master/output_images/estimate_test.png)

