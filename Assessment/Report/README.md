# Deep learning approach of Crop disease on phone

Jingqi Cheng

Edge Impulse link: https://studio.edgeimpulse.com/studio/199147

## Introduction
- Overview of what the project does

My project enable the users to real-time detect the crop disease for main diseases of 4 main type of crops leaves(potato, corn, rice and wheat), by using a phone for input images. 

- Inspiration for making the project 

Agriculture is the largest employment sector in many developing countries, such as Bangladesh, making up 14.2 percent of Bangladesh's GDP in 2017 and employing about 42.7 percent of the workforce (Wikipedia, 2023). Crop diseases can cause significant damage to agriculture, resulting in huge economic losses and food scarcity. Early detection of crop diseases is essential to prevent the spread of diseases and to take appropriate measures to control them. Traditional methods of disease detection are time-consuming and require expertise, making it difficult for farmers to identify diseases quickly (Fang & Ramasamy, 2015). To address this issue, deep learning-based approaches have been proposed as a promising solution for crop disease detection.

- Examples that it is based on. 

J.et. al. utilized convolutional neural network (CNN)-based pre-trained models for efficient plant disease identification and focused on fine tuning the hyperparameters of popular pre-trained models, such as DenseNet-121, ResNet-50, VGG-16, and Inception V4. They used PlantVillage dataset, which consists of total 54,305 image in 38 classes. They have proved that DenseNet-121 achieved 99.81% higher classification accuracy (J. et al., 2022).


## Research Question
Can a deep learning model accurately detect and classify multiple types of crop diseases using a limited dataset?

## Application Overview
The purpose of this project is to classify the leaves into different types of health conditions.
The image input is getting from the end device, in this project, I build on the mobile phone. When the image has been captured, Preprocessing and feature extraction are performed in Image processing block in edge impulse (Edge Impulse Edge Impulse Documentation (2022a)), aim to take the raw image and improve image features by suppressing unwanted distortions, resizing and/or enhancing important features, making the data more suited to the model and improving performance. The TensorFlow model will run the image classification model which is a deep learning model built, trained and converted to multiple TensorFlow models, such as Conv2d model, in the Edge impulse environment (Edge Impulse Edge Impulse Documentation (2022a)). Then the model will take the input real-time image from the phone and classify it into the categories of different types crop disease and output the result of the name of the class on the phone automatically.

<img width="500" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/dl.png">


## Data
The dataset used in this project is from the Kaggle dataset New Bangladeshi Crop Disease (https://www.kaggle.com/datasets/nafishamoin/new-bangladeshi-crop-disease). This dataset consists of 10427 items in 14 classes. The detailed classes and corresponding item counts per class is displayed below in the table. Below showed some sample images of the dataset.

For preprocessing the data, I used the image block in Edge Impulse. The image block allows the preparation of image as changing the image height, width and colour. As most of the deep learning models assume a square shape input image, so the resize of the image are to the same aspect ratio, in this project, the image size is resize to 32 x 32 or 96 x 96. For image colour, RGB or greyscale channels can be choose, by collapse RGB channel to greyscale channel, the dimensional reduction has been applied, this can make the training problem simpler for model to process. The data is then split into 80% for the training dataset, 20% for the testing dataset, and the 20% of the training dataset is used for validation.

|  Class   | Item count  |
|  ----  | ----  |
| Corn_common_rust  | 1192 |
| Corn_leaf_spot  | 513 |
| Corn_northern_leaf_blight  | 985 |
| Corn_Healthy  | 1162 |
| Potato_early_blight  | 1000 |
| Potato_late_blight  | 1000 |
| Potato_Healthy  | 152 |
| Rice_brown_spot  | 613 |
| Corn_leaf_blast  | 977 |
| Corn_neck_blast  | 1000 |
| Corn_Healthy  | 1488 |
| Wheat_brown_rust  | 902 |
| Wheat_yellow_rust  | 924 |
| Wheat_Healthy  | 1116 |

<img width="200" alt="image" src="/https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/corncommonrust.jpg"> <img width="200" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/corngreyleaf.jpg"> <img width="200" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/cornhealthy.jpg"> <img width="200" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/cornnorthernleaf.jpg">

<img width="200" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/potatoearly.jpg"> <img width="200" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/potatolate.jpg"> 

<img width="200" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/ricebrown.jpg"> <img width="200" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/ricehealthy.jpg"> <img width="200" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/riceleafblast.jpg"> <img width="200" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/riceneckblast.jpg">

<img width="200" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/wheatbrownrust.jpg"> <img width="200" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/wheathealthy.jpg"> <img width="200" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/wheatyellowrust.jpg">

## Model
In order to get the best performance on accuracy on the testing dataset. I have tested on three different models, grayscale-conv2d-d95, rgb-conv2d-a07 and rgb-conv2d-afa. All models are based on conv2d neural layer, the conv2d layer combined with pooling layers, flatten layer and dropout layer to build a complete neural network model. The Conv2D layer can be configured by setting parameters such as the number of filters and pooling.

<img width="500" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/conv2d.png">

In general, increasing the number of filters can improve accuracy, but may result in slower training times. Using a larger kernel size can capture more global features in the input image, but may result in lower accuracy on more detailed features.
Ultimately, the best approach is to experiment with different parameter settings and evaluate the performance of the model on a validation set to determine which combination of parameters works best for your specific problem. 
The grayscale-conv2d-d95 model is designed to classify grayscale image using convolutional layer, in general, it is useful when colour information is not necessary or low power device.
The rgb-conv2d-a07 model has a deep architecture and require more computing memory and power.
The main difference between rgb-conv2d-afa and rgb-conv2d-a07 is their architecture and training dataset.
The rgb-conv2d-afa model has lower accuracy than rgb-conv2d-a07 for complex image classification in most cases but in my project, rgb-conv2d-afa model has slightly higher accuracy on training dataset, this might because the used dataset is relatively small and simple.
More experiments can be found in the next section.



## Experiments
During the experiments, I run the EON Tuner, which A tool in Edge Impulse that is used to optimise machine learning models for microcontrollers and other resource-constrained devices. 
This involves finding the right balance between model accuracy and model size, memory usage, and computational resources.
This optimises the hyper parameters such as learning rate, batch size and regularisation parameters, along with the architecture of the neural network to find the best model that fits the device requirements of memory usage and computational resources. The result of the EON Tuner is showed below.

<img width="500" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/eon.png">

I have run the EON Tuner twice, and find out slightly different results each time, so I used the EON Tuner result as a starting point to tune the 9 parameters on different models in the Edge Impulse environment. 
For the experiment, I first tested on the grayscale-conv2d-d95 model, I found that when the training cycle and learning rate increased respectively, the validation accuracy and test accuracy has increased significantly to 78.2%. This model has the lowest accuracy among the three models, this proved the dataset should not be reduced to greyscale channel.

<img width="500" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/grey.png">

According to the result of EON Tuner, I did some experiments on rgb-conv2d-a07 model, apart from changing the training cycle and learning rate, I changed the image size between 32 x 32 and 96 x 96, but the result shows by increasing the size of the image, there’s little impact on the result of accuracy. The best result of accuracy occurs when increasing the training cycle.

<img width="500" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/a07.png">

Then, for the optimized performing rgb-conv2d-afa model, I have tried to auto-balance the dataset before modelling, increase the number of convolutional layer and increase the dropout rate. The result showed there’s little impact on accuracy when use auto-balance dataset and increase the dropout rate, but optimised result when one convolutional layer. Also showed increase the number of training cycle increased the peak RAM and flash usage.

<img width="500" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/afa.png">


## Results and Observations
According to the confusion matrix generated by edge impulse based on the optimized model results, we can find that the labels used to classify the rice class have the lowest accuracy, especially rice_brown_spot and rice_leaf_blast. Most of these two types of labels will be predicted in the test set and validation set. Rice_healthy label.

<img width="500" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/confusion.jpg">

When I build this project to the mobile phone for testing, other class labels can be quickly and accurately classified, but I found that the accuracy of all classes in rice class is low, and most of them will be classified as tags under the potato class . This might because there are major types of diseases that look very different for rice. In order to solve this problem, the project can name only a few blast and brown spot diseases occurred by fungus infection, and tungro and ragged stunt developed from viruses. 

<img width="500" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/result1.jpg">

<img width="500" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/result2.jpg">

<img width="500" alt="image" src="https://github.com/Naomi1122/casa0018/blob/main/Assessment/Report/image/result3.jpg">

Future improvement

- Object detection
Multiple leaves in different environments can be detected in a single image. Useful to increase the production speed in real time.
- Model
Experiments with model not based on conv2d layer
- Dataset
 Narrower down the different classes by only selected a single type of crops to start the project to test for most suitable parameters set and model.
- Link
Train the model so it can give out advice on treating the crops if disease is detected. Make the production more artificial intelligent, less human resources needed.
- Deploy
Deploy on more devices, microcontroller, so more portable as crop production is in the field.

## Bibliography
1. Edge Impulse Documentation (2022a) Impulse design, Edge Impulse. Available at: https://docs.edgeimpulse.com/docs/edge-impulse-studio/impulse-design (Accessed: March 26, 2023).
2. Fang, Y. and Ramasamy, R.P. (2015) Current and prospective methods for plant disease detection, Biosensors. U.S. National Library of Medicine. Available at: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4600171/ (Accessed: April 27, 2023). 
3. J., A. et al. (2022) Deep learning-based leaf disease detection in crops using images for agricultural applications, MDPI. Multidisciplinary Digital Publishing Institute. Available at: https://www.mdpi.com/2073-4395/12/10/2395#:~:text=Early%20diagnosis%20of%20plant%20diseases,classification%20and%20object%20detection%20systems. (Accessed: April 27, 2023). 
4. Wikipedia, W. (2023) Agriculture in Bangladesh, Wikipedia. Wikimedia Foundation. Available at: https://en.wikipedia.org/wiki/Agriculture_in_Bangladesh (Accessed: April 27, 2023). 
----

## Declaration of Authorship

I, Jingqi Cheng, confirm that the work presented in this assessment is my own. Where information has been derived from other sources, I confirm that this has been indicated in the work.


Jingqi Cheng

27/04/2023
