# Smoking-Retection-AI-Robot
We built a CNN model using AlexNet to classify smoking images, which performed well in tests and practical applications. Our research shows it can achieve good results on devices with limited computing resources and has promising practical applications.
Project Report - Smoker Detection

# Abstract
This project focuses on the binary classification of smoking images and explores and optimizes two classic CNN models, VGG and AlexNet. Based on real-world situations, we ultimately use the AlexNet framework to build a model that performs well in multiple indicator tests and practical applications. Our exploration and related conclusions also demonstrate that our designed CNN model can achieve relatively optimal results even on devices with limited computing resources, and has good practical application prospects.
Introduction
# Background:
The widespread presence of negative content on social media can have a profound impact on the mental health and guidance of young people. To address this issue, social media platforms have introduced a “teen mode” that filters out harmful content such as violence and smoking.
Our team is developing an AI-powered monitoring system to detect inappropriate images, specifically those depicting smoking behavior or cigarettes. This tool will serve as an aid for image classification in “teen mode” on social media platforms.
We believe that this technology can help protect the well-being of young people and make a positive contribution to society.

# Overview of the technology:
Nowadays, the application of image detection and classification in AI should be contributed to deep learning methods, which mainly uses convolutional neural networks (CNNs) for extracting features from images. 
CNN mimics the biological visual system's neural network structure that automatically learns hierarchical feature representations from raw images with strong generalization ability and robustness. 
Meanwhile, many new network structures are being developed and applied along with CNN, through multiple optimization methods, data augmentation techniques based on deep learning, to further improve performance efficiency in image recognition/classification tasks. For instance, VGG improves model performance by increasing the number of layers in the network; AlexNet introduces ‘relu’ function to maximize the output trend and accelerate the training procedure.
Novelty:
·Our model aimed for a binary-classification task on images only, which is specialized and trained for smoking detection specifically.
·Our model pursues maximum environmental adaptability, making it possible to achieve relatively optimal results even on devices with insufficient computing resource. This ensures high availability even in image recognition tasks based on local rather than server-based systems.
·Our exploration procedure includes comparison among multiple AI models, allowing for an in-depth and large-scale analysis on this specific tasks.

# Methodology
Description of dataset:
Our dataset contains 3,275 image files, classified into 2 categories: smoking(1996) and not smoking(1267). This image contains a wide variety of objects and the number of samples for these 2 class is imbalanced. Therefore, we assume that the model may encounter many challenges during the model training.

Data Preprocessing:
As the code shown, we use Keras' module for image data preprocessing to preprocess the images to a certain extent, including but not limited to data standardization, horizontal flipping, distortion, and other operations to enhance the model's generalization ability and improve the training effectiveness of the model.

Training Environment: 
Platform 1:
Processor: Intel Core i9-9980HK
GPU: NVIDIA GeForce GTX 1060 6G
Memory: 32 Gigabytes

Platform 2:
Processor: Intel Core i7-9750H
GPU: NVIDIA GeForce GTX 1650 4G
Memory: 16 Gigabytes

Software Version:
Microsoft Visual Studio Code 1.77.3
Python 3.9.16
tensorflow-gpu 2.6.0
Keras 2.9.0
CUDA 11.2.0
cuDNN 8.8.1.3

(All the following model exploration procedure has been tested on both platforms)

# Model exploration:
VGG
The VGG (Visual Geometry Group) model is a deep convolutional neural network architecture developed by a research team at the University of Oxford. 
Technological principles:
The basic idea of the VGG model is to use a series of convolutional layers and pooling layers to extract features from an image, and then map these features to different classes through fully connected layers. The main feature of the VGG model is the use of very small convolutional filters (3x3) and stacking multiple convolutional layers, which allows for a deeper network with fewer parameters than other architectures like AlexNet.
Selection reason:
·Simple and easy-to-understand architecture.
·Alleged one of the best performances on various image classification.
·Highly modular, easier to modify and adapt for different tasks.
Test procedure:
To shorten the training procedure and improve accuracy, we are going to firstly apply public pre-trained VGG model trained on ImageNet, to validate its overall performance on our dataset, and then fine-tune it by our dataset to improve its performance.
Through exploration, we found that this model can easily run into overfitting on our relatively small dataset. After examination, for the avoidance of overfitting, we set batch-size of training procedure to 16, which is a reasonable number for this mission.
Without Fine-tuning:

BATCH SIZE = 16
Train Loss:  6.506694793701172
Train Accuracy:  0.5093750357627869
--------------------
Validation Loss:  6.068174362182617
Validation Accuracy:  0.53125
--------------------
Test Loss:  5.530672073364258
Test Accuracy:  0.5779816508293152

Result of test accuracy is surely disappointing, indicated that pre-trained VGG model does not fit for our goal. What’s more, the curve of validation accuracy also indicated that overfitting situation clearly occurred during the training process, for the figure of training accuracy failed to climb up and maintained an upward trend.
Fine tuning is obviously needed.
With Fine-tuning:
Based on the pre-trained model, we added multiple dense and dropouts on top of the model and unlock certain layers for training purpose, allowing it to train itself. 
The result of our procedure is as below:

Batch size: 16:

Train Loss:  2.6246659755706787
Train Accuracy:  0.703125
--------------------
Validation Loss:  2.494419813156128
Validation Accuracy:  0.699999988079071
--------------------
Test Loss:  2.6440773010253906
Test Accuracy:  0.6941896080970764

The validation accuracy climbed up to by a fascinating 12%, to 69%. The overfitting situation occurred later at around epoch 14, when the training and validation accuracy are both struggling to climb up any further. 
As a result, the fine-tuning process successfully improve the performance of our model. However, overfitting problem still rose, and the overall outcome with a 69% accuracy is clearly far from our goal.
Conclusion:
VGG model is quite easy to build and modify. But for our binary classification problem, its performance contains several fatal drawbacks:
·It may have reliable performance on multiple classification. But for our binary classification, its performance is clearly poorer. We assumed the reason as that it is not well-suited for tasks that require real-time predictions like ours. 
·The overfitting problem is hard to overcome on our dataset, which contains less than 3000 images, due to the large number of parameters.
·The required computational source is too big for us. The average time-assumption of a 5-epochs training procedure is over 30 minutes on our hardware with GPU acceleration on.

Therefore, the VGG model architecture may not be suitable for our binary classification task.


AlexNet
AlexNet is a very classic deep convolutional neural network (CNN) architecture developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in 2012.

Technological principles:

"The traditional AlexNet has 8 layers, including 5 convolutional layers and 3 fully connected layers, and introduces non-linearity using Rectified Linear Unit (ReLU) activation function. The framework of AlexNet has many advantages over VGG in image binary classification problems."

Selection reason:

Compared to VGG, AlexNet is a shallower architecture with relatively smaller computational complexity, because of which we assumed that its training and prediction speeds may be faster than VGG. The smaller computational complexity also allows it to be run and optimized on GPUs with lower computing power.

Based on the obvious shortcomings exposed in the previous exploration of VGG, taking into account the actual situation of the dataset as well as the advantages and disadvantages of the AlexNet model, this project will continue to explore the use of a CNN framework based on the second model, AlexNet, in the hope of achieving even better performance.

Model Design and Exploration Process:

Firstly, regarding the dataset, we used Keras' ImageDataGenerator tool to perform appropriate image rotation and flipping, random cropping and scaling, in order to generate more data samples and increase the model's robustness to resist the impact of noise and deformation factors in practical situations, while the computing power allows.

Secondly, for the model construction, after researching and adjusting 57 model frameworks, we made the following modifications based on the characteristics and size of the dataset and obtained the relatively optimal model. The research process and related experimental data records are as follows:

# Optimization approach:

Firstly, we used the basic code framework of AlexNet for initial training and visualized the Training and Validation Accuracy indicators of the model. We can judge to what extent our model learned the characteristics of the dataset and its generalization ability by the trend of the line graph, so as to determine some relatively optimal parameters.

1.The existing AlexNet model with three fully connected layers is highly prone to overfitting for our dataset. The model often experiences sharp fluctuations in validation accuracy around 0.7 after about 5 epochs, even when the training accuracy steadily rises to 0.85-0.90.
Solution: Try removing the last fully connected layer.

2.Next, even with two fully connected layers, a similar situation as in the first stage occurred, and we also found that the model's training speed was too slow. In addition to the problems mentioned earlier, we also hope to reduce the use of the GPU appropriately.
Solution: Try modifying the original batch size from 32 to 16.

3.At this point, the Training and Validation Accuracy indicators of our model have improved significantly compared to before, and the validation accuracy has also increased to around 0.75. However, we still observed overfitting when increasing the number of training epochs.
Solution: We tried reducing the number of neurons in the first layer of the original AlexNet from 96 to 48.

In addition to the major framework modifications mentioned above, we also made modifications and attempted many details, including but not limited to the padding state of some layers, the increase or decrease of the number of neurons in the fully connected layers, the insertion of BatchNormalization() in the neural network, adjustment of the Dropout value, and the selection of activation functions...

We generated a model for each modification mentioned in the previous section, saved each experimental model. Based on the performance of the visualization graphs and test accuracy values of each model, we listed the top 5 models with the best performance(Sort by Test Accuracy):

No. of Model	Train Accuracy	Validation Accuracy	Test Accuracy
Model 16	0.9186	0.7577	0.7767
Model 34	0.8465	0.7826	0.7645
Model 07	0.8060	0.7050	0.7523
Model 27	0.7925	0.7236	0.7302
Model 45	0.7671	0.7422	0.7248
...	...	...	...

*The explorations in the above experimental stages are all based on the optimal solution of the model under controlled relative variables.



Model 16 has the highest validation accuracy during our exploration process. The final Training and Validation Accuracy visualization results are shown below, and overall, the validation accuracy of this model steadily increased as the number of epochs increased. However, we found that this model had a little overfitting after epoch 15, and the validation loss remained relatively high at epoch 35.

The Model 16


The Model 34
After careful consideration, we ultimately chose Model 34, which maintains a relatively high level of accuracy while slowly decreasing the validation loss and avoiding overfitting.



The final "sequential" model generated by Model34 is shown below:


To improve accuracy, the structure of this model is more complex than our expectation, which illustrates that the training task based on this dataset is very challenging. However, it also demonstrates that, relatively speaking, AlexNet will outperform VGG in terms of overall performance and structure construction on this task.

Model Evaluation:
After finalizing the model, we conducted a series of evaluations around it:

1.Precision&Recall&f1-score ,accuracy ,macro & weighted avg



In this model, the accuracy is 0.76, which means it correctly classified 76% of the total 327 samples. The precision is 0.81 and the recall is 0.81, indicating that the model can accurately identify most of the smoking samples. The F1 score is a weighted average of precision and recall and combines these two metrics to some extent. In this example, the model has an F1 score of 0.81, indicating that it predicts smoking samples relatively accurately.

Due to the imbalance of our dataset, with smoking (1996) - no smoking (1267), we also calculated two other metrics, macro avg and weighted avg. The precision, recall, and F1 score of macro avg are 0.75, 0.75, and 0.75, respectively, indicating that the model's performance in predicting smoking samples is still good enough.

2.Confusion Matrix


We then generated the confusion matrix for this binary classification model. In this matrix, there were 88 samples with the true label "not_smoking" correctly predicted as "not_smoking," and 39 samples incorrectly predicted as "smoking." For samples with the true label "smoking," 38 were incorrectly predicted as "not_smoking," and 162 were correctly predicted as "smoking." Based on the above metrics, we can conclude that this model performed well in identifying smoking samples, but there is still room for improvement.

3.ROC curve


The ROC curve is a method for measuring the performance of a binary classification model. Based on the curve generated by our model, the AUC is 0.82, indicating relatively high accuracy and good performance.


Real-World Test



In real-world testing, we adjusted the threshold of the model to improve its accuracy in judging smoking images found on social media, and achieved good results. Considering the real-world situation, theoretically we need to set a higher threshold for the binary classification model, which will increase the probability of false positives when judged as negative.

This process inspires us that in the later stage of model optimization, we can also use some unmarked datasets to manually screen out images that are suspected to be smoking and annotate them. We can evaluate the performance of the model using the annotated dataset and then adjust the threshold using similar methods. This method requires more time and labor costs, but it can ensure higher detection accuracy.

# Conclusion:
During the exploration process of this project, we explored and optimized two CNN frameworks, VGG and AlexNet, based on the specific application scenario and GPU requirements. Through modifications to the framework and adjustments to various hyperparameters, we ultimately obtained a modified AlexNet model that performs relatively well and meets our expectations The model has been evaluated through various metrics and actual testing, showing good potential and application value. Our model's final performance exceeded that of the MobileNetV2 model fine-tuned on Kaggle, demonstrating that our exploration has yielded relatively better results with limited GPU resources. 
We have also seen some attempts based on other more recent models, which achieved higher prediction accuracy than ours (but most of them also had higher loss values), such as EfficientNet. However, on our platform with limited computing power, we always encountered situations where the model could not be loaded, or training could not be completed. Therefore, we can assert that, trying to train more advanced models on a platform with higher computational power may be possible, but under current conditions, our model is already a relatively well-balanced solution in terms of performance and power requirements, which also meets the initial goal of exploring relatively more general training solutions.

# Future Prospects:
Although we have obtained relatively good results and models, their performance is slightly worse compared to other mainstream research directions such as cat and dog recognition, where accuracy can reach 95%. Our situation may be a relatively difficult issue to improve the accuracy of the model. 
First of all, from the perspective of the dataset (except for some problems of data imbalance and small number of samples), only the features of smoke or cigarettes are not enough to determine whether the person in the picture is smoking. By checking the dataset images, we can even see some other animal images and special situations such as "burning cigarettes dropped on the ground", which also increase the difficulty of model training.
Nevertheless, there are still some areas for improvement in our process. Regarding the dataset, we can increase the number of training images and add specific content to some of the training images, such as adding some human face images to the no-smoking dataset for training, in order to enhance the model's ability to understand images. (Due to limited GPU computing power, we have made some attempts at the beginning and obtained relatively good results but have not thoroughly optimized.) 
In addition to the models and frameworks mentioned above, we can also try other frameworks in order to demonstrate relatively better performance.

# Additional document
Due to the size of the model is too large to upload, please visit Google Cloud Disk
https://drive.google.com/file/d/1TJsRbJmGgVDYJWMexd0fMXim37rKyTBG/view?usp=share_link
