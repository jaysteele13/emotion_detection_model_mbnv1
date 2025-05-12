## Summary

This repo tracks the progress of **classifying emotions** using **transfer learning** on **Deep Convulutional Neural Network**, **MobileNetV1**.

This repository follows the process training and testing a multi-classification model for 7 emotions on an open source dataset known as '[FaceRec2](https://universe.roboflow.com/project-1-7fc2a/face-recognition-uy5cg)' 

I have been lucky enough to have been given access to **[Kelvin2](https://ni-hpc.github.io/nihpc-documentation/Connecting%20to%20Kelvin2/)**. A high performance cluster to aid in training and processing my model.

## Accuracy and Context to the Problem

I highest accuracy I achieved was **68.9%** which is **.3%** greater than the maximum accuracy the dataset creator achieved.

It's clear by the confusion matrix that it is not only a ***tuning issue***. But a greater problem. 
1. *Anger & Disgust* are commonly onfused which is a staple in this linear multi-classification problem where perhaps my attention layer couldn't differentiate them to the best of its ability.
2. There are some faces which I would say are **mislabled** however this leads me into my third and final point.
3. **EMOTIONS ARE NOT BINARY**. It is clear that some people may see one emotion when another sees a different one.

There is a multitude of reasons for this, namely emotions are more nuanced than 7 linear ones that I have chosen. **People can feel and look as they are experiencing more than one emotion**, and based on certain cultures, emotions can vary greatly in terms of percieving emotions. I found this in the *TFEID* dataset where those of taiwanese culture often look more sad than scared to me personally. However as I say, these observations differ greatly amongst the individual.

## Dataset Label Distribution
I experimented with adding personal 'in-the-wild' images to the dataset, using a ResNet-50 pipeline to augment the dataset in an attempt to improve its ability to identify more common weight characteristics in real-world images.
<img width="947" alt="facerec2_with_additions" src="https://github.com/user-attachments/assets/11c9cc9f-31fa-4300-ac9d-60a3a03cb640" />



## Statistics
<img width="671" alt="confusion_matrix" src="https://github.com/user-attachments/assets/76314489-c26a-4a79-9a0e-212d71bba5e4" />

<img width="675" alt="accuracy" src="https://github.com/user-attachments/assets/923ca665-3826-4aa3-b9d9-41d8f0bfb65b" />

<img width="854" alt="graph_loss" src="https://github.com/user-attachments/assets/fe7f67a0-d9b2-4a4b-89be-d57a962d4dda" />


