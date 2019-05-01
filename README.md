# MURA-PROJECT
## What is MURA?
"MURA (musculoskeletal radiographs) is a large dataset of bone X-rays. Algorithms are tasked with determining whether an X-ray study is normal or abnormal.
Musculoskeletal conditions affect more than 1.7 billion people worldwide, and are the most common cause of severe, long-term pain and disability, with 30 million emergency department visits annually and increasing. We hope that our dataset can lead to significant advances in medical imaging technologies which can diagnose at the level of experts, towards improving healthcare access in parts of the world where access to skilled radiologists is limited.
MURA is one of the largest public radiographic image datasets. We're making this dataset available to the community and hosting a competition to see if your models can perform as well as radiologists on the task." (https://stanfordmlgroup.github.io/competitions/mura/)

## Approach:
The purpose of this program is to detect the abnormality  the upper limb parts in the xrays privided in MURA dataset.  To approach that, first it identifies the part in the xray as finger, hand, wrist, forearm, elbow, humerus or shoulder.  After that, the corresponding model is selected to predict the abnormality in that part.  The model then take the arithmetic mean of all the xrays in the study and produce the final prediction. (for more information about the structure of the dataset, please go to: https://stanfordmlgroup.github.io/competitions/mura/)

## How to run the codes?

1. first, the dataset need to be rearranged to have (train, valid and test) sets for each model. For example, the dataset for the upper lim part classification need to have 7 classes:
           1. XR_ELBOW
           2. XR_FINGER
           3. XR_FOREARM
           4. XR_HAND
           5. XR_HUMERUS
           6. XR_SHOULDER
           7. XR_WRIST
while the dataset of the elbow (for example) model need 2 classes:
           1. negative
           2. positive
           
2.  Then run the upper limb parts classifier and save the .h5 and .json files.

3.  Run the 7 abnormality detectors as well and save the corresonding .h5 and .json files.

4.  Now everything is ready to run the Main program.ipynb that will load all the saved models and test the overall performance of the trained models.

## Simple web_app user interface
The code xrayforweb.py uses flask api to post the prediction results of a user selected image.  The posted result is sent to the predict.html for user visualization.  Just make sure to place all the trained models in h5/ folder and place the predict.html file in /static folder.



 
