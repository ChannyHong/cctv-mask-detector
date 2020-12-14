# cctv-mask-detector

This is the project page for the CCTV Mask Detector project that I did as part of my internship at Superb AI during the fall of 2020. 

Copyright 2020 Superb AI, Inc.\
The code, cache files, and the models are all released under the Apache 2.0 license.\
Authors: Channy Hong

---

## Background

The year 2020 was shaped by the coronavirus pandemic in ways that pervaded our everyday lives. In particular, mask-wearing became the social norm for everyone when going outside. And because humans are animals of habits, one of the most common "oops, forgot to..." moments this year perhaps has been forgetting to put on one's mask before leaving the door. The same could be said for the wonderful folks as Superb AI as well, and to ensure that everyone is safe from this dangerous disease, we came up with a very simple ML solution to this problem.

On a separate note, I also wanted to internally test out the usefulness of Superb AI's Suite within a machine learning project workflow. During my internship, I was exposed to the various ways in which machine learning projects en route to deployment face. And perhaps one of the most common challenges faced was in continuously integrating the data annotation process, which is especially relevant in concept drift scenarios.

Concept drift occurs when deployed models assume that the incoming live data will be similar to that it was trained on. The fact of the matter is that it often is not the case - in some cases even impossible - that the data that our deployed model will predict on will be of the same distribution to the training data used to train it. In almost all cases, a subset of the live data must be continuosuly annotated and fed back into the training loop to make sure that our model doesn't degrade over time.

## Big Picture

The big picture idea behind this project is to hook up our Arlo security camera with an ML model that detects whether folks leaving the office has their mask on or not. At the top level, our machine learning model would consist of the following:

1. Face recognition model that first identifies all the faces in a frame (bounding box).
2. Mask detection model that then classifies whether a given face is (a) correctly masked (b) incorrectly masked or (c) not masked

And if less than 75% of folks in the frame are correctly masked, then the model automatically triggers an announcement that yells "Don't forget to wear your mask correctly before going outside!"

## Implementation Strategy In Details

Now let's get down to the knitty-gritty. At first glance, this should not be an extremely difficult model to implement. 

As for the face recognition model, there already are many awesome face detection models (pretrained) available out and about the internet from which I can just pick out the face recognition portion. MTCNN (multi-task convolutional neural network) seems to do the trick well, and I ultimately decided to use [timesler's facenet-pytorch project](https://github.com/timesler/facenet-pytorch) as a starting point. The repository comes with a pretrained MTCNN model that can be adapted for our use.

The mask detection model should be even easier to implement. [cabani's MaskedFace-Net](https://github.com/cabani/MaskedFace-Net) project provides correctly masked and incorrectly masked versions of the [FFHQ-dataset](https://github.com/NVlabs/ffhq-dataset)(the original would be the not masked class) which can be our training data for the mask detection three-way classifier model. We are just going to use a very simple CNN with two convolutional layers and two pooling layers, given the simplicity of the task at hand.

Bootstrapping this without any annotating on our end, our project spec looks as the following:

Data:
- 'correctly masked' class: from [cabani's MaskedFace-Net](https://github.com/cabani/MaskedFace-Net)
- 'incorrectly masked' class: from [cabani's MaskedFace-Net](https://github.com/cabani/MaskedFace-Net)
- 'not masked' class: from [FFHQ-dataset](https://github.com/NVlabs/ffhq-dataset)

Model Training
1. Use the pretrained MTCNN face recognition model from [timesler's facenet-pytorch project](https://github.com/timesler/facenet-pytorch) to extract bounding boxes.
2. Train the mask detection classifier using the resulting bounding boxes from the previous step.

[SHOW SIMPLE DIAGRAM HERE]

## Implementation Code

I used Python 3.7.9 for this project. Make sure you have the following libraries installed in your environment:
- torch
- facenet_pytorch

**Project & Data Setup**

First create an empty folder called 'mask_dataset':
```
mkdir mask_dataset
```

Then, download the [MaskedFace-Net](https://github.com/cabani/MaskedFace-Net) project into the "mask_dataset" folder. Make sure to have separate folders for each mask detection class, and also within it, one for each 'train' and 'test'. The folder structure should look as the following:
```
mask_dataset
    L correctly_masked
        L train
            L img1.jpg
            L img2.jpg
            L ...
        L test
            L ...
    L incorrectly_masked
        L ...
    L not_masked
        L ...
```

**Extract Face Bounding Boxes**

```
python extract_faces.py
```

**Train Mask Detection Classifier**

First, we create a 'models' folder where we will be storing the output .pt model file.
```
mkdir models
```

Now run the training script on the training dataset.
```
python train_mask_detection_model.py
```

When the training is complete, the output .pt model file will be in our 'models' folder. 

## Sanity Check

Now, we are going to test our MTCNN face recognizer + mask detection CNN models on the test dataset.

```
python test.py \
--data_dir=mask_dataset
```

## Preliminary Testing

Now, we are going to test our MTCNN face recognizer + mask detection CNN models on our Arlo security camera footage. Here's a raw footage of me walking out of the office.

[Channy gif]

Right from the get-go, it's very apparent that . Similar to concept drift, I suspect that our model is going to have a hard time dealing with the low resolution and the . Similar to concept drift, the difference in the distribution of this live data from the training dataset will 

Since our footage is a video file, we are going to need to deconstruct them into individual frames before feeding them into our model. Then we draw bounding boxes + classify them frame by frame, and then we can reconstruct them back into video files for viewing. Upon deployment, this all can happen on-the-go using the live feed.

Deconstruct video files:
```
python deconstruct.py \
--data_dir=arlo_footage/raw_videos
``` 

Feed Arlo footage frames into our model:
```
python test.py \
--data_dir=arlo_footage/deconstructed_frames
``` 

Reconstruct video files:
```
python reconstruct.py
--data_dir=arlo_footage/output_frames
``` 

The reconstructed vidoes should now be in the 'reconstructed_videos' folder. An example of a reconstruction:

[gif image here showing utter failure]

As expected, our model does a terrible job at the moment even placing bounding boxes around faces. What should we do next?

## Finetuning To The Rescue


Implementation in details



If you take a look at the Arlo footage, you will be able to tell straightaway that the 

However, we know 


As with most projects, I am going to 



So in summary, the 





In order to 



common problems that 

I was especialy exposed to the common problems that machine learning projects face upon deployment, which is that 
