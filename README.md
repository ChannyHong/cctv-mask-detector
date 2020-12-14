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
python test.py
```

## Preliminary Testing

Now, we are going to test our MTCNN face recognizer + mask detection CNN models on our Arlo security camera footage. Here's a raw footage of me walking out of the office.

[Channy gif]

Right from the get-go, it's very apparent that . Similar to concept drift, our models may have a hard time picking up on the 

 I am pretty certain that our model is not going to do super well 

Since our footage is a video file, we are going to need to deconstruct them into individual frames before feeding them into our model.


However, 

[gif image here showing utter failure]


## Finetuning To The Rescue


Implementation in details



If you take a look at the Arlo footage, you will be able to tell straightaway that the 

However, we know 


As with most projects, I am going to 



So in summary, the 





In order to 



common problems that 

I was especialy exposed to the common problems that machine learning projects face upon deployment, which is that 




Script for training the ISR Encoder. Requires monolingual corpora cache files for training.

**Prerequisites**:

The following cache files saved in the 'data_dir' directory:
- Monolingual corpora sentences cache files, as "mc_##.npy" (e.g. "mc_en.npy") where ## corresponds to [ISO 639-1 Code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) of each 'train_languages'; refer to Parsing and Caching Scripts section below.
- (If do_mid_train_eval) XNLI dev examples cache file, as ["DEV.npy"](https://drive.google.com/uc?export=download&id=1VOZqXGrLRjVbSmf-wB9ETrBtGfQPNZ7L); refer to Parsing and Caching Scripts section below.

(If do_mid_train_eval,) The following model files in the 'mid_train_eval_nli_model_path' directory (the trailing 'nli_solver' is the model name and not part of the directory):
-  English NLI solver model files, as ["nli_solver.meta"](https://drive.google.com/uc?export=download&id=1RroiNFZVxap9FPCwSkS78CiXfoxHkgzw), ["nli_solver.index"](https://drive.google.com/uc?export=download&id=1sptgnDG8lhj415OVnjHt25peORMy8a8E), and ["nli_solver.data-00000-of-00001"](https://drive.google.com/uc?export=download&id=1nDtQFFOM7EnA8sX5viyXcrwkWoGsw1JE) (note that 'mid_train_eval_nli_target_language' should be fixed as English when using this NLI solver).
```
python train_isr.py \
  --data_dir=data \
  --output_dir=outputs/isr_training_model \
  --train_languages=English,Spanish,German,Chinese,Arabic \
  --embedding_size=768 \
  --train_batch_size=32 \
  --Dis_Gen_train_ratio=10 \
  --Dis_learning_rate=0.00001 \
  --Gen_learning_rate=0.00001 \
  --lambda_Dis_cls=1.0 \
  --lambda_Dis_gp=1.0 \
  --lambda_Gen_cls=10.0 \
  --lambda_Gen_rec=1.0 \
  --lambda_Gen_isr=1.0 \
  --beta1=0.5 \
  --beta2=0.999 \
  --num_train_epochs=100 \
  --save_checkpoints_steps=5000 \
  --log_losses=True
  --do_mid_train_eval=True \
  --run_mid_train_eval_steps=5000 \
  --mid_train_eval_nli_target_language=English \
  --mid_train_eval_nli_model_path=nli_solver_path/nli_solver
```

## Tensorboard

Script to access the Tensorboard logs of the various losses (and if do_mid_train_eval, mid train evaluation accuracies) to help decide when to halt training of ISR Encoder. In our study, we stopped training when the generator seemed to be reasonably functional in generating sentences of correct target domain (classification task accuracy) without losing the semantics of the original sentence (English NLI solving accuracy).

```
tensorboard --port=6006 --logdir=outputs/isr_training_model
```

---

## Classifier Training

Code for training a classifier on top of fixed ISR Encoder. Requires NLI training examples (mostly available in high-resource language, i.e. English) for training.

**Prerequisites**:

The following cache files saved in the 'data_dir' directory:
- NLI training examples cache file(s), as "bse_##.npy" (e.g. "bse_en.npy") where ## corresponds to [ISO 639-1 Code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) of each 'xnli_train_languages'; refer to Parsing and Caching Scripts section below. Theoretically, NLI training examples from multiple languages can be used jointly from training the classifier on top of ISR (while the underlying assumption is that only English training examples are widely available currently).
- (if do_mid_train_eval) XNLI dev examples cache file, as ["DEV.npy"](https://drive.google.com/uc?export=download&id=1VOZqXGrLRjVbSmf-wB9ETrBtGfQPNZ7L).

The following files in the 'isr_encoder_dir' directory:
- The ISR Encoder model files outputted from the ISR Encoder Training section above, as "isr_encoder.meta", "isr_encoder.index", "isr_encoder.data-00000-of-00001". Alternatively, the ISR Encoder trained during our study can downloaded here: ["isr_encoder.meta"](https://drive.google.com/uc?export=download&id=1LJ9l-r2OoBPAt-7kNTWr34W7Et-gzhFN), ["isr_encoder.index"](https://drive.google.com/uc?export=download&id=1PIOseaAo37SeKe7_lbSGqQdFI-4h0ywH), and ["isr_encoder.data-00000-of-00001"](https://drive.google.com/uc?export=download&id=1Y0IyQOKZsknMEhQTzdGFyj9zFDtRi2PW).
- The language reference file, as "language_reference.json". The language reference file corresponding to our study's ISR Encoder can be downloaded here: ["language_reference.json"](https://drive.google.com/uc?export=download&id=1Owm6Hv6KKE1NLGhTtgAGINfZc94LHYA_)

```
python train_classifier.py \
  --data_dir=data \
  --isr_encoder_dir=isr_encoder_dir \
  --isr_encoder_name=isr_encoder \
  --output_dir=outputs/custom_output_model_name \
  --xnli_train_languages=English \
  --embedding_size=768 \
  --train_batch_size=32 \
  --dropout_rate=0.5 \
  --learning_rate=0.00005 \
  --beta1=0.9 \
  --beta2=0.999 \
  --num_train_epochs=100 \
  --save_checkpoints_steps=5000 \
  --log_losses=True \
  --do_mid_train_eval=True \
  --mid_train_xnli_eval_languages=English,Spanish,German,Chinese,Arabic \
  --run_mid_train_eval_steps=5000 \
  --mid_train_eval_batch_size=32
```
### Tensorboard

Script to access the Tensorboard logs of the classifier loss and training accuracy (and if do_mid_train_eval, evaluation accuracies on dev examples).

```
tensorboard --port=6006 --logdir=outputs/custom_output_model_name
```
---

## Parsing and Caching Scripts

### Producing a monolingual corpora cache file from Wikipedia dump

**1. Download the [Wikipedia dump](https://dumps.wikimedia.org/) of the language of interest (.XML file).**

**2. Use [WikiExtractor](https://github.com/attardi/wikiextractor) to extract and clean text from the XML file, outputting a file (e.g. "wiki_00") in the "AA" folder within the 'output' directory. The "100G" 'bytes' parameter in our sample usage is to ensure that only 1 file is outputted (rather than broken up into multiple)**:

**Prerequisites**:
- The downloaded dump file (e.g. "en_dump.xml") in the current directory.
```
python WikiExtractor.py \
 --output=en_extracted \
 --bytes=100G \
en_dump.xml
```

**3. Run mc_custom_extraction.py on once-extracted file to perform custom extraction and cleanup to output a .txt file.**

**Prerequisites**:
- The once-extracted dump file renamed to its [ISO 639-1 Code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) (e.g. for once extracted English dump file, renamed to "en" from "wiki_00") in the 'source_file_path' directory.

```
python mc_custom_extraction.py \
  --source_file_path=once_extracted \
  --output_dir=custom_extracted \
  --language=en \
  --char_count_lower_bound=4 \
  --char_count_upper_bound=385 \
  --output_num_examples=392702
```

The monolingual corpora .txt files used in our study can be downloaded here:\
[mc_en.txt](https://drive.google.com/uc?export=download&id=1SkZKzfMY2X5_1XNOvfIE5RSjiA34ec6z)\
[mc_es.txt](https://drive.google.com/uc?export=download&id=1LsoXQgGGp5n_Ks1sFO4AHTe9UH6QUYA7)\
[mc_de.txt](https://drive.google.com/uc?export=download&id=1Mz-wxBcgkMKruep59LB3RgWXnYs15h3_)\
[mc_zh.txt](https://drive.google.com/uc?export=download&id=1jXYRLgox3R_K46uDhOlhzUEY0gYax1pR)\
[mc_ar.txt](https://drive.google.com/uc?export=download&id=1NsRpJvjcjhx4lYc6Of4JfsXVPZi92LW2)

**4. Run bse_cache.py to produce cache files.**

**Prerequisites**:
- [bert-as-service](https://github.com/hanxiao/bert-as-service) installed.
- [BERT-Base, Multilingual Cased model](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) (refer to [BERT Multilingual GitHub page](https://github.com/google-research/bert/blob/master/multilingual.md) for more details) saved in the 'bert_dir' directory.
- The custom extracted .txt file in the 'data_dir' directory, as "mc_##.txt" where ## corresponds to [ISO 639-1 Code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) of the text.

```
python bse_cache.py \
  --data_dir=custom_extracted \
  --language=English \
  --data_type=mc \
  --output_dir=bse_cache_files \
  --bert_dir=../pretrained_models/multi_cased_L-12_H-768_A-12
```

The monolingual corpora cache files used in our study can be downloaded here:\
[mc_en.npy](https://drive.google.com/uc?export=download&id=1LArWH8bU2sL0o-Ih2y2re2osxRGnv8bJ)\
[mc_es.npy](https://drive.google.com/uc?export=download&id=1_PEgGCv7e4YJKWhDBhKseHc9GyYpF2Cj)\
[mc_de.npy](https://drive.google.com/uc?export=download&id=1lcUBEKOry8JscOsM4P3NKeWI0ZZfj8i4)\
[mc_zh.npy](https://drive.google.com/uc?export=download&id=1JZSHRXR_JAelUspEgl2OAS4irIuhdGC2)\
[mc_ar.npy](https://drive.google.com/uc?export=download&id=1VfUl8B0c0o1KqtxHHLXNlGELw9djOuhX)

### Producing a NLI examples cache file from XNLI dataset

**1. Download the [XNLI dev and test examples](https://www.nyu.edu/projects/bowman/xnli/XNLI-1.0.zip) ("xnli.dev.tsv" and "xnli.test.tsv") from the [XNLI project page](https://www.nyu.edu/projects/bowman/xnli/). Also download the [XNLI machine translated training examples](https://www.nyu.edu/projects/bowman/xnli/XNLI-MT-1.0.zip), which includes the original English MNLI training examples (as "multinli.train.en.tsv").**

**2. Run bse_cache.py to produce cache files.**

#### _English MNLI training examples_

**Prerequisites**:
- [bert-as-service](https://github.com/hanxiao/bert-as-service) installed.
- [BERT-Base, Multilingual Cased model](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) saved in the 'bert_dir' directory.
- The English MNLI training examples file in the 'data_dir' directory, as "multinli.train.en.tsv".
```
python bse_cache.py \
  --data_dir=xnli_data \
  --language=English \
  --data_type=mnli \
  --output_dir=bse_cache_files \
  --bert_dir=../pretrained_models/multi_cased_L-12_H-768_A-12
```
The English MNLI training examples cache file used in our study can be downloaded here: [bse_en.npy](https://drive.google.com/uc?export=download&id=1dzOhSUraOtwhSjReoQhISsMeAnqpXhS5)

#### _XNLI dev examples_
**Prerequisites**:
- [bert-as-service](https://github.com/hanxiao/bert-as-service) installed.
- [BERT-Base, Multilingual Cased model](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) saved in the 'bert_dir' directory.
- The XNLI dev examples file in the 'data_dir' directory, as "xnli.dev.tsv".
```
python bse_cache.py \
  --data_dir=xnli_data \
  --data_type=dev \
  --output_dir=bse_cache_files \
  --bert_dir=../pretrained_models/multi_cased_L-12_H-768_A-12
```
The XNLI dev examples cache file used in our study can be downloaded here: [DEV.npy](https://drive.google.com/uc?export=download&id=1VOZqXGrLRjVbSmf-wB9ETrBtGfQPNZ7L)

#### _XNLI test examples_
**Prerequisites**:
- [bert-as-service](https://github.com/hanxiao/bert-as-service) installed.
- [BERT-Base, Multilingual Cased model](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) saved in the 'bert_dir' directory.
- The XNLI test examples file in the 'data_dir' directory, as "xnli.test.tsv".
```
python bse_cache.py \
  --data_dir=xnli_data \
  --data_type=test \
  --output_dir=bse_cache_files \
  --bert_dir=../pretrained_models/multi_cased_L-12_H-768_A-12
```
The XNLI test examples cache file used in our study can be downloaded here: [TEST.npy](https://drive.google.com/uc?export=download&id=12u5oTmpGZ0hpZTNyY_ABSQRTvU6U--rm)