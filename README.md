# Affect-In-Tweets

*Yang Chen*



## Challenge Link

* Semeval 2018 Task 1: Affect in Tweets

  * https://competitions.codalab.org/competitions/17751

* WASSA 2017 twitter emotion detection

  * https://competitions.codalab.org/competitions/16380

    ​

## Introduction

* cnn-sentence
  * the cnn sentence model implemented for emotion regression task (subtask 1)
* baselines
  * baseline models implemented to compare with cnn sentence model (subtask 1)
  * go to this folder to see instructions for run these models
* EI-OC-EN
  * cnn sentence model for EI-OC-EN emotion classification task (subtask 2)
  * go to this folder to see instructions for run these models
* wordvec
  * Folder to save word vector models
  * Too big to attach, install from google drive first as introduced below
* results
  * results will be saved here as the format:
    * {emotion}_pred.txt
  * compress them to get a zip file to submit on the challenge website



## Model Installation

* Install the model from: https://drive.google.com/open?id=1Fy013eAX_NYR85_F0TdHMWxPk_m4levy

* Save it to the wordvec sub folder.

  ​



## Run

In linux or mac:

```
sudo chmod +x run.sh
./run.sh
```

### Results

- All results will be saved at ../results with the format: '../results/{}-pred.txt'.format(emotion), compress them and then submit the zip file to the challenge website to see evaluation results.

  ​

## Environment requirement

- python 2.7
- scipy
- numpy
- tensorflow
- gensim
- At least 12 G Memory

### Library Installation(Use sudo if needed):

```
pip install scipy
pip install numpy
pip install gensim
pip install tensorflow
```



## Test Environment

The code is fully tested on:

```
Ubuntu Linux 16.04 64bit
Python 2.7
Memory size: 32G
CPU: Intel 8700k
```

