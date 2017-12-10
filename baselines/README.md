# Baseline Models

*Yang Chen*



This folder is for baseline model I implemented to compare with the cnn sentence model.



## Model Installation

* Install the model from: https://drive.google.com/open?id=1Fy013eAX_NYR85_F0TdHMWxPk_m4levy
* Save it to the ../wordvec sub folder.




## Run

In linux or mac:

```
sudo chmod +x run.sh
./run.sh
```

### Models

* The default model used is lightGBM, uncomment corresponding part and comment the lightGBM part to use other baseline models.
* Other baseline modes include:
  * xgboost
  * Linear Regression
  * Support Vector Machine

### Results

* All results will be saved at ../results with the format: '../results/{}-pred.txt'.format(emotion), compress them and then submit the zip file to the challenge website to see evaluation results.



## Environment requirement

- python 2.7
- scipy
- numpy
- tensorflow
- gensim
- pandas 0.19.0
- xgboost
- lightgbm
- sklearn
- At least 12 G Memory

### Library Installation(Use sudo if needed):

```
pip install scipy
pip install numpy
pip install gensim
pip install tensorflow
pip install pandas==0.19.0
pip install xgboost
pip install sklearn
pip install lightgbm
```



## Test Environment

The code is fully tested on:

```
Ubuntu Linux 16.04 64bit
Python 2.7
Memory size: 32G
CPU: Intel 8700k
```

