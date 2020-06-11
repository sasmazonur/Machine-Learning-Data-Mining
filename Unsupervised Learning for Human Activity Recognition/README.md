<p align="center"><img width=20.5% src="https://upload.wikimedia.org/wikipedia/en/thumb/0/07/Oregon_State_College_of_Engineering_Logo.jpg/220px-Oregon_State_College_of_Engineering_Logo.jpg"></p>


## Implementation assignment 4
This assignment we will work with the Samsung Human Activity Recognition dataset to practice unsupervised and also dimensional reduction methods. More specifically we are interested in applying k-means clustering and Principal Component Analysis (PCA) methods.

## Description of the Dataset:
The data set provided are in two parts:
* <b>(x train.txt):</b> Contains 7352 rows (samples) each with 561 features.
* <b>(y train.txt):</b> Contains 7352 rows of labels corresponding to the rows in the (x train.txt).

## Requirements
You will need to install the following packages:
```
python3 -m venv env
source env/bin/activate
pip install numpy
pip install pandas
pip install sklearn
pip install pip
pip install matplotlib
pip install seaborn
```

## Running the program
To run the program and generate the output:
```
python3 main.py --all=1
```
Make sure your working directory is the directory that main.py is in or you may get a path error about not being able to find the training data.
