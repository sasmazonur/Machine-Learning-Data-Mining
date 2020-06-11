<p align="center"><img width=20.5% src="https://upload.wikimedia.org/wikipedia/en/thumb/0/07/Oregon_State_College_of_Engineering_Logo.jpg/220px-Oregon_State_College_of_Engineering_Logo.jpg"></p>


## Implementation assignment 2

In this program implements the Naive Bayes (Multinomial) classifier for Sentiment Analysis of an IMDB movie review dataset (a highly polarized dataset with 50000 movie reviews). 
The primary task is to classify the reviews into negative and positive.

## Description of the Dataset:
The data set provided are in two parts:
* <b>IMDB.csv:</b> This contains a single column called Reviews where each row contains a movies review. There are total of 50K rows. 
The first 30K rows should be used as your Training set (to train your model). The next 10K should be used as the validation set (use this for parameter tuning). 
And the last 10K rows should be used as the test set (predict the labels).
* <b>IMDB labels.csv:</b> This contains 40K labels. Please use the first 30K labels for the training data and the last 10K labels for validation data. The labels for test data is not provided, we will use that to evaluate your predicted labels.


## Data cleaning and generating BOW representation:
### <b>Data Cleaning:</b>
Pre-processing is need to makes the texts cleaner and easy to process. The reviews columns are comments provided by users about the movie. These are known as ”dirty text” that required further cleaning. 

#### Typical cleaning steps include:
* a) Removing html tags 
* b) Removing special characters 
* c) Converting text to lower case 
* d) Replacing punctuation characters with spaces
* e) Removing stopwords (i.e. articles, pronouns from consideration.)


### <b>Generating BOW representation:</b>
To transform from variable length reviews to fixed-length vectors, we use the Bag Of Words technique. 
It uses a list of words called ”vocabulary”, so that given an input text we can output a vector of word counts for each word in the vocabulary. 


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
```

## Running the program
To run the program and generate the output:
```
python3 group47_imp2.py
```
A variable can be toggled in the program code to run on a smaller data set for code testing, to reduce the runtime of the program since it is normally very long. 
