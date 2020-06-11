import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
import re


MAX_FT = 2000
MAX_DF = 1
MIN_DF = 1


# Set _DEV = 1 to use smaller data for quick code tests
_DEV = 1 ## TODO change to 0
if _DEV == 1:
    for i in range(5):
        print("WARNING Running in Dev Mode")
    print("")


def clean_text(text):
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    # convert text to lowercase
    text = text.strip().lower()
    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
    return text



def generate_bow(input_text, vocabulary):
    vectorizer = CountVectorizer(analyzer = "word",
        tokenizer = None,
        preprocessor= None,
        stop_words = None,
        max_features = MAX_FT,
        max_df = MAX_DF,
        min_df = MIN_DF)
    # Get vocabulary, features, and word count
    vocab = vectorizer.fit(input_text)
    feature_names = vectorizer.get_feature_names()
    wordCount = vectorizer.transform(input_text)
    return vocab, wordCount, feature_names



def get_data():
    if _DEV ==1:
        imdb_data = pd.read_csv('IMDB_dev.csv', delimiter=',')
        imdb_label = pd.read_csv('IMDB_labels_dev.csv', delimiter=',')
    else:
        imdb_data = pd.read_csv('IMDB.csv', delimiter=',')
        imdb_label = pd.read_csv('IMDB_labels.csv', delimiter=',')
    return imdb_data, imdb_label



def split_data(imdb_data, imdb_label):
    if _DEV == 1: # Cut down on run-time by using tiny data
        training_set = list(imdb_data['review'][:3000])
        label_train = list(imdb_label['sentiment'][:3000])
        validation_set = list(imdb_data['review'][3000:4000])
        label_validation = list(imdb_label['sentiment'][3000:4000])
        test_set = list(imdb_data['review'][4000:])
    else:
        training_set = list(imdb_data['review'][:30000])
        label_train = list(imdb_label['sentiment'][:30000])
        validation_set = list(imdb_data['review'][30000:40000])
        label_validation = list(imdb_label['sentiment'][30000:40000])
        test_set = list(imdb_data['review'][40000:])
    return training_set, label_train, validation_set, label_validation, test_set



def get_vocab(data):
    vectorizer = CountVectorizer(stop_words="english", preprocessor=clean_text)
    vectorizer.fit(data)
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    return [inv_vocab[i] for i in range(len(inv_vocab))]


# Main Function
def main():
    # Import Data -> WordCount(Count Vectorizer) -> Term Frequency & Inverse Document Frequency -> Naive Bayes Classifier
    # Import and split data
    imdb_data, imdb_label = get_data()
    training_set, label_train, validation_set, label_validation, test_set = split_data(imdb_data, imdb_label)
    # Generate vocabulary and clean input
    vocabulary = get_vocab(imdb_data['review'])
    cleanSentences = [clean_text(imdb_data['review'][i]) for i in range(0, imdb_data['review'].size)]
    
    ## TODO Q1 BOW representation for all 50k reviews.
    bagOfWords, wordCount, feature_names = generate_bow(cleanSentences, vocabulary)
    
    ## TODO Q2 Train a multi-nomial Naive Bayes classifier with Laplace smooth with α = 1 on the training set.
    # Sentiment analysis
    
    ## TODO Q3 Apply the learned Naive Bayes model to the validation set (the next 10k reviews) and report the validation accuracy of the your model.

    ## TODO Q4 Tuning smoothing parameter alpha. Train the Naive Bayes classifier with different values of α between 0 to 2 (incrementing by 0.2).

    ## TODO Q5 Tune your heart out.
    
    if _DEV == 1:
        print("\nEND MAIN")


if __name__ == "__main__":
    main()
    if _DEV == 1:
        print("\nEXITING")
