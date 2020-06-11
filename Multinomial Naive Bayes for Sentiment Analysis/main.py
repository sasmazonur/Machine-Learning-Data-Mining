# group47_imp2.py
# Adrian, Tristan, Onur; 2020.04.26
# Applies a Naive Bayes classifier to IMDB movie reviews for sentiment analysis.

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import matplotlib.pyplot as plt

"""try:
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
except:
    import nltk
    nltk.download("punkt")
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize"""



# Vectorizer Parameters
MAX_FT = 2000
MAX_DF = 0.4
MIN_DF = 0.04
ALPHA = 1


# Set _DEV = 1 to use smaller data for quick code tests
_DEV = 0
if _DEV == 1:
    for i in range(5):
        print("WARNING Running in Dev Mode")
    print("")


"""
Removes HTML tags and special characters, forces lower case
"""
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    text = text.strip().lower()
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
    
    ## TODO Porter stemming or lemmatizing? ## NOT HERE IT'S SLOW
    """ps = PorterStemmer()
    text = word_tokenize(text)
    text = [ps.stem(text[i]) for i in range(0, len(text))]
    text = " ".join(text)"""
    
    return text


"""
Generate the bag-of-words representation of input text
"""
def generate_bow(input_text, vocabulary):
    vectorizer = CountVectorizer(analyzer = "word",
        tokenizer = None,
        preprocessor= None,
        stop_words = None,
        vocabulary = vocabulary)
    # Get vocabulary, features, and word count
    vocab = vectorizer.fit(input_text)
    feature_names = vectorizer.get_feature_names()
    wordCount = vectorizer.transform(input_text)
    return vocab, wordCount, feature_names


"""
Load data from CSV
"""
def get_data():
    if _DEV ==1:
        imdb_data = pd.read_csv('IMDB_dev.csv', delimiter=',')
        imdb_label = pd.read_csv('IMDB_labels_dev.csv', delimiter=',')
    else:
        imdb_data = pd.read_csv('IMDB.csv', delimiter=',')
        imdb_label = pd.read_csv('IMDB_labels.csv', delimiter=',')
    return imdb_data, imdb_label


"""
Split data into train, validate, and test sets
"""
def split_data(data, label):
    
    ## TODO randomize rows?
    
    if _DEV == 1: # Cut down on run-time by using tiny data
        training_set = list(data[:3000])
        label_train = list(label[:3000])
        validation_set = list(data[3000:4000])
        label_validation = list(label[3000:4000])
        test_set = list(data[4000:])
        
    else:
        training_set = list(data[:30000])
        label_train = list(label[:30000])
        validation_set = list(data[30000:40000])
        label_validation = list(label[30000:40000])
        test_set = list(data[40000:])
    return training_set, label_train, validation_set, label_validation, test_set


"""
Process data to extract vocabulary
"""
def get_vocab(data, MIN_DF = MIN_DF, MAX_DF = MAX_DF):
    vectorizer = CountVectorizer(stop_words="english", preprocessor=clean_text, min_df = MIN_DF, max_df = MAX_DF, max_features = MAX_FT)
    vectorizer.fit(data)
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    return [inv_vocab[i] for i in range(len(inv_vocab))]


"""
MAIN FUNCTION
Import Data -> WordCount(Count Vectorizer) -> Term Frequency & Inverse Document Frequency -> Naive Bayes Classifier
"""
def _main():
    # Import and split data
    imdb_data, imdb_label = get_data()
    #print(vocabulary)
    cleanSentences = [clean_text(imdb_data['review'][i]) for i in range(0, imdb_data['review'].size)]
    
    training_data, training_labels, val_data, val_labels, test_data = split_data(cleanSentences, [imdb_label['sentiment'][i] for i in range(0, imdb_label['sentiment'].size)])
    
    MAX_DFi = MAX_DF
    MIN_DFi = MIN_DF
    ALPHAi = ALPHA
    
    ## Q3: Demonstrate Classifier
    if _DEV ==1:
        vocabulary = get_vocab(imdb_data['review'][:3000], MIN_DFi, MAX_DFi)
    else:
        vocabulary = get_vocab(imdb_data['review'], MIN_DFi, MAX_DFi)
    #_, wordCount, _ = generate_bow(cleanSentences, vocabulary, MIN_DF)
    
    _, train_matrix, _ = generate_bow(training_data, vocabulary)
    _, val_matrix, _ = generate_bow(val_data, vocabulary)
    #_, test_matrix, _ = generate_bow(test_data, vocabulary)
    
    # Sentiment analysis
    classifier = MultinomialNB(ALPHAi)
    if _DEV == 1:
        print("ALPHA: {}".format(ALPHAi))
        print("MIN_DF: {}".format(MIN_DFi))
        print("MAX_DF: {}".format(MAX_DFi))
    
    classifier.fit(train_matrix, training_labels)
    if _DEV == 1:
        print("Val Matrix Dimensions:")
        print(val_matrix.shape)
        print("Vocab Length: {}".format(len(vocabulary)))
    check_array = classifier.predict(val_matrix)
    for i in range(0, len(check_array)):
        if check_array[i] == "positive":
            check_array[i] = 1
        else:
            check_array[i] = 0
    np.savetxt("test-prediction1.csv", check_array.astype(int))
    
    ## Q4: Tune Alpha
    alpha_accuracy = []
    best_accuracy = -1
    preds = []
    for ALPHAi in np.arange(0, 2, 0.2):
        if _DEV ==1:
            vocabulary = get_vocab(imdb_data['review'][:3000], MIN_DFi, MAX_DFi)
        else:
            vocabulary = get_vocab(imdb_data['review'], MIN_DFi, MAX_DFi)
        #_, wordCount, _ = generate_bow(cleanSentences, vocabulary, MIN_DF)
        
        _, train_matrix, _ = generate_bow(training_data, vocabulary)
        _, val_matrix, _ = generate_bow(val_data, vocabulary)
        #_, test_matrix, _ = generate_bow(test_data, vocabulary)
        
        # Sentiment analysis
        classifier = MultinomialNB(ALPHAi)
        if _DEV == 1:
            print("ALPHA: {}".format(ALPHAi))
            print("MIN_DF: {}".format(MIN_DFi))
            print("MAX_DF: {}".format(MAX_DFi))
        
        try:
            classifier.fit(train_matrix, training_labels)
            if _DEV == 1:
                print("Val Matrix Dimensions:")
                print(val_matrix.shape)
                print("Vocab Length: {}".format(len(vocabulary)))
            check_array = classifier.predict(val_matrix)
            logicals = check_array == val_labels
            alpha_accuracy.append(np.sum(logicals)/len(logicals))
            if _DEV == 1:
                print("ACCURACY: {}".format(np.sum(logicals)/len(logicals)))
        except Exception as e:
            print("FAIL: {}".format(e))
        if alpha_accuracy[-1] > best_accuracy:
            best_accuracy = alpha_accuracy[-1]
            preds = check_array
    for i in range(0, len(preds)):
        if preds[i] == "positive":
            preds[i] = 1
        else:
            preds[i] = 0
    np.savetxt("test-prediction2.csv", preds.astype(int))
    plt.plot(np.arange(0,2,0.2), alpha_accuracy)
    plt.title("Tuning Alpha")
    plt.xlabel("Alpha")
    plt.ylabel("Accuracy")
    plt.show()
    
    ## Q5: Tune Like an Orchestra
    df_accuracy_best = -1
    MIN_DF_best = -1
    MAX_DF_best = -1
    preds = []
    for MIN_DFi in np.arange(0.01, 0.06, 0.005):#np.arange(0.03, 0.05, 0.001):
        for MAX_DFi in np.arange(MIN_DFi*2, 0.9, 0.05):
            if _DEV ==1:
                vocabulary = get_vocab(imdb_data['review'][:3000], MIN_DFi, MAX_DFi)
            else:
                vocabulary = get_vocab(imdb_data['review'], MIN_DFi, MAX_DFi)
            #_, wordCount, _ = generate_bow(cleanSentences, vocabulary, MIN_DF)
            
            _, train_matrix, _ = generate_bow(training_data, vocabulary)
            _, val_matrix, _ = generate_bow(val_data, vocabulary)
            #_, test_matrix, _ = generate_bow(test_data, vocabulary)
            
            # Sentiment analysis
            classifier = MultinomialNB(ALPHAi)
            if _DEV == 1:
                print("ALPHA: {}".format(ALPHAi))
                print("MIN_DF: {}".format(MIN_DFi))
                print("MAX_DF: {}".format(MAX_DFi))
            
            try:
                classifier.fit(train_matrix, training_labels)
                if _DEV == 1:
                    print("Val Matrix Dimensions:")
                    print(val_matrix.shape)
                    print("Vocab Length: {}".format(len(vocabulary)))
                check_array = classifier.predict(val_matrix)
                logicals = check_array == val_labels
                df_accuracy = np.sum(logicals)/len(logicals)
                if df_accuracy > df_accuracy_best:
                    df_accuracy_best = df_accuracy
                    MIN_DF_best = MIN_DFi
                    MAX_DF_best = MAX_DFi
                    preds = check_array
                if _DEV == 1:
                    print("ACCURACY: {}".format(np.sum(logicals)/len(logicals)))
            except Exception as e:
                print("FAIL: {}".format(e))
                
    print("Best accuracy ({}) at DF_MIN {} and DF_MAX {}".format(df_accuracy_best, MIN_DF_best, MAX_DF_best))
    for i in range(0, len(preds)):
        if preds[i] == "positive":
            preds[i] = 1
        else:
            preds[i] = 0
    np.savetxt("test-prediction3.csv", preds.astype(int))


if __name__ == "__main__":
    _main()
    if _DEV == 1:
        print("\nEXITING")
