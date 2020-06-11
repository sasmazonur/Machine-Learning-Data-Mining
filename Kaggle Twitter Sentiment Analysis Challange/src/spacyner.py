# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 21:34:07 2020

@author: Adrian Henle
"""

import pandas as pd
try:
    import spacy
except:
    import pip
    pip.main(["install", spacy])
try:
    import tqdm
except:
    import pip
    pip.main(["install", tqdm])
import random
from spacy.util import minibatch, compounding
import os


def clean_training_data(sentiment, rawdata):
    """
    Cleans training data for sentiment-specific NER.......
    """
    
    train_data = []
    for _, row in rawdata.iterrows():
        if row.sentiment == sentiment:
            start = row.text.find(row.selected_text)
            # Create selected_text column with index bounds for entity recognition
            train_data.append((row.text, {"entities": [[start, start + len(row.selected_text), 'selected_text']]}))
            
    return train_data


def save_model(output_dir, nlp, new_model_name):
    """
    Saves model.
    """
    
    #output_dir = "../working/{}".format(output_dir) # Format path
    if not os.path.exists(output_dir): # Make sure path exists
        os.makedirs(output_dir)
    nlp.meta["name"] = new_model_name # Apply name
    nlp.to_disk(output_dir) # Save


def train(train_data, output_dir, n_iter, model=None):
    """
    Train n_iter NLP models.
    """
    
    # Load or create NLP model
    if model is not None:
        nlp = spacy.load(output_dir)
    else:
        nlp = spacy.blank("en")
    
    # Pipeline components
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")
    
    # Label
    for _, annotations in train_data:
        for entity in annotations.get("entities"):
            ner.add_label(entity[2])

    # Disable other piplines (only train NER)
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        if model is None:
            nlp.begin_training()
        else:
            nlp.resume_training()

        # Train NER
        for itn in tqdm.tqdm(range(n_iter)):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))    
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, drop=0.5, losses=losses)
            
    save_model(output_dir, nlp, "st_ner")
    
    ## TODO play with minibatch size, nlp.update drop


def train_NER(df_train, n_iter):
    """
    Trains the negative and positive sentiment NER models.
    """
    
    train(clean_training_data("positive", df_train), "models/model_pos", n_iter)
    train(clean_training_data("negative", df_train), "models/model_neg", n_iter)


def predict_entities(text, model):
    """
    Predict the sentiment of input text based on input model.
    """
    
    doc = model(text)
    
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if new_int not in ent_array:
            ent_array.append([start, end, ent.label_])
    
    return text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text


def get_selected_texts(df_test):
    selected_texts = []

    model_pos = spacy.load("models/model_pos")
    model_neg = spacy.load("models/model_neg")
        
    for index, row in df_test.iterrows():
        text = row.text
        if row.sentiment == 'neutral' or len(text.split()) <= 2:
            selected_texts.append(text)
        elif row.sentiment == 'positive':
            selected_texts.append(predict_entities(text, model_pos))
        else:
            selected_texts.append(predict_entities(text, model_neg))
            
    return selected_texts


def save_submission(df_test):
    df_submission = pd.DataFrame()
    df_submission['selected_text'] = df_test['selected_text']
    pd.DataFrame({'textID': df_test.textID, 'selected_text': df_submission.selected_text}).to_csv('submission.csv', index=False)


def main(n_iter = 15):
    # Get data from Kaggle competition
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    
    # Train the NER models for negative and positive sentiment
    train_NER(df_train, n_iter) ## TODO play with number of iterations
    
    # Get selected texts
    df_test['selected_text'] = get_selected_texts(df_test)
    
    # Save results to submission.csv
    save_submission(df_test)
