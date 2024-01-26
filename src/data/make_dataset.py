# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import re
import unicodedata
import contractions

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    # Load data
    df = pd.read_json(input_filepath, lines=True)
    # Remove all records with no headline text. Clean and split data
    df = df[df['headline'] != '']
    # Split the data.
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['is_sarcastic', 'article_link']), df['is_sarcastic'], test_size=0.3, random_state=42, stratify =df['is_sarcastic'])
    write_train_test_data(output_filepath, X_train, X_test, y_train, y_test)

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def pre_process_corpus(docs):
    norm_docs = []
    for doc in docs:
        doc = strip_html_tags(doc)
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = doc.lower()
        doc = remove_accented_chars(doc)
        doc = contractions.fix(doc)
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z0-9\s]', ' ', doc, flags=re.I|re.A)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()
        norm_docs.append(doc)
    return norm_docs

def generate_train_test_addresses(processed_path):
    if not os.path.exists(processed_path):
        os.mkdir(processed_path)
    output_paths = dict()
    output_paths['train_feature_path'] = processed_path + '/X_train.csv'
    output_paths['test_feature_path'] = processed_path + '/X_test.csv'
    output_paths['train_target_path'] = processed_path + '/y_train.csv'
    output_paths['test_target_path'] = processed_path + '/y_test.csv'
    return output_paths

def write_train_test_data(output_filepath, X_train, X_test, y_train, y_test):
    output_paths = generate_train_test_addresses(output_filepath)
    X_train.to_csv(output_paths['train_feature_path'], index=False)
    X_test.to_csv(output_paths['test_feature_path'], index=False)
    y_train.to_csv(output_paths['train_target_path'], index=False)
    y_test.to_csv(output_paths['test_target_path'], index=False)

def read_train_data(output_filepath):
    output_paths = generate_train_test_addresses(output_filepath)
    X_train = pd.read_csv(output_paths['train_feature_path'])
    y_train = pd.read_csv(output_paths['train_target_path'])
    return X_train, y_train

def read_test_data(output_filepath):
    output_paths = generate_train_test_addresses(output_filepath)
    X_test =pd.read_csv(output_paths['test_feature_path'])
    y_test = pd.read_csv(output_paths['test_target_path'])
    return X_test, y_test



def read_train_test_data(output_filepath):
    output_paths = generate_train_test_addresses(output_filepath)
    X_train = pd.read_csv(output_paths['train_feature_path'])
    X_test =pd.read_csv(output_paths['test_feature_path'])
    y_train = pd.read_csv(output_paths['train_target_path'])
    y_test = pd.read_csv(output_paths['test_target_path'])
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    data_path = os.getenv("DATA_PATH")
    processed_path = os.getenv("PROCESSED_PATH")
    main(data_path, processed_path)
