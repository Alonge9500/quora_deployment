import pandas as pd
import numpy as np
import re
import contractions
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
import Levenshtein
from gensim.models import Word2Vec
import pickle
import texthero as hero


feature_scaler_path = os.path.join(os.path.dirname(__file__), '..', 'pickles', 'scaler.pkl')
word2vec_path = os.path.join(os.path.dirname(__file__), '..', 'pickles', 'word2vec_model.pkl')


with open(feature_scaler_path,'rb')as f:
    scaler = pickle.load(f)


with open(word2vec_path,'rb')as z:
    word2vec = pickle.load(z)
    

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def first_features(question1,question2):

    question1 = str(question1)
    question2 = str(question2)

    # Feature calculations

    tokens1 = word_tokenize(question1)
    tokens2 = word_tokenize(question2)
    common_tokens = len(set(tokens1) & set(tokens2))
    percentage_common_tokens = common_tokens / min(len(tokens1), len(tokens2))

    question1_length = len(tokens1)
    question2_length = len(tokens2)
    length_difference = abs(question1_length - question2_length)

    num_capital_letters1 = sum(1 for char in question1 if char.isupper())
    num_capital_letters2 = sum(1 for char in question2 if char.isupper())
    num_question_marks1 = question1.count('?')
    num_question_marks2 = question2.count('?')

    starts_with_are = question1.lower().startswith('are') or question2.lower().startswith('are')
    starts_with_can = question1.lower().startswith('can') or question2.lower().startswith('can')
    starts_with_how = question1.lower().startswith('how') or question2.lower().startswith('how')

    feature_row = [percentage_common_tokens, question1_length,
                    question2_length, length_difference, num_capital_letters1, num_capital_letters2,
                    num_question_marks1, num_question_marks2, starts_with_are, starts_with_can, starts_with_how]

    return feature_row
def clean(question):

    custom_pipeline = [
                    ppe.remove_whitespace,
                    ppe.remove_punctuation,
                    ppe.remove_digits,
                    ppe.fillna,
                    ppe.remove_whitespace,
                    ppe.remove_brackets,
                    ppe.lowercase,
                    ppe.remove_diacritics,
                    ]
    return hero.clean(question,custome_pipeline)
    

def remove_html_tags(text):
    # Use BeautifulSoup library to remove HTML tags from the text
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    return cleaned_text

def tokenize_text(text):
    # Use NLTK's word_tokenize function to tokenize the text
    tokens = word_tokenize(text)
    return tokens
    

def remove_urls(text):
    if isinstance(text, str):
        url_pattern = re.compile(r'http\S+|www\S+')
        return url_pattern.sub('', text)
    else:
        return text


def remove_special_characters(text):
    if isinstance(text, str):
        special_chars_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        return special_chars_pattern.sub('', text)
    else:
        return text


def preprocess_text(text):
  if isinstance(text, str):
    # Apply the custom preprocessing steps here
    text = remove_html_tags(text)
    text = tokenize_text(text)
    text = remove_urls(text)
    text = remove_special_characters(text)
    # Add any additional preprocessing steps you need
    
    return text
    
nltk.download('stopwords')
nltk.download('wordnet')

# Function to remove stopwords and perform lemmatization
def text_preprocess(text, flag='lem', remove_stopwords=False):
    if text is None:
        return ''
    
    if isinstance(text, list):
        text = ' '.join(text)
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Remove stopwords if specified
    if remove_stopwords:
        tokens = [token for token in text.split() if token.lower() not in stop_words]
    else:
        tokens = text.split()

    # Apply lemmatization if specified
    if flag == 'lem':
        processed_text = ' '.join([lemmatizer.lemmatize(token) for token in tokens])
    else:
        processed_text = ' '.join(tokens)

    # Return the preprocessed text
    return processed_text
    

def word_count(text1, text2):
    return len(text1.split()), len(text2.split())

def sentence_count(text1, text2):
    return len(text1.split('.')), len(text2.split('.'))


def avg_word_length(text1, text2):
    if len(text1) > 0 and len(text2) > 0:
        total_length1 = sum(len(word) for word in text1.split())
        total_length2 = sum(len(word) for word in text2.split())
        return total_length1 / len(text1.split()), total_length2 / len(text2.split())
    else:
        return 0, 0 

def unique_word_count(text1, text2):
  words1 = set(text1)
  words2 = set(text2)
  return len(words1.union(words2))

def similar_word_count(text1, text2):
  words1 = set(text1)
  words2 = set(text2)
  return len(words1.intersection(words2))


def fuzzy_word_partial_ratio(text1, text2):
  return fuzz.patial_ratio(text1, text2)

def token_set_ratio(text1, text2):
  return fuzz.token_set_ratio(text1, text2)

X_train['token_set_ratio'] = X_train.apply(lambda row: fuzz.token_set_ratio(row['clean_text1'], row['clean_text2']), axis=1)
X_test['token_set_ratio'] = X_test.apply(lambda row: fuzz.token_set_ratio(row['clean_text1'], row['clean_text2']), axis=1)

def token_sort_ratio(text1, text2):
  return fuzz.token_sort_ratio(text1, text2)

def word_overlap(text1, text2):
    words1 = set(text1)
    words2 = set(text2)
    
    if len(words1) == 0 or len(words2) == 0:
        return 0.0
    
    overlap = words1.intersection(words2)
    return len(overlap) / (len(words1) + len(words2))

def jaccard_similarity(text1, text2):
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if len(words1) == 0 and len(words2) == 0:
        return 0.0
    
    jaccard_sim = len(words1.intersection(words2)) / len(words1.union(words2))
    return jaccard_sim


X_train['jaccard_similarity'] = X_train.apply(lambda row: jaccard_similarity(row['clean_text1'], row['clean_text2']), axis=1)
X_test['jaccard_similarity'] = X_test.apply(lambda row: jaccard_similarity(row['clean_text1'], row['clean_text2']), axis=1)


def levenshtein_distance(text1, text2):
    return fuzz.ratio(text1, text2)


def final_features(question1,question2):
    features1 = first_features(question1,question2)
    
    clean_tex1 = clean(question1)
    clean_tex2 = clean(question2)
    
    clean_tex1 = preprocess_text(question1)
    clean_tex2 = preprocess_text(question2)
    
    clean_tex1 = text_preprocess(question1)
    clean_tex2 = text_preprocess(question2)
    
    word_count1,word_count2 = zip(word_count(clean_text1,clean_text2))
    sentence_count1,sentence_count2 = zip(sentence_count(clean_text1,clean_text2))
    avg_word_length1,avg_word_length2 = zip(avg_word_length(clean_text1,clean_text2))
    unique_word_count = unique_word_count(clean_text1,clean_text2)
    similar_word_count = similar_word_count(clean_text1,clean_text2)
    fuzzy_word_partial_ratio = fuzzy_word_partial_ratio(clean_text1,clean_text2)
    token_set_ratio = token_set_ratio(clean_text1,clean_text2)
    token_sort_ratio = token_sort_ratio(clean_text1,clean_text2)
    word_overlap = word_overlap(clean_text1,clean_text2)
    levenshtein_distance = levenshtein_distance(clean_text1,clean_text2)
    
    
    
