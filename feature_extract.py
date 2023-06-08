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
from nltk import ngrams
from nltk import FreqDist
from fuzzywuzzy import fuzz
from gensim.models import Word2Vec
import pickle
import texthero as hero
from texthero import preprocessing as ppe
from sklearn.metrics.pairwise import cosine_similarity



feature_scaler_path = os.path.join(os.path.dirname(__file__), 'pickles', 'scaler.pkl')
word2vec_path = os.path.join(os.path.dirname(__file__), 'pickles', 'word2vec_model.pkl')


with open(feature_scaler_path,'rb')as f:
    scaler = pickle.load(f)


with open(word2vec_path,'rb')as z:
    word2vec = pickle.load(z)
    

def download_all():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')


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

    starts_with_are = int(question1.lower().startswith('are') or question2.lower().startswith('are'))
    starts_with_can = int(question1.lower().startswith('can') or question2.lower().startswith('can'))
    starts_with_how = int(question1.lower().startswith('how') or question2.lower().startswith('how'))

    feature_row = [percentage_common_tokens, question1_length,
                    question2_length, length_difference, num_capital_letters1, num_capital_letters2,
                    num_question_marks1, num_question_marks2, starts_with_are, starts_with_can, starts_with_how]

    return feature_row
def clean(question):    
    question = str(question).lower()
    question= question.replace(",000,000", "m")
    question= question.replace(",000", "k")
    question= question.replace("′", "'")
    question= question.replace("’", "'")
    question= question.replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")
    question= question.replace("n't", " not").replace("what's", "what is").replace("it's", "it is")
    question= question.replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")
    question= question.replace("he's", "he is").replace("she's", "she is").replace("'s", " own")
    question= question.replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")
    question= question.replace("€", " euro ").replace("'ll", " will")
    question= re.sub(r"([0-9]+)000000", r"\1m", question)
    question= re.sub(r"([0-9]+)000", r"\1k", question)
    question= re.sub(r"http\S+", "", question)
    question= re.sub('\W', ' ', question)
    
    return question
    

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    return cleaned_text

def tokenize_text(text):
   
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
    


def text_preprocess(text, flag='lem', remove_stopwords=True):
    if text is None:
        return ''
    
    if isinstance(text, list):
        text = ' '.join(text)
    
    stop_wordss = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Remove stopwords if specified
    if remove_stopwords:
        tokens = [token for token in text.split() if token.lower() not in stop_wordss]
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
  return fuzz.partial_ratio(text1, text2)

def token_set_ratio(text1, text2):
  return fuzz.token_set_ratio(text1, text2)

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


def levenshtein_distance(text1, text2):
    return fuzz.ratio(text1, text2)


def length_ratio(text1, text2):
    length2 = len(text2)
    if length2 == 0:
        return 0
    return (len(text1) / length2)


# Common n-grams
def common_ngrams(text1, text2, n):
    ngrams1 = set(ngrams(text1, n))
    ngrams2 = set(ngrams(text2, n))
    if len(ngrams2) == 0:
        return 0
    return len(ngrams1.intersection(ngrams2)) / len(ngrams2)

def average_word_frequency(text):
    words = text.split()
    freq_dist = FreqDist(words)
    total_frequency = sum(freq_dist.values())
    return total_frequency / (len(words) +0.001)

def average_word_frequency_diff(text1, text2):
    freq1 = average_word_frequency(text1)
    freq2 = average_word_frequency(text2)
    return freq1 - freq2

def embedding(tokens):
    embeddings = [word2vec.wv[word] for word in tokens if word in word2vec.wv]
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(200)
    
def cos_similarity(embedding1,embedding2):
    similarity_score = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity_score


def final_features(question1,question2):
    #download_all() #If you are running for the first time in a new venv kindly uncomment this line
    features1 = first_features(question1,question2)
    
    clean_text1 = clean(question1)
    clean_text2 = clean(question1)
    
    clean_text1 = preprocess_text(clean_text1)
    clean_text2 = preprocess_text(clean_text2)
    
    clean_text1 = text_preprocess(clean_text1)
    clean_text2 = text_preprocess(clean_text2)
    
    word_count1,word_count2 = word_count(clean_text1,clean_text2)
    sentence_count1,sentence_count2 = sentence_count(clean_text1,clean_text2)
    avg_word_length1,avg_word_length2 = avg_word_length(clean_text1,clean_text2)
    unique_word_count_ = unique_word_count(clean_text1,clean_text2)
    similar_word_count_ = similar_word_count(clean_text1,clean_text2)
    fuzzy_word_partial_ratio_ = fuzzy_word_partial_ratio(clean_text1,clean_text2)
    token_set_ratio_ = token_set_ratio(clean_text1,clean_text2)
    token_sort_ratio_ = token_sort_ratio(clean_text1,clean_text2)
    word_overlap_ = word_overlap(clean_text1,clean_text2)
    jaccard_similarity_ = jaccard_similarity(clean_text1,clean_text2)
    levenshtein_distance_ = levenshtein_distance(clean_text1,clean_text2)
    length_ratio_ = length_ratio(clean_text1,clean_text2)
    common_2grams = common_ngrams(clean_text1,clean_text2, 2)
    common_3grams = common_ngrams(clean_text1,clean_text2, 3)
    average_word_frequency1 = average_word_frequency(clean_text1)
    average_word_frequency2 = average_word_frequency(clean_text2)
    average_word_frequency_diff = abs(average_word_frequency1 - average_word_frequency2)
    
    tokens1 = word_tokenize(clean_text1)
    tokens2 = word_tokenize(clean_text2)
    embedding1 = embedding(tokens1)
    embedding2 = embedding(tokens2)
    
    cos_similarity_ = cos_similarity(embedding1,embedding2)
    
    features1.extend([word_count1,word_count2,sentence_count1,sentence_count2,
                                     avg_word_length1,avg_word_length2,unique_word_count_,
                                     similar_word_count_,fuzzy_word_partial_ratio_,token_set_ratio_,
                                     token_sort_ratio_,word_overlap_,jaccard_similarity_,levenshtein_distance_,
                                     length_ratio_,common_2grams,common_3grams,average_word_frequency1,average_word_frequency2,
                                     average_word_frequency_diff,cos_similarity_])
    

    shape_orig = np.array(features1).shape
    scaled_features = scaler.transform(np.array(features1).reshape(1,-1))
    scaled_features = scaled_features.reshape(shape_orig).tolist()
    
    finalfeatures = []
    finalfeatures.extend(embedding1)
    finalfeatures.extend(embedding2)
    finalfeatures.extend(scaled_features)
    
    return finalfeatures
    
    
    
    
