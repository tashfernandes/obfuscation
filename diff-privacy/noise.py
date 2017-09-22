from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics

import numpy as np
import argparse
import os
from multiprocessing import Pool

import json
import math

from util import *
from word2vec import *

from time import time

#
# This code loads reuters files, converts to bag of topic words and runs a classifier over them.
#

# Generate a noise angle in radians, making sure it is within the correct range.
# Return the cosine of the angle so we can use it with similarity measures in Word2Vec
# The closer the output is to 1, the more similar it is.
def generate_laplace_similarity(scale):
    theta = np.random.laplace(scale=scale)
    while math.fabs(theta) > (math.pi / 4):
        theta = np.random.laplace(scale=scale)
    return math.cos(theta + (math.pi/4))

def classify(train, test):
    # Build the classifier using the top n features for train and test and predict a result
    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(train['x'], train['y'])
    pred = classifier.predict(test['x'])
    score = metrics.accuracy_score(test['y'], pred)
    print("accuracy:   %0.3f" % score)
    return score

# Run a classifier over original train and test docs using limited features
def run_standard_classifier(num_features, train_x, train_y, test_x):
    # Build the classifier using the top n features for train and test and predict a result
    vectorizer = CountVectorizer(max_df=0.5,stop_words='english')
    X_train = vectorizer.fit_transform(train_x)
    X_test = vectorizer.transform(test_x)
    
    ch2 = SelectKBest(chi2, k=num_features)
    X_train_best = ch2.fit_transform(X_train, train_y)
    X_test_best = ch2.transform(X_test)
    
    f_names = vectorizer.get_feature_names()
    arr = ch2.get_support(indices=True)
    f_names = [ f_names[i] for i in arr ]
    f_names = np.asarray(f_names)
    return f_names, X_train_best, X_test_best

# Create classifier using original training data and bow test data using pre-learned features
def run_bow_classifier(data_train, data_test, feature_names):
    new_data_test = []
    for test_doc in data_test:
        bow = generate_bow_doc(test_doc, feature_names)
        new_data_test.append(" ".join(bow))
        
    vectorizer = CountVectorizer(max_df=0.5,stop_words='english')
    X_train = vectorizer.fit_transform(data_train)
    X_test = vectorizer.transform(new_data_test)
    return X_train, X_test, new_data_test

# Create a noisy bag of words document given a bow document
def create_noisy_doc(doc, scale):
    newdoc = []
    for word in doc:
        if word != 'XXXX' and word in model.vocab:
            noise = generate_laplace_similarity(scale) 
            syn = find_closest_word(word, noise)
            newdoc.append(syn)
    return newdoc

# For every test document, create a noisy bag of words document and then
# run the classifier on the noisy document.
# Feature_names is a list of features to create a topic bow
def run_noisy_classifier(data_train, bow_test, scale):
    new_data_test = []
    
    for test_doc in bow_test:
        noisy_bow = create_noisy_doc(test_doc.split(), scale)
        noisy_bow = " ".join(noisy_bow)
        noisy_bow = re.sub(r"_", r" ", noisy_bow)
        new_data_test.append(noisy_bow)
      
    # Train on the original data, test on the new data
    #vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
    vectorizer = CountVectorizer(max_df=0.5,stop_words='english')
    X_train = vectorizer.fit_transform(data_train)
    X_test = vectorizer.transform(new_data_test)
    return X_train, X_test, new_data_test

def load_training_data(homedir=""):
    indir = homedir+'datasets/reuters/corpus-final/'
    datafiles, targets = load_data(indir)
    return load_xml(datafiles, targets)

def load_test_data(homedir=""):
    indir = homedir+'datasets/reuters/corpus-test/'
    datafiles, targets = load_data(indir)
    return load_xml(datafiles, targets)

def load_bow_data(homedir=""):
    indir = homedir+"datasets/reuters/problem-bow/unknown/"
    feature_file = homedir+"datasets/reuters/problem-bow/features.txt"

    features = []
    with open(features_file, 'r') as f:
        s = f.read()
        features.extend(s.split(' '))
    data, filenames = load_text_data(indir)
    return data, filenames, features

def load_func_data(homedir=""):
    indir = homedir+"datasets/reuters/problem-func/unknown/"
    return load_text_data(indir)


def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
    return data

def obfuscate(data):
    infile, outdir, scale = data
    doc = load_file(infile)
    print(doc)
    #noisy_doc = create_noisy_doc(doc.split(), scale, model)
    #noisy_doc = " ".join(noisy_doc)
    #noisy_doc = re.sub(r"_", r" ", noisy_doc)
    #return noisy_doc


def main(input_dir, output_dir):

    files = os.listdir(input_dir)
    model = load_model()
    radius = 1.5
    epsilon = 10.0
    scale = float(radius/epsilon)

    params = [(os.path.join(input_dir, file), output_dir, scale) for file in files]
    pool = Pool()
    pool.map(obfuscate, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", action="store", help="input directory")
    parser.add_argument("-o", action="store", help="output directory")
    args = vars(parser.parse_args())

    input_dir = args["i"]
    output_dir = args["o"]

    if not input_dir or not output_dir:
        parser.print_help()
    else:
        main(input_dir, output_dir)
