from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics

import numpy as np

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
    while math.abs(theta) > (math.pi / 4):
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
def create_noisy_doc(doc):
    newdoc = []
    for word in doc:
        noise = generate_laplace_similarity(0.5) # arbitrary epsilon!!!
        if word in model.vocab:
            syn = find_closest_word(word, noise)
            newdoc.append(syn)
    return newdoc

# For every test document, create a noisy bag of words document and then
# run the classifier on the noisy document.
# Feature_names is a list of features to create a topic bow
def run_noisy_classifier(data_train, bow_test):
    new_data_test = []
    
    for test_doc in bow_test:
        noisy_bow = create_noisy_doc(test_doc)
        noisy_bow = " ".join(noisy_bow)
        noisy_bow = re.sub(r"_", r" ", noisy_bow)
        new_data_test.append(noisy_bow)
      
    # Train on the original data, test on the new data
    #vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
    vectorizer = CountVectorizer(max_df=0.5,stop_words='english')
    X_train = vectorizer.fit_transform(data_train)
    X_test = vectorizer.transform(new_data_test)
    return X_train, X_test, new_data_test

def load_training_data():
    indir = 'datasets/reuters/corpus-final/'
    datafiles, targets = load_data(indir)
    return load_xml(datafiles, targets)

def load_test_data():
    indir = 'datasets/reuters/corpus-test/'
    datafiles, targets = load_data(indir)
    return load_xml(datafiles, targets)
    
# Load files for processing.
data_train, data_authors, y_train, train_files = load_training_data()
data_test, data_test_authors, y_test, test_files = load_test_data()

NUM_FEATURES = 100

feature_names, X_train_std, X_test_std = run_standard_classifier(NUM_FEATURES, data_train, y_train, data_test) 

train = {'x' : X_train_std, 'y' : y_train}
test = {'x' : X_test_std, 'y': y_test}
std_score = classify(train, test)

# Build a classifier using the top n features for test only and predict a result
X_train_bow, X_test_bow, bow_test = run_bow_classifier(data_train, data_test, feature_names) 
train = {'x' : X_train_bow, 'y' : y_train}
test = {'x' : X_test_bow, 'y': y_test}
bow_score = classify(train, test)

X_train_noisy, X_test_noisy, noisy_bow_test = run_noisy_classifier(data_train, bow_test)
train = {'x' : X_train_noisy, 'y' : y_train}
test = {'x' : X_test_noisy, 'y': y_test}
noisy_score = classify(train, test) 
#save_json(noisy_bow_test, "datasets/reuters/corpus-noisy/noisy_test_" + str(NUM_FEATURES) + ".json")

#for i in range(10):
#    X_train_noisy, X_test_noisy, noisy_bow_test = run_noisy_classifier(data_train, bow_test)
#    train = {'x' : X_train_noisy, 'y' : y_train}
#    test = {'x' : X_test_noisy, 'y': y_test}
#    noisy_score = classify(train, test) # 0.599
#    if i == 0:
#        save_json(noisy_bow_test, "datasets/reuters/corpus-noisy/noisy_test_" + str(NUM_FEATURES) + ".json")

