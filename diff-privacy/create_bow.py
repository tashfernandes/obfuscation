#!/usr/local/bin/python

#
# Takes as input a directory in koppel format and creates bow docs from all docs found
#

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics

from util import *
import re
import numpy as np
import argparse
import os

def load_text(f):
    with open(f, 'r', encoding='utf-8') as dfile:
        s = dfile.read()
    return s

def save_text(f,t):
    with open(f, 'w', encoding='utf-8') as dfile:
        dfile.write(t)

def get_files(d):
    files = []
    for f in os.listdir(d):
        if f.endswith(".txt"):
            files.append(os.path.join(d, f))
    return files

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

def generate_bow_doc(doc, feature_names):
    vectorizer = CountVectorizer(max_df=0.5,stop_words='english')
    tokeniser = vectorizer.build_analyzer()
    bow = [w for w in tokeniser(doc) if w in feature_names]
    return bow

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

# Remove non-function words from the document - these are just the stopwords
def obfuscate(doc):
    vectorizer = CountVectorizer(max_df=0.5,stop_words='english')
    tokeniser = vectorizer.build_analyzer()
    nf_doc = [w for w in tokeniser(doc)]
    return ' '.join(nf_doc)

def main(indir, outdir, num_features):
    files, authors = load_data(indir)
    train_authors = []
    train_x = []
    train_files = []
    test_x = []
    test_files = []
    train_y = []
    
    i = 0
    for f in files:
        if re.search(r'unknown', f):
            test_x.append(load_text(f)) 
            base = os.path.basename(f)
            test_files.append(base)
        else:
            train_x.append(load_text(f))
            base = os.path.basename(f)
            train_authors.append(authors[i])
            train_files.append(base)
            c = re.search(r'^\d', base)
            train_y.append(c.group(0))
        i += 1

    feature_names, new_xtrain, new_xtest = run_standard_classifier(num_features, train_x, train_y, test_x)
    new_data = []
    for idx in range(len(train_x)):
        doc = train_x[idx]
        bow = generate_bow_doc(doc, feature_names)
        bow = " ".join(bow)
        filepath = os.path.join(outdir, train_authors[idx])
        if not os.path.isdir(filepath):
            os.makedirs(filepath)
        filepath = os.path.join(filepath, train_files[idx])
        print("Saving to " + filepath)
        save_text(filepath, bow)       

    for idx in range(len(test_x)):
        doc = test_x[idx]
        bow = generate_bow_doc(doc, feature_names)
        bow = " ".join(bow)
        filepath = os.path.join(outdir, 'unknown')
        if not os.path.isdir(filepath):
            os.makedirs(filepath)
        filepath = os.path.join(filepath, test_files[idx])
        print("Saving to " + filepath)
        save_text(filepath, bow)       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", action="store", help="path to input files")
    parser.add_argument("-o", action="store", help="path to output files")
    parser.add_argument("-n", action="store", help="number of features")
    args = vars(parser.parse_args())

    input_dir = args["i"]
    output_dir = args["o"]
    num_features = args["n"]

    if not input_dir or not output_dir or not num_features:
        parser.print_help()
    else:
        main(input_dir, output_dir, int(num_features))
