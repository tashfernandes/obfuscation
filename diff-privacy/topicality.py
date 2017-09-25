#!/usr/local/bin/python

#
# Run topic classifier on directory
#

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectFromModel
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

def classify(train, test):
    # Build the classifier using the top n features for train and test and predict a result
    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(train['x'], train['y'])
    pred = classifier.predict(test['x'])
    score = metrics.accuracy_score(test['y'], pred)
    print("accuracy:   %0.3f" % score)
    return score

def load_train_test(files):
    train_x = []
    train_files = []
    test_x = []
    test_files = []
    train_y = []
    test_y = []
    
    i = 0
    for f in files:
        if re.search(r'unknown', f):
            test_x.append(load_text(f)) 
            base = os.path.basename(f)
            c = re.search(r'^C\d*', base)
            test_y.append(c.group(0))
            test_files.append(base)
        else:
            train_x.append(load_text(f))
            base = os.path.basename(f)
            train_files.append(base)
            c = re.search(r'^C\d*', base)
            train_y.append(c.group(0))
        i += 1

    train_data = { 'x' : train_x, 'y' : train_y, 'files' : train_files }
    test_data = { 'x' : test_x, 'y': test_y, 'files' : test_files }

    return (train_data, test_data)


def main(indir):
    files, authors = load_data(indir)
    train_data, test_data = load_train_test(files)

    vectorizer = CountVectorizer(max_df=0.5,stop_words='english')
    X_train = vectorizer.fit_transform(train_data['x'])
    X_test = vectorizer.transform(test_data['x'])
    train = {'x' : X_train, 'y' : train_data['y'] }
    test = {'x' : X_test, 'y': test_data['y'] }
    score = classify(train, test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", action="store", help="path to input files")
    args = vars(parser.parse_args())

    input_dir = args["i"]

    if not input_dir:
        parser.print_help()
    else:
        main(input_dir)
