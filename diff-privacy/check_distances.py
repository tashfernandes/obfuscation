from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import argparse
import re
import random
import json
import gensim
import os
from util import *

# This program calculates the wmd distance for a random sample of files, and also
# the corresponding ruzicka distance. The wmd distance is calculated with respect to
# every other file in the training set. Using these distances we can plot wmd
# distance against ruzicka distance to see if there is a correlation.

def char_wb_ngrams(text_document):
    # normalize white spaces
    _white_spaces = re.compile(r"\s\s+")
    text_document = _white_spaces.sub(" ", text_document)

    min_n, max_n = (4, 4)
    ngrams = []

    # bind method outside of loop to reduce overhead
    ngrams_append = ngrams.append

    for w in text_document.split():
        #w = ' ' + w + ' '
        w_len = len(w)
        for n in range(min_n, max_n + 1):
            offset = 0
            ngrams_append(w[offset:offset + n])
            while offset + n < w_len:
                offset += 1
                ngrams_append(w[offset:offset + n])
            if offset == 0:   # count a short word (w_len < n) only once
                break
    return ngrams

def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def main(input_dir, output_dir, out_file):
    # 1. Need to generate wmd. Load files, pick first one as references, and
    # calculate distances of other files from this one.

    # Let's not use a reference, just calculate pairwise distances
    data = []
    filenames = os.listdir(input_dir)
    for f in filenames:
        d = load_file(os.path.join(input_dir, f))
        data.append(d)

    model = gensim.models.KeyedVectors.load_word2vec_format('../../word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    model.init_sims(replace=True)

    distances = []

    for i in range(len(data)):
        doc1 = data[i].split()
        for j in range(i+1, len(data)):
            doc2 = data[j].split()
            dist = model.wv.wmdistance(doc1, doc2)
            distances.append( { 'd1' : filenames[i], 'd2' : filenames[j], 'distance' : dist }) 

    # 2. Calculate the ruzicka distance for these files
    n=4
    featureLength = 20000
    v = CountVectorizer(analyzer=char_wb_ngrams, ngram_range=(n,n), max_features=featureLength)
    ngrams = v.fit_transform(data)
   
    idx = 0
    for i in range(len(filenames)):
        row1 = i
        for j in range(i+1, len(filenames)):
            row2 = j
            minsum = ngrams[[row1, row2]].min(0).sum()
            maxsum = ngrams[[row1, row2]].max(0).sum()
            ruzicka = 1.0 - ( float(minsum) / maxsum)
            distances[idx]['ruzicka'] = ruzicka
            idx += 1

    outpath = os.path.join(output_dir, out_file)   
    with open(outpath, 'w') as f:
        json.dump(distances, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", action="store", help="input directory")
    parser.add_argument("-o", action="store", help="output directory")
    parser.add_argument("-f", action="store", help="output filename")
    args = vars(parser.parse_args())

    input_dir = args["i"]
    output_dir = args["o"]
    out_file = args["f"]

    if not input_dir or not output_dir or not out_file:
        parser.print_help()
    else:
        main(input_dir, output_dir, out_file)
