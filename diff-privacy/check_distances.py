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
    filenames = os.listdir(input_dir)
    reference_file = filenames[0]
    filenames = filenames[1:]

    model = gensim.models.KeyedVectors.load_word2vec_format('../../word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    model.init_sims(replace=True)

    distances = []
    data = []

    doc1 = load_file(os.path.join(input_dir, reference_file))
    data.append(doc1)
    doc1 = doc1.split()
    for f in filenames:
        doc2 = load_file(os.path.join(input_dir, f))
        data.append(doc2)
        doc2 = doc2.split()
        #dist = random.randint(1,100)
        dist = model.wv.wmdistance(doc1, doc2)
        print("Adding distance b/w " + reference_file + " and " + f + " of " + str(dist))
        distances.append( { 'd1' : reference_file, 'd2' : f, 'distance' : dist }) 

    # 2. Calculate the ruzicka distance for these files
    n=4
    featureLength = 20000
    v = CountVectorizer(analyzer=char_wb_ngrams, ngram_range=(n,n), max_features=featureLength)
    ngrams = v.fit_transform(data)
   
    row1 = 0
    for i in range(len(filenames)):
        row2 = i+1
        minsum = ngrams[[row1, row2]].min(0).sum()
        maxsum = ngrams[[row1, row2]].max(0).sum()
        ruzicka = 1.0 - ( float(minsum) / maxsum)
        distances[i]['ruzicka'] = ruzicka

    outfile = os.path.join(output_dir, 'distances.json')   
    with open(outfile, 'w') as f:
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
