#!/usr/local/bin/python

# Obfuscation code
# Takes as input a directory containing files to obfuscate. Removes stopwords from files.
#

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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

# Remove non-function words from the document - these are just the stopwords
def obfuscate(doc):
    vectorizer = CountVectorizer(max_df=0.5,stop_words='english')
    tokeniser = vectorizer.build_analyzer()
    nf_doc = [w for w in tokeniser(doc)]
    return ' '.join(nf_doc)

def main(indir):
    files = get_files(indir)
    for f in files:
        t = load_text(f)
        new_t = obfuscate(t)
        save_text(f, new_t)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", action="store", help="path to files to overwrite")
    args = vars(parser.parse_args())

    input_dir = args["i"]

    if not input_dir:
        parser.print_help()
    else:
        main(input_dir)
