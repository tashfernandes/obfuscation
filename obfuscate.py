#!/usr/local/bin/python

# Obfuscation code
# Takes as input a directory containing files to obfuscate. Removes stopwords from files.
#

import argparse
import os
import codecs
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words('english'))

def load_text(f):
    dfile = codecs.open(f, "r", "utf-8")
    s = dfile.read()
    dfile.close()
    return s

def save_text(f,t):
    dfile = codecs.open(f, "w", "utf-8")
    dfile.write(t)
    dfile.close()

def get_files(d):
    files = []
    for f in os.listdir(d):
        if f.endswith(".txt"):
            files.append(os.path.join(d, f))
    return files

def obfuscate(t):
    '''Removes stopwords from the text in t'''
    global STOP_WORDS
    words = word_tokenize(t)
    filtered = [w for w in words if not w.lower() in STOP_WORDS]
    return ' '.join(filtered)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", action="store", help="path to directory of files to obfuscate")
    args = vars(parser.parse_args())

    indir = args["i"]
    if indir == None:
        parser.print_help()
        return

    files = get_files(indir)
    for f in files:
        t = load_text(f)
        new_t = obfuscate(t)
        save_text(f, new_t)

main()
