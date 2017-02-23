#!/usr/local/bin/python

# Obfuscation code
# Takes as input a directory containing files to obfuscate. Produces list of words in file in random order.
#

import argparse
import os
import codecs
from nltk.tokenize import word_tokenize

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
    '''Randomises the text and outputs bag of words'''
    words = word_tokenize(t)
    bag = set(words)
    return ' '.join(bag)
    

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
