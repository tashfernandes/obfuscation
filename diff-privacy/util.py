import re
import html
import numpy as np
import json
from os import environ, listdir, makedirs
from os.path import dirname, exists, expanduser, isdir, join, splitext, basename

def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

def fixname(name):
    final = name.lower()
    final = re.sub(r"[^a-zA-Z]", r'-', final)
    return final

def save_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f)

def load_json(file):
    data = None
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def load_data(indir):
    target_names = []
    filenames = []
    
    folders = [f for f in sorted(listdir(indir))
               if isdir(join(indir, f))]

    for label, folder in enumerate(folders):
        target_names.append(folder)
        folder_path = join(indir, folder)
        documents = [join(folder_path, d)
                     for d in sorted(listdir(folder_path))]
        filenames.extend(documents)

    filenames = np.array(filenames)  
    return filenames, target_names

def load_xml(datafiles, targets):
    data_train = []
    data_authors = []
    data_targets = []
    data_files = []
    
    target_dict = {}
    for i in range(len(targets)):
        target_dict[targets[i]] = i

    for xmlfile in datafiles:
        data = []
        with open(xmlfile, 'r') as f:
            data.append(f.read())
    
        s = " ".join(data) 
   
        author = re.search('<byline>(.*)</byline>', s)
        text = re.search('<text>(.*)</text>',s,re.DOTALL)
        result = html.unescape(text.group(1))
        # Figure out what class this is by the filename
        c = re.search('/(C[\d]+)/', xmlfile)
    
        data_train.append(striphtml(result))
        author_fullname = fixname(author.group(1))
        data_authors.append(author_fullname)
        data_targets.append(target_dict[c.group(1)])
        data_files.append(xmlfile)
        
    return data_train, data_authors, data_targets, data_files
