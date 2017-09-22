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

def load_text_data(indir):
    # filenames contains list of files in indir directory
    filenames = np.array(sorted(listdir(indir)))
    # target names contains a list of category names - which are first elements of the filename
    # eg. filename C11-34567.txt has category C11
    #target_names = list(set(map(lambda x: x.split('-')[0], filenames)))
    bow_data = []
    for bowfile in filenames:
        data = []
        fullfile = join(indir, bowfile)
        with open(fullfile, 'r') as f:
            data.append(f.read())

        s = " ".join(data)
        bow_data.append(s)

    return bow_data, filenames
    

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
        
    return data_train, data_authors, data_targets, data_files, targets


# Convert documents to bag of words and write out
def write_training_data(data, outdir, metadata):
    found_authors = []
    
    target_names = data['target_names']

    for idx in range(len(data['data'])):
        doc = data['data'][idx]
        author = data['authors'][idx]
        target = data['targets'][idx]
        filename = data['files'][idx]
        
        authordir = outdir + author + '/'
        if not exists(authordir):
            makedirs(authordir)
        if not author in found_authors:
            found_authors.append(author)
            metadata['candidate-authors'].append({"author-name" : author })
        
        # write out the file
        fname = basename(filename)
        fname = target_names[target] + '-' + fname
        fname = re.sub(r".xml", r'.txt', fname)
        fpath = authordir + fname
    
        with open(fpath, 'w') as f:
            f.write(doc)
            
    return metadata

def write_obf_data(data, outdir, metadata, grounddata):
    unkdir = outdir + "unknown/"
    if not exists(unkdir):
        makedirs(unkdir)
        
    for idx in range(len(data['data'])):
        doc = data['data'][idx]
        author = data['authors'][idx]
        filename = data['files'][idx]
        
        # write out the file
        fname = basename(filename)
        fname = re.sub(r".xml", r'.txt', fname)
        fpath = unkdir + fname
    
        # If this path exists then it might have been used for a different topic.
        # Skip it..
    
        #print("Checking " + fpath)
        #if not exists(fpath):
        metadata['unknown-texts'].append({"unknown-text" : fname })
        grounddata['ground-truth'].append({"unknown-text" : fname, "true-author" : author})
        
        with open(fpath, 'w') as f:
            f.write(doc)
        #else:
        #    print("Found " + fpath + " already, skipping.")
            
    return metadata, grounddata


def write_test_data(data, outdir, metadata, grounddata):
    unkdir = outdir + "unknown/"
    if not exists(unkdir):
        makedirs(unkdir)
        
    target_names = data['target_names']

    for idx in range(len(data['data'])):
        doc = data['data'][idx]
        author = data['authors'][idx]
        target = data['targets'][idx]
        filename = data['files'][idx]
        
        # write out the file
        fname = basename(filename)
        fname = target_names[target] + '-' + fname
        fname = re.sub(r".xml", r'.txt', fname)
        fpath = unkdir + fname
    
        # If this path exists then it might have been used for a different topic.
        # Skip it..
    
        #print("Checking " + fpath)
        #if not exists(fpath):
        metadata['unknown-texts'].append({"unknown-text" : fname })
        grounddata['ground-truth'].append({"unknown-text" : fname, "true-author" : author})
        
        with open(fpath, 'w') as f:
            f.write(doc)
        #else:
        #    print("Found " + fpath + " already, skipping.")
            
    return metadata, grounddata

