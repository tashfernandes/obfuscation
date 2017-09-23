from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import gensim

# Load Google's pre-trained Word2Vec model.
model = None

def load_model(homedir):
    global model
    model = gensim.models.KeyedVectors.load_word2vec_format(homedir+'word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    model.init_sims(replace=True)
    return model

def generate_bow_doc(doc, feature_names):
    vectorizer = CountVectorizer(max_df=0.5,stop_words='english')
    tokeniser = vectorizer.build_analyzer()
    bow = [w for w in tokeniser(doc) if w in feature_names and w in model.vocab]
    return bow

def find_closest_word(word, sim):
    global saved_lookups
    search_size = 20000
    closest_matches = []
    loops = 0
    
    closest_matches = model.most_similar(positive=[word], topn=search_size)
        
    while (True):
        # sim_scores contains the closest match similarity scores for each potential match
        sim_scores = np.array([ m[1] for m in closest_matches])
        # best_val contains the index of the closest score
        best_val = np.abs(sim_scores - sim).argmin()
        # If this index is NOT at the end, choose it
        loops += 1
        if loops > 100 or best_val < (len(closest_matches) - 1):
            return closest_matches[best_val][0]

        # the best index was at the end.. check the next array just in case it contains closer words
        closest_matches = model.most_similar(positive=[word], topn=search_size*2)
        # Don't use this for now...
        closest_matches = closest_matches[(search_size-1):]        
        search_size = search_size * 2 


