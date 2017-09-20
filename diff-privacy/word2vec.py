from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('../../word2vec/GoogleNews-vectors-negative300.bin', binary=True)
model.init_sims(replace=True)

# Finds the closest word in Word2Vec to the given word with the given cosine similarity.
saved_lookups = {}

def generate_bow_doc(doc, feature_names):
    vectorizer = CountVectorizer(max_df=0.5,stop_words='english')
    tokeniser = vectorizer.build_analyzer()
    bow = [w for w in tokeniser(doc) if w in feature_names and word in model.vocab]
    return bow

# This code finds the closest word (in terms of cosine similarity) at distance sim to the given word
def find_closest_word(word, sim):
    global saved_lookups
    search_size = 200
    closest_matches = []
    loops = 0
    
    if word in saved_lookups:
        closest_matches = saved_lookups[word]
        search_size = len(closest_matches)
    else:
        closest_matches = model.most_similar(positive=[word], topn=search_size)
        saved_lookups[word] = closest_matches
        
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
        saved_lookups[word] = closest_matches
        closest_matches = closest_matches[(search_size-1):]        
        search_size = search_size * 2 

