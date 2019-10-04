def get_noun_freq(text, vocab):
    noun_freq = []
    for w in vocab:
        if w[1] == "n":
            noun_freq.append((w, text.count(w)))
    noun_freq.sort(key = lambda x: x[1], reverse=True)
    return noun_freq

def get_tfidf_noun(text, vocab, noun_freq):
    import numpy as np
    tfidf_noun = []
    n = len(vocab)
    for noun in noun_freq:
        tf = np.log(1 + noun[1])
        idf = np.log((n + 1) / noun[1])
        tfidf_noun.append((noun[0], tf * idf))
    tfidf_noun.sort(key = lambda x: x[1], reverse=True)
    return tfidf_noun

def load_list_pickle(filename):
    import pickle
    with open("./checkpoints/"+filename+'.txt', "rb") as fp:
        l = pickle.load(fp)
    return l

text = load_list_pickle("lemmatized_text")
vocab = list(set(text))
vocab.sort()
noun_freq = get_noun_freq(text, vocab)
tfidf_noun = get_tfidf_noun(text, vocab, noun_freq)
print(">>>Frequency noun list (25 most common):")
for n in noun_freq[:25]:
    print(n)
print("\n\n>>>tf-idf noun list (first 25):")
for n in tfidf_noun[:25]:
    print(n)