import nltk
from nltk.corpus import cess_esp

def store_pickle_file(filename, obj):
    from pickle import dump
    output = open("./checkpoints/"+filename+".pkl", "wb")
    dump(obj, output, -1)
    output.close()

# Creating the default tagger
default_tagger = nltk.DefaultTagger('S')

# Creating a REGEX tagger
patterns = [(r'.*o$', 'NMS'), (r'.*os$', 'NMP'), (r'.*a$', 'NFS'), (r'.*as$', 'NFP')]
regex_tagger = nltk.RegexpTagger(patterns, backoff=default_tagger)

# Creating, training an UnigramTagger on cess_esp sentences 
tagged_sents = cess_esp.tagged_sents()
unigram_tagger = nltk.UnigramTagger(tagged_sents, backoff=regex_tagger)

# Saving to disk the general tagger
store_pickle_file("tagger", unigram_tagger)
print("done")