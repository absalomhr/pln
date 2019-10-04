def load_pickle_file(filename):
    import pickle
    with open("./checkpoints/"+filename+'.pkl', "rb") as fp:
        obj = pickle.load(fp)
    return obj

def load_list_pickle(filename):
    import pickle
    with open("./checkpoints/"+filename+'.txt', "rb") as fp:
        l = pickle.load(fp)
    return l

def save_list(lst, filename):
    import pickle    
    with open("./checkpoints/"+filename+'.txt', "wb") as fp:
        pickle.dump(lst, fp)

def lemmatize(tagged_text):
    lemmas = load_pickle_file("lemmas_dict")

    new_tokens = []

    for t in tagged_text:
        temp = t[0] + " " + t[1]
        if (temp in lemmas):
            new_tokens.append((lemmas[temp], t[1]))
        else:
            new_tokens.append(t)

    return new_tokens

tagged_text_list = load_list_pickle("tagged_text")
lemmatized_text = lemmatize(tagged_text_list)
save_list(lemmatized_text, "lemmatized_text")
print("done")