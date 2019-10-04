from prettytable import PrettyTable

def remove_html(str_text):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(str_text, 'lxml')
    return soup.get_text()

def str_to_sentences(str_text):
    import nltk
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return sent_tokenizer.tokenize(str_text)

def tokenize_sentence(sentence):
    from nltk.tokenize import word_tokenize
    return word_tokenize(sentence)

def remove_special_chars(text_list):
    import re
    new_tokens = []
    for tpl in text_list:
        clean_token = ""
        for c in tpl[0]:
            if re.match(r'[a-záéíóúñü]', c):
                clean_token += c
        if clean_token != '':
            new_tokens.append((clean_token, tpl[1][0].lower()))
    return new_tokens

def remove_stopwords(text_list):
    from nltk.corpus import stopwords
    stopwords_spa = set(stopwords.words('spanish'))
    return [tpl for tpl in text_list if tpl[0] not in stopwords_spa]

def load_pickle_file(filename):
    import pickle
    with open("./checkpoints/"+filename+'.pkl', "rb") as fp:
        obj = pickle.load(fp)
    return obj

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

def normalize_article(article):
    # Removing html tags
    clean_text = remove_html(article)
    # Sentence segmentation
    text_sents = str_to_sentences(clean_text)
    # Tokenize each sentence
    for i in range(len(text_sents)):
        text_sents[i] =  tokenize_sentence(text_sents[i])
    # POS tagging
    tagger = load_pickle_file("tagger")
    tagged_text = []
    for sentence in text_sents:
        tagged_text += tagger.tag(sentence)
    # Make all text lower case
    tagged_text = [(tpl[0].lower(), tpl[1]) for tpl in tagged_text]

    # Remove special characters from each token
    clean_tagged_text = remove_special_chars(tagged_text)

    # Remove stopwords
    clean_tagged_text = remove_stopwords(clean_tagged_text)

    result = lemmatize(clean_tagged_text)

    return result

def text_to_articles(raw):
    temp = raw.split("<h3>")
    articles = []
    for a in temp:
        if "</h3>" in a:
            a = "<h3>" + a
            articles.append(a) 
    articles = [normalize_article(a) for a in articles]
    return articles

def save_list(filename, lst):
    import pickle    
    with open("./checkpoints/"+filename+'.txt', "wb") as fp:
        pickle.dump(lst, fp)

def load_list_pickle(filename):
    import pickle
    with open("./checkpoints/"+filename+'.txt', "rb") as fp:
        l = pickle.load(fp)
    return l

# corpus_root = './texts'
# filename = 'e960401_mod.htm'
# f = open(corpus_root +"/"+ filename, encoding='utf-8')
# raw_text = f.read()
# articles = text_to_articles(raw_text)
# save_list("articles", articles)

articles = load_list_pickle("articles")
print("There are:",len(articles), "articles")
topics = ["Artículos", ('méxico', 'n'), ('gobierno', 'n'), ('internet', 'n'), ('política', 'n')]
t = PrettyTable(topics)
for i, a in enumerate(articles):
    total_count = 0
    row = []
    for j, to in enumerate(topics):
        if j == 0:
            continue
        count = a.count(to)
        row.append(float(count))
        total_count += count
    if (total_count != 0):
        row = [r / total_count for r in row]
    t.add_row([i + 1] + row)
print(t)