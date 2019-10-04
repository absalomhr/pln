def store_pickle_file(filename, obj):
    from pickle import dump
    output = open("checkpoints/"+filename+".pkl", "wb")
    dump(obj, output, -1)
    output.close()


def get_lemmas_dict (filename):
    f = open(filename, encoding='latin-1')
    lines = f.readlines()
    lines = [l.strip() for l in lines]

    lemmas = {}

    for l in lines:
        if l != '':
            words = l.split()
            words = [w.strip() for w in words]
            wordform = words[0] + ' ' + words[-2][0].lower()
            wordform = wordform.replace('#', '')
            lemmas[wordform] = words[-1]

    return lemmas


text_dir = "./texts"
filename = "generate.txt"
lemmas = get_lemmas_dict(text_dir+'/'+filename)
store_pickle_file("lemmas_dict", lemmas)
print("done")