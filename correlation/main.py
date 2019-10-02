import numpy as np

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

vocab = load_list_pickle("vocabulary")
idf_vector = load_pickle_file("idf_vector")
okapi_vec = load_pickle_file("okapi_vec")
palabra = "empresa"
etiqueta = "n"

idf_arr = []
for key, value in idf_vector.items():
    idf_arr.append(value)

res = np.multiply(idf_arr, okapi_vec[(palabra, etiqueta)])

res_dict = {}

for i, k in enumerate(idf_vector.keys()):
    res_dict[k] = res[i]

sorted_res = sorted(res_dict.items() , reverse=True, key=lambda x: x[1])
for s in sorted_res:
    print(s)