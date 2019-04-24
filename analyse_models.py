from gensim.models import Word2Vec
import numpy as np
import logging

def analyseModel(model):
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
    word_vectors = model.wv
    number_of_samples = len(word_vectors.vocab) // 10
    length = str(len(word_vectors.vocab))
    randomNumbers = np.random.randint(low=1, high=length, size=number_of_samples) 
    totalSum = 0
    for n in randomNumbers:
        word = word_vectors.index2word[int(n)]
        values = [value[1] for value in word_vectors.most_similar(word)]
        totalSum += sum(values) / len(values)
    return(totalSum / number_of_samples)

def print_model_parameters(model):
    parameters = ""
    parameters += "\t" + str(model.corpus_total_words)
    parameters += "\t" + str(model.epochs)
    parameters += "\t" + str(model.layer1_size)
    parameters += "\t" + str(model.min_alpha)
    parameters += "\t" + str(model.min_count)
    parameters += "\t" + str(model.sample)
    parameters += "\t" + str(model.negative)
    parameters += "\t" + str(analyseModel(model))
    return parameters

import os
model_analysis = "filename" + "\t" + "vocabulary size" + "\t" + "epochs" + "\t" + "hidden layer size" + "\t" + "learning rate" + "\t" + "min count" + "\t" + "downsampling" + "\t" + "negative sampling" + "\t" + "model quality"  
rootdir = 'models/'
extensions = ('.model')
for dirpath, dirs, files in os.walk(rootdir):
    if ".git" in dirs:
        dirs.remove(".git")
    if ".ipynb_checkpoints" in dirs:
        dirs.remove(".ipynb_checkpoints")
    for file in files:
        ext = os.path.splitext(file)[-1].lower()
        if ext in extensions:
            model_analysis += "\n" + str(file) 
            model = Word2Vec.load(os.path.join(dirpath, file))
            model_analysis += print_model_parameters(model)
f= open("model-analysis.csv","w", encoding="utf-8")
f.write(model_analysis)
f.close()