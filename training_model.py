import pandas as pd
import gensim
import multiprocessing
from preprocess_sparql_output import preprocess_input
from gensim.models import Word2Vec
cores = multiprocessing.cpu_count()
import logging
def read_metadata_record(row,column_name):
    return gensim.utils.simple_preprocess(str(row[column_name]).encode('utf-8'))

def training(documents,size, sample, mincount, negative, outputname):
    print("output file name will be: " + outputname)
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
    model = Word2Vec(min_count=mincount,
                     window=5,
                     size=size,
                     sample=sample,#6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=negative,#negative=20,
                     workers=cores-1)
    model.build_vocab(documents, progress_per=10000)
    model.train(sentences=documents, total_examples=len(documents), epochs=model.iter)
    model.save(outputname)

input = preprocess_input()
use_gensim_preprocessing = False
documents = []
for index, row in input.iterrows():
    if(use_gensim_preprocessing == False):
        documents.append(row['keywords'].split())
    else:
        documents.append(read_metadata_record(row,"keywords"))
trainingSettings = pd.read_csv('training-settings.csv','utf-8',',')
for i, row in trainingSettings.iterrows():
    size = row['hidden layer size']
    sample = row['downsampling']
    mincount = row['min count']
    negative = row['negative sampling']
    outputname = "models/" + row['filename']
    training(documents, size, sample, mincount, negative, outputname)