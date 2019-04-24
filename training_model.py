import gensim
import multiprocessing
from gensim.models import Word2Vec
from preprocess_sparql_output import preprocess_input
input = preprocess_input()
cores = multiprocessing.cpu_count() # Count the number of cores in a computer
def read_metadata_record(row,column_name):
    return gensim.utils.simple_preprocess(str(row[column_name]).encode('utf-8'))

def training(size, sample, use_gensim_preprocessing, mincount, negative, outputname):
    documents = []
    for index, row in input.iterrows():
        if(use_gensim_preprocessing == False):
            documents.append(row['keywords'].split())
        else:
            documents.append(read_metadata_record(row,"keywords"))
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
    pass

use_gensim_preprocessing = False
documents = []
for index, row in input.iterrows():
    if(use_gensim_preprocessing == False):
        documents.append(row['keywords'].split())
    else:
        documents.append(read_metadata_record(row,"keywords"))
model = Word2Vec(workers=cores-1)
model.build_vocab(documents, progress_per=10000)
model.train(sentences=documents, total_examples=len(documents), epochs=model.iter)