# code based on https://radimrehurek.com/gensim/models/doc2vec.html
import gensim
import random
import pickle
from gensim.models.doc2vec import Doc2Vec
import collections
import multiprocessing
from preprocess_sparql_output import preprocess_input
cores = multiprocessing.cpu_count()
import logging
def read_metadata_record(row,column_name):
    return gensim.utils.simple_preprocess(str(row[column_name]).encode('utf-8'))
input = preprocess_input()
use_gensim_preprocessing = False
def read_corpus():
    for index, row in input.iterrows():
        yield gensim.models.doc2vec.TaggedDocument(row['keywords'].split(), [index])
documents = list(read_corpus())
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
#model = gensim.models.doc2vec.Doc2Vec()
#model.build_vocab(documents)
#print('epochs:' + str(model.epochs))
#model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
#model.save('doc2vec_model.model')
model = Doc2Vec.load('doc2vec_model.model')
ranks = []
second_ranks = []
for doc_id in range(len(documents)):
    print('processing  doc {} of {}'.format(doc_id,len(documents)))
    inferred_vector = model.infer_vector(documents[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    second_ranks.append(sims[1])
collections.Counter(ranks)
with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([documents, ranks, second_ranks],f)
print('Document ({}): «{}»\n'.format(doc_id, ' '.join(documents[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(documents[sims[index][0]].words)))
doc_id = random.randint(0, len(documents) - 1)
print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(documents[doc_id].words)))
sim_id = second_ranks[doc_id]
print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(documents[sim_id[0]].words)))
