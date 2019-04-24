import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
model = Word2Vec.load("models/defaults-without-gensim-preprocessing.model")
vocabulary = model.wv.vocab.keys()
#output = ""
#type(vocabulary)
#f= open("vocabulary-defaults.txt","w", encoding="utf-8")
#for x in vocabulary:
#    output += str (x) + "\n"
#f.write(output)
#f.close()
def tsnescatterplot(model, word):
    arrays = np.empty((0, 100), dtype='f')
    word_labels = [word]
    color_list  = ['red']
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    close_words = model.wv.most_similar([word])
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    reduc = PCA(n_components=9).fit_transform(arrays)
    np.set_printoptions(suppress=True)
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    for line in range(0, df.shape[0]):
         title = df["words"][line].title()
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + re.sub('_',' ',title),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(10)    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
    plt.title('t-SNE visualization for {}'.format(word.title()))
    plt.savefig("close-words.png", format="png")
    
tsnescatterplot(model, 'unemployment')