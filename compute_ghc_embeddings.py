# coding: utf-8

# ## Carrega Arquivos

# In[1]:

#!wget -c https://github.com/nathanshartmann/portuguese_word_embeddings/raw/master/preprocessing.py -O preprocessing.py

# ## Define funções de Pré-Processamento

# In[2]:


import pandas as pd
import gensim
import nltk
nltk.download('punkt')
import preprocessing
import unicodedata
nltk.download('stopwords')
from nltk.corpus import stopwords

def clean_row(row):
    return preprocessing.clean_text(row[3])

def remove_accents(row):
    nfkd_form = unicodedata.normalize('NFKD', row[3])
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')

def narratives2tokens(narratives, accents=False):

    txt = []
    final = []

    if accents:
        txt = narratives.apply(remove_accents, axis=1)
    else:
        txt = narratives.apply(clean_row, axis=1)

    for line in txt:
        for sent in sent_tokenizer.tokenize(line):
            if sent.count(' ') >= 3 and sent[-1] in ['.', '!', '?', ';']:
                if sent[0:2] == '- ':
                    sent = sent[2:]
                elif sent[0] == ' ' or sent[0] == '-':
                    sent = sent[1:]
            final.append(sent)

    reg_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    sentences_tokens = []
    tokens_count = 0
    for sentence in final:
        tokens = reg_tokenizer.tokenize(sentence)
        tokens_count += len(tokens)
        sentences_tokens.append(tokens) 
  
    return sentences_tokens, len(final), tokens_count


# ## Monta Vocabulário

# In[25]:


corpus = pd.read_csv('narratives.csv.gz', nrows=200000, skiprows=1, header=None)
sent_tokens_vocab, sent_count, tokens_count = narratives2tokens(corpus, True)
print('Building Vocab...', '\t Sentences: ', sent_count, '\t Tokens:', tokens_count) ## 232s


# In[26]:


model_w2v = gensim.models.Word2Vec(size=300, workers=12, iter=10, negative=5, min_count=5)
model_w2v.build_vocab(sent_tokens_vocab)

model_ft = gensim.models.FastText(size=300, workers=12, iter=10, min_count=5)
model_ft.build_vocab(sent_tokens_vocab)

del(sent_tokens_vocab)

print(len(model_w2v.wv.vocab), len(model_ft.wv.vocab))  ## 25s


# ## Treina modelo Word2Vec e FastText

# In[27]:


import time, sys, gc, os

sent_total = 0
tokens_total = 0
nrows = 10000 ## 50000

start = time.time()
end = time.time()
time_total = round(end - start,3)

for i in range(0,3):  # 0,76
    start = time.time()
    gc.collect()

    corpus = pd.read_csv('narratives.csv.gz', nrows=nrows, skiprows=1+(i*nrows), header=None)
    sent_tokens, sent_count, tokens_count = narratives2tokens(corpus, True)
    sent_total += sent_count
    tokens_total += tokens_count

    sys.stdout.write('Iteration: '+str(i)+'\t Sentences: '+str(sent_count)+'\t Tokens:'+str(tokens_count))

    model_w2v.train(sent_tokens, total_examples=int(sent_count), epochs=10)

    end = time.time()
    time_total = round(end - start,3)
    sys.stdout.write('\t TimeW2V: ' + str(time_total))
    sys.stdout.flush()

    model_ft.build_vocab(sent_tokens, update=True)
    model_ft.train(sent_tokens, total_examples=int(sent_count), epochs=10)

    end = time.time()
    time_total = round(end - start,3)
    sys.stdout.write('\t TimeFT: ' + str(time_total) + '\n')
    sys.stdout.flush()

    os.system('rm *.npy')
    os.system('rm *.model')

    model_w2v.save('health_word2vec_300_'+str(i)+'.model')

    model_ft.save('health_fasttext_300_'+str(i)+'.model')

print('TOTAL: \t\t Sentences: ', sent_total, '\t Tokens:', tokens_total, '\t Time:', time_total)

# In[28]:

os.system('rm *.npy')
os.system('rm *.model')

model_w2v.save('health_word2vec_300.model')
model_ft.save('health_fasttext_300.model')
