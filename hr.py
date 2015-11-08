import csv
import re
import string
import requests
import numpy as np
import scipy.sparse
from bisect import bisect_left, bisect_right
from collections import Counter
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from lxml import html

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

# remove url's, hashtags, and @
def cleaner(corpus):
    for i in range(len(corpus)):
        corpus[i] = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', corpus[i])
        corpus[i] = corpus[i].replace('@', '')
        corpus[i] = corpus[i].replace('#', '')
        corpus[i] = corpus[i].replace('RT', '')
        corpus[i] = corpus[i].replace('amp', '')
    return corpus

def searchSorted(data, x):
    i = bisect_left(data, x)
    j = bisect_right(data, x)
    idx = arange(i,j)
    return idx

def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]

with open('/home/b/Documents/humanRights/human_rights_training_sample_8-18-15.csv') as csvfile: 
    reader = csv.reader(csvfile)
    next(reader, None) # skip headers
    hr = [row[1] for row in reader]

hr = cleaner(hr)
    
with open('/home/b/Documents/humanRights/non_hr_training_sample_8-21-15.csv') as csvfile:
    reader = csv.reader(csvfile)
    non_hr = [row[11] for row in reader]

non_hr = cleaner(non_hr)

all_tweets = hr + non_hr


c_vectorizer = CountVectorizer(tokenizer = tokenize, stop_words = 'english', strip_accents = 'unicode')

counts = c_vectorizer.fit_transform(hr)
vocab = c_vectorizer.vocabulary_
vocab_counts = zip(c_vectorizer.get_feature_names(),
                   np.asarray(counts.sum(axis=0)).ravel())
vocab_counts = list(vocab_counts)
vocab_counts.sort(key=lambda tup:tup[1], reverse=True)

# vocab_counts = sorted(vocab_counts, key = lambda tup: tup[1], reverse = True)
vocabulary = [word[0] for word in vocab_counts[0:200]]
# vocabulary.append('human right')
# vocabulary.append('human rights')

# c_vectorizer = CountVectorizer(strip_accents = 'unicode',
#                                stop_words = 'english')
labeled = c_vectorizer.fit_transform(all_tweets)


vocab_l = c_vectorizer.vocabulary_

# convert csr into stuff with row, column indices
testdata = scipy.sparse.coo_matrix(labeled)

N = labeled.shape[0]
avg_dl = mean(list(Counter(coo.row).values()))


# use UN human rights declaration for vocab
page = requests.get('http://www.amnestyusa.org/research/human-rights-basics/universal-declaration-of-human-rights')
tree = html.fromstring(page.content)

page_intro = ' '.join(tree.xpath('/html/body/div[2]/div[2]/div[2]/div[3]/div/div/div/div/div[2]/p/text()'))
page_list = ' '.join(tree.xpath('/html/body/div[2]/div[2]/div[2]/div[3]/div/div/div/div/div[2]/ol/li/text()'))
page_text = page_intro + page_list
page_text = re.sub('\s+', ' ', page_text)

words = set(tokenize(page_text))
dec_vocab = [word.lower() for word in words if word not in stopwords.words('english')]


def IDF(query_word, N, coo, vocab):
    # get the index that sklearn created for each word
    q_idx = vocab[query_word]
    # count the number of documents with the query word
    n_q = sum(coo.col == q_idx)
    idf = log((N - n_q + .5)/(n_q + 0.5))
    return idf
    
def score(query_list, document, coo, col_index, vocab, idf, avg_dl, D, k=1.2, b=.75):
    Score = 0
    # find the document of interest
    rows = searchSorted(coo.row, document)
    for query in query_list:
        q_idx = vocab[query]
        # columns = np.where(coo.col == q_idx)
        columns = col_index[q_idx]
        f_qi_idx = np.intersect1d(columns[0], rows, assume_unique = True)
        if len(f_qi_idx) != 0:
            f_qi = coo.data[int(f_qi_idx)]
            Score  += idf[query] * f_qi * (k+1) / (f_qi + k * (1-b+b*D/avg_dl))
    return Score

def BM25(query_list, docs, k=1.2, b=.75):
    vectorizer = CountVectorizer(tokenizer = tokenize,
                                 strip_accents = 'unicode',
                                 stop_words = 'english',
                                 vocabulary = query_list)
    vectorized = vectorizer.fit_transform(docs)
    vocab = vectorizer.vocabulary_
    # convert csr into stuff with row, column indices
    data = scipy.sparse.coo_matrix(vectorized)
    N = vectorized.shape[0]
    avg_dl = mean(list(Counter(data.row).values()))
    # length of document
    D = [len(doc.split()) for doc in docs]
    scores = np.zeros(N)
    idf = dict()
    for query in query_list:
        idf[query] = IDF(query, N, data, vocab)
    all_columns = get_indices_sparse(data.col)
    for i in range(N):
        if i % 1000 == 0:
            print('Finished with document', i)
        scores[i] = score(query_list, i, data, all_columns, vocab, idf, avg_dl, D[i], k, b)
    return scores

with open('second_chunk_200k-1mil.csv') as csvfile:
    reader = csv.reader(csvfile)
    new = [row[11] for row in reader]

new = cleaner(new)

a = BM25(['human', 'rights'], new, vocab)

vectorizer = CountVectorizer(tokenizer = tokenize,
                             strip_accents = 'unicode',
                             stop_words = 'english')
new_v = vectorizer.fit_transform(new[:10000])
vocab_n = vectorizer.vocabulary_

query_list = ['human', 'judge']
idf = dict()
for query in query_list:
    idf[query] = IDF(query, N, coo, vocab)


start = time.clock()
np.intersect1d(columns[0], rows[0])
end = time.clock()
print(end-start)


start = time.clock()
np.setdiff1d(columns[0], rows[0])
end = time.clock()
print(end-start)

result = collections.defaultdict(list)

condlist = [coo.row==0, coo.col==325]
choiselist = [coo.row, coo.col]

query_list = ['judg', 'law', 'human', 'fed']
for query in query_list:
    print(IDF(query, 1000, testdata, vocab))



def find_indices(data, q_idx):
    return [elem for i, elem in enumerate(data.col) if i==q_idx]
