from nltk.tokenize import sent_tokenize, word_tokenize
from keras.utils import to_categorical as to_ct
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.utils import np_utils
from collections import Counter
from corus import load_lenta
from tqdm import notebook
from nltk import bigrams
from nltk import ngrams
import re
import nltk
import corus
import string
import operator
import numpy as np
import pandas as pd
import tensorflow as tf
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')



def tokenize(corpus):
    """Tokenize the corpus text.
    :param corpus: list containing a string of text (example: ["I like playing football with my friends"])
    :return corpus_tokenized: indexed list of words in the corpus, in the same order as the original corpus (the example above would return [[1, 2, 3, 4]])
    :return V: size of vocabulary
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    corpus_tokenized = tokenizer.texts_to_sequences(corpus)
    V = len(tokenizer.word_index)
    return corpus_tokenized, V

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def corpus2io(corpus_tokenized, V, window_size):
    """Converts corpus text into context and center words
    # Arguments
        corpus_tokenized: corpus text
        window_size: size of context window
    # Returns
        context and center words (arrays)
    """
    for words in corpus_tokenized:
        L = len(words)
        for index, word in enumerate(words):
            contexts = []
            labels = []
            s = index - window_size
            e = index + window_size + 1
            contexts.append([words[i]-1 for i in range(s, e) if 0 <= i < L and i != index])
            labels.append(word-1)
            x = np_utils.to_categorical(contexts, V)
            y = np_utils.to_categorical(labels, V)
            yield (x, y.ravel())

            
def softmax(x):
    """Calculate softmax based probability for given input vector
    # Arguments
        x: numpy array/list
    # Returns
        softmax of input array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def cbow(context, label, W1, W2, loss, lr):
    """
    Implementation of Continuous-Bag-of-Words Word2Vec model
    :param context: all the context words (these represent the inputs)
    :param label: the center word (this represents the label)
    :param W1: weights from the input to the hidden layer
    :param W2: weights from the hidden to the output layer
    :param loss: float that represents the current value of the loss function
    :return: updated weights and loss
    """
    x = np.mean(context, axis=(0, 1))
    h = np.dot(W1.T, x)
    u = np.dot(W2.T, h)
    y_pred = softmax(u)

    e = -label + y_pred
    dW2 = np.outer(h, e)
    dW1 = np.outer(x, np.dot(W2, e))

    new_W1 = W1 - lr * dW1
    new_W2 = W2 - lr * dW2

    loss += -float(u[label == 1]) + np.log(np.sum(np.exp(u)))

    return new_W1, new_W2, loss


def text_prepare(text, language='russian', delete_stop_words=False):
    """
        text: a string
        
        return: modified string
    """
    lemmatizer = WordNetLemmatizer()

    # 1. Перевести символы в нижний регистр
    text = text.lower() #your code
    
    # 2.1 Заменить символы пунктуации на пробелы
    text = re.sub(r'[{}]'.format(string.punctuation), ' ', text)
    
    
    
    # 2.2 Удалить "плохие" символы
    text = re.sub('[^A-Za-z0-9]' if language == 'english' else '[^А-яа-я]', ' ', text)

    
    # 3. Применить WordNetLemmatizer
    word_list = nltk.word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    
    # 4. Удалить стопслова.
    if delete_stop_words:
        stopWords = set(stopwords.words(language))
        for stopWord in stopWords:
            text = re.sub(r'\b{}\b'.format(stopWord), '', text)
        
    # 5. Удаляю пробелы у получая просто строку слов через пробел
    text = ' '.join(text.split())
    
    return text


def get_text(path='../../../data/lenta-ru-news.csv.gz', 
                        amount_of_sentense=1000, 
                        verbose=True, 
                        show_how_much=1000, **kwargs):
    records = load_lenta(path)
    a = []
    count = 1
    try:
        while True and count != amount_of_sentense:
            item = next(records).text
            if verbose:
                print(f'Sentence {count}') if count % show_how_much == 0 else 'pass'
            a.append(text_prepare(item))
            count +=1
    except StopIteration:
        pass
    finally:
        del records
    return a

class Word2Vec():
    
    def __init__(self, d=50, h=5):
        self.d = d
        self.h = h
        
    def fit(self, coprusI, num_epochs=1, lr=0.1):
        np.random.seed(100)
        print('Start counting dictionary')
        self.corpus_tokenized, self.V = tokenize(coprusI)
        print(f'Vocabulary size {self.V}')
        my = zip(self.corpus_tokenized, coprusI)
        print('All Words')
        self.vocabulary = {}
        self.r = {}
        for i, j in my:
            u = j.split()
            for m, n in zip(i, u):
                self.vocabulary[n] = m-1
                self.r[m-1] = n
        print('Start fitting')
        E = np.random.rand(self.V, self.d)
        C = np.random.rand(self.d, self.V)
        print(f'Emb: {E.shape}, Context: {C.shape}, {len(list(self.vocabulary.keys()))}')
        loss = 0.
        for num in range(num_epochs):
            print(f'epoch {num}')
            for i, (context, label) in enumerate(corpus2io(self.corpus_tokenized, self.V, self.h)):
                E, C, loss = cbow(context, label, E, C, loss, lr)
                if i % 1000 ==0 :
                    print(f"\n\t loss = {loss}\n")
                    print(f'Word {i}')
        self.embedding = E
        self.context = C
        
    def predict(self, x):
        prob = softmax(np.dot(self.context.T, np.dot(self.embedding.T, to_ct(self.vocabulary[x], num_classes=self.V))))
        return self.r[np.argmax(prob)]
    
    def emb(self, word):
        if word in self.vocabulary:
            
            return self.embedding[self.vocabulary[word]-1]
        else:
            return f'No {word}'
