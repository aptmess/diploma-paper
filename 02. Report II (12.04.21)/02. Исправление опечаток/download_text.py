import numpy as np
import pandas as pd
import operator
import os

# Corus - NLP datasets
import corus
from corus import load_lenta

#NLTK - Natural Language Tool Kit
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import words
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import bigrams
from nltk import ngrams

#Other
from collections import Counter
import re
import string
from tqdm import notebook

def download_file(file_name: str):
    os.system(f'wget -P ~/IHaskell/word_data/ {file_name}')

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


def get_grams_from_text(path='lenta-ru-news.csv.gz', 
                        n=[1, 2], 
                        amount_of_sentense=1000, 
                        verbose=True, 
                        show_how_much=1000, **kwargs):
    records = load_lenta(path)
    grams, count = {}, 1
    flatten = lambda l: [' '.join(item) for sublist in l for item in sublist]
    try:
        while True and count != amount_of_sentense:
            item = next(records).text
            if verbose:
                print(f'Sentence {count}') if count % show_how_much == 0 else 'pass'
            #for i in np.arange(1, n+1):
            for i in n:
                if i not in list(grams.keys()):
                    grams[i] = Counter()
                ngram = [list(ngrams(text_prepare(sentense, **kwargs).lower().split(), n=i)) for sentense in nltk.sent_tokenize(item)]
                grams[i] += Counter(flatten(ngram))
            count +=1
    except StopIteration:
        pass
    finally:
        del records
    return grams


def predict(corpus, sentence, n=3):
    sen = text_prepare(sentence)
    cor = corpus.copy()
    rev = sen.split()[::-1]
    s = sum(list(cor[2].values()))
    s1 = sum(list(cor[1].values()))
    d = {}
    for key, value in list(cor[1].items()):
        a = []
        for i in np.arange(1, n+1):
            v = cor[2][f'{rev[i-1]} {key}']
            a.append(np.log(v / s) if v!=0 else np.log(0.000001))
        d[key] = sum([np.log(value / s1)] + a)    
    return sentence + ' ' + max(d.items(), key=operator.itemgetter(1))[0]