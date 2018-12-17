# -*- coding: utf-8 -*-
'''
Module with preprocessing methods to prepare a list of documents 
to train LDA topic model.
'''

__author__ = "Abelino Jiménez"

import os
import glob
import re
import numpy as np
import nltk

from collections import Counter
from nltk.corpus import stopwords



'''
GLOBAL VARIABLES
'''
LANGUAGE = 'finnish'
SPANISH_EXTRA_STOPWORDS = ['está','va','si']
FINNISH_EXTRA_STOPWORDS = ['myös','voi','voidaan','avulla','käytetään',
'välillä','saa','vain','syntyy','toisiaan','eri','ero','sanotaan',
'oleva','tulee','kaikki','aina', 'niinku']

if LANGUAGE == 'spanish':
    STOPWORDS = stopwords.words(LANGUAGE) + SPANISH_EXTRA_STOPWORDS
    STEMMER = nltk.SnowballStemmer(LANGUAGE)
    from .spanish_expressions_sets import sets_to_be_replaced
elif LANGUAGE == 'finnish':
    STOPWORDS = stopwords.words(LANGUAGE) + FINNISH_EXTRA_STOPWORDS
    STEMMER = nltk.SnowballStemmer(LANGUAGE)
    from .finnish_expressions_sets import sets_to_be_replaced

'''
paths
'''
root_path = os.getcwd()
data_path = os.path.join(root_path,'..','data')
names_path = os.path.join(data_path,LANGUAGE+'_names.txt')

# Load commons names used in LANGUAGE
with open(names_path,'r') as f:
    NAMES = [x.rstrip() for x in reversed(f.readlines())]

def read_texts(folder):
    texts = []
    for root, subFolder, files in os.walk(folder):
        for item in files:
            if item.endswith(".txt") :
                fileNamePath = str(os.path.join(root,item))
                with open(fileNamePath,"rb") as f:
                    text = f.read()
                    try:
                        text1 = text.decode()
                    except UnicodeDecodeError:
                        text1 = text.decode('ISO 8859-1')
                texts.append(text1)
    return texts

# replace words with tags according to the sets imported
def replace_words(x):
    for key,value in sets_to_be_replaced.items():
        text = ' '+key+' '
        for y in value:
            my_regex = r'\s'+re.escape(y)+r'[\s.,;\-!:]'
            x = re.sub(my_regex,text,x)
    return x

def brute_stemming(word):
    if word == word.lower():
        return word[:7]
    else:
        return word

def stemming(word):
    if len(word) > 4 and word == word.lower():
        return STEMMER.stem(word)
    return word


def detect_names(x):
    text = ' A_NAME '
    for y in NAMES:
        my_regex = r'\s'+re.escape(y)+r'[\s.,;\-!:]+'
        x = re.sub(my_regex,text,x)
    return x

def replace_symbols(x):
    x = x.replace(',',' ')
    x = x.replace('\x97',' ')
    x = x.replace('*',' ')
    x = x.replace(':',' ')
    x = x.replace(';',' ')
    x = x.replace('.',' ')
    x = x.replace('-',' ')
    x = x.replace('???',' QUESTION_SYMBOL ')
    x = x.replace('??',' QUESTION_SYMBOL ')
    x = x.replace('?',' QUESTION_SYMBOL ')
    x = x.replace(')',' ')
    x = x.replace('(',' ')
    x = x.replace('[',' ')
    x = x.replace(']',' ')
    x = x.replace('==',' ')
    x = x.replace('{',' ')
    x = x.replace('}',' ')
    x = x.replace('|',' ')
    x = x.replace('=',' EQUAL_SYMBOL')

    return x
'''
 Preprocessing routine
'''
def preprocessing(x,remove_stop_words=False,with_stemming=False):
    # x: string. A sentence.
    # output: list of preprocessed words in order.

    # replace all names with a tag 
    x = detect_names(x)
    # lowercase
    x = x.lower()
    # recover tag
    x = re.sub('a_name','A_NAME',x)
    # replace special words with tags
    x = " "+x # add space just in case
    x = replace_words(x)
    # replace symbol /
    x = re.sub(' / ',' ',x)
    # replace different numbers with the tag NUMBER
    x = re.sub('[-+]?\d*,?\d+',' NUMBER ',x)
    # replace chars with )
    x = re.sub('(\W/($|\s))',' ',x)
    # replace ...
    x = re.sub('…',' ',x)
    # remove symbols
    x = replace_symbols(x)
    # replace special words with tags again
    x = replace_words(x)
    # remove stop words
    word_list = x.split()
    if remove_stop_words:
        word_list = [x for x in word_list if x not in STOPWORDS]
    # stemming
    if with_stemming:
        word_list = [stemming(x) for x in word_list]
    return word_list

def clean_last_lines(lines):
    if len(lines)>2 and lines[-2].count('_')>3:
        return lines[:-2]
    lines[-1] = lines[-1].split('_')[0][:-1]
    return lines

def split_uppercase_words(line):
    words = line.split()
    final_words = []
    for word in words:
        new_word = ""
        last_word = ""
        for char in word:
            if char.isupper():
                if new_word!="" and last_word.islower():
                    final_words.append(new_word)
                    new_word = ""
            if char.islower() and last_word!="" and last_word.isupper() and len(new_word)>1:
                final_words.append(new_word)
                new_word = ""
            new_word+= char
            last_word = char
        final_words.append(new_word)
    return " ".join(final_words)

'''
If the text is a textbook, then removes the name of the textbook
from the last line.

'''
def finnish_preprocessing(text,is_textbook=False):
    lines = text.splitlines()
    final_lines = []
    if len(lines)>0:
        if is_textbook:
            lines = clean_last_lines(lines)
        connecting_line = False
        last_line = ""
        for line in lines:
            if not is_textbook:
                line  = re.sub(r'\[[ |\w|?|:|,|.]*\]', '', line)
                line_cleaned = line                 
            else:
                line_cleaned = split_uppercase_words(line)
            if line_cleaned.endswith('-'):
                connecting_line = True
                last_line += line_cleaned[:-1] # remove dash
                continue
            elif connecting_line == True:
                line_cleaned = last_line + line_cleaned
                connecting_line = False
                last_line = ""
            final_lines.append(line_cleaned)
    return "\n".join(final_lines)

def preprocessing_example(x,remove_stop_words=False,with_stemming=False):
    # x: string. A sentence.

    print('Original text:')
    print('\t'+x)
    # replace all names with a tag 
    x = detect_names(x)
    print('Nombres:')
    print('\t'+x)
    # lowercase
    x = x.lower()
    print('lowercase:')
    print('\t'+x)
    x = replace_words(x)
    print("Replace words")
    print('\t'+x)
    # replace different numbers with the tag NUMBER
    x = re.sub('[-+]?\d*,?\d+',' NUMBER ',x)
    print('Numbers to NUMBER:')
    print('\t'+x)
    x = x.replace(',',' ')
    x = x.replace('*',' ')
    x = x.replace(':',' ')
    x = x.replace(';',' ')
    x = x.replace('.',' ')
    x = x.replace('-',' ')
    print('Remove Symbols:')
    print('\t'+x)
    # remove stop words
    wordList = x.split()
    if remove_stop_words:
        wordList = [x for x in wordList if x not in STOPWORDS]
        print("Remove Stop Words:")
        print(wordList)
    # stemming
    if with_stemming:
        wordList = [stemming(x) for x in wordList]
        print("Stemming:")
        print(wordList)