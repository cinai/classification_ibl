# -*- coding: utf-8 -*-
'''
Module with preprocessing methods to prepare a data from utterances
to an excel table with features for classification and labels
'''

__author__ = "Catalina Espinoza"

'''import os
import glob
import re
import numpy as np
import nltk

from collections import Counter
from nltk.corpus import stopwords'''
import pandas as pd
import numpy as np

#



def get_topic_distribution_utterance(utterance,ldamodel,dictionary):
    bow = dictionary.doc2bow(utterance)
    T = ldamodel.get_document_topics(bow,minimum_probability=0,minimum_phi_value=0.001)
    return [x[1] for x in T]

def get_data_next_utterances(dict_utterances,groups_to_filter,lda_model,dictionary):
    #X = get_topic_distribution_by_list(doc_clean)
    phases = []
    ut_order = []
    n_words = []
    X = []
    utterances = []
    for g in dict_utterances:
        if g in groups_to_filter:
            n_utterances = len(dict_utterances[g]['clean_utterances'])
            for i,phrase in enumerate(dict_utterances[g]['clean_utterances']):
                if i == 0:
                    before_phrase = phrase
                else:
                    before_phrase = dict_utterances[g]['clean_utterances'][i-1]
                if i+1 == n_utterances:
                    after_phrase = phrase
                else:
                    after_phrase = dict_utterances[g]['clean_utterances'][i+1]
                T_before = get_topic_distribution_utterance(before_phrase,lda_model,dictionary)
                T_phrase = get_topic_distribution_utterance(phrase,lda_model,dictionary)
                T_after = get_topic_distribution_utterance(after_phrase,lda_model,dictionary)
                X.append(T_before+T_phrase+T_after)
                phases.append(dict_utterances[g]['phases'][i])
                ut_order.append(dict_utterances[g]['ut_order'][i])
                n_words.append(len(dict_utterances[g]['clean_utterances'][i]))
                utterance = dict_utterances[g]['clean_utterances'][i]
                utterances.append(" ".join(utterance))
    return X,phases,utterances,n_words,ut_order

def get_data(dict_utterances,groups_to_filter,lda_model,dictionary):
    phases = []
    ut_order = []
    n_words = []
    X = []
    utterances = []
    for g in dict_utterances:
        if g in groups_to_filter:
            for i,v in enumerate(dict_utterances[g]['phases']):
                phases.append(v)
                ut_order.append(dict_utterances[g]['ut_order'][i])
                n_words.append(len(dict_utterances[g]['clean_utterances'][i]))
                utterance = dict_utterances[g]['clean_utterances'][i]
                utterances.append(" ".join(utterance))
                X.append(get_topic_distribution_utterance(utterance,lda_model,dictionary))
    return X,phases,utterances,n_words,ut_order

def get_data_window(dict_utterances,groups_to_filter,lda_model,dictionary,size_window):
    phases = []
    ut_order = []
    n_words = []
    X = []
    utterances = []
    for g in dict_utterances:
        if g in groups_to_filter:
            utterances_window = []
            for i,v in enumerate(dict_utterances[g]['phases']):
                phases.append(v)
                ut_order.append(dict_utterances[g]['ut_order'][i])
                n_words.append(len(dict_utterances[g]['clean_utterances'][i]))
                utterance = dict_utterances[g]['clean_utterances'][i]
                if len(utterances_window)>=size_window:
                    utterances_window.pop()
                utterances_window = utterance + utterances_window
                utterances.append(" ".join(utterances_window))
                X.append(get_topic_distribution_utterance(utterance,lda_model,dictionary))
    return X,phases,utterances,n_words,ut_order

def build_simple_df(dict_utterances,groups_to_filter,lda_model,dictionary):
    labels = list(map(lambda x:'Topic {}'.format(x+1),range(lda_model.num_topics)))
    X,phases,utterance,n_words,ut_order = get_data(
        dict_utterances,groups_to_filter,lda_model,dictionary)
    df = pd.DataFrame(X,columns=labels)
    df['phase'] = phases
    df['phase_1'] = list(map(lambda x: 1 if x==1 else 0,phases))
    df['phase_2'] = list(map(lambda x: 1 if x==2 else 0,phases))
    df['phase_3'] = list(map(lambda x: 1 if x==3 else 0,phases))
    df['phase_4'] = list(map(lambda x: 1 if x==4 else 0,phases))
    df['phase_5'] = list(map(lambda x: 1 if x==5 else 0,phases))
    df['utterance'] = utterance
    df['length utterance'] = normalize_values(n_words)
    df['utterance_relative_time'] = ut_order
    return df

def build_simplest_df(dict_utterances,groups_to_filter,lda_model,dictionary):
    labels = list(map(lambda x:'Topic {}'.format(x+1),range(lda_model.num_topics)))
    X,phases,utterance,n_words,ut_order = get_data(
        dict_utterances,groups_to_filter,lda_model,dictionary)
    df = pd.DataFrame(X,columns=labels)
    df['length utterance'] = normalize_values(n_words)
    df['utterance_relative_time'] = ut_order
    df['phase'] = phases
    return df

def build_simplest_next_utterances_df(dict_utterances,groups_to_filter,lda_model,dictionary):
    labels = list(map(lambda x:'Topic before {}'.format(x+1),range(lda_model.num_topics)))
    labels += list(map(lambda x:'Topic {}'.format(x+1),range(lda_model.num_topics)))
    labels += list(map(lambda x:'Topic after {}'.format(x+1),range(lda_model.num_topics)))
    X,phases,utterance,n_words,ut_order = get_data_next_utterances(
        dict_utterances,groups_to_filter,lda_model,dictionary)
    df = pd.DataFrame(X,columns=labels)
    df['length utterance']= normalize_values(n_words)
    df['utterance_relative_time']=ut_order
    df['phase'] = phases
    return df

def build_next_utterances_df(dict_utterances,groups_to_filter,lda_model,dictionary):
    labels = list(map(lambda x:'Topic before {}'.format(x+1),range(lda_model.num_topics)))
    labels += list(map(lambda x:'Topic {}'.format(x+1),range(lda_model.num_topics)))
    labels += list(map(lambda x:'Topic after {}'.format(x+1),range(lda_model.num_topics)))
    X,phases,utterance,n_words,ut_order = get_data_next_utterances(
        dict_utterances,groups_to_filter,lda_model,dictionary)
    df = pd.DataFrame(X,columns=labels)
    df['phase'] = phases
    df['phase_1'] = list(map(lambda x: 1 if x==1 else 0,phases))
    df['phase_2'] = list(map(lambda x: 1 if x==2 else 0,phases))
    df['phase_3'] = list(map(lambda x: 1 if x==3 else 0,phases))
    df['phase_4'] = list(map(lambda x: 1 if x==4 else 0,phases))
    df['phase_5'] = list(map(lambda x: 1 if x==5 else 0,phases))
    df['utterance'] = utterance
    df['length utterance']= normalize_values(n_words)
    df['utterance_relative_time']=ut_order
    return df

# Make a list with num_topics labels plus the pair combination 
# of those labels.
def get_labels_co_occurrence(num_topics):
    labels_num_topics = list(map(lambda x:'Topic {}'.format(x+1),range(num_topics)))
    labels = list(map(lambda x:'Topic {}'.format(x+1),range(num_topics)))
    for i,label_1 in enumerate(labels_num_topics):
        for j,label_2 in enumerate(labels_num_topics):
            if i>j:
                labels.append("{}-{}".format(label_1,label_2))
    return labels

def to_matrix(X):
    X_matrix = np.zeros((len(X),len(X[0])))
    for i in range(X_matrix.shape[0]):
        for j in range(X_matrix.shape[1]):
            X_matrix[i,j] = X[i][j]
    return X_matrix

def get_co_occurrence(X):
    X_co_occurrence = np.zeros((X.shape[0],X.shape[1]+int((X.shape[1]*X.shape[1]-X.shape[1])/2)))
    for i in range(X.shape[0]):
        count = 0
        for j in range(X.shape[1]):
            X_co_occurrence[i,count] = X[i,count]
            count += 1
        for j in range(X.shape[1]):
            for k in range(X.shape[1]):
                if j > k:
                    X_co_occurrence[i,count] = X[i,j]*X[i,k]
                    count+=1
    return X_co_occurrence

def normalize_values(values):
    aux_values = list(map(lambda x: (x-np.mean(values))/np.std(values),values))
    new_values = [(x-np.min(aux_values))/(np.max(aux_values)-np.min(aux_values)) for x in aux_values]
    return new_values

def build_windows_df(dict_utterances,groups_to_filter,lda_model,dictionary,size_window):
    labels = list(map(lambda x:'Topic {}'.format(x+1),range(lda_model.num_topics)))
    X,phases,utterance,n_words,ut_order = get_data_window(
        dict_utterances,groups_to_filter,lda_model,dictionary)
    df = pd.DataFrame(X,columns=labels)
    df['phase'] = phases
    df['phase_1'] = list(map(lambda x: 1 if x==1 else 0,phases))
    df['phase_2'] = list(map(lambda x: 1 if x==2 else 0,phases))
    df['phase_3'] = list(map(lambda x: 1 if x==3 else 0,phases))
    df['phase_4'] = list(map(lambda x: 1 if x==4 else 0,phases))
    df['phase_5'] = list(map(lambda x: 1 if x==5 else 0,phases))
    df['utterance'] = utterance
    df['length utterance'] = normalize_values(n_words)
    df['utterance_relative_time'] = ut_order
    return df

def build_co_occurrence_df(dict_utterances,groups_to_filter,lda_model,dictionary):
    labels = get_labels_co_occurrence(lda_model.num_topics)
    X_simple,phases,utterance,n_words,ut_order = get_data(
        dict_utterances,groups_to_filter,lda_model,dictionary)
    X_matrix = to_matrix(X_simple)
    X_co_occurrence = get_co_occurrence(X_matrix)
    df = pd.DataFrame(X_co_occurrence,columns=labels)
    df['phase'] = phases
    df['phase_1'] = list(map(lambda x: 1 if x==1 else 0,phases))
    df['phase_2'] = list(map(lambda x: 1 if x==2 else 0,phases))
    df['phase_3'] = list(map(lambda x: 1 if x==3 else 0,phases))
    df['phase_4'] = list(map(lambda x: 1 if x==4 else 0,phases))
    df['phase_5'] = list(map(lambda x: 1 if x==5 else 0,phases))
    df['utterance'] = utterance
    df['length utterance']=normalize_values(n_words)
    df['utterance_relative_time']=ut_order
    return df