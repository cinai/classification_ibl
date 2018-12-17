'''
Markov Model (MM) classifier
This module trains an MM classifier that is used to classify 
all the utterances from an IBL transcription.
'''

import sys
import os
import numpy as np
import pandas as pd 

root_path = os.path.join(os.getcwd(),'..')
sys.path.append(root_path)

from src import phase_classification as pc


'''
Gets the Transition probability matrix from a list
of phases. There are 5 phases, integers from 1 to 5.
'''
def get_tpm(phases):
	the_phases = list(set(phases))
	n = len(the_phases)
	tpm = np.zeros((n,n))
	p_i = phases[0]
	for p_j in phases[1:]:
		tpm[p_i-1,p_j-1] += 1
	return tpm

def get_sum_tpm(list_tpm):
	total = np.sum(list_tpm,0)

def get_normalized_tpm(tpm):
	# I didnt know what to do

def train(self,training_set_list):
    training_set = pc.join_list_df(training_set_list)
    X_train = [x[:-1] for x in training_set]
    y_train = [x[-1] for x in training_set]
