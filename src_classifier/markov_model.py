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
class MM:
    THE_PHASES = range(1,6)

    def get_tpm(self,phases):
        n = len(self.THE_PHASES)
        tpm = np.zeros((n,n))
        p_i = phases[0]-1
        for p_j in phases[1:]:
            tpm[p_i-1,p_j-1] += 1
            p_i = p_j
        return tpm

    def get_sum_tpm(self,list_tpm):
        total = np.sum(list_tpm,0)
        return total

    def get_normalized_tpm(self,tpm):
        pass
        # I didnt know what to do because we don't really
        # want to guess what phase is next

    def train(self,training_set_list):
        tpms = []
        for df in training_set_list:
            tpms.append(self.get_tpm(df.phase))
        # summarize tpm
        total_tpm = self.get_sum_tpm(tpms)
        print(total_tpm)
        # get mean topic distribution
        topic_means = []
        covariance_matrices = []
        for p in self.THE_PHASES:
            tm,cv = self.get_mean_and_covar_per_phase(training_set_list,p)
            topic_means.append(tm)
            covariance_matrices.append(cv)
        return total_tpm,pd.DataFrame(topic_means),covariance_matrices
        #training_set = pc.join_list_df(training_set_list)
        #X_train = [x[:-1] for x in training_set]
        #y_train = [x[-1] for x in training_set]


    def get_distributions_per_phase(self,list_df,phase):
        topic_distribution = []#np.zeros((1,list_df[0].shape[1]-1))
        for df in list_df:
            X = df.values
            for i in range(X.shape[0]):
                if X[i][-1] != phase:
                    continue
                topic_distribution.append(X[i][:-1])
        return np.array(topic_distribution)


    def get_mean_and_covar_per_phase(self,list_df,p):
        topic_distribution = self.get_distributions_per_phase(list_df,p)
        tm = np.mean(topic_distribution,0)
        cv = np.cov(topic_distribution.T)
        return tm,cv
