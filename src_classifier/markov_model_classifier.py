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
from .hmm import HMM
from scipy.stats import multivariate_normal

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
        norm_tpm = np.zeros(tpm.shape)
        for i in range(len(tpm)):
            norm_tpm[i,:] = tpm[i,:]/sum(tpm[i,:])
        return norm_tpm

    def train(self,training_set_list):
        # get prior
        p0 = [0 for i in self.THE_PHASES]
        for df in training_set_list:
            p0[df.phase.values[0]-1] += 1.0
        total = sum(p0)     
        p0 = [p0_i/total for p0_i in p0]
        # get tpm
        tpms = []
        for df in training_set_list:
            tpms.append(self.get_tpm(df.phase))
        # summarize tpm
        total_tpm = self.get_normalized_tpm(self.get_sum_tpm(tpms))
        #print(total_tpm)
        # get mean and cov from topic distribution
        t_m,vh = self.get_means_and_covar_matrices(training_set_list)
        topic_means,covariance_matrices = t_m
        
        '''
        topic_means = []
        covariance_matrices = []
        for p in self.THE_PHASES:
            tm,cv = self.get_means_and_covar_matrices_per_phase(training_set_list)
            topic_means.append(tm)
            covariance_matrices.append(cv)
        '''
        # define multivariate distribution
        f = multivariate_normal.pdf
        # build hidden markov model
        hmm = HMM(p0,total_tpm,f,topic_means,covariance_matrices,vh)
        return hmm
        #return total_tpm,topic_means,covariance_matrices
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


    def get_mean_and_covar_per_phase(self,observations):
        tm = np.mean(observations,0)
        cv = np.cov(observations.T)
        return tm,cv

    def kn2(self,observations):
        topic_means = []
        covariance_matrices = []
        for p in self.THE_PHASES:
            try:
                tm,cv = self.get_mean_and_covar_per_phase(observations[p-1])
            except:
                print(p)
                print(observations)
                raise
            topic_means.append(tm)
            covariance_matrices.append(cv)
        return topic_means,covariance_matrices

    def kn(self,observations):
        t,c = self.kn2(observations)
        for c_i in c:
            c_i =np.diag(np.diag(c_i))
        return t,c

    def n2(self,observations):
        topic_means = []
        for p in self.THE_PHASES:
            topic_means.append(np.mean(observations[p-1],0))
        obs = self.group_observations(observations)
        c = np.cov(obs.T)
        covariance_matrices = [c for i in self.THE_PHASES]
        return topic_means,covariance_matrices

    def n(self,observations):
        t,c = self.n2(observations)
        for c_i in c:
            c_i =np.diag(np.diag(c_i))
        return t,c

    def group_observations(self,observations):
        return np.array([o for o_i in observations for o in o_i])

    def project_observations(self,observations,vh):
        result = []
        for phase in range(len(observations)):
            aux = []
            for j in range(len(observations[phase])):
                aux.append(np.matmul(vh,observations[phase][j]))
            result.append(np.array(aux))
        return result

    def get_means_and_covar_matrices(self,training_set_list,f=None):
        if not f:
            f = self.kn2
        observations = []
        for i in self.THE_PHASES:
            observations.append(self.get_distributions_per_phase(training_set_list,i))
        obs = self.group_observations(observations)
        u, s, vh = np.linalg.svd(obs)
        vh = vh[:20,:]
        observations = self.project_observations(observations,vh)
        return f(observations),vh

    def test(self,test_set_list):
        return self.results(test_set_list)

    def results(self,test_set_list):
        reality = []
        prediction = []
        for test in test_set_list:
            aux_real = list(test.phase.values)
            features = test.values[:,:-1]
            reality += aux_real
            aux_prediction = hmm.get_sequence_of_states(features)
            prediction += aux_prediction
        cm = confusion_matrix(reality, prediction)
        try:
            df = pd.DataFrame(cm,columns=["Predicted {}".format(i) for i in labels])
            df.index = labels
        except ValueError:
            df = pd.DataFrame(cm)
        text = classification_report(reality,prediction)
        text += "\n accuracy: {}".format(accuracy_score(reality, prediction))
        text += "\n mean error: {}".format(pc.mean_error(reality,prediction))
        return df,text
