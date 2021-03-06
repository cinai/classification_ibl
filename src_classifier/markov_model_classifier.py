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
from src_classifier.hmm import HMM
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix,\
                            classification_report,\
                            accuracy_score

'''
Gets the Transition probability matrix from a list
of phases. There are 5 phases, integers from 1 to 5.
'''
class MM:
    THE_PHASES = range(1,6)

    def __str__(self):
        return "Markov model"

    def get_tpm(self,phases):
        n = len(self.THE_PHASES)
        tpm = np.zeros((n,n))
        p_i = phases[0]-1
        for p_j in phases[1:]:
            tpm[p_i-1,p_j-1] += 1
            p_i = p_j
        for p_i in range(n):
            for p_j in range(n):
                if tpm[p_i,p_j] < pow(10,-8):
                    tpm[p_i,p_j] = pow(10,-8)
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
        sum_tpm = self.get_sum_tpm(tpms)
        total_tpm = self.get_normalized_tpm(sum_tpm)
        #print(total_tpm)
        # get mean and cov from topic distribution
        t_m,vh = self.get_means_and_covar_matrices(training_set_list,vh_i=5)
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
        f = multivariate_normal.logpdf
        # build hidden markov model
        hmm = HMM(p0,total_tpm,f,topic_means,covariance_matrices,vh)
        self.hmm = hmm
        #return hmm
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
                o = observations[p-1]
                tm,cv = self.get_mean_and_covar_per_phase(o)
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

    def get_means_and_covar_matrices(self,training_set_list,f=None,vh_i=2):
        if not f:
            f = self.n
        observations = []
        for i in self.THE_PHASES:
            observations.append(
                self.get_distributions_per_phase(training_set_list,i))
        obs = self.group_observations(observations)
        u, s, vh = np.linalg.svd(obs)
        vh = vh[:vh_i,:]
        observations = self.project_observations(observations,vh)
        return f(observations),vh

    def get_test_means_and_covar_matrices(self,training_set_list,f=None,vh_i=2):
        if not f:
            f = self.n
        observations = []
        for i in self.THE_PHASES:
            observations.append(
                self.get_distributions_per_phase(training_set_list,i))
        n = len(observations[0][0])
        return f(observations),np.eye(n,n)

    def test(self,test_set_list):
        return self.results(test_set_list)

    def map_states_to_phases(self,states):
        sequence = []
        for s in states:
            sequence.append(self.THE_PHASES[s])
        return sequence

    def results(self,test_set_list):
        reality = []
        prediction = []
        for i,test in enumerate(test_set_list):
            aux_real = list(test.phase.values)
            features = test.values[:,:-1]
            reality += aux_real
            aux_prediction = self.map_states_to_phases(
                self.hmm.get_sequence_of_states(features))
            prediction += aux_prediction
            #print("Test {} accuracy: {}".format(i,accuracy_score(aux_real,aux_prediction)))
            #print("Test {} cm:\n {}".format(i,confusion_matrix(aux_real,aux_prediction)))

        cm = confusion_matrix(reality, prediction)
        try:
            cols = ["Predicted {}".format(i) for i in self.THE_PHASES]
            df = pd.DataFrame(cm,columns=cols)
            df.index = self.THE_PHASES
        except ValueError:
            df = pd.DataFrame(cm)
        report = classification_report(reality,prediction) 
        accuracy = accuracy_score(reality, prediction)
        mean_error = pc.mean_error(reality,prediction)
        return df,accuracy,mean_error,report

