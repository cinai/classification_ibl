'''
Markov Model (MM) tester
This module test if the MM behaves reasonable.
'''

import os
import sys
import argparse
import numpy as np
import pandas as pd

root_path = os.path.join(os.getcwd(),'..')
sys.path.append(root_path)

from src_classifier.hmm import HMM
from src_classifier.markov_model_classifier import MM
from scipy.stats import multivariate_normal

root_path = os.path.join(os.getcwd(),'..')
data_path = os.path.join(root_path,'data')
tables_path = os.path.join(data_path,'tables')
results_path = os.path.join(root_path,'results')

def read_df_list(path_list):
    df_list = []
    for p in path_list:
        df_list.append(pd.read_excel(p))
    return df_list

def format_df(matrix):
    cols = ["Phase {}".format(i) for i in range(1,6)]
    df = pd.DataFrame(matrix,columns=cols)
    df.index = cols
    return df

def test_from_1_to_4():
    #p0 = [.90,0.1,0,0,0]
    p0 = [1.,0.,0.,0.,0.]
    #p0 = [0.,0.,0.,1.,0.]
    print("p0: {}".format(p0))
    # get tpm
    tpm = np.zeros((5,5))
    tpm[0,3] = 1.
    tpm[1,4] = 1.
    tpm[2,2] = 1.
    tpm[3,3] = 1.
    tpm[4,1] = 1.   
    print("tpm: \n")
    print(format_df(tpm))
    return p0,tpm


def test_from_3_to_5():
    p0 = [0.,0.,1.,0.,0.]
    print("p0: {}".format(p0))
    tpm = np.zeros((5,5))
    tpm[0,3] = 1.
    tpm[1,4] = 1.
    tpm[2,1] = 1.
    tpm[3,4] = 1.
    tpm[4,4] = 1.

    print("tpm: \n")
    print(format_df(tpm))
    return p0,tpm


def test_tpm():
    p0 = [0.2,0.2,.2,0.2,0.2]
    print("p0: {}".format(p0))
    tpm = .2*np.ones((5,5))
    '''
    tpm[0,0] = 1.
    tpm[1,1] = 1.
    tpm[2,2] = 1.
    tpm[3,3] = 1.
    tpm[4,4] = 1.
    '''
    print("tpm: \n")
    print(format_df(tpm))
    return p0,tpm

def get_sets_list(means,cov):
    # create 5 distributions with different means same covariance
    n = cov.shape[0]
    # generate traindata from those distributions
    training_set_list = []
    cols = ["v_{}".format(i) for i in range(n)] + ['phase']
    for i,m in enumerate(means):
        df = []
        for j in range(200):
            value = np.random.multivariate_normal(m,cov)
            value = np.append(value,i+1)
            df.append(value)
        df = pd.DataFrame(np.array(df))
        df.columns = cols
        training_set_list.append(df)
    # generate testdata from those distributions
    test_set_list = []
    for i,m in enumerate(means):
        df = []
        for j in range(200):
            value = np.random.multivariate_normal(m,cov)
            value = np.append(value,i+1)
            df.append(value)
        df = pd.DataFrame(np.array(df))
        df.columns = cols
        test_set_list.append(df)

    return training_set_list,test_set_list

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'experiment_path', help='path in which there are train and \
        test folders with group discussions')
    args = parser.parse_args()

    folder_path = args.experiment_path

    test_set_list = []
    for f in os.listdir(os.path.join(tables_path,folder_path,'test',)):
        test_set_list.append(os.path.join(tables_path,folder_path,'test',f))
    training_set_list = []
    for f in os.listdir(os.path.join(tables_path,folder_path,'train')):
        training_set_list.append(os.path.join(tables_path,folder_path,'train',f))
    training_set_list = read_df_list(training_set_list)
    test_set_list = read_df_list(test_set_list)

    '''
    n = 3
    cov = np.eye(n,n)
    means = [e*np.ones(n) for e in np.arange(-40,60,20)]
    training_set_list,test_set_list = get_sets_list(means,cov)

    ''' 
    p0,tpm = test_from_1_to_4()
    f = multivariate_normal.pdf
    mm = MM()
    t_m,vh = mm.get_means_and_covar_matrices(training_set_list,vh_i=5)
    topic_means,covariance_matrices = t_m
    hmm = HMM(p0,tpm,f,topic_means,covariance_matrices,vh)
    mm.hmm = hmm

    p0,tpm = test_from_3_to_5()
    mm = MM()
    hmm = HMM(p0,tpm,f,topic_means,covariance_matrices,vh)
    mm.hmm = hmm
'''
    p0,tpm = test_from_1_to_4()
    f = multivariate_normal.pdf
    mm = MM()
    t_m,vh = mm.get_test_means_and_covar_matrices(training_set_list,vh_i=62)
    topic_means,covariance_matrices = t_m
    hmm = HMM(p0,tpm,f,topic_means,covariance_matrices,vh)
    mm.hmm = hmm
    df,text = mm.results(test_set_list)
    print(df)
    print(text)
