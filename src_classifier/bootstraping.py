'''
 Modulo que ejecuta testeo bootstrapping entre distintos clasificadores 
 de fases IBL y almacena los resultados en archivos excel y txt
'''

import os
import argparse
import sys
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from pandas import ExcelWriter
from random_classifier import Biggest,Select2
from svm import SVM
from nearest_svm import Nearest_Phase
from combination_svm import LC_SVM
from markov_model_classifier import MM


classifiers = [Biggest(),Select2(-1,-1),Select2(1,5),SVM(),SVM(True),
            Nearest_Phase(),Nearest_Phase(True),LC_SVM(),LC_SVM(True),MM()]
#classifiers = [MM()]
#classifiers = [LC_SVM(),LC_SVM(True)]
'''
paths
'''
root_path = os.path.join(os.getcwd(),'..')
data_path = os.path.join(root_path,'data')
tables_path = os.path.join(data_path,'tables')
results_path = os.path.join(root_path,'results')

def save_xls(list_dfs, xls_path):
    writer = ExcelWriter(xls_path)
    for n, df in enumerate(list_dfs):
        sheet_name = str(classifiers[n])
        df.to_excel(writer,sheet_name)
    writer.save()

def read_df_list(path_list):
    df_list = []
    for p in path_list:
        df_list.append(pd.read_excel(p))
    return df_list

def save_metrics(folder_path,accuracies,mean_errors,reports,text_name):
    end_text = ""
    for i,c in enumerate(classifiers):
        end_text += "\n {} ".format(str(c))
        #end_text += "\n {}".format(reports[i])
        end_text += "\n accuracy: {}".format(accuracies[i])
        end_text += "\n mean error: {}".format(mean_errors[i])
    with open(os.path.join(results_path,text_name),'w') as f:
        f.write(end_text)

def save_results(folder_path,accuracies,mean_errors,reports,results_c):
    # save metrics
    text_name = 'classifiers_results_{}_{}.txt'.format(folder_path,
                                            datetime.now().timestamp())
    save_metrics(folder_path,accuracies,mean_errors,reports,text_name)
    # save excel
    xls_name = 'classifiers_results_{}_{}.xlsx'.format(folder_path,
                                                datetime.now().timestamp())
    save_xls(results_c,os.path.join(results_path,xls_name))

def run_classifiers(training_set_list,test_set_list,folder_path,write=True):
    # train classifiers
    train_set_list_df = read_df_list(training_set_list)
    for c in classifiers:
        if write:
            print("Training {}".format(str(c)))
        c.train(train_set_list_df)
    # test classifiers
    test_set_list_df = read_df_list(test_set_list)
    end_text = ""
    results_c = []
    accuracies = []
    mean_errors = []
    reports = []
    for c in classifiers:
        if write:
            print("Testing {}".format(str(c)))
        aux_df,acc,mean_error,report = c.test(test_set_list_df)
        results_c.append(aux_df)
        accuracies.append(acc)
        mean_errors.append(mean_error)
        reports.append(report)
    # save results
    if write:
        save_results(folder_path,accuracies,mean_errors,reports,results_c)
    return results_c,accuracies,mean_errors,reports


def bootstraping(list_files,folder_path):
    # Get combinations
    comb = itertools.combinations(list_files, 9)
    list_split = [[x, tuple(y for y in list_files if y not in x)] for x in comb]    
    results = []
    # Iterate over the combinations of runs
    for i,combination in enumerate(list_split):
        # calculate the results of each run
        training_list = []
        for f_i in combination[0]:
            path_f_i = os.path.join(tables_path,folder_path,'all',f_i)
            training_list.append(path_f_i)
        test_list = []
        for f_i in combination[1]:
            path_f_i = os.path.join(tables_path,folder_path,'all',f_i)
            test_list.append(path_f_i)
        r,a,e,reports = run_classifiers(training_list,test_list,folder_path,write=False)
        results.append([r,a,e,reports])
        print("Done combination {}".format(i), end='\r',flush=True)
    # Summarize the results of each algorithm (mean and std)
    results_c = []
    accuracies = []
    mean_errors = []
    reports = []
    for i,c in enumerate(classifiers):
        df_sum = results[0][0][i]
        accuracy = [results[0][1][i]]
        error = [results[0][2][i]]
        for j in range(1,len(results)):
            df_sum += results[j][0][i]
            accuracy.append(results[j][1][i])
            error.append(results[j][2][i])
        results_c.append(df_sum)
        accuracies.append(np.mean(accuracy))
        mean_errors.append(np.mean(error))
        a_text = "\n".join(["\n".join(e[3]) for e in results])
        reports.append(a_text)
    save_results(folder_path,accuracies,mean_errors,reports,results_c)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'experiment_path', help='path in which there are train and \
        test folders with group discussions')
    parser.add_argument(
        'classifier', nargs='?',help='test particular classifier',)
    args = parser.parse_args()

    folder_path = args.experiment_path

    all_files = []
    for f in os.listdir(os.path.join(tables_path,folder_path,'all')):
        all_files.append(os.path.join(tables_path,folder_path,'all',f))
    bootstraping(all_files,folder_path)