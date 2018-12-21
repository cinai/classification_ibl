'''
 Modulo que ejecuta distintos clasificadores de fases IBL
 y almacena los resultados en archivos excel y txt
'''

import os
import argparse
import pandas as pd
import sys
from datetime import datetime
from pandas import ExcelWriter
from random_classifier import Biggest,Select2
from svm import SVM
from nearest_svm import Nearest_Phase
from combination_svm import LC_SVM
from markov_model_classifier import MM


#classifiers = [Biggest(),Select2(-1,-1),Select2(1,5),SVM(),SVM(True),
#            Nearest_Phase(),Nearest_Phase(True),LC_SVM(),LC_SVM(True),
classifiers = [MM()]
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

def run_classifiers(training_set_list,test_set_list,folder_path):
    trained_c = []
    results_c = []
    # get train df
    train_set_list_df = read_df_list(training_set_list)
    # train classifiers
    for c in classifiers:
        # get train df
        print("Training {}".format(str(c)))
        c.train(train_set_list_df)
    # get test df
    test_set_list_df = read_df_list(test_set_list)
    # test classifiers
    result_df = []
    end_text = ""
    for c in classifiers:
        print("Testing {}".format(str(c)))
        aux_df,text_result = c.test(test_set_list_df)
        results_c.append(aux_df)
        end_text += "\n {} \n".format(str(c))
        end_text+= text_result
    # save results
    xls_name = 'classifiers_results_{}_{}.xlsx'.format(folder_path,
                                            datetime.now().timestamp())
    text_name = 'classifiers_results_{}_{}.txt'.format(folder_path,
                                            datetime.now().timestamp())
    with open(os.path.join(results_path,text_name),'w') as f:
        f.write(end_text) 
    save_xls(results_c,os.path.join(results_path,xls_name))
    print(end_text)



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
    '''
    training_set_list = []
    for f in os.listdir(os.path.join(tables_path,folder_path,'train')):
        training_set_list.append(os.path.join(tables_path,folder_path,'train',f))
    df_list = read_df_list(training_set_list)
    for df in df_list:
        print(df.shape)
        print(df.columns)
    '''
    if args.classifier != None:
        training_set_list = []
        for f in os.listdir(os.path.join(tables_path,folder_path,'train')):
            training_set_list.append(os.path.join(tables_path,folder_path,'train',f))
        df_list = read_df_list(training_set_list)
        classifier = eval(args.classifier+'()')
        topics = classifier.train(df_list)
        xls_name = 'topics_means_by_phase_{}_{}.xlsx'.format(folder_path,
                                            datetime.now().timestamp())
        save_xls([topics],xls_name)
    else:
        test_set_list = []
        for f in os.listdir(os.path.join(tables_path,folder_path,'test',)):
            test_set_list.append(os.path.join(tables_path,folder_path,'test',f))
        training_set_list = []
        for f in os.listdir(os.path.join(tables_path,folder_path,'train')):
            training_set_list.append(os.path.join(tables_path,folder_path,'train',f))
        run_classifiers(training_set_list,test_set_list,folder_path)
