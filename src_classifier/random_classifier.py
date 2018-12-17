'''
Classifiers that don't use more information than the phase
proportions in the training to predict phases in the test set
'''
import sys
import os
import numpy as np
import pandas as pd 

root_path = os.path.join(os.getcwd(),'..')
sys.path.append(root_path)

from src import phase_classification as pc

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


SEED = 10
labels = ["Phase {}".format(i) for i in range(1,6)]

class MissingParameter(Exception):
    def __init__(self):
        pass

def get_sorted_phases(train_set):
    the_keys = list(set(train_set['phase']))
    total_samples = 0
    class_samples = {}
    for key in the_keys:
        n = list(train_set.phase.values).count(key)
        #print("key {}, total {}".format(key,n))
        total_samples += n
        class_samples[key] = n
    proportions = []
    for key in range(1,len(the_keys)+1):
        proportion = round(class_samples[key]*1.0/total_samples,2)
        proportions.append(proportion)
    return np.argsort(proportions)
    
def find_threshold(list_df,p1,p2):
    accuracy_try = []
    steps = np.arange(0,1.0,0.025)
    y_all = [df.phase.values for df in list_df]
    for step in steps:
        accuracy_try_aux = []
        for i in range(len(y_all)):
            test_prediction = []
            for j in range(len(y_all[i])):
                step_j = j*1.0/len(y_all[i])
                if step_j > step:
                    test_prediction.append(p2)
                else:
                    test_prediction.append(p1)
            accuracy_try_aux.append(np.sum(confusion_matrix(y_all[i], test_prediction).diagonal())/len(y_all[i]))
        accuracy_try.append(np.mean(accuracy_try_aux))
    return steps[np.argmax(accuracy_try)]

class Biggest:

    biggest_phase = -1

    def __str__(self):
        if self.biggest_phase == -1:
            return "Select biggest phase"
        else:
            return "Select biggest phase {}".format(self.biggest_phase)

    def train(self,training_set_list):
        columns = training_set_list[0].columns
        training_set = pd.DataFrame(pc.join_list_df(training_set_list))
        training_set.columns = columns
        self.biggest_phase = get_sorted_phases(training_set)[-1]+1

    def results(self,test_set_list):
        reality = []
        prediction = []
        for test in test_set_list:
            aux = list(test.phase.values)
            reality += aux
            prediction += [self.biggest_phase for p in aux]
        cm = confusion_matrix(reality, prediction)
        df = pd.DataFrame(cm,columns=["Predicted {}".format(i) for i in labels])
        df.index = labels
        text = classification_report(reality,prediction)
        text += "\n accuracy: {}".format(accuracy_score(reality, prediction))
        text += "\n mean error: {}".format(pc.mean_error(reality,prediction))
        return df,text

    def test(self,test_set_list):
        if self.biggest_phase != -1:
            return self.results(test_set_list)
        else:
            raise MissingParameter()


class Select2:
    threshold = -1.0
    p1 = -1
    p2 = -1

    def __init__(self,p1,p2):
        if (p1==-1 or p1>0) and (p2==-1 or p2>0):
            self.p1 = p1
            self.p2 = p2
        else:
            raise MissingParameter()

    def __str__(self):
        if self.p1 == -1:
            return "Select two phases"
        else:
            return "Select two phases {}-{}".format(self.p1,self.p2)

    def train(self,training_set_list):
        columns = training_set_list[0].columns
        training_set = pd.DataFrame(pc.join_list_df(training_set_list))
        training_set.columns = columns
        if self.p1 <= -1:
            sorted_phases = get_sorted_phases(training_set)
            self.p1 = sorted_phases[-2]+1
            self.p2 = sorted_phases[-1]+1
        self.threshold = find_threshold(training_set_list,self.p1,self.p2)


    def get_prediction(self,n):
        pred = []
        for j in range(n):
            step_j = j*1.0/n
            if step_j > self.threshold:
                pred.append(self.p2)
            else:
                pred.append(self.p1)
        return pred

    def results(self,test_set_list):
        reality = []
        prediction = []
        for test in test_set_list:
            aux = list(test.phase.values)
            reality += aux
            prediction += self.get_prediction(len(aux))
        cm = confusion_matrix(reality, prediction)
        df = pd.DataFrame(cm,columns=["Predicted {}".format(i) for i in labels])
        df.index = labels
        text = classification_report(reality,prediction)
        text += "\n accuracy: {}".format(accuracy_score(reality, prediction))
        text += "\n mean error: {}".format(pc.mean_error(reality,prediction))
        return df,text

    def test(self,test_set_list):
        if self.p1 != -1:
            return self.results(test_set_list)
        else:
            raise MissingParameter()

