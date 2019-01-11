'''
Nearest phase classifier
This module trains an SVM classifier
that is used to classify all the utterances with a degree
of certainty in a first step. If there are utterance that
are not classified after the first attempt, then the code
assigned to that utterance is selected from the two nearest 
codified phases.
'''

import sys
import os
import random
import numpy as np
import pandas as pd 

root_path = os.path.join(os.getcwd(),'..')
sys.path.append(root_path)

from src import phase_classification as pc

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


SEED = 10
labels = ["Phase {}".format(i) for i in range(1,6)]

class MissingParameter(Exception):
    def __init__(self):
        pass

def find_closest_phase(index,first_layer,backwards=True):
    if first_layer[index] != -1:
        raise ValueError("Phase[index] should always be -1")
    if backwards:
        for i in range(index-1,-1,-1):
            if first_layer[i] != -1:
                return first_layer[i]
    else:
        for i in range(index+1,len(first_layer),1):
            if first_layer[i] != -1:
                return first_layer[i]
    return -1

class Nearest_Phase:

    svm = None
    threshold = 0.0

    def __init__(self,class_weight=False):
        self.class_weight = class_weight

    def __str__(self):
        return "Nearest_Phase-class_weight-{}".format(self.class_weight)

    def first_layer_classifier(self,features,svm=None):
        if svm:
            svc = svm
        else:
            svc = self.svm
        threshold = self.threshold
        pred_val = svc.predict_proba(features)
        #n_classes = len(svc.class_weight_)
        pred_labels = []
        for val in pred_val:
            if np.max(val) > threshold:
                pred_labels.append(np.argmax(val)+1)
            else:
                pred_labels.append(-1)
        return pred_labels

    def second_layer_classifier(self,features,first_layer,svm=None):
        second_layer = []
        if svm:
            svc = svm
        else:
            svc = self.svm
        for i in range(len(first_layer)):
            if first_layer[i] != -1:
                second_layer.append(first_layer[i])
            else:
                last_phase = find_closest_phase(i,first_layer,backwards=True)
                next_phase = find_closest_phase(i,first_layer,backwards=False)
                pred_val_i = svc.predict_proba([features[i]])[0]
                if last_phase == -1 and next_phase == -1:
                    print("This is wroong")
                if pred_val_i[last_phase-1] > pred_val_i[next_phase-1]:
                    second_layer.append(last_phase)
                else:
                    second_layer.append(next_phase)
        return second_layer

    def find_threshold(self,train,val,seed=None):
        if seed:
            random.seed(seed)
        ratio = np.arange(0.2,0.7,0.05)
        # Train
        training_set = pc.join_list_df(train)
        X_train = [x[:-1] for x in training_set]
        y_train = [x[-1] for x in training_set]
        if self.class_weight:
            svc = SVC(kernel='linear',random_state=SEED,probability=True,class_weight='balanced')
        else:
            svc = SVC(kernel='linear',random_state=SEED,probability=True,class_weight=None)
        svc.fit(X_train, y_train)
        ratios_results = []
        val_set = pc.join_list_df(val)
        X_val = [x[:-1] for x in val_set]
        y_val = [x[-1] for x in val_set]
        border_amount = len(y_val)*0.3
        for r in ratio:
            self.threshold = r
            first_layer = self.first_layer_classifier(X_val,svc)
            prediction = self.second_layer_classifier(X_val,first_layer,svc)
            cm_d = np.sum(confusion_matrix(y_val,prediction).diagonal())*1.0
            if cm_d/len(val) > border_amount:
                ratios_results.append(accuracy_score(y_val,prediction))
            else:
                ratios_results.append(0.0)
        return ratio[np.argmax(ratios_results)]


    def train(self,training_set_list):
        train,val = pc.split_df_list(training_set_list,0.2,SEED)
        training_set = pc.join_list_df(train)

        # find threshold to split using train and val set
        self.t = self.find_threshold(train,val,SEED)
        # train svm with all the data 
        training_set = pc.join_list_df(training_set_list)
        X_train = [x[:-1] for x in training_set]
        y_train = [x[-1] for x in training_set]
        if self.class_weight:
            svc = SVC(kernel='linear',random_state=SEED,probability=True,class_weight='balanced')
        else:
            svc = SVC(kernel='linear',random_state=SEED,probability=True,class_weight=None)
        svc.fit(X_train, y_train)
        self.svm = svc


    def results(self,test_set_list):
        reality = []
        prediction = []
        for test in test_set_list:
            aux_real = list(test.phase.values)
            features = test.values[:,:-1]
            reality += aux_real
            prediction_1 = self.first_layer_classifier(features)
            aux_prediction = self.second_layer_classifier(features,prediction_1)
            prediction += aux_prediction
        cm = confusion_matrix(reality, prediction)
        try:
            df = pd.DataFrame(cm,columns=["Predicted {}".format(i) for i in labels])
            df.index = labels
        except ValueError:
            df = pd.DataFrame(cm)
        report = classification_report(reality,prediction) 
        accuracy = accuracy_score(reality, prediction)
        mean_error = pc.mean_error(reality,prediction)
        return df,accuracy,mean_error,report


    def test(self,test_set_list):
        return self.results(test_set_list)
