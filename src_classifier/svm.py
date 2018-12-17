'''
Simple SVM classifier
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

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


SEED = 10
labels = ["Phase {}".format(i) for i in range(1,6)]

class MissingParameter(Exception):
    def __init__(self):
        pass

def get_proportion(phases):
    the_keys = list(set(phases))
    total_samples = 0
    class_samples = {}
    for key in the_keys:
        n = phases.count(key)
        total_samples += n
        class_samples[key] = n
    class_weight = {}
    for key in the_keys:
        class_weight[key] = 1000.0/class_samples[key]
    return class_weight

class SVM:

    svm = None

    def __init__(self,class_weight=False):
        self.class_weight = class_weight

    def __str__(self):
        return "SVM-class_weight-{}".format(self.class_weight)

    def first_layer_classifier(self,features,threshold):
        svc = self.svm
        pred_val = svc.predict_proba(features)
        #n_classes = len(svc.class_weight_)
        pred_labels = []
        for val in pred_val:
            if np.max(val) > threshold:
                pred_labels.append(np.argmax(val)+1)
            else:
                pred_labels.append(-1)
        return pred_labels

    def train(self,training_set_list):
        training_set = pc.join_list_df(training_set_list)
        X_train = [x[:-1] for x in training_set]
        y_train = [x[-1] for x in training_set]
        class_weight = get_proportion(y_train)
        if self.class_weight:
            svc = SVC(kernel='linear',random_state=SEED,probability=True,class_weight=class_weight)
        else:
            svc = SVC(kernel='linear',random_state=SEED,probability=True,class_weight=None)
        svc.fit(X_train, y_train)
        self.svm = svc


    def results(self,test_set_list):
        reality = []
        prediction = []
        for test in test_set_list:
            aux = list(test.phase.values)
            features = test.values[:,:-1]
            reality += aux
            prediction += self.first_layer_classifier(features,0.0)

        cm = confusion_matrix(reality, prediction)
        df = pd.DataFrame(cm,columns=["Predicted {}".format(i) for i in labels])
        df.index = labels
        text = classification_report(reality,prediction)
        text += "\n accuracy: {}".format(accuracy_score(reality, prediction))
        text += "\n mean error: {}".format(pc.mean_error(reality,prediction))
        return df,text

    def test(self,test_set_list):
        return self.results(test_set_list)
