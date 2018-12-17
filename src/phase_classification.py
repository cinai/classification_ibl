import os
import sys
import pickle 
import random
import numpy as np
from scipy.optimize import minimize

root_path = os.path.dirname(os.path.abspath(os.getcwd()))
data_path = os.path.join(root_path,'data')
tables_path = os.path.join(data_path,'tables')
results_path = os.path.join(root_path,'results')
output_path =os.path.join(results_path,'tables')

def load_classifier(name_classifier):
    with open(os.path.join(data_path,name_classifier),'rb') as f:
        svc = pickle.load(f)
    return svc

def split_df(df,val_set_threshold):
    train_set = {}
    validation_set = {}
    for name, group in df.groupby(['phase']):
        train_set[name]=[]
        validation_set[name]=[]
        n = len(group)
        ra = random.sample(range(n),int(n*val_set_threshold))
        print("validation set phase {}: {}".format(name,int(n*val_set_threshold)))
        count = 0
        group = group.reset_index()
        for i,row in group.iterrows():
            if i in ra:
                count+=1
                validation_set[name].append(row.values)
            else:
                train_set[name].append(row.values)
    return train_set,validation_set

def split_df_test(df):
    train_set = {1:[],2:[],3:[],4:[],5:[]}
    train_set[5]
    for i,row in df.iterrows():
        train_set[int(row.phase)].append(row.values)
    return train_set

def how_many_discussions(df):
    time = df.utterance_relative_time
    n = 1
    last_time = time[0]
    indices = [0,]
    for i,row in df.iterrows():
        if time[i]<last_time:
            n+=1
            indices.append(i)
        last_time = time[i]
    indices.append(len(df))
    return n,indices

def split_df_list(list_df,threshold,seed=None):
    if seed:
        random.seed(seed)
    n = len(list_df)
    test_set_index = random.sample(range(n),int(n*threshold))
    training_set = []
    test_set = []
    for i in range(n):
        if i in test_set_index:
            test_set.append(list_df[i])
        else:
            training_set.append(list_df[i])
    return training_set,test_set

def split_df_discussions(df,threshold,seed=None):
    if seed:
        random.seed(seed)
    n,indices = how_many_discussions(df)
    test_set = random.sample(range(n),int(n*threshold))
    dfs = []
    test_dfs = []
    for i_0,index_1 in enumerate(indices[1:]):
        index_0 = indices[i_0]
        df_aux = df[index_0:index_1].reset_index(drop=True)
        if i_0 in test_set:
            test_dfs.append(df_aux)
        else:
            dfs.append(df_aux)
    return dfs,test_dfs

def get_data_from_dict(a_dict,filter_rows):
    y = []
    X = []
    for key in a_dict:
        rows = a_dict[key]
        for row in rows:
            X.append(row[filter_rows])
            y.append(key)
    return X,y

def get_data_from_list_df(list_df,filter_rows,row_label):
    y = []
    X = []
    for i in range(len(list_df)):
        df = list_df[i]
        aux_y = []
        aux_X = []
        for i,row in df.iterrows():
            aux_X.append(row[filter_rows])
            aux_y.append(row[row_label])
        y.append(aux_y)
        X.append(aux_X)
    return X,y

def join_list_df(list_df):
    X = []
    for i in range(len(list_df)):
        df = list_df[i]
        for i,row in df.iterrows():
            X.append(row)
    return X

def get_joined_data_from_df(list_df,filter_rows,row_label):
    X,y = get_data_from_list_df(list_df,filter_rows,row_label)
    X_joined = []
    y_joined = []
    for i,x in enumerate(X):
        for j,x_j in enumerate(x):
            X_joined.append(x_j)
            y_joined.append(y[i][j])
    return X_joined,y_joined

def get_svc(name_classifier):
    if type(name_classifier) == str:
        # if string use the given name or default svc already loaded
        if name_classifier== "":
            name_classifier = "classifier_svm.pickle"
        svc = load_classifier(name_classifier)
    else:
        # if not string is svc
        svc = name_classifier
    return svc

def first_layer_classifier(features,threshold,name_classifier=""):
    svc = get_svc(name_classifier)
    pred_val = svc.predict_proba(features)
    #n_classes = len(svc.class_weight_)
    pred_labels = []
    for val in pred_val:
        if np.max(val) > threshold:
            pred_labels.append(np.argmax(val)+1)
        else:
            pred_labels.append(-1)
    return pred_labels

def find_closest_phase(index,df,backwards=True):
    phases = df.first_layer.values
    time = df.utterance_relative_time
    if phases[index] != -1:
        raise ValueError("Phase[index] should always be -1")
    last_time = time[index]
    if backwards:
        for i in range(index-1,-1,-1):
            if time[i] > last_time:
                return -1
            last_time = time[i]
            if phases[i] != -1:
                return phases[i]
    else:
        for i in range(index+1,len(phases),1):
            if time[i] < last_time:
                return -1
            last_time = time[i]
            if phases[i] != -1:
                return phases[i]
    return -1


def second_layer_classifier_max_border(features,df,name_classifier=""):
    second_layer = []
    svc = get_svc(name_classifier)
    for i,row in df.iterrows():
        if row.first_layer != -1:
            second_layer.append(row.first_layer)
        else:
            last_phase = find_closest_phase(i,df,backwards=True)
            next_phase = find_closest_phase(i,df,backwards=False)
            pred_val_i = svc.predict_proba([features[i]])[0]
            if pred_val_i[last_phase-1] > pred_val_i[next_phase-1]:
                second_layer.append(last_phase)
            else:
                second_layer.append(next_phase)
    return second_layer

def get_linear_system(features,svc,phases):
    solutions = []
    vectors_1 = []
    vectors_2 = []
    vectors_3 = []
    for i in range(len(features)):
        val_i = svc.predict_proba([features[i]])[0]
        if i == 0:
            val_last_i = val_i
        else:
            val_last_i = svc.predict_proba([features[i-1]])[0]
        if i+1 == len(features):
            val_next_i = val_i
        else:
            val_next_i = svc.predict_proba([features[i+1]])[0]
        try:
            solution_i = unit_vector(int(phases[i]))
        except:
            print(phases[i])

        solutions.append(solution_i)
        vectors_1.append(val_last_i)
        vectors_2.append(val_i)
        vectors_3.append(val_next_i)
    return solutions,vectors_1,vectors_2,vectors_3

def second_layer_combination_before_after_splitted(features,df,name_classifier=""):
    second_layer = []
    svc = get_svc(name_classifier)
    phases = df.first_layer.values
    s,v1,v2,v3 = get_linear_system(features,svc,phases)
    s_extended = []
    v1_extended = []
    v2_extended = []
    v3_extended = []
    for i in range(len(s)):
        for j in range(len(s[0])):
            s_extended.append(s[i][j])
            v1_extended.append(v1[i][j])
            v2_extended.append(v2[i][j])
            v3_extended.append(v3[i][j])
    return s_extended,v1_extended,v2_extended,v3_extended#second_layer

def get_elements_linear_system(s,v1,v2,v3):
    As = []
    bs = []
    for i in range(len(s)):
        A = np.zeros((len(v1[i]),3))
        for j in range(len(v1[i])):
            A[j,0] = v1[i][j]
            A[j,1] = v2[i][j]  
            A[j,2] = v3[i][j]
        As.append(A)
        bs.append(s[i])
    return As,bs

def get_solution_linear_system(s,v1,v2,v3):
    As,bs = get_elements_linear_system(s,v1,v2,v3)
    fun = lambda x: np.sqrt(np.sum([np.square(np.linalg.norm(np.dot(As[i],x)-bs[i])) for i in range(len(s))])/(len(s)*2))
    cons = ({'type': 'eq', 'fun': lambda x:  x[0]+x[1]+x[2]-1})
    sol = minimize(fun, np.zeros(3), method='SLSQP', bounds=[(0.,1.) for x in range(3)],constraints=cons)
    return sol,As

def get_phase_from_linear_combination(sol,As):
    phases = []
    for i in range(len(As)):
        aux = np.dot(As[i],sol)
        index = np.argmax(aux)
        phases.append(index+1)
    return phases

def second_layer_combination_before_after(features,df,name_classifier=""):
    svc = get_svc(name_classifier)
    phases = df.phase.values
    s,v1,v2,v3 = get_linear_system(features,svc,phases)
    sol,As = get_solution_linear_system(s,v1,v2,v3)
    second_layer = get_phase_from_linear_combination(sol.x,As)
    return second_layer,sol.x

def combination_before_after(features,phases,name_classifier=""):
    svc = get_svc(name_classifier)
    s,v1,v2,v3 = get_linear_system(features,svc,phases)
    sol,As = get_solution_linear_system(s,v1,v2,v3)
    second_layer = get_phase_from_linear_combination(sol.x,As)
    return second_layer,sol.x

def second_layer_combination_test(features,coeff,name_classifier=""):
    svc = get_svc(name_classifier)
    phases_dummy = [-1 for i in range(len(features))]
    s,v1,v2,v3 = get_linear_system(features,svc,phases_dummy)
    As,_ = get_elements_linear_system(s,v1,v2,v3)
    second_layer = get_phase_from_linear_combination(coeff,As)
    return second_layer

def unit_vector(n):
    vector = np.zeros((5))
    vector[n-1] = 1
    return vector

def second_layer_combination_before_after_try(features,df,filter_rows,name_classifier=""):
    svc = get_svc(name_classifier)
    phases = df.phase.values
    s,v1,v2,v3 = get_linear_system_try(df,svc,phases,filter_rows)
    sol,As = get_solution_linear_system(s,v1,v2,v3)
    second_layer = get_phase_from_linear_combination(sol.x,As)
    return second_layer,sol.x

def get_linear_system_try(df,svc,phases,filter_rows):
    solutions = []
    vectors_1 = []
    vectors_2 = []
    vectors_3 = []
    features = f = df.values[:,filter_rows]
    for i in range(len(df)):
        val_i = svc.predict_proba([features[i]])[0]
        if i == 0:
            val_last_i = val_i
        else:
            val_last_i = svc.predict_proba([features[i-1]])[0]
        if i+1 == len(features):
            val_next_i = val_i
        else:
            val_next_i = svc.predict_proba([features[i+1]])[0]
        solution_i = unit_vector(phases[i])
        solutions.append(solution_i)
        vectors_1.append(val_last_i)
        vectors_2.append(val_i)
        vectors_3.append(val_next_i)
    return solutions,vectors_1,vectors_2,vectors_3


def mean_error(reality,prediction):
    real = [unit_vector(x) for x in reality]
    pred = [unit_vector(x) for x in prediction]
    square_diff = [np.square(pred[i]-real[i]) for i in range(len(real))]
    return np.sqrt(np.sum(square_diff)/(len(real)*2))
