#!/usr/bin/env python
# coding: utf-8

# In[4]:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import roc_curve, roc_auc_score
import time
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.utils import resample
from scipy.stats import ttest_ind

def score(l1,t):
    """
    Calculate the U score for a given permutation and parameter values.

    Parameters:
        l1 (list): Ground truth permutation.
        t (list): values corresponding to the sample permutation.

    Returns:
        int: U score.
    """
    import itertools
    # Generate all possible combinations of length 2 from l1
    combinations = list(itertools.combinations(l1, 2))
    #col_map (dict): Mapping of parameter index to its value and position of t.
    col_map = {i: col for i, col in enumerate(t)}
    # Create a dictionary to map parameter index to its value and position
    dic = {}
    for key, value in col_map.items():
        dic[key] = [l1.index(key), value]

    u = 0
    for i in combinations:  
        if (( dic[i[1]][0] > dic[i[0]][0]) & (dic[i[0]][1] > dic[i[1]][1])) or ((dic[i[1]][0] < dic[i[0]][0]) & (dic[i[0]][1] < dic[i[1]][1])):
            u += 1  
        if ((dic[i[1]][0] < dic[i[0]][0]) & (dic[i[0]][1] > dic[i[1]][1])) or ((dic[i[1]][0] > dic[i[0]][0]) & (dic[i[0]][1] < dic[i[1]][1])):
            u += -1
        else:
            u += 0

    return u


# In[5]:
from joblib import Parallel, delayed
import multiprocessing

def cal_score(l1, selected,n_case,n_control):
    """
    Calculate the U score for observations in parallel.

    Parameters:
        l1 (list): Ground truth permutation.
        selected (DataFrame): Selected data containing observations and parameters.
        n_case (int): Number of cases.
        n_control (int): Number of controls.

    Returns:
        tuple: U score, saved U values, U case, U control.
    """
    def compute_score(r, t):
        #case
        if selected.iloc[r, -1] == 1:
            return score(l1, t), r,1
        #control
        elif selected.iloc[r, -1] == 0:
            return score(l1, t), r,0

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(compute_score)(r, selected.iloc[r, :-1].tolist())
        for r in range(len(selected))
    )
    
    u_case = sum(result[0] for result in results if result[2] == 1)/n_case
    
    u_control = sum(result[0] for result in results if result[2] == 0)/n_control
   
    save_u = {result[1]: result[0] for result in results}
    u = u_case - u_control
    return round(u,2), save_u, round(u_case,2), round(u_control,2)


# In[6]:

from sklearn.metrics import roc_auc_score
def cal_auc(data,permutation,save_best_u):
    """
    Calculate the AUC score for a given data subset and permutation.

    Parameters:
        data (DataFrame): The dataset containing features and the target variable.
        permutation (list): A list of column indices representing the feature order.
        save_best_u (dict): A dictionary mapping indices to their respective scores.

    Returns:
        float: The calculated AUC score.
    """
    # Select the subset of data based on the given permutation
    subset = data.iloc[:, permutation]
    # Add the target variable ('cancer') to the subset
    subset['cancer'] = list(data[data.shape[1]-1])
    # Add the 'score' column from save_best_u values
    subset['score'] = list(save_best_u.values())
    # Calculate the AUC score using the cancer labels and the scores
    auc = roc_auc_score(subset['cancer'],subset['score'])
    return auc

# In[7]:
def cal_auc_ci(data, permutation, save_best_u, n_bootstrap=1000, confidence_level=0.95):
    """
    Calculate the AUC score and its confidence interval using bootstrapping.

    Parameters:
        data (DataFrame): The dataset containing features and the target variable.
        permutation (list): A list of column indices representing the feature subset.
        save_best_u (dict): A dictionary mapping indices to their respective scores.
        n_bootstrap (int): Number of bootstrap samples to generate for confidence interval calculation (default: 1000).
        confidence_level (float): Confidence level for the confidence interval (default: 0.95).

    Returns:
        tuple: A tuple containing the mean AUC, lower bound, and upper bound of the confidence interval.
    """
    subset_40 = data.iloc[:, permutation]
    index = len(data.columns) - 1
    subset_40['cancer'] = data.iloc[:, index]
    subset_40['score'] = list(save_best_u.values())

    aucs = []
    i = 0
    while i < n_bootstrap:
        # Sampling with replacement
        bootstrap_sample = subset_40.sample(n=len(subset_40), replace=True)
        # Handle the case where there are not enough unique classes
        if len(np.unique(bootstrap_sample['cancer'])) < 2:
            continue

        auc = roc_auc_score(bootstrap_sample['cancer'], bootstrap_sample['score'])
        aucs.append(auc)
        i += 1

    # Compute the lower and upper percentile CI bounds
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile
    lower_bound = np.percentile(aucs, lower_percentile * 100)
    upper_bound = np.percentile(aucs, upper_percentile * 100)
    return np.mean(aucs), lower_bound, upper_bound


# In[8]:
import random
import time
def rbl(data,g,train,val,val2,test):
    """
    Find the optimal permutation.

    Parameters:
        data(dataframe):input dataframe
        num_val(int):the number of validation dataset to be sampled
        g(float): after g percentage of iterations, check the performance

    Returns:
        train_auc_save(list):saved train auc 
        val_auc_save(list):saved validation auc
        val_auc_save2(list):saved validation2 auc
        test_auc_save(list):saved test auc
        save[best_score](list):best permutation
    """
    index = len(train.columns) - 1
    train_freq = train.iloc[:, index].value_counts()
    val_freq = val.iloc[:, index].value_counts()
    val2_freq = val2.iloc[:, index].value_counts()
    test_freq = test.iloc[:, index].value_counts()

    # Extract class frequencies for each set
    train_freq0, train_freq1 = train_freq[0], train_freq[1]
    val_freq0, val_freq1 = val_freq[0], val_freq[1]
    val2_freq0, val2_freq1 = val2_freq[0], val2_freq[1]
    test_freq0, test_freq1 = test_freq[0], test_freq[1]
    
    # Standardize column indices
    for dataset in [train, val, val2, test]:
        dataset.columns = range(len(dataset.columns))
    
    #find a better initilized score
    # Initialize lists to store scores and AUCs
    num_features = range(0,len(train.columns)-1)
    cur_perm = list(num_features)
    auc_base_train = float('-inf')
    auc_base_val = float('-inf')
    auc_base_val2 = float('-inf')
    best_permu_base = []
    
    start_time = time.time()  # Record the start time
    
    for k in range(0,100):
        random.seed(k)
        random.shuffle(cur_perm)
        # Calculate scores and AUCs
        score_b,save_b, _, _= cal_score(cur_perm,train,train_freq1,train_freq0)
        auc_train = cal_auc(train,cur_perm,save_b)
        score_val_b,save_u_v_b, _, _ = cal_score(cur_perm,val,val_freq1,val_freq0)
        auc_val = cal_auc(val,cur_perm,save_u_v_b)
        score_val_b2,save_u_v_b2, _, _ = cal_score(cur_perm,val2,val2_freq1,val2_freq0)
        auc_val2 = cal_auc(val2,cur_perm,save_u_v_b2)
        
        # Check if scores and AUCs meet conditions
        if auc_train > 0.5 and auc_val > 0.5 and auc_val2 > 0.5 and ((auc_train+auc_val+auc_val2)/3 > (auc_base_train+auc_base_val+auc_base_val2)/3):
            print(auc_train)
            print(auc_val)
            print(auc_val2)
            best_permu_base = cur_perm.copy() 

    #add early stop
    #change initial list as previous permutation
    #if find a good initilization
    if best_permu_base:
        cur_perm = best_permu_base
    else:
        random.seed(200)
        random.shuffle(cur_perm)
        cur_perm = list(num_features)

    # Initialize variables for MCMC
    best_score = float('-inf')
    save = {}
    save_score = []
    save_u = {}
    score_train_prev, score_val_prev, score_val2_prev = float('-inf'), float('-inf'), float('-inf')
    visit = set()
    train_auc_save = []
    val_auc_save = []
    val_auc_save2 = []
    test_auc_save = []
    
    #for early stop
    cur = float('inf')
    cur_auc = 0
    # Main loop for MCMC
    for k in range(0,5001):
        random.seed(k)
        i = random.randint(0, len(cur_perm) - 1)
        j = random.randint(0, len(cur_perm) - 1)
        
        #swap the two elements
        cur_perm[i], cur_perm[j] = cur_perm[j], cur_perm[i]
        
        #skip this i and j if they have been checked before
        if tuple(cur_perm) in visit or i == j:
            continue
        # Add the current permutation to the visit set
        visit.add(tuple(cur_perm))
        
        #save train
        score_train, save_u, _, _  = cal_score(cur_perm,train,train_freq1,train_freq0)
        auc_train = cal_auc(train,cur_perm,save_u)
        train_auc_save.append(round(auc_train,2))

        #save val
        score_val, save_u_v, _, _= cal_score(cur_perm,val,val_freq1,val_freq0)
        auc_val = cal_auc(val,cur_perm,save_u_v)
        val_auc_save.append(round(auc_val,2))

        #save val2
        score_val2, save_u_v2, _, _ = cal_score(cur_perm,val2,val2_freq1,val2_freq0)
        auc_val2 = cal_auc(val2,cur_perm,save_u_v2)
        val_auc_save2.append(round(auc_val2,2))
        
        #save test
        score_test, save_u_t, _, _ = cal_score(cur_perm,test,test_freq1,test_freq0)
        auc_test = cal_auc(test,cur_perm,save_u_t)
        test_auc_save.append(round(auc_test,2))
        
        #MCMC
        if (score_train > score_train_prev) and (score_val > score_val_prev) and (score_val2 > score_val2_prev):
            score_train_prev = score_train
            score_val_prev = score_val
            score_val2_prev = score_val2
            best_score = score_train
            my_list_copy = cur_perm.copy()
            save[best_score] = my_list_copy
            save_score.append(best_score)
 
        else:
            cur_perm[i], cur_perm[j] = cur_perm[j], cur_perm[i]
        #early stop
        if k>0 and k%500 == 0:
            print(k)
            cur = k
            cur_auc = auc_train

        if k >= cur + cur * g and cur_auc >= auc_train:
            print(f"Stopping at iteration {k} because AUC did not increase.")
            break

        if k %100 == 0:
            current_time = time.time() - start_time  # Calculate the elapsed time
            print("Iteration:", k)
            print("Time taken:", current_time, "seconds")

    end_time = time.time()  # Record the end time
    total_time = end_time - start_time  # Calculate the total time

    print("Total time taken:", total_time, "seconds")        

    return train_auc_save,val_auc_save,val_auc_save2,test_auc_save,save[best_score]



# In[9]:
def cal_auc_per_data(df,train,val,val2,test):
    """
    Parameters:
        data(dataframe):input dataframe

    Returns:
        AUC(int): train AUC
        AUC(int): test AUC
        best_perm(list):best permutation
    """

    # Calculate class frequencies
    index = len(train.columns) - 1
    train_freq = train.iloc[:, index].value_counts()
    val_freq = val.iloc[:, index].value_counts()
    val2_freq = val2.iloc[:, index].value_counts()
    test_freq = test.iloc[:, index].value_counts()

    # Extract class frequencies for each set
    train_freq0, train_freq1 = train_freq[0], train_freq[1]
    val_freq0, val_freq1 = val_freq[0], val_freq[1]
    val2_freq0, val2_freq1 = val2_freq[0], val2_freq[1]
    test_freq0, test_freq1 = test_freq[0], test_freq[1]

    # Initialize variables to track best parameters and results
    best_g = 0
    best_val = 0
    g_range = [0.2,0.3]
    for i in g_range:
        a,b,c,d,e = rbl(df,i,train,val,val2,test)
        # Calculate AUC on validation set with the obtained permutation
        score_val, save_u_v, _, _ = cal_score(e,val,val_freq1,val_freq0)
        # Update best parameter and AUC if current AUC is higher
        auc_val = cal_auc(val,e,save_u_v)
        if auc_val > best_val:
            best_g = i
            best_val = auc_val
            best_perm = e

            # If AUC is perfect, stop optimization
            if best_val == 1.0:
                break

    # Calculate train auc
    _, save_u, _, _ = cal_score(best_perm,train,train_freq1,train_freq0)
    train_auc = cal_auc(train,best_perm,save_u)

    # Calculate test auc
    _, save_u_t,_, _ = cal_score(best_perm,test,test_freq1,test_freq0)
    test_auc = cal_auc(test,best_perm,save_u_t)
    
    return train_auc, test_auc, best_perm



