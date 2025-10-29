import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_auc_score
import time
from joblib import Parallel, delayed
import multiprocessing

def score(l1, t):
    """
    Calculate the U score for a given permutation and parameter values.

    Parameters:
        l1 (list): Ground truth permutation.
        t (list): values corresponding to the sample permutation.

    Returns:
        int: U score.
    """
    n = len(l1)
    # Create a fast lookup for original positions of elements in l1
    # This is O(n)
    l1_pos_map = {val: i for i, val in enumerate(l1)}

    # Create a dictionary to map parameter index to its value and position in l1
    # This is O(n)
    dic = {}
    for i, value_t in enumerate(t):
        # i is the feature index (0 to n-1)
        # value_t is the value for that feature in the current sample
        # l1_pos_map[i] gives the ground truth position of feature 'i'
        dic[i] = [l1_pos_map[i], value_t]

    u = 0
    # The combinations generation is still O(n^2)
    # This loop is O(n^2)
    for i_idx in range(n):
        for j_idx in range(i_idx + 1, n):
            # i_idx and j_idx are the feature indices (e.g., 0, 1, 2...)

            # Access elements from dic based on feature indices
            l1_pos_i = dic[i_idx][0]
            val_t_i = dic[i_idx][1]

            l1_pos_j = dic[j_idx][0]
            val_t_j = dic[j_idx][1]

            # Conditions based on ground truth positions and sample values
            if ((l1_pos_j > l1_pos_i) and (val_t_i > val_t_j)) or \
               ((l1_pos_j < l1_pos_i) and (val_t_i < val_t_j)):
                u += 1
            elif ((l1_pos_j < l1_pos_i) and (val_t_i > val_t_j)) or \
                 ((l1_pos_j > l1_pos_i) and (val_t_i < val_t_j)):
                u += -1
    return u

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


def cal_auc(data, permutation, save_best_u):
    """
    Calculate the AUC score for a given data subset and permutation.

    Parameters:
        data (DataFrame): The dataset containing features and the target variable.
                          Expected to have integer column names after preprocessing.
        permutation (list): A list of column indices representing the feature order
                            (these are the feature columns).
        save_best_u (dict): A dictionary mapping original row indices
                            (or some identifier) to their respective calculated scores.
                            Assumes values are ordered correctly if converted to a list.

    Returns:
        float: The calculated AUC score.
    """
    # Get the target variable (last column, as columns were renamed to integers 0, 1, ..., N-1)
    target_variable = data.iloc[:, -1] # -1 refers to the last column

    # Convert the scores from the dictionary to a list.
    scores = list(save_best_u.values())

    # Calculate the AUC score directly
    auc = roc_auc_score(target_variable, scores)

    return auc

def cal_auc_ci(data, permutation, n_bootstrap=1000, confidence_level=0.95):
    """
    Estimates the 95% confidence interval for test AUC under a fixed permutation using bootstrapping.
    Calculates U-scores for the entire dataset once, then samples by looking up pre-computed scores.

    Parameters:
        data (DataFrame): The input dataset where the last column is the 'cancer' label.
        permutation (list of int): A list of feature column indices (excluding the last label column).
        n_bootstrap (int): The number of bootstrap samples to draw.
        confidence_level (float): The desired confidence level, e.g., 0.95.

    Returns:
        (mean_auc, lower_bound, upper_bound, aucs_list)
        mean_auc (float): The mean AUC across all bootstrap samples.
        lower_bound (float): The lower bound of the confidence interval.
        upper_bound (float): The upper bound of the confidence interval.
        aucs_list (list): A list of all AUC values calculated during bootstrapping.
    """

     # Reset DataFrame index and rename the label column for consistency
    df = data.reset_index(drop=True).copy()
    idx_last = df.shape[1] - 1
    df = df.rename(columns={df.columns[idx_last]: 'cancer'})

    # Run cal_score once on the entire DataFrame to get save_u_full (row_id -> U-score)
    # Get class frequencies for the full dataset
    freq_full = df['cancer'].value_counts().sort_index()
    n_neg_full = int(freq_full.get(0, 0))
    n_pos_full = int(freq_full.get(1, 0))

    # cal_score(permutation, selected_df, n_pos, n_neg) 返回 (_, save_u, ...)，
    _, save_u_full, _, _ = cal_score(permutation, df, n_pos_full, n_neg_full)

    # Bootstrap loop for AUC estimation by looking up pre-computed scores
    aucs = []
    tries = 0
    while len(aucs) < n_bootstrap:
        # Sample with replacement using 'tries' as random_state for reproducibility and distinct samples
        boot = df.sample(n=len(df), replace=True, random_state=tries)
        tries += 1

        # Skip if the sample contains only one class label
        if boot['cancer'].nunique() < 2:
            continue

        # Get the original indices of the rows in the current bootstrap sample.
        orig_indices = boot.index.copy()  

        # Reset the index of the bootstrap sample to ensure it's 0-based for proper column assignment
        boot = boot.reset_index(drop=True)

        # Pull in the pre-computed U-scores from `save_u_full` using the original indices.
        # `orig_indices` at position `i` corresponds to the row in `df` from which `boot`'s `i`-th row was sampled.
        boot['score'] = orig_indices.map(lambda r: save_u_full[r])

        # Calculate AUC for the current bootstrap sample
        auc_i = roc_auc_score(boot['cancer'], boot['score'])
        aucs.append(auc_i)

    # Calculate confidence interval using percentiles
    lower_p = (1 - confidence_level)/2 * 100
    upper_p = (1 + confidence_level)/2 * 100
    lower_bound = np.percentile(aucs, lower_p)
    upper_bound = np.percentile(aucs, upper_p)

    return np.mean(aucs), lower_bound, upper_bound, aucs

def rbl(train,val,val2,g = 0.2, initial_iter = 100, mcmc_iter = 5000):
    """
    Find the optimal permutation.

    Parameters:
        data(dataframe):input dataframe. The last column must be a binary outcome variable with values 0 and 1.
        num_val(int):the number of validation dataset to be sampled
        g(float): after g percentage of iterations, check the performance
        initial_iter (int): number of random permutations to try initially to find a strong starting point
        mcmc_iter (int): maximum number of iterations for the Markov Chain Monte Carlo (MCMC) search process to refine the permutation and maximize the score.

    Returns:
        train_auc_save(list):saved train auc 
        val_auc_save(list):saved validation auc
        val_auc_save2(list):saved validation2 auc
        save[best_score](list):best permutation
    """

    index = len(train.columns) - 1
    train_freq = train.iloc[:, index].value_counts()
    val_freq = val.iloc[:, index].value_counts()
    val2_freq = val2.iloc[:, index].value_counts()

    # Extract class frequencies for each set
    train_freq0, train_freq1 = train_freq[0], train_freq[1]
    val_freq0, val_freq1 = val_freq[0], val_freq[1]
    val2_freq0, val2_freq1 = val2_freq[0], val2_freq[1]
    
    train = train.copy()
    val = val.copy()
    val2 = val2.copy()

    # Standardize column indices
    for dataset in [train, val, val2]:
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
    
    for k in range(initial_iter):
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
            best_permu_base = cur_perm.copy() 
            auc_base_train = auc_train
            auc_base_val = auc_val
            auc_base_val2 = auc_val2

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
    
    #for early stop
    cur = float('inf')
    cur_auc = 0
    # Main loop for MCMC
    for k in range(mcmc_iter):
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

    return train_auc_save,val_auc_save,val_auc_save2,save[best_score]

def cal_auc_per_data(train,val,val2,test,g_range = [0.2],initial_iter = 100, mcmc_iter = 5000):
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
    
    for i in g_range:
        _,_,_,perm = rbl(train,val,val2, g = i, initial_iter = initial_iter, mcmc_iter = mcmc_iter)
        
        # Calculate AUC on validation set with the obtained permutation
        score_val, save_u_v, _, _ = cal_score(perm,val,val_freq1,val_freq0)
        # Update best parameter and AUC if current AUC is higher
        auc_val = cal_auc(val,perm,save_u_v)
        if auc_val > best_val:
            best_g = i
            best_val = auc_val
            best_perm = perm

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



