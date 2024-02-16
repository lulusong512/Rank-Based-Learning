#!/usr/bin/env python
# coding: utf-8

# In[197]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import random


# In[791]:


def simulation_data(random_seed, num_significant_features):
    import numpy as np
    import pandas as pd

    # Set seed for reproducibility
    np.random.seed(random_seed)

    # Define the number of features, cases, and controls
    num_cases = 30
    num_controls = 30
    num_features = 300

    # Produce case and control data
    case_data = np.zeros((num_cases, num_features))
    control_data = np.zeros((num_controls, num_features))

    # Generate indices of significant features related to the outcome
    significant_feature_indices = np.random.choice(num_features, num_significant_features, replace=False)

    # Generate data for each feature for cases and controls
    for j in range(num_features):
        # Generate different normal distribution parameters for each feature
        mean_case = np.random.uniform(50, 70)
        std_case = np.random.uniform(0, 50)
        mean_control = np.random.uniform(50, 70)  
        std_control = np.random.uniform(0, 50)

        # Generate data for cases and controls
        case_data[:, j] = np.random.normal(loc=mean_case, scale=std_case, size=num_cases)
        control_data[:, j] = np.random.normal(loc=mean_control, scale=std_control, size=num_controls)
        
        # For features significantly related to the outcome, add additional noise to enhance correlation
        if j in significant_feature_indices:
            case_data[:, j] += np.random.normal(loc=50, scale=40, size=num_cases)
            #control_data[:, j] += np.random.normal(loc=20, scale=15, size=num_controls)

    # Create DataFrame for cases and controls
    case_df = pd.DataFrame(case_data, columns=[f'feature_{i}' for i in range(num_features)])
    control_df = pd.DataFrame(control_data, columns=[f'feature_{i}' for i in range(num_features)])

    # Add the outcome column
    case_df['cancer'] = 1
    control_df['cancer'] = 0

    # Concatenate case and control datasets
    simulated_data = pd.concat([case_df, control_df], ignore_index=True)
    return case_df, control_df, simulated_data


# In[792]:


case_df, control_df, simulated_data = simulation_data(107,0)


# In[793]:


# Change values less than 0 to 0
simulated_data[simulated_data< 0] = 0
case_df[case_df< 0] = 0
control_df[control_df< 0] = 0
#simulated_data


# In[794]:


case_df


# In[795]:


# Plotting the histograms for the case group
import matplotlib.pyplot as plt

for i in range(0, case_df.shape[1]-1):
    plt.hist(case_df.iloc[:, i], bins=20, density=True, alpha=0.6, label=f"Case", color='blue')

# Plotting the histograms for the control group
for i in range(0, control_df.shape[1]-1):
    plt.hist(control_df.iloc[:, i], bins=20, density=True, alpha=0.6, label=f"Control", color='orange')

# Getting the handles and labels for the legend
handles, labels = plt.gca().get_legend_handles_labels()

# Showing only two legends with specified colors
plt.legend([handles[0],handles[301]], [labels[0],labels[301]], loc='upper right')  # Use [:2] to select the first two handles and labels

plt.xlabel('Value')
plt.ylabel('Probability')
plt.title('Histogram of Generated Data')
# Set the y-axis limit
plt.ylim(0, 4.6)
plt.show()


# In[798]:


from scipy.stats import ttest_ind
def calculate_num_significant_feature(case_df, control_df):
    """
    Calculate the number of significant features

    Parameters:
        case_df (dataframe)
        control_df (dataframe)

    Returns:
        int: the number of significant features
    """

    significant_features_p_values = []

    # Loop through each feature index
    for feature_index in range(case_df.shape[1] - 1):
        case_feature_values = case_df[f'feature_{feature_index}']
        control_feature_values = control_df[f'feature_{feature_index}']

        # Calculate the significance of the difference between case and control groups using t-test
        t_statistic, p_value = ttest_ind(case_feature_values, control_feature_values)

        # Record the result
        significant_features_p_values.append((feature_index, p_value))

    from statsmodels.stats.multitest import multipletests

    # Get p-values of significant features
    p_values = [p_value for _, p_value in significant_features_p_values]

    # Bonferroni correction
    adjusted_p_values = multipletests(p_values, method='bonferroni')[1]

    # Count the number of significant features
    num_significant_feature = 0
    for p_value in adjusted_p_values:
        if p_value < 0.05:
            num_significant_feature += 1

    return num_significant_feature



# In[755]:


"""
def score1(l1, t):
    # Import itertools module for combinations
    import itertools
    
    # Generate all possible combinations of length 2 from l1
    combinations = list(itertools.combinations(l1, 2))
    #col_map (dict): Mapping of parameter index to its value and position of t.
    col_map = {i: col for i, col in enumerate(t)}
    print(col_map)
    # Create a dictionary to map parameter index to its value and position
    dic = {}
    n = len(col_map)
    for key, value in col_map.items():
        dic[key] = [n-l1[key], value]
    print(dic)
    # Initialize the U score
    u = 0
    
    # Iterate through all combinations
    for i in combinations:  
        # Check if the values in the combination follow the specified order
        if ((dic[i[1]][0] > dic[i[0]][0]) & (dic[i[1]][1] > dic[i[0]][1])) or ((dic[i[1]][0] < dic[i[0]][0]) & (dic[i[1]][1] < dic[i[0]][1])):
            # Increment the U score if the order is correct
            u += 1
        # Check if the values in the combination follow the opposite order
        elif ((dic[i[1]][0] < dic[i[0]][0]) & (dic[i[1]][1] > dic[i[0]][1])) or ((dic[i[1]][0] > dic[i[0]][0]) & (dic[i[1]][1] < dic[i[0]][1])):
            # Decrement the U score if the order is opposite
            u += -1
        else:
            # Do nothing if the values are equal
            u += 0
    
    # Return the calculated U score
    return u
"""


# In[839]:


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
    #print(dic)

    u = 0
    for i in combinations:  
        if (( dic[i[1]][0] > dic[i[0]][0]) & (dic[i[0]][1] > dic[i[1]][1])) or ((dic[i[1]][0] < dic[i[0]][0]) & (dic[i[0]][1] < dic[i[1]][1])):
            u += 1
    #print(u,i dic[i[1]][0],dic[i[0]][0],dic[i[0]][1],dic[i[1]][1])
        if ((dic[i[1]][0] < dic[i[0]][0]) & (dic[i[0]][1] > dic[i[1]][1])) or ((dic[i[1]][0] > dic[i[0]][0]) & (dic[i[0]][1] < dic[i[1]][1])):
            u += -1
    #print(u,i[1],i[0],t[i[0]-1],t[i[1]-1])
        else:
            u += 0
        #print(u,i,dic[i[1]][0],dic[i[0]][0],dic[i[0]][1],dic[i[1]][1])
    # Return the calculated U score
    return u


# In[840]:


from joblib import Parallel, delayed
import multiprocessing

def cal_score3(l1, selected,n_case,n_control):
    """
    Calculate the U score for observations in parallel.

    Parameters:
        l1 (list): Ground truth permutation.
        selected (DataFrame): Selected data containing observations and parameters.
        n_case (int): Number of cases.
        n_control (int): Number of controls.
        score_function (function): Function to compute the U score.

    Returns:
        tuple: U score, saved U values, U case, U control.
    """
    def compute_score(r, t):
        #print(f"Col map: {col_map}")
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
    #print(results)
    u_case = sum(result[0] for result in results if result[2] == 1)/n_case
    #print(sum(result[0] for result in results if result[2] == 1),n_case)
    u_control = sum(result[0] for result in results if result[2] == 0)/n_control
    #print(sum(result[0] for result in results if result[2] == 0),n_control)
    save_u = {result[1]: result[0] for result in results}
    u = u_case - u_control
    return round(u,2), save_u, round(u_case,2), round(u_control,2)


# In[801]:


#try a simple example
a = [0,2,1]#acb


# In[802]:


# Define the data as a dictionary
data = {'a': [1,5],
        'b': [3,2],
        'c': [2,3],
       'cancer':[1,0]}
#case: bca
#control:acb
# Create the DataFrame
train = pd.DataFrame(data)
train


# In[803]:


cal_score3(a,train,1,1)


# In[804]:


l1 = [2,1,0]#cba
#l1 = [0,1,2]#abc
t = [1, 3, 2]#bca

score(l1,t)  


# In[805]:


#try a simple example
# Create a dictionary with sample data

data = {
    'Name': ['A', 'B', 'C', 'D','E','F'],
    'case1': [100, 50, 20, 10, 5, 1],
    'case2': [50, 100, 10, 5, 1, 20],
    'case3': [100, 50, 1, 5, 20, 10],
    'case4': [100, 50, 20, 1, 5, 10],
    'control1': [1, 5, 10, 20, 50, 100],
    'control2': [5, 1, 20, 10, 100, 50],
    'control3': [20, 10, 1, 5, 50, 100],
    'control4': [10, 1, 5, 20, 100, 50],
}

# Create a data frame from the dictionary
dft = pd.DataFrame(data)

# Print the data frame
print(dft)


# In[806]:


dft1 = dft.transpose()
dft1


# In[807]:


dft1.columns = dft1.iloc[0]
dft1 = dft1.drop(dft1.index[0])
dft1


# In[808]:


dft1['cancer'] = 0  # initialize the column to 0
dft1.iloc[0:4, -1] = 1
dft1


# In[809]:


a = [0,1,2,3,4,5]
cal_score3(a,dft1,4,4)


# In[810]:


dft1.iloc[0,:-1].tolist()


# In[811]:


score(a,dft1.iloc[0,:-1].tolist())


# In[812]:


import random
import time

# Initialize variables and data
a = range(0, len(dft1.columns) - 1)
my_list = list(a)

# Define a set to store generated permutations
visit = set()

# Set random seed for reproducibility
random.seed(807)
best_score, save_u, u_case, u_control = cal_score3(my_list, dft1, train_freq1, train_freq0)
#best_score = float('-inf')
save = {}
save_score = {}
save_best_u = {}
score_u = [best_score]
save_score[best_score] = my_list.copy()
# Start time
start_time = time.time()

# Iterate for a certain number of iterations
for k in range(200):
    # Randomly select two indices to swap
    i = random.randint(0, len(my_list) - 1)
    j = random.randint(0, len(my_list) - 1)
    #print(i,j)
    # Skip this iteration if permutation has been visited before
    if tuple(my_list) in visit or i == j:
        continue
    
    # Add the current permutation to the visit set
    visit.add(tuple(my_list))
    
    # Swap elements at indices i and j
    my_list[i], my_list[j] = my_list[j], my_list[i]
    
    # Calculate score and other metrics
    score1, save_u, u_case, u_control = cal_score3(my_list, dft1, train_freq1, train_freq0)
    score_u.append(score1)
    # Update best score if applicable
    if score1 > best_score:
        best_score = score1
        save_score[best_score] = my_list.copy()
        save_best_u = save_u.copy()
    
    # Print iteration and time information
    if k % 100 == 0:
        current_time = time.time() - start_time
        print("Iteration:", k)
        print("Time taken:", current_time, "seconds")

# End time
end_time = time.time()
total_time = end_time - start_time
print("Best score:", best_score)
print("Total time taken:", total_time, "seconds")


# In[813]:


best_score


# In[814]:


save_score


# In[815]:


save_score[best_score]


# In[817]:


from sklearn.metrics import roc_curve, roc_auc_score
#calculate AUC 
def cal_auc(data,permutation,save_u):
    """
    Parameters:
        data(dataframe):input dataframe
        permutation(list):permutation of a list
        save_u(dic): U values for each observation

    Returns:
        AUC(int)
    """
    subset= data.iloc[:, permutation]
    subset['cancer'] = list(data['cancer'])
    subset['score'] = list(save_u.values())

    fpr, tpr, thresholds = roc_curve(subset['cancer'],subset['score'])
    auc = roc_auc_score(subset['cancer'],subset['score'])
    return auc


# In[837]:


def rbl(data,num_val,g):
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
    import random
    X_train, test = train_test_split(data, test_size=0.3, random_state=42)
    # Split temp set into validation and test sets (70% train, 30% validation)
    train, val = train_test_split(X_train, test_size=0.3, random_state=4)
    
    # Randomly sample validation data
    val2 = train.sample(num_val,random_state=807)
    # Calculate class frequencies
    train_freq = train['cancer'].value_counts()
    val_freq = val['cancer'].value_counts()
    val2_freq = val2['cancer'].value_counts()
    test_freq = test['cancer'].value_counts()
    # Extract class frequencies for each set
    train_freq0 = train_freq[0]
    train_freq1 = train_freq[1]
    val_freq0 = val_freq[0]
    val_freq1 = val_freq[1]
    val2_freq0 = val2_freq[0]
    val2_freq1 = val2_freq[1]
    test_freq0 = test_freq[0]
    test_freq1 = test_freq[1]
    
    #find a better initilized score
    # Initialize lists to store scores and AUCs
    a = range(0,len(train.columns)-1)
    my_list = list(a)
    score_save_b = []
    train_auc_b = []
    score_u_v_b = []
    val_auc_b = []
    score_u_v_b2 = []
    val_auc_b2 = []
    auc_base_train = 0
    auc_base_val = 0
    auc_base_val2 = 0
    best_permu_base = []
    import time
    start_time = time.time()  # Record the start time

    for k in range(0,30):
        random.seed(k)
        random.shuffle(my_list)
        # Calculate scores and AUCs
        score_b,save_b,u_case_b,u_control_b = cal_score3(my_list,train,train_freq1,train_freq0)
        score_save_b.append(score_b)
        auc_train = cal_auc(train,my_list,save_b)
        train_auc_b.append(round(auc_train,2))

        score_val_b,save_u_v_b,u_case_v_b,u_control_v_b = cal_score3(my_list,val,val_freq1,val_freq0)
        score_u_v_b.append(score_val_b)
        auc_val = cal_auc(val,my_list,save_u_v_b)
        val_auc_b.append(round(auc_val,2))

        score_val_b2,save_u_v_b2,u_case_v_b2,u_control_v_b2 = cal_score3(my_list,val2,val2_freq1,val2_freq0)
        score_u_v_b2.append(score_val_b2)
        auc_val2 = cal_auc(val2,my_list,save_u_v_b2)
        val_auc_b2.append(round(auc_val2,2))
        
        # Check if scores and AUCs meet conditions
        if auc_train > 0.5 and auc_val > 0.5 and auc_val2 > 0.5 and ((auc_train+auc_val+auc_val2)/3 > (auc_base_train+auc_base_val+auc_base_val2)/3):
            #print(score_b)
            print(auc_train)
            print(auc_val)
            print(auc_val2)
            #print(cal_score3(my_list,train,17,16))
            auc_base_train = auc_train
            auc_base_val = auc_val
            auc_base_val2 = auc_val2
            #print(my_list)
            best_permu_base = my_list.copy() 
            #print(k)
            best_save_b = save_b
    

    #add early stop
    #train
    #import random
    
    a = range(0,len(train.columns)-1)
    #change initial list as previous permutation
    #if find a good initilization
    if best_permu_base:
        my_list = best_permu_base
    else:
        random.seed(200)
        random.shuffle(my_list)
        my_list = list(a)
    """
    my_list = list(a)
    random.seed(807)
    random.shuffle(my_list)
    """
    # Initialize variables for MCMC
    best_score = float('-inf')

    #best_converted = 0
    save = {}
    save_score = []
    save_u = {}
    #save_best_u = {}

    prob = 0
    beta_range = [0.1,1,5,10]
    score0 = float('-inf')
    #val
    score_val0 = float('-inf')
    score_val20 = float('-inf')
    best_beta = 1
    beta = 0
    #use visit to list to avoid duplication
    visit = set()
    score_u = []
    score_case = []
    score_control = []
    train_auc_save = []

    score_u_v = []
    score_case_v = []
    score_control_v = []
    val_auc_save = []

    score_u_v2 = []
    score_case_v2 = []
    score_control_v2 = []
    val_auc_save2 = []

    score_u_t = []
    score_case_t = []
    score_control_t = []
    test_auc_save = []

    #for early stop
    cur = float('inf')
    cur_auc = 0
    # Main loop for MCMC
    for k in range(2001):
        random.seed(k)
        
        i = random.randint(0, len(my_list) - 1)
        j = random.randint(0, len(my_list) - 1)

        #swap the two elements
        my_list[i], my_list[j] = my_list[j], my_list[i]
        #print('a'+ str(my_list))
        
            #skip this i and j if they have been checked before
        if tuple(my_list) in visit or i == j:
            continue
        # Add the current permutation to the visit set
        visit.add(tuple(my_list))
        
        #save train
        score1,save_u,u_case,u_control = cal_score3(my_list,train,train_freq1,train_freq0)
        score_u.append(score1)
        score_case.append(u_case)
        score_control.append(u_control)
        auc_train = cal_auc(train,my_list,save_u)
        #print("train auc:", auc_train)
        train_auc_save.append(round(auc_train,2))

        #save val
        score_val,save_u_v,u_case_v,u_control_v = cal_score3(my_list,val,val_freq1,val_freq0)
        score_u_v.append(score_val)
        score_case_v.append(u_case_v)
        score_control_v.append(u_control_v)
        auc_val = cal_auc(val,my_list,save_u_v)
        #print("val auc:", auc_val)
        val_auc_save.append(round(auc_val,2))

        #save val2
        score_val2,save_u_v2,u_case_v2,u_control_v2 = cal_score3(my_list,val2,val2_freq1,val2_freq0)
        score_u_v2.append(score_val2)
        score_case_v2.append(u_case_v2)
        score_control_v2.append(u_control_v2)
        auc_val2 = cal_auc(val2,my_list,save_u_v2)
        #print("val auc2:", auc_val2)
        val_auc_save2.append(round(auc_val2,2))

        #save test
        score_t,save_u_t,u_case_t,u_control_t = cal_score3(my_list,test,test_freq1,test_freq0)
        score_u_t.append(score_t)
        score_case_t.append(u_case_t)
        score_control_t.append(u_control_t)
        auc_test = cal_auc(test,my_list,save_u_t)
        #print("test auc:", auc_test)
        test_auc_save.append(round(auc_test,2))

        #MCMC
        if (score1 > score0) and (score_val > score_val0) and (score_val2 > score_val20):
            #swap i and j
            #my_list[i], my_list[j] = my_list[j], my_list[i]
            #prob = 1
            #update scores
            score0 = score1
            score_val0 = score_val
            score_val20 = score_val2

            save_best_u = []
            best_score = score1
            best_score0 = score_val
            #best_converted = i
            my_list_copy = my_list.copy()
            save[best_score] = my_list_copy
            #print(len(my_list_copy))
            save_score.append(best_score)
            save_best_u = save_u   

            best_beta = beta
        else:
            my_list[i], my_list[j] = my_list[j], my_list[i]
        #early stop
        if k>0 and k%500 == 0:
            print(k)
            cur = k
            cur_auc = auc_train
            #print(k * 0.3)
        #print(cur_auc, auc_val)
        if k >= cur + cur * g and cur_auc >= auc_train:
            print(f"Stopping at iteration {k} because AUC did not increase.")
            break

        #print(k)
        if k %100 == 0:
            current_time = time.time() - start_time  # Calculate the elapsed time
            print("Iteration:", k)
            print("Time taken:", current_time, "seconds")

    end_time = time.time()  # Record the end time
    total_time = end_time - start_time  # Calculate the total time

    print("Total time taken:", total_time, "seconds")        
    
    return train_auc_save,val_auc_save,val_auc_save2,test_auc_save,save[best_score]


# In[841]:


rbl(simulated_data, num_val=15, g=0.2)


# In[829]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import random
import time
def cal_auc_per_data(df,num_val,split_per):
    """
    Parameters:
        data(dataframe):input dataframe
        num_val(int):the number of validation dataset to be sampled
        split_per(float): the percentage to split train and test data

    Returns:
        AUC(int): train AUC
        AUC(int): test AUC
    """
    # Split data into train, validation, and test sets
    X_train, test = train_test_split(df, test_size=split_per, random_state=42)
    # Split temp set into validation and test sets (70% train, 30% validation)
    train, val = train_test_split(X_train, test_size=0.3, random_state=4)
    # Randomly sample validation data
    val2 = train.sample(num_val,random_state=807)
    # Calculate class frequencies
    train_freq = train['cancer'].value_counts()
    val_freq = val['cancer'].value_counts()
    val2_freq = val2['cancer'].value_counts()
    test_freq = test['cancer'].value_counts()
    # Extract class frequencies for each set
    train_freq0 = train_freq[0]
    train_freq1 = train_freq[1]
    val_freq0 = val_freq[0]
    val_freq1 = val_freq[1]
    val2_freq0 = val2_freq[0]
    val2_freq1 = val2_freq[1]
    test_freq0 = test_freq[0]
    test_freq1 = test_freq[1]

    # Initialize variables to track best parameters and results
    best_g = 0
    best_val = 0
    g_range = [0.2,0.3]
    for i in g_range:
        a,b,c,d,e = rbl(df,num_val,i)
        #print(e)
        # Calculate AUC on validation set with the obtained permutation
        score_val,save_u_v,u_case_v,u_control_v = cal_score3(e,val,val_freq1,val_freq0)
        # Update best parameter and AUC if current AUC is higher
        auc_val = cal_auc(val,e,save_u_v)
        if auc_val > best_val:
            best_g = i
            best_val = auc_val
            best_perm = e
            print("best_g: "+ str(i))
            print("best_val: "+ str(best_val))
            # If AUC is perfect, stop optimization
            if best_val == 1.0:
                break

    print('best permutation: '+ str(best_perm))
    #calculate train auc
    score1,save_u,u_case,u_control = cal_score3(best_perm,train,train_freq1,train_freq0)
    train_auc = cal_auc(train,best_perm,save_u)
    #calculate test auc
    score_t,save_u_t,u_case_t,u_control_t = cal_score3(best_perm,test,test_freq1,test_freq0)
    test_auc = cal_auc(test,best_perm,save_u_t)
    
    return train_auc, test_auc


# In[842]:


cal_auc_per_data(simulated_data, num_val=15, split_per=0.3)


# In[843]:


#compare with LR
def LR_SF(n_fea,trainX,testX,trainY,testY):
    # Create a logistic regression model
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=10000)
    #select features
    from sklearn.feature_selection import RFE
    from sklearn.metrics import roc_auc_score
    
    rfe = RFE(model, n_features_to_select=n_fea)
    rfe.fit(trainX,trainY)
    true_indices = [index for index, value in enumerate(list(rfe.support_)) if value == True]
    selected_elements = [trainX.columns[i] for i in true_indices]
    X_train1 = trainX[selected_elements]
    X_test1 = testX[selected_elements]
    #fit LR on selected dataset
    LR = LogisticRegression(penalty=None,random_state=42,max_iter=10000,solver='saga')
    LR.fit(X_train1, trainY)
    y_pred_train =LR.predict_proba(X_train1)[:, 1]  # Probabilities of positive class
    train_auc = roc_auc_score(trainY, y_pred_train)
    #print("Train AUC:", train_auc)
    y_pred_test =LR.predict_proba(X_test1)[:, 1]  # Probabilities of positive class
    test_auc = auc = roc_auc_score(testY, y_pred_test)
    #print("Validation AUC:", test_auc)
    return train_auc,test_auc


# In[844]:


def LR_SIMU(df,split_per):
    df_lr = df.fillna(0)
    y = df_lr['cancer']
    X = df_lr.drop('cancer', axis=1)
    trainX,testX,trainY, testY = train_test_split(X,y, test_size=split_per, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(trainX,trainY, test_size=0.3, random_state=4)
    from sklearn.linear_model import LogisticRegression
    best_auc = 0
    best_n = 0
    for n in range(1,11):
        train_auc,val_auc = LR_SF(n, X_train, X_val, Y_train, Y_val)
        if val_auc > best_auc:
            best_n = n
            #print('number of features: ' + str(n))
            best_auc = val_auc
            #print(best_auc)
    
    from sklearn.feature_selection import RFE
    model = LogisticRegression(max_iter=10000)
    print('best_n: ' + str(best_n))
    rfe = RFE(model, n_features_to_select=best_n)
    rfe.fit(X_train,Y_train)
    true_indices = [index for index, value in enumerate(list(rfe.support_)) if value == True]
    selected_elements = [X_train.columns[i] for i in true_indices]
    X_train1 = X_train[selected_elements]
    X_test1 = testX[selected_elements]
    #fit LR on selected dataset
    LR = LogisticRegression(penalty=None,random_state=42,max_iter=10000,solver='saga')
    LR.fit(X_train1, Y_train)
    y_pred_train =LR.predict_proba(X_train1)[:, 1]  # Probabilities of positive class
    train_auc = roc_auc_score(Y_train, y_pred_train)
    #print("Train AUC:", round(train_auc,2))
    y_pred_test =LR.predict_proba(X_test1)[:, 1]  # Probabilities of positive class
    test_auc = auc = roc_auc_score(testY, y_pred_test)
    #print("Test AUC:", round(test_auc,2))
    save = {}
    save['train_auc'] = train_auc
    save['test_auc'] = test_auc
    return save


# In[845]:


#calculate AUCA for logistic regression
def auc_lr(seed_range,num_significant_features):
    save_auc_lr = {}
    for i in seed_range:
        print(f"\nRun {i}:")
        case_df, control_df, simulated_data = simulation_data(i,num_significant_features)
        simulated_data[simulated_data< 0] = 0
        save_auc_lr[i] = {}
        auc = LR_SIMU(simulated_data,0.3) 
        #print(auc)
        train_auc, test_auc = auc['train_auc'], auc['test_auc']
        save_auc_lr[i]['train_auc'] = round(train_auc,2)
        save_auc_lr[i]['test_auc'] = round(test_auc,2)
        # Print or use the results as needed
        print(f"Train AUC for DataFrame {i }: {round(train_auc,2)}")
        print(f"Test AUC for DataFrame {i }: {round(test_auc,2)}")
    return save_auc_lr 


# In[846]:


n = 0
seed_range = []
for i in range(1000):
    #print(i)
    np.random.seed(i)
    case_df, control_df, simulated_data = simulation_data(i,num_significant_features=100)
    simulated_data[simulated_data< 0] = 0
    case_df[case_df< 0] = 0
    control_df[control_df< 0] = 0
    num_significant_feature = calculate_num_significant_feature(case_df,control_df)
    print(num_significant_feature)
    if num_significant_feature == 30:
        seed_range.append(i)
        n += 1
        print("n:"+str(n))
        if n == 10:
            break


# In[847]:


#calculate AUC for rank based learning
def auc_rbl(seed_range,num_significant_features):
    save_auc = {}
    for i in seed_range:
        print(f"\nRun {i}:")
        case_df, control_df, simulated_data = simulation_data(i,num_significant_features)
        simulated_data[simulated_data< 0] = 0
        save_auc[i] = {}
        train_auc, test_auc = cal_auc_per_data(simulated_data, num_val=15, split_per=0.3)
        save_auc[i]['train_auc'] = round(train_auc,2)
        save_auc[i]['test_auc'] = round(test_auc,2)
        # Print or use the results as needed
        print(f"Train AUC for DataFrame {i }: {round(train_auc,2)}")
        print(f"Test AUC for DataFrame {i }: {round(test_auc,2)}")
    return save_auc


# In[824]:


simulated_data


# In[848]:


def find_seed(real_num_feature,num_significant_features):
    n = 0
    seed_range = []
    for i in range(1000):
        #print(i)
        np.random.seed(i)
        case_df, control_df, simulated_data = simulation_data(i,num_significant_features)
        simulated_data[simulated_data< 0] = 0
        case_df[case_df< 0] = 0
        control_df[control_df< 0] = 0
        num_significant_feature = calculate_num_significant_feature(case_df,control_df)
        #print(num_significant_feature)
        if num_significant_feature == real_num_feature:
            seed_range.append(i)
            n += 1
            print("n:"+str(n))
            if n == 10:
                break
    return seed_range,num_significant_feature


# In[849]:


real_num_feature = 10


# In[850]:


seed_range,num_significant_feature = find_seed(real_num_feature ,0)
seed_range


# In[88]:


seed_range


# In[ ]:


#print output
import sys
sys.stdout = open('/rsrch5/home/biostatistics/lsong3/SIMU_TD_OUTPUT.log', 'w')
sys.stderr = open('/rsrch5/home/biostatistics/lsong3/SIMU_TD_ERROR.log', 'w')


# In[263]:


#30 sig features
real_num_feature = 30
seed_range30,num_significant_feature30 = find_seed(real_num_feature ,30)
seed_range30


# In[223]:


print('real_num_feature_lr:'+ str(real_num_feature))
save_auc_lr30 = auc_lr(seed_range30,num_significant_feature30)
print(save_auc_lr30)


# In[229]:


df_result = pd.DataFrame(save_auc_lr30)
df_result['auc_avg'] = df_result.mean(axis=1)
df_result


# In[172]:


print('real_num_feature_rbl:'+ str(real_num_feature))
save_auc30 = auc_rbl(seed_range30,num_significant_feature30)
print(save_auc30)


# In[265]:


num_significant_feature30


# In[235]:


save_auc30 = {89: {'train_auc': 1.0, 'test_auc': 0.98}, 252: {'train_auc': 1.0, 'test_auc': 1.0}, 483: {'train_auc': 1.0, 'test_auc': 1.0}, 505: {'train_auc': 1.0, 'test_auc': 1.0}, 608: {'train_auc': 1.0, 'test_auc': 1.0}, 626: {'train_auc': 1.0, 'test_auc': 0.99}, 664: {'train_auc': 1.0, 'test_auc': 0.98}, 745: {'train_auc': 1.0, 'test_auc': 1.0}, 784: {'train_auc': 1.0, 'test_auc': 1.0}, 983: {'train_auc': 1.0, 'test_auc': 1.0}}
df_result = pd.DataFrame(save_auc30)
df_result['auc_avg'] = df_result.mean(axis=1)
df_result


# In[176]:


#50 sig features
real_num_feature = 50
seed_range50,num_significant_feature50 = find_seed(real_num_feature ,150)
seed_range50 


# In[224]:


print('real_num_feature_lr:'+ str(real_num_feature))
save_auc_lr50 = auc_lr(seed_range50,num_significant_feature50)
print(save_auc_lr50)


# In[230]:


df_result = pd.DataFrame(save_auc_lr50)
df_result['auc_avg'] = df_result.mean(axis=1)
df_result


# In[178]:


print('real_num_feature_rbl:'+ str(real_num_feature))
save_auc50 = auc_rbl(seed_range50,num_significant_feature50)
print(save_auc50)


# In[236]:


save_auc50 = {46: {'train_auc': 1.0, 'test_auc': 1.0}, 139: {'train_auc': 1.0, 'test_auc': 1.0}, 196: {'train_auc': 1.0, 'test_auc': 0.96}, 213: {'train_auc': 1.0, 'test_auc': 0.98}, 296: {'train_auc': 1.0, 'test_auc': 1.0}, 514: {'train_auc': 1.0, 'test_auc': 0.98}, 530: {'train_auc': 1.0, 'test_auc': 0.89}, 632: {'train_auc': 1.0, 'test_auc': 0.98}, 659: {'train_auc': 0.99, 'test_auc': 0.91}, 808: {'train_auc': 1.0, 'test_auc': 0.95}}

df_result = pd.DataFrame(save_auc50)
df_result['auc_avg'] = df_result.mean(axis=1)
df_result


# In[182]:


#70 sig features
real_num_feature = 70
seed_range70,num_significant_feature70 = find_seed(real_num_feature ,260)
seed_range70


# In[225]:


print('real_num_feature_lr:'+ str(real_num_feature))
save_auc_lr70 = auc_lr(seed_range70,num_significant_feature70)
print(save_auc_lr70)


# In[231]:


df_result = pd.DataFrame(save_auc_lr70)
df_result['auc_avg'] = df_result.mean(axis=1)
df_result


# In[ ]:


print('real_num_feature_rbl:'+ str(real_num_feature))
save_auc70 = auc_rbl(seed_range70,num_significant_feature70)
print(save_auc70)


# In[237]:


save_auc70 = {36: {'train_auc': 1.0, 'test_auc': 0.98}, 72: {'train_auc': 1.0, 'test_auc': 0.98}, 90: {'train_auc': 1.0, 'test_auc': 0.82}, 204: {'train_auc': 1.0, 'test_auc': 0.96}, 328: {'train_auc': 1.0, 'test_auc': 0.96}, 346: {'train_auc': 1.0, 'test_auc': 1.0}, 565: {'train_auc': 1.0, 'test_auc': 0.94}, 696: {'train_auc': 0.99, 'test_auc': 0.98}, 700: {'train_auc': 0.99, 'test_auc': 1.0}, 734: {'train_auc': 1.0, 'test_auc': 1.0}}
df_result = pd.DataFrame(save_auc70)
df_result['auc_avg'] = df_result.mean(axis=1)
df_result


# In[189]:


#100 sig features
real_num_feature = 100
seed_range100,num_significant_feature100 = find_seed(real_num_feature ,120)
seed_range100


# In[226]:


print('real_num_feature_lr:'+ str(real_num_feature))
save_auc_lr100 = auc_lr(seed_range100,num_significant_feature100)
print(save_auc_lr100)


# In[232]:


df_result = pd.DataFrame(save_auc_lr100)
df_result['auc_avg'] = df_result.mean(axis=1)
df_result


# In[191]:


print('real_num_feature_rbl:'+ str(real_num_feature))
save_auc100 = auc_rbl(seed_range100,num_significant_feature100)
print(save_auc100)


# In[238]:


save_auc100 = {21: {'train_auc': 1.0, 'test_auc': 1.0}, 64: {'train_auc': 1.0, 'test_auc': 1.0}, 73: {'train_auc': 1.0, 'test_auc': 1.0}, 98: {'train_auc': 1.0, 'test_auc': 1.0}, 104: {'train_auc': 1.0, 'test_auc': 1.0}, 114: {'train_auc': 1.0, 'test_auc': 1.0}, 115: {'train_auc': 1.0, 'test_auc': 1.0}, 146: {'train_auc': 1.0, 'test_auc': 1.0}, 195: {'train_auc': 1.0, 'test_auc': 1.0}, 198: {'train_auc': 1.0, 'test_auc': 1.0}}

df_result = pd.DataFrame(save_auc100)
df_result['auc_avg'] = df_result.mean(axis=1)
df_result


# In[194]:


#150 sig features
real_num_feature = 150
seed_range150,num_significant_feature150 = find_seed(real_num_feature ,175)
seed_range150


# In[227]:


print('real_num_feature_lr:'+ str(real_num_feature))
save_auc_lr150 = auc_lr(seed_range150,num_significant_feature150)
print(save_auc_lr150)


# In[233]:


df_result = pd.DataFrame(save_auc_lr150)
df_result['auc_avg'] = df_result.mean(axis=1)
df_result


# In[ ]:


print('real_num_feature_rbl:'+ str(real_num_feature))
save_auc150 = auc_rbl(seed_range150,num_significant_feature150)
print(save_auc150)


# In[239]:


save_auc150 = {23: {'train_auc': 1.0, 'test_auc': 1.0}, 58: {'train_auc': 1.0, 'test_auc': 1.0}, 61: {'train_auc': 1.0, 'test_auc': 1.0}, 83: {'train_auc': 1.0, 'test_auc': 1.0}, 89: {'train_auc': 1.0, 'test_auc': 1.0}, 129: {'train_auc': 1.0, 'test_auc': 1.0}, 131: {'train_auc': 1.0, 'test_auc': 1.0}, 168: {'train_auc': 1.0, 'test_auc': 1.0}, 188: {'train_auc': 1.0, 'test_auc': 1.0}, 213: {'train_auc': 1.0, 'test_auc': 1.0}}
df_result = pd.DataFrame(save_auc150)
df_result['auc_avg'] = df_result.mean(axis=1)
df_result


# In[196]:


#200 sig features
real_num_feature = 200
seed_range200,num_significant_feature200 = find_seed(real_num_feature ,250)
seed_range200


# In[228]:


print('real_num_feature_lr:'+ str(real_num_feature))
save_auc_lr200 = auc_lr(seed_range200,num_significant_feature200)
print(save_auc_lr200)


# In[234]:


df_result = pd.DataFrame(save_auc_lr200)
df_result['auc_avg'] = df_result.mean(axis=1)
df_result


# In[ ]:


print('real_num_feature_rbl:'+ str(real_num_feature))
save_auc200 = auc_rbl(seed_range200,num_significant_feature200)
print(save_auc200)


# In[240]:


save_auc200 = {5: {'train_auc': 1.0, 'test_auc': 1.0}, 12: {'train_auc': 1.0, 'test_auc': 0.99}, 47: {'train_auc': 1.0, 'test_auc': 1.0}, 57: {'train_auc': 1.0, 'test_auc': 1.0}, 62: {'train_auc': 1.0, 'test_auc': 1.0}, 85: {'train_auc': 0.99, 'test_auc': 0.99}, 104: {'train_auc': 1.0, 'test_auc': 1.0}, 106: {'train_auc': 1.0, 'test_auc': 1.0}, 111: {'train_auc': 1.0, 'test_auc': 1.0}, 113: {'train_auc': 1.0, 'test_auc': 1.0}}

df_result = pd.DataFrame(save_auc200)
df_result['auc_avg'] = df_result.mean(axis=1)
df_result


# In[103]:


"""
print('real_num_feature:'+ str(real_num_feature))
save_auc_lr = auc_lr(seed_range,num_significant_features)
print(save_auc_lr)
"""


# In[104]:


#save_auc_lr


# In[ ]:


#print(save_auc_lr)


# In[99]:


"""
lr_res = pd.DataFrame(save_auc_lr)
lr_res
"""


# In[100]:


"""
lr_res['auc_avg'] = lr_res.mean(axis=1)
lr_res
"""


# In[105]:





# In[106]:





# In[96]:


"""
save_auc
rbl_res = pd.DataFrame(save_auc)
rbl_res
"""


# In[98]:


"""
rbl_res['auc_avg'] = rbl_res.mean(axis=1)
rbl_res
"""


# In[ ]:




