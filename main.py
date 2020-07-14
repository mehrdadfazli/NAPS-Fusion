# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:17:55 2019

@author: 14342
"""
from __future__ import absolute_import, division, print_function
from pyds_local import *
from Naive_Adaptive_Sensor_Fusion import *
from config import *

from timeit import default_timer as timer
import random
import glob
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import warnings
from imblearn.over_sampling import SMOTE
import timeit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from multiprocessing import Pool, TimeoutError
#import tensorflow as tf

warnings.filterwarnings("ignore")


def readdata_csv(data_dir):
    """This function gets the directory of the datasets and returns the dataset
    containing information of all 60 users
    
    Input:
        data_dir[string]: holds the directory of all the csv files (60)
        
    Output:
        grand_dataset[dict]: a dictionary of all the users' data. The format is:
            grand_dataset:{'uuid1': dataframe of csv file of the user 1
                           'uuid2': dataframe of csv file of the user 2
                           'uuid3': dataframe of csv file of the user 3
                           ...}
    """
    length_uuids = 36 # number of characters for each uuid
    data_list = glob.glob(os.path.join(os.getcwd(), data_dir, "*.csv"))
    # grand_dataset is a dict. that holds the uuids and correspondong datast
    grand_dataset = {}
    for i in range(len(data_list)):
#    for i in range(5):
        # dismantles the file name and picks only uuids (first 36 characters)
        uuid = os.path.basename(data_list[i])[:length_uuids]    
        dataset_ith = pd.read_csv(data_list[i])
        print('User {}/{}  -> Shape of the data     {}'.format(i+1, \
              len(data_list), dataset_ith.shape))
        grand_dataset[uuid] = dataset_ith
    return(grand_dataset)
    
def Set_Act_Sens():
    """This function defines two dictionaries for activities and sensors. Each
    dictionaray holds the the range of columns for the specified sensor or 
    activity.
    
    Input:
    Output:
        Activities[dict]: a dictionary of the activities and their corresponding
                        column number
        Sensors[dict]: a dictionary of the sensors and their corresponding range
                        of features
    """
    Activities = {}
    Activities['label:LYING_DOWN'] = 226
    Activities['label:SITTING'] = 227
    Activities['label:FIX_walking'] = 228
    Activities['label:FIX_running'] = 229
    Activities['label:BICYCLING'] = 230
    Activities['label:SLEEPING'] = 231
    Activities['label:OR_standing'] = 270
    
    
    Sensors = {}
    Sensors['Acc'] = list(range(1,27))
    Sensors['Gyro'] = list(range(27,53))
#    Sensors['Mag'] = list(range(53,84))
    Sensors['W_acc'] = list(range(84,130))
#    Sensors['Compass'] = list(range(130,139))
    Sensors['Loc'] = list(range(139,156))
    Sensors['Aud'] = list(range(156,182))
#    Sensors['AP'] = list(range(182,184))
    Sensors['PS'] = list(np.append(range(184,210),range(218,226)))
#    Sensors['LF'] = list(range(210,218))
    
    return(Activities,Sensors)
    
def Response_Merger(data, cols_to_merge):
    """
    This function takes in the dataset and a list of columns of different labels 
    to merge and combnine them using a logical OR to give back one column. ex. 
    l1+l2+l3 -> {l1,l2,l3}

    Parameters
    ----------
    data : dataframe
        dataframe of the dataset (ex. training data).
    cols_to_merge : list
        a list of the columns to merge with a logical OR. like:
        ['Lying_down','Sleeping'].

    Returns
    -------
    merged_label: dataframe
        a dataframe with only one column whose values are binary

    """
    data = data[cols_to_merge].fillna(0)
    
    merged_label = data[cols_to_merge[0]]
    
    for i in range(1,len(cols_to_merge)):
        merged_label = np.logical_or(merged_label, data[cols_to_merge[i]])*1
    
    merged_label = merged_label.to_frame()
    merged_label.columns=['Merged_label']

     
#    col_name = ''.join(cols_to_merge[:])
#    cols_to_merge = add_label(cols_to_merge)
#        
#    # First we impute the NaN with 0 in all the columns that are about to be merged
#    data = data[cols_to_merge].fillna(0)
#    
#    # Now find the logical OR of the desired columns (labels)
#    merged_label = data[cols_to_merge[0]]
#    merged_label.name = col_name
#    
#    for i in range(len(cols_to_merge)):
#        merged_label = np.logical_or(merged_label, data[cols_to_merge[i]])
    return(merged_label)

def Xy(data, feature_sets, feature_set_idx, response_perms_1, response_perms_2,\
       response_perm_idx, impute = False):
    """This function takes data, feature sets matrix, respnse perms, the index
       of the row of the desired feature set, and the index of the desired rows
       of the response variables and gives back the X and y"""
    # Maybe add sorting later
    # Maybe some more manupulations on response variable
    X = data.iloc[:,list(feature_sets[feature_set_idx,:])]
#    X.loc[:,:] = preprocessing.scale(X)
    
    if impute is not False:
        X = X.fillna(0)
    
    y1 = Response_Merger(data, response_perms_1[response_perm_idx])
    y2 = Response_Merger(data, response_perms_2[response_perm_idx])
    
    aux = y1 + y2
    indices = np.where(aux ==1)[0]
    
    if len(indices) > 1:
        y1 = y1.loc[indices,:]
        y2 = y2.loc[indices,:]
        X = X.loc[indices,:]
    
    y = y1
    
    xy = pd.concat([X,y],axis=1)
    xy = xy.dropna()
    
    y = xy.iloc[:,-1]
    X = xy.iloc[:,0:-1]
    
    return(X, y, y1, y2)

def train_test_spl(test_fold, num_folds, fold_dir, grand_dataset):
    """This function takes the number of test fold (ranging from 0 to 4) and
    number of folds (in this case 5) and directory where the folds' uuids are
    and the dataset, and returns train and test datasets
    
    Input:
        test_fold_idx[integer]: an integer indicating the index of the test fold
        fold_dir[string]: holds the directory in which the folds' uuids are
        grand_dataset[dict]: a dictionary of all users' data. (essentially the
                             output of readdata_csv())
    Output:
        train_dataset[pandas.dataframe]: dataframe of the train dataset
        test_dataset[pandas.dataframe]: dataframe of the test dataset
    """
    train_dataset = pd.DataFrame()
    test_dataset = pd.DataFrame()
    folds_uuids = get_folds_uuids(fold_dir)
    
    # Dividing the folds uuids into train and test (the L denotes they are still lists)
    test_uuids_L = [folds_uuids[test_fold]]
    del(folds_uuids[test_fold])
    train_uuids_L = folds_uuids
    
    # Transforming the list of arrays of uuids into a single uuids np.array
    test_uuids = np.vstack(test_uuids_L)
    train_uuids = np.vstack(train_uuids_L)
    
    # Now collecting the test and train dataset using concatenating
    for i in train_uuids:
        train_dataset = pd.concat([train_dataset,grand_dataset[i[0]]], axis=0, \
                                  ignore_index=True)
    
    for j in test_uuids:
        test_dataset = pd.concat([test_dataset,grand_dataset[j[0]]], axis=0, \
                                 ignore_index=True)
        
    return(train_dataset,test_dataset)

def get_folds_uuids(fold_dir):
    """
    The function gets the directory where the the folds text files are located
    and returns a list of five np.arrays in each of them the uuids of the
    corresponding fold are stored.
    
    Input:
        fold_dir[string]: holds the directory in which folds are
    
    Output:
        folds_uuids[list]: a list of numpy arrays. Each array holds the uuids
                    in that fold. ex.
                    folds_uuids = [('uuid1','uuid2',...,'uuid12'),
                                   ('uuid13','uuid14',...,'uuid24'),
                                   ...,
                                   ('uuid49','uuid50',...,'uuid60')]
    """
    num_folds = 5
    # folds_uuids is gonna be a list of np.arrays. each array is a set of uuids
    folds_uuids = [0,1,2,3,4]
    # This loop reads all 5 test folds (iphone and android) and stores uuids
    for i in range(0,num_folds):
        filename = 'fold_{}_test_android_uuids.txt'.format(i)
        filepath = os.path.join(fold_dir, filename)
        # aux1 is the uuids of ith test fold for "android"
        aux1 = pd.read_csv(filepath,header=None,delimiter='\n')
        aux1 = aux1.values
        
        filename = 'fold_%s_test_iphone_uuids.txt' %i
        filepath = os.path.join(fold_dir, filename)
        # aux2 is the uuids of ith test fold for "iphone"
        aux2 = pd.read_csv(filepath,header=None,delimiter='\n')
        aux2 = aux2.values
        
        # Then we concatenate them
        folds_uuids[i] = np.concatenate((aux1,aux2),axis=0)
        
    return(folds_uuids)

def train_NAPS_Models(train_dataset, feature_sets, j, Response_Perm_1, \
                      Response_Perm_2, i, bagging_R, num_bags, impute = True):
    
    t0 = timeit.default_timer()
    X_train, y_train, y1, y2 = Xy(train_dataset, feature_sets, j, \
                        Response_Perm_1, Response_Perm_2, i, impute = True)
    
    t1 = timeit.default_timer()
    print('\tgetting training data took : ', t1-t0)
    
    X_train_tmp, y_train_tmp = smt.fit_sample(X_train, y_train)
    t2 = timeit.default_timer()
    print('\tSMOTE took : ', t2-t1)
    
    X_train = pd.DataFrame(X_train_tmp, columns = X_train.columns)
    y_train = pd.DataFrame(y_train_tmp, columns = ['Merged_label'])
    
    
    U2 = Uncertainty_Bias([y1, y2])
    
    t3 = timeit.default_timer()
    #create a model and train it
    NAPS_sample = DS_Model(Response_Perm_1[i], Response_Perm_2[i], \
               X_train, y_train, j)
    t4 = timeit.default_timer()
    print('\tCreating the DS model took : ', t4-t3)
    
    NAPS_sample.Bags_Trainer(X_train, y_train, bagging_R, num_bags)
    t5 = timeit.default_timer()
    print('\tTraining bags took : ', t5-t4)
    
    NAPS_sample.Uncertainty_B = U2
    
    return(NAPS_sample)

#=============================================================================#
#--------------------------| Tensorflow for LR |------------------------------#
#=============================================================================#
 
# num_classes = 10 # 0 to 9 digits
# # num_features = 784 # 28*28
# learning_rate = 0.01
# training_steps = 1000
# batch_size = 256
# display_step = 50

# W = tf.Variable(tf.ones([num_features, num_classes]), name="weight")
# b = tf.Variable(tf.zeros([num_classes]), name="bias")

# def logistic_regression(x):
#     # Apply softmax to normalize the logits to a probability distribution.
#     return tf.nn.softmax(tf.matmul(x, W) + b)

# def cross_entropy(y_pred, y_true):
#     # Encode label to a one hot vector.
#     y_true = tf.one_hot(y_true, depth=num_classes)
#     # Clip prediction values to avoid log(0) error.
#     y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
#     # Compute cross-entropy.
#     return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

# def accuracy(y_pred, y_true):
#     # Predicted class is the index of the highest score in prediction vector (i.e. argmax).
#     correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
#     return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# # Stochastic gradient descent optimizer.
# optimizer = tf.optimizers.SGD(learning_rate)



#=============================================================================#
#-------------------------------| INPUTS |------------------------------------#
#=============================================================================#
start_time = timeit.default_timer()


#=============================================================================#
#-------------------------| Reading in the data |-----------------------------#
#=============================================================================#


Activities, Sensors = Set_Act_Sens() #creating two dicts for sensor and activity

print('\n#------------- Reading in the Data of Users -------------#\n')
dataset_uuid = readdata_csv(data_dir) #reading all data and storing in "dataset" a DF
stop1 = timeit.default_timer()

print('Reading the data took:   ', int(stop1 - start_time))

uuids = list(dataset_uuid.keys())

print('\n#-------------- Combining the Data of Users-------------#\n')

#We concatenate the data of all participants (60) to get one dataset for all      
dataset_ag = dataset_uuid[uuids[0]] #"dataset_ag" is the aggregation of all user's data
for i in range(1,len(uuids)):
    dataset_ag = pd.concat([dataset_ag, dataset_uuid[uuids[i]]], axis=0, ignore_index=True)

dataset_ag.iloc[:,feature_range] = preprocessing.scale(dataset_ag.iloc[:,feature_range])
stop2 = timeit.default_timer()

print('Combining the data took:   ', int(stop2 - stop1))

#=============================================================================#
#-----------------------------| DST Setups |----------------------------------#
#=============================================================================#

#We create feature sets, a sample mass function (initialized to 0) and response
#permutations 1 and 2 in which corresponding elements are exclusive and exhaustive

feature_sets = feature_set(sensors_to_fuse, feature_sets_st, Sensors, feature_sets_count)
mass_template = BPA_builder(FOD)
Response_Perm_1, Response_Perm_2 = pair_resp(mass_template)

num_p = len(FOD)
num_fs = len(feature_sets)
num_rp = len(Response_Perm_1)
num_folds = 5

smt = SMOTE()

#find the train_dataset
#at personal level:
print('\n#-------------- Obtaining Training Dataset -------------#\n')

train_dataset, test_dataset = train_test_spl(0,num_folds,cvdir,dataset_uuid)

stop3 = timeit.default_timer()
print('Obtaining the training dataset took:   ', int(stop3 - stop2))

print('Training dataset has  {}  samples'.format(len(train_dataset)))


#=============================================================================#
#------------------| Creating and Training all the models |-------------------#
#=============================================================================#

#------------------------ Parallelization goes here  -------------------------#

#impute = True
#NAPS_Model = []
#with Pool(processes= num_prc) as pool:
#    pool.map(f, [1,2,3])
#
#if __name__ == '__main__':
#    # start 4 worker processes
#    with Pool(processes=4) as pool:
#
#        # print "[0, 1, 4,..., 81]"
#        print(pool.map(f, range(10)))
#    NAPS_Model += pool.map(NAPS_Models_Trainer, (num_rp, fs_range, train_dataset\
#                                                  , feature_sets, Response_Perm_1\
#                                                  , Response_Perm_2, impute))

#NAPS_Models = NAPS_Models_Trainer(num_rp, fs_range, train_dataset, feature_sets\
#                                  , Response_Perm_1, Response_Perm_2, impute = True)

print('\n#-------------- Creating and Training Models -------------#\n')

NAPS_models = []
print('\nLooping over Response Permutations \n ')

for i in range(num_rp): #i runs over response permutations
    
    start_rp = timer()
    
    print('\nResponse Permutation {}/{}'.format(i+1,num_rp))
    print('\n\tLooping over feature sets')
    NAPS_models.append([])
    
    progress = ProgressBar(num_fs, fmt = ProgressBar.FULL)
    
    for j in range(num_fs): #j runs over feature sets
        NAPS_models[i].append([])
        
        NAPS_models[i][j] = \
            train_NAPS_Models(train_dataset, feature_sets, j, Response_Perm_1, \
                      Response_Perm_2, i, bagging_R, num_bags, impute = True)
        #find X and y
#         X_train, y_train, y1, y2 = Xy(train_dataset, feature_sets, j, \
#                         Response_Perm_1, Response_Perm_2, i, impute = True)
# #        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
#         X_train_tmp, y_train_tmp = smt.fit_sample(X_train, y_train)
        
#         X_train = pd.DataFrame(X_train_tmp, columns = X_train.columns)
#         y_train = pd.DataFrame(y_train_tmp, columns = ['Merged_label'])
        
        
#         if j == 0:
#             U2 = Uncertainty_Bias([y1, y2])

#         #create a model and train it
#         NAPS_models[i][j] = DS_Model(Response_Perm_1[i], Response_Perm_2[i], \
#                    X_train, y_train, j)
#         NAPS_models[i][j].Bags_Trainer(X_train, y_train, bagging_R, num_bags)
#         NAPS_models[i][j].Uncertainty_B = U2
        
        progress.current += 1
        progress()
    
    progress.done()
    
    stop_rp = timer()
    
    print("\n It took : ", stop_rp - start_rp)

stop4 = timeit.default_timer()
print('Training all models took:   ', int(stop4 - stop3))
    
#=============================================================================#
#------------------| Model Selection and Testing Models |---------------------#
#=============================================================================#

print('\n#-------------- Testing the Models -------------#\n')
test_dataset[FOD] = test_dataset[FOD].fillna(0)
test_dataset = test_dataset[np.sum(test_dataset[FOD], axis=1) != 0]

y_test_ag = np.zeros([len(test_dataset),len(FOD)])
y_pred_ag = np.zeros([len(test_dataset),len(FOD)])


for t in range(len(test_dataset)):
# for t in range(50):
    print(t,'/',len(test_dataset))
    test_sample = test_dataset.iloc[t,:]
    test_sample = test_sample.to_frame().transpose()
    y_test_ag[t,:] = np.floor(test_sample[FOD].fillna(0).values)
    
    assert np.sum(y_test_ag == 1)
    
    Uncertainty_Mat = np.ones([num_rp, num_fs])
    
    for i in range(num_rp):
        for j in range(num_fs):
            X_test, y_test, y1, y2 = Xy(test_sample, feature_sets, j, \
                                    Response_Perm_1, Response_Perm_2, i, impute = True)
            if len(X_test) != 0 or len(y_test) != 0:
                Uncertainty_Mat[i][j] = (NAPS_models[i][j].Uncertainty_B +\
                                NAPS_models[i][j].Uncertainty_Context(X_test))/2
            NAPS_models[i][j].Mass_Function_Setter(Uncertainty_Mat[i][j], X_test)
            
    #=========\ Model Selection/==========#
    
    Selected_Models_idx = Model_Selector(Uncertainty_Mat, models_per_rp, num_rp, 1)
    y_pred_ag[t,:] = Fuse_and_Predict(Selected_Models_idx, NAPS_models, FOD, num_p, \
              num_rp, models_per_rp)

stop5 = timeit.default_timer()
print('Testing took:   ', int(stop5 - stop4))



conf_mat = confusion_matrix(y_test_ag, y_pred_ag)
accuracy = accuracy_score(y_test_ag, y_pred_ag)
balanced_accuracy = balanced_accuracy_score(y_test_ag, y_pred_ag)
f1 = f1_score(y_test_ag, y_pred_ag)



#!!!!!!!!! Model Selection based on the uncertainty should be fixed !!!!!!!!















