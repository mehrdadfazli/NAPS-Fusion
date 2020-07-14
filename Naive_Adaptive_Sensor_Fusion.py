# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:13:08 2019

@author: 14342
"""
from pyds_local import *

import random
import numpy as np
import sys
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
import itertools
import sys
import re

class DS_Model:
    
    def __init__(self, resp_var_1, resp_var_2, X_train, y_train, feature_set_idx):
        """
        Initializing a DS model

        Parameters
        ----------
        resp_var_1 : list
        resp_var_2 : list
            These two are the response variables that we want to build a model
            on. For example if we are building a model to classify {c1} vs {c2,c3}
            then [c1] would be our resp_var_1 and [c2,c3] would be our resp_var_2.
            
        X_train : pd.dataframe
            training features in a dataframe format.
        y_train : pd.dataframe
            training labels as a vector.
        feature_set_idx : integer
            Index of the feature set (out of all the randomly created feature sets).

        Returns
        -------
        None.

        """
        self.Feature_Set_idx = feature_set_idx
        self.Response_Variables = [resp_var_1,resp_var_2]
        self.clf = LogisticRegression(class_weight='balanced',solver='saga', n_jobs=-1).fit(X_train, y_train)
        self.Bags = []                          #list of the bags for this model
        self.Uncertainty_B = 0                  #Uncertainty of the biased model
        self.mass = MassFunction()              #Mass function of the model
    
    def Mass_Function_Setter(self, uncertainty, X):
        """
        We used pyds package (a dempster shafer package) to define the mass function
        given the probabilities and uncertainty.
        """
        probability = self.clf.predict_proba(X)
        self.mass[self.Response_Variables[0]] = probability[0,0]*(1-uncertainty)
        self.mass[self.Response_Variables[1]] = probability[0,1]*(1-uncertainty)
    
    def Bags_Trainer(self, X_train, y_train, ratio, num_bags):
        """
        This function trains bags

        Parameters
        ----------
        X_train : pd.dataframe
            training features in a dataframe format.
        y_train : pd.dataframe
            training labels as a vector.
        ratio : float
            Ratio of the generated bagging size to the actual training dataset.
        num_bags : integer
            number of the bags to generate.

        Returns
        -------
        None.

        """
        for i in range(num_bags):
            self.Bags.append(clone(self.clf))
            indices = random.choices(list(range(len(X_train))), k=int(ratio*(len(X_train))))
            X_train_Bag = X_train.iloc[indices,:]
            y_train_Bag = y_train.iloc[indices]
            
            self.Bags[i].fit(X_train_Bag, y_train_Bag)
    
    def Uncertainty_Context(self, X_test_single):
        """
        This function calculates the uncertainty of the contextual meaning by
        calculating the votes from all the bags.

        Parameters
        ----------
        X_test_single : pd.series
            one single test example.

        Returns
        -------
        None.

        """
        
        C = len(self.Response_Variables) #Number of the classes
        V = np.zeros(C) #Vote vector
        T = len(self.Bags) #total number of the bags
        
        for i in range(T):
            V[int(self.Bags[i].predict(X_test_single))] += 1
        
        Uncertainty_c = 1 - np.sqrt(np.sum(np.power((V/T-1/C),2)))/np.sqrt((C-1)/C)
        return(Uncertainty_c)
        
        
        

class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d remaining feature sets'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)


def feature_set(sensors_to_fuse, st_feat, Sensors, feat_set_count=100):
    """
    This function takes a list of name of the sensors to fuse "sensors_to_fuse
    and structure of the reduced feature space also as a list (like [s1,s2,..])
    where "si" is the number of the features of the ith sensor to be used and 
    also number of the reduced feature sets to create. Then it returns a matrix
    with each row being a feature set number of columns
       
    Parameters
    ----------
    sensors_to_fuse: list
        a list of the name of the sensors to fuse. like ['Acc','Gyro','PS','Aud']
    st_feat: list
        the structure of a feature set. like [s1,s2,..] where "si" is 
        the number of the features of the ith sensor
    feat_set_count: integer
        the number of randome feature sets to create. It is a number like 100
    Sensors: dict
        the sensors dictionary which is the output of the labeling function
       
    Returns
    -------
        selected_feats: np.array 2D
            a matrix that each row represents one set of features and the 
            values in each rows are the index of the columns of the data to
            be used as features
       
    """

    #Making sure that the length of the "st_feat" is equal to the length of the
    #"sensors_to_fuse
    assert len(st_feat) == len(sensors_to_fuse)
    
    #Creating and initializing the "selected_feats" to 0
    selected_feats = np.zeros([feat_set_count,sum(st_feat)])
    
    #A for loop to create the desired number of the random reduced feature sets
    for j in range(feat_set_count):
        
        #"col" stores the index of the randomly generated columns (features)
        col = []
        for i in range(len(st_feat)):
            
            #"aux" stores the index of the generated columns of one sensor
            aux = random.sample(Sensors[sensors_to_fuse[i]],st_feat[i])
            aux.sort()
            col.extend(aux)
        selected_feats[j,:] = col
    return(selected_feats)


def BPA_builder(labels):
    """This function creates the Basic Probability Assignment (BPA) matrix given
       a list of labels (classes)
       
       Input:
           labels: a list of the target labels to be used as our propositins.
                   In other words it is the Frame of discernment (FOD).
                   like: ['Walking','Lying_down']
       Ouput:
           mass: a dataframe that has all the subsets of the the FOD. each row
                 respresent a sebset with a binary vector like:
                     
                     Lying_down     Sleeping    mass
                     0              0           0
                     0              1           0
                     1              0           0
                     1              1           0
                 
                 ex. the last row represents {'Lying_down','Sleeping'} and m is
                 the corresponding basic beleif assignment or mass function.
                 also all masses are initialized with 0
                   
    """
       
    mass = pd.DataFrame(columns=labels)
    perms = list(itertools.product([0, 1], repeat=len(labels)))
    for i in range(len(perms)):
        mass.loc[i] = perms[i]
    mass['m'] = 0
    return(mass)
    
    
def pair_resp(BPA):
    """This function takes in the mss function (dataframe in fact) and returns
       two lists. Each element in each of the lists is itself a list of labels
       representing one subset of the FOD or one member of the power set. Note
       that perms_set_1[i] and perms_set_2[i] are complementary subsets.
       
       Input:
           BPA: the mass dataframe. output of the BPA_builder()
       
       Output:
           perms_set_1: a list of lists
           perms_set_2: the same
           
       ex. target labels = 'Lying_down','Sleeping','Walking'
           perms_set_1[2]=['Lying_down','Sleeping'] 
           perms_set_2[2]=['Walking']
    
    """
    
    perms_set_1 = []
    perms_set_2 = []
    
    #Here we don't need the last column of the mass which is the valuses of mass
    #function. So we get rid of them so that they won't interfere later
    BPA = BPA.iloc[:,:-1]
    
    #It is enough to go up to the half of the rows of the mass to generate the
    #two classes of permustations. The rest are just the complementary to the
    #first half. In fact if the powerset has 2**n elements, we have 2**(n-1) 
    #complementary pairs
    for i in range(BPA.shape[0]//2):
        assert (sum(BPA.loc[i,:])+sum(BPA.loc[BPA.shape[0]-i-1,:])) == BPA.shape[1]
        perms_set_1.append([])
        perms_set_2.append([])
        
        for j in range(BPA.shape[1]):
            if BPA.iloc[i,j] == 1:
                perms_set_1[i].append(BPA.columns[j])
            else:
                perms_set_2[i].append(BPA.columns[j])
    
    perms_set_1.remove(perms_set_1[0])
    perms_set_2.remove(perms_set_2[0])
    
    return(perms_set_1,perms_set_2)


def Uncertainty_Bias(response_variables):
    """
    This function calculates the "uncertainty of the biased model" based on the
    frequency of the classes.
    
    Input:
        response_variables[list of pd.Dataframe]: each element of the list is 
        itself a dataframe of one class (response variable).
    
    Output:
        U[float]: uncertainty of the biased model
    """
    
    C = len(response_variables) #number of the classes (response variables)
    I = np.zeros(C) #an array of the frequencies of the classes
    
    S = 0 #total samples in the dataset
    for i in range(C):
        I[i] = int(np.sum(response_variables[i] == 1))
        S += I[i]
    
    U = np.sqrt(np.sum(np.power((I/S-1/C),2)))/np.sqrt((C-1)/C)
    return (U)

    
def Model_Selector(uncertainty_m, models_per_rp, num_rp, fs_axis):
    """This function takes the total uncertainty matrix and chooses the least
    uncertain models for every response permutation.
    
    Input:
        uncertainty_m[np.array 2d]: Matrix of the total uncertainty for all models.
        models_per_rp[int]: Number of the models to select for each response permutation
        num_rp[int]: Number of the response variables (Also the length of the 
              uncertainty matrix along one of the dimensions)
        fs_axis[int]: The axis of the uncertainty matrix along which feature sets
            are laid
    
    Output:
        selected_models_idx[np.array 2d]: A 2d array that holds the feature set
            indices for each response permutation
    """
    
    selected_models_idx = np.zeros([num_rp, models_per_rp])
    
    index_m = np.argsort(uncertainty_m, axis= fs_axis)
    
    for i in range(models_per_rp):
        selected_models_idx[:,i] = np.argwhere(index_m == i)[:,1]
    
    return(selected_models_idx)
        
def Fuse_and_Predict(selected_models_idx, Models, FOD, num_classes, num_rp, models_per_rp):
    assert len(selected_models_idx) == num_rp
    y_pred = np.zeros([1,num_classes])
    
    fs_idx = int(selected_models_idx[0][0])
    combined_mass = Models[0][fs_idx].mass
    for i in range(num_rp):
        for j in range(models_per_rp):
            if i == 0 and j == 0:
                continue
            fs_idx = int(selected_models_idx[i][j])
            combined_mass = combined_mass & Models[i][fs_idx].mass
    
    for i in range(num_classes):
        y_pred[0,i] = combined_mass[{FOD[i]}]
    
    y_pred_aux = np.zeros_like(y_pred)
    y_pred_aux[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
    y_pred = y_pred_aux
    
    return(y_pred)
    
# def NAPS_Models_Trainer(num_rp, fs_range, train_dataset, feature_sets, Response_Perm_1, \
#                         Response_Perm_2, impute = True):
#     """
#     This function trains NAPS models.

#     Parameters
#     ----------
#     num_rp : int
#         number of the response permutations (which is 2**(n-1)-1 for n classes).
#     fs_range : range or list
#         a list of int from 0 to m (m being the total number of the feature sets).
#     train_dataset : pd.dataframe
#         dataframe of the training features.
#     feature_sets : np.array 2D
#         matrix of the selected features.
#     Response_Perm_1 : list
#         lsit of the response permutations.
#     Response_Perm_2 : list
#         complementary list of the Response_Perm_1.
#     impute : bool, optional
#         Whether to impute or discard missing features. The default is True.

#     Returns
#     -------
#     None.

#     """
#     NAPS_models = []
#     print('\nLooping over Response Permutations \n ')
    
#     for i in range(num_rp):
#         print('\nResponse Permutation {}/{}'.format(i+1,num_rp))
#         print('\n\tLooping over feature sets')
#         NAPS_models.append([])
                
#         for j in fs_range:
#             print('\t\tFeature Set {}/{}'.format(j+1, len(fs_range)))
#             NAPS_models[i].append([])
#             #find X and y
#             X_train, y_train, y1, y2 = Xy(train_dataset, feature_sets, j, \
#                             Response_Perm_1, Response_Perm_2, i, impute = True)
#     #        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            
#             X_train_tmp, y_train_tmp = smt.fit_sample(X_train, y_train)
            
#             X_train = pd.DataFrame(X_train_tmp, columns = X_train.columns)
#             y_train = pd.DataFrame(y_train_tmp, columns = ['Merged_label'])
            
            
#             if j == 0:
#                 U2 = Uncertainty_Bias([y1, y2])
    
#             #create a model and train it
#             NAPS_models[i][j] = DS_Model(Response_Perm_1[i], Response_Perm_2[i], \
#                        X_train, y_train, j)
#             NAPS_models[i][j].Bags_Trainer(X_train, y_train, bagging_R, num_bags)
#             NAPS_models[i][j].Uncertainty_B = U2
    
#     return(NAPS_models)
        
    
    
    
    
    

