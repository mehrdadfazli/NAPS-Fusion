# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 21:05:15 2020

@author: 14342
"""
data_dir = "C:\\Mehrdad\\DARPA-WASH\\datasets\\"
cvdir = "C:\\Mehrdad\\DARPA-WASH\\cv5Folds\\cv_5_folds\\"

sensors_to_fuse = ['Acc','Gyro','W_acc','Aud'] 
#there are also "Mag", "Loc", "AP", "PS", "LF", "Compass" sensors
FOD = ['label:LYING_DOWN','label:SITTING','label:OR_standing','label:FIX_walking']#\
#      ,'label:FIX_running','label:BICYCLING']
feature_sets_st = [3,3,3,3] #feature set structure
feature_sets_count = 10
bagging_R = 0.6  #bagging ratio
num_bags = 4
models_per_rp = 2  #number of models to select for the fusion per response permutation
feature_range = range(1,225) #range of the column number of all features
num_prc = 6  #number of processors to split the job during parallelization

