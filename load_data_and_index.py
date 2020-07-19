# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:09:49 2020

@author: Mohamed OUERFELLI
"""

from sklearn.base import BaseEstimator
import os
import pickle
import numpy as np
import sys
import nibabel as nb
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp
from nilearn import plotting , datasets, image
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_roi, plot_epi, show
from sklearn.utils import resample , check_random_state, check_array
from sklearn.model_selection import train_test_split
from scipy import sparse
import sklearn.externals.joblib as joblib
from sklearn.externals.joblib import Memory
from nilearn._utils.cache_mixin import cache
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics import make_scorer
from scipy.sparse import csgraph, coo_matrix, dia_matrix
from sklearn.model_selection import  GridSearchCV, StratifiedKFold, StratifiedShuffleSplit,RepeatedStratifiedKFold
from sklearn.svm import SVR
from sklearn.cluster import FeatureAgglomeration 
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


os.path.join(os.path.dirname(__file__))


cachedir = 'C:\cach'
memory = Memory(cachedir, verbose=0)
if not os.path.exists("INDEX"):
    os.makedirs("INDEX")

#define path 
path_atlas = 'JHU-ICBM-labels-1mm.nii.gz'
masker_load = nb.load(path_atlas)
mask_extract = image.math_img("img>0", img=masker_load)
nb.save(mask_extract, "mask.nii.gz")
mask_path='mask.nii.gz'
nifti_masker = NiftiMasker(mask_img=mask_path).fit()
path_label = 'TC_Lionel.xls'
  

def read_nifti_path(reference,path):
    
    """
    read nifti images while keeping the path structure    
    
    Parameters
    ----------
    reference : str,
       subject id
    Returns
    -------
    img : object,
      Nifti Object of current subject image
    
    """
    path2= "%s/%s" % (path, reference)
    for elementt in os.listdir(path2):
        path3= "%s/%s/%s" % (path2, elementt,'misc')
        for elementtt in os.listdir(path3):
            path4= "%s/%s/%s" % (path3, elementtt,'nii')
            for elementtttt in os.listdir(path4):
                path5 = "%s/%s/%s/%s" % (path4, elementtttt,'tbss','FA')
                if path=='FA_view':
                    example_filename = os.path.join(path5, 'FA_subj_FA_to_target.nii.gz')
                elif path=='MD_view':
                    example_filename = os.path.join(path5, 'FA_subj_to_target_MD.nii.gz')   
                elif path=='L1_view':
                    example_filename = os.path.join(path5, 'FA_subj_to_target_L1.nii.gz')   
                elif path=='Lt_view':
                    example_filename = os.path.join(path5, 'FA_subj_to_target_Lt.nii.gz')   
                img = nb.load(example_filename)
                return img
            
       
        

def transform_with_mask(ID_patient,path): 
    """
    apply mask on nifti image object

    Parameters
    ----------
    ID_patient : str,
       list of ID_patient of subjects who are not excluded and who have outcomes
    Returns
    -------
    X : matrix,
      matrix of shape (number of subjects * number of voxels)
    """
    
    image = []
    masker =[]
    for element in os.listdir(path):  # loop into all patients
        if element in ID_patient :
            img = read_nifti_path(element,path)
            image.append(img)         #Contains all images 3D
    for element in image:  # loop into all patients
        masker.append(nifti_masker.fit_transform(element))    #apply mask

    m=np.array(masker) 
    m= m.reshape(m.shape[0],m.shape[2])

    return  m



def read_labels(filename):

    """
    read the Outcomes of subjects who are not excluded and who have a nifti images

    Parameters
    ----------
    filename : str,
       file .xls
       
    Returns
    -------
    label : int,
      list of Gose (Outcome)
    idP : str 
      list of subjects who are not excluded and who have a nifti images
        
    
    """

    data = pd.read_excel(filename)
    outcome=[]
    ID_PATIENT = []
    for i in range (data.shape[0]):
        a= str(data['Observations'][i])
        b= str(data['Outcome'][i])
        c= str(data['DISPO'][i])
        if a.startswith('EXC') or a.startswith('exc')  or a.startswith('Exc') :
         
            continue
        elif b != 'nan' and b!= '-3' and c == 'YES':
            outcome.append(data['Outcome'][i])
            ID_PATIENT.append(data['Patient ID'][i])
    label = np.array(outcome).reshape(len(outcome), 1)
    idP = np.array(ID_PATIENT).reshape(len(ID_PATIENT), 1)
    label=label.reshape(len(label),)

    return label , idP
    
    
    
def fold_idx(X,y,k,stratif_externe=False):
    """K-Folds cross validation iterator.

    Parameters
    ----------
    X:
    y: Outcome (Gose)
    k : int, default 5
    stratif_externe : bool, default False

    Yields
    -------
    train_idx, test_idx
    """
        

    if stratif_externe:
        cv = StratifiedKFold(n_splits=k,shuffle=True,random_state=42)

    else:
        cv = StratifiedShuffleSplit(n_splits=k,test_size=0.5)

    train_idx=[]
    test_idx =[]
    for train_index, test_index in cv.split(X,y):
        train_idx.append(train_index)
        test_idx.append(test_index)
    return train_idx, test_idx


     
def stock_index(X,Outcome,K,J):

    FULLtrain_idx, Test_idx = fold_idx(X,Outcome,K,True)    #StratifiedKFold : split data 
 
    pickle.dump(FULLtrain_idx, open('INDEX/Fulltrain_idx', 'wb'))     #Save index of Fulltrain_idx
    pickle.dump(Test_idx, open('INDEX/Test_idx', 'wb'))                 #Save index of Test_idx

    train_index=[]
    val_index=[]
    for train,test in zip(FULLtrain_idx,Test_idx):
        FX_train, X_test, Fy_train, y_test = X[train], X[test], Outcome[train], Outcome[test]
        train_idx, val_idx = fold_idx(FX_train,Fy_train,J,False)    #StratifiedKFold : split data 
        train_index.append(train_idx)
        val_index.append(val_idx)
    pickle.dump(train_index, open('INDEX/train_idx', 'wb'))     #Save index of Fulltrain_idx
    pickle.dump(val_index, open('INDEX/val_idx', 'wb'))                 #Save index of Test_idx
    


if __name__ == "__main__":
    K= sys.argv[1]    #put argument: the number of folds of external cross val
    J= sys.argv[2]
    Outcome , ID_patient = read_labels(path_label)   
    X1 = transform_with_mask(ID_patient,'FA_view')    #Apply mask to data
    X2 = transform_with_mask(ID_patient,'MD_view')    #Apply mask to data
    X3 = transform_with_mask(ID_patient,'L1_view')    #Apply mask to data
    X4 = transform_with_mask(ID_patient,'Lt_view')    #Apply mask to data
    pickle.dump(X1, open('X1', 'wb'))
    pickle.dump(X2, open('X2', 'wb'))
    pickle.dump(X3, open('X3', 'wb'))
    pickle.dump(X4, open('X4', 'wb'))
    stock_index(X1,Outcome,int(K),int(J))
















