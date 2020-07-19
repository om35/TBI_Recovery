"""
TBI_Recovery:  MeCA Team - Institut de Neurosciences de la Timone
Author:
    Mohamed Ouerfelli  & Sylvain Takerkart
"""
import time
import os
import pickle
import sys
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import pandas as pd
from nilearn import  image
from nilearn.input_data import NiftiMasker
from sklearn.model_selection import  GridSearchCV
from sklearn.svm import LinearSVR
from sklearn.cluster import FeatureAgglomeration 
from scipy.sparse import csgraph, coo_matrix, dia_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn=warn

os.path.join(os.path.dirname(__file__))
if not os.path.exists("OUTPUT_ward"):
    os.makedirs("OUTPUT_ward")

#define path 
path_atlas = 'JHU-ICBM-labels-1mm.nii.gz'
masker_load = nb.load(path_atlas)
mask_extract = image.math_img("img>0", img=masker_load)
nb.save(mask_extract, "mask.nii.gz")
mask_path='mask.nii.gz'
nifti_masker = NiftiMasker(mask_img=mask_path).fit()
path_label = 'TC_Lionel.xls'


            
def read_labels(filename):
    """"
    Input : filename .xls
    Output : vector Y(Gose) and the id_patient used.
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
        elif b != 'nan' and  b!='-3' and c == 'YES':
            outcome.append(data['Outcome'][i])
            ID_PATIENT.append(data['Patient ID'][i])
    label = np.array(outcome).reshape(len(outcome), 1)
    idP = np.array(ID_PATIENT).reshape(len(ID_PATIENT), 1)
    label=label.reshape(len(label),)

    return label , idP

def compute_connectivity():
    """
    Compute the weighted adjacency graph 
    """
    from sklearn.feature_extraction import image
    mask= np.asarray(nifti_masker.mask_img_.get_data()).astype(bool)
    mask = np.asarray(mask).astype(bool)
    shape = mask.shape
    connectivity = image.grid_to_graph(
        n_x=shape[0], n_y=shape[1], n_z=shape[2], mask=mask)
  
    return connectivity


def run_fixed_parcellation(model,X,Outcome):
    connectivity= compute_connectivity()
    train_idx=pickle.load(open('INDEX/Fulltrain_idx', 'rb'))
    test_idx = pickle.load(open('INDEX/Test_idx', 'rb'))
    scores=[]
    coef=[]
  
    train=train_idx[0]
    test=test_idx[0]
    for train,test in zip(train_idx,test_idx):
        X_fulltrain, X_test, y_fulltrain, y_test = X[train], X[test], Outcome[train], Outcome[test]
        X_train, X_val, y_train, y_val = train_test_split(X_fulltrain, y_fulltrain,stratify=y_fulltrain, test_size=0.5)
        liste_ward_label=[]    
        liste_r2=[]
        liste_coef=[]
        intercp=[]
        parcels=[50,100,200,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,8000,9000] 
        L= [0.0001,0.001,0.01,10,100,150,170,200,250,300,350,370,390,400,420,450,470,500,550,570,600,650,670,700,750,800,820,850,900,1000,1500,2000,3000,5000]

        for p in parcels:    
            ward = FeatureAgglomeration(n_clusters=p,connectivity=connectivity)
            X_red= ward.fit_transform(X_train)
            liste_ward_label.append(np.ravel(ward.labels_))
            for l in L:
                mod = model.set_params(C = l)
                mod.fit(X_red,y_train)
                predict = mod.predict(ward.transform(X_val)) 
                liste_r2.append(r2_score(y_val,predict))
                liste_coef.append(mod.coef_)
                intercp.append(mod.intercept_)
        Best_R2 = max(liste_r2)
        Best_index =liste_r2.index(Best_R2)
        intercept= intercp[Best_index]
        labels= liste_ward_label[int(Best_index /len(L))]
        Best_coef= liste_coef[Best_index]
        coef.append(Best_coef)
        n_parcels= Best_coef.shape[0]
        n_voxels = len(labels)
        #Return to voxel space
        incidence = coo_matrix(
                (np.ones(n_voxels), (labels, np.arange(n_voxels))),
                shape=(n_parcels, n_voxels), dtype=np.float32).tocsc()
        inv_sum_col = dia_matrix(
                (np.array(1. / incidence.sum(axis=1)).squeeze(), 0),
                shape=(n_parcels, n_parcels))    
        incidence = inv_sum_col * incidence
        w_aprox = Best_coef * incidence
        y_pred= X_test @ w_aprox.T + intercept
        scores.append(r2_score(y_test, y_pred ))
        
    pickle.dump(scores, open('OUTPUT_ward/R2_scs', 'wb'))
    #pickle.dump(coef, open('OUTPUT_ward/weights', 'wb'))
    pickle.dump(w_aprox, open('OUTPUT_ward/weight_map', 'wb'))
    neg = nifti_masker.inverse_transform(w_aprox)
    nb.save(neg, "OUTPUT_ward/ward_weight_map.nii.gz")



if __name__ == "__main__":
    warnings.warn = warn
    L = sys.argv[1]
    Outcome , ID_patient = read_labels(path_label)  
    X1= pickle.load(open('X1', 'rb'))
    X2= pickle.load(open('X2', 'rb'))
    X3= pickle.load(open('X3', 'rb'))
    X4= pickle.load(open('X4', 'rb')) 
    X1_normalized=normalize(X1,norm='l2',axis=1)
    X2_normalized=normalize(X2,norm='l2',axis=1) 
    X3_normalized=normalize(X3,norm='l2',axis=1)
    X4_normalized=normalize(X4,norm='l2',axis=1)      
    
    if 'FA' in L:
        run_fixed_parcellation(LinearSVR(),X1_normalized,Outcome)   
    elif 'MD' in L:
        run_fixed_parcellation(LinearSVR(),X2_normalized,Outcome)   
    elif 'L1' in L:
        run_fixed_parcellation(LinearSVR(),X3_normalized,Outcome)   
    elif 'Lt' in L:
        run_fixed_parcellation(LinearSVR(),X4_normalized,Outcome)   
        
