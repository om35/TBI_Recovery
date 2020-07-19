# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:09:49 2020

@author: Mohamed Ouerfelli
"""

from sklearn.base import BaseEstimator
import os
import pickle
import numpy as np
import time
import scipy.io as sio
import nibabel as nb
import matplotlib.pyplot as plt
import pandas as pd
from nilearn import plotting , datasets, image
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_roi
from scipy import sparse
import sklearn.externals.joblib as joblib
from sklearn.externals.joblib import Memory
from nilearn._utils.cache_mixin import cache
import sys
from scipy.sparse import csgraph, coo_matrix, dia_matrix
from sklearn.svm import SVR, LinearSVR
from sklearn.cluster import FeatureAgglomeration 
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

os.path.join(os.path.dirname(__file__))


OUtput= "OUTPUT_precoce_2"


#Define Folder
if not os.path.exists(OUtput):
    os.makedirs(OUtput)
if not os.path.exists("INDEX"):
    os.makedirs("INDEX")
if not os.path.exists("%s/Nifti" %OUtput):
    os.makedirs("%s/Nifti" %OUtput)   
if not os.path.exists("%s/clustering" %OUtput):
    os.makedirs("%s/clustering" %OUtput)
if not os.path.exists("%s/R2_MATRIX" %OUtput):
    os.makedirs("%s/R2_MATRIX" %OUtput)
if not os.path.exists("%s/Weight" %OUtput):
    os.makedirs("%s/Weight" %OUtput)
if not os.path.exists("%s/Intercept" %OUtput):
    os.makedirs("%s/Intercept" %OUtput)
if not os.path.exists("%s/Folds" %OUtput):
    os.makedirs("%s/Folds" %OUtput)
if not os.path.exists("%s/FINAL" %OUtput):
    os.makedirs("%s/FINAL" %OUtput)
    
    
#define path 
path_atlas = 'JHU-ICBM-labels-1mm.nii.gz'       #Atlas file : Nifti image
masker_load = nb.load(path_atlas)               #Load atlas
mask_extract = image.math_img("img>0", img=masker_load)      #extract white matter mask
nb.save(mask_extract, "mask.nii.gz")
mask_path='mask.nii.gz'
nifti_masker = NiftiMasker(mask_img=mask_path).fit()   
path_label = 'TC_Lionel.xls'                     # GOSE : Outcome file
  
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
    
def mean_weights(weights):
    
    """
    Compute the average weight from an array of many weights computed on each parcellation 
    
    Parameters
    ----------
    weights : array,
       array of weights (nb_parcellations X nb_voxels)
       
    Returns
    -------
    W : vector,
      W_bagg : average of weights
    """
    coef = []
    for i in range(weights.shape[1]):
        coef.append(np.mean(weights[:,i]))  
    W = np.array(coef).reshape(1,-1)
    return W


def compute_connectivity():
    """
    Compute connectivity : useful for the FeatureAgglomeration function : spatial constraints
    """
    from sklearn.feature_extraction import image
    mask= np.asarray(nifti_masker.mask_img_.get_data()).astype(bool)
    shape = mask.shape
    connectivity = image.grid_to_graph(
        n_x=shape[0], n_y=shape[1], n_z=shape[2], mask=mask).tocsr()
    return connectivity
    

    
def save_clustering(labels_,j,fold,parcels):
    """
    This function is used to save clustering plot.
    input : labels,index of parcellation,index of iteration of cross val externe , number of parcels.
    Output: plot and save clustering associated with the index of parcellation and iteration of cross val and the number of parcels.
    """
    y = labels_.reshape(1,len(labels_))
    plot = nifti_masker.inverse_transform(y)
    plotting.plot_roi(plot, title="Ward FeatureAgglomeration",  display_mode='xz',output_file='%s/clustering/00000%s_fold_parcellation_%s_0000%s_parcels.png' %(OUtput,fold+1,j+1,parcels) , cut_coords=(5,  9))
    plot.to_filename('%s/Nifti/00000%s_fold_parcellation_%s_0000%s_parcels.nii.gz' %(OUtput,fold+1,j+1,parcels))



class FrEM(BaseEstimator):
    """
    Ensemble Learning method
    Parameters
    ----------
    data : array_like, shape=(n_samples, n_voxels)
      Masked subject images as an array.
    y : Vector of Gose (label)
    b : number of parcellation= number of iteration of cross validation
    fold: index of current iteration of external cross validation
    memory : instance of joblib.Memory or string
      Used to cache the masking process.
      By default, no caching is done. If a string is given, it is the
      path to the caching directory.
    n_jobs : int,
      Number of parallel workers.
      If 0 is provided, all CPUs are used.
      A negative number indicates that all the CPUs except (|n_jobs| - 1) ones
      will be used.
    """
    def __init__(self, data=None,y=None,fold=None, memory=None,  n_jobs=1):
        self.data = data
        self.y= y 
        self.fold=fold
        self.memory = memory
        self.n_jobs = n_jobs


    def fit(self,model,X1,X2,y):
        
        """
        Fit to data, then perform the ensemble learning 
        
        Parameters
        ----------
        model : regression model
        X : 2D array
        y : label vector
        """
        
        #PART 1: Load index
        train_idx = pickle.load(open('INDEX/train_idx', 'rb'))
        val_idx = pickle.load(open('INDEX/val_idx', 'rb'))
        J= len(val_idx[0])
        liste_w_aprox=[] 
        intercp=[]
        parcelles=[]
        connectivity= compute_connectivity()
        for train,val, j in zip(train_idx[self.fold],val_idx[self.fold], range(J)):
            X_train_1, X_val_1, y_train_1, y_val_1 = X1[train], X1[val], y[train], y[val]
            X_train_2, X_val_2, y_train_2, y_val_2 = X2[train], X2[val], y[train], y[val]
            X_train = np.concatenate((X_train_1,X_train_2),axis=0)
            
            liste_ward_label=[]    
            liste_r2=[]
            liste_coef=[]
            intercept=[]
            parcels=[50,100,200,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,9000] 

            #PART2: Hyperparameter selection and training regresion model 
            for p in parcels:
                ward = FeatureAgglomeration(n_clusters=p,connectivity=connectivity,linkage='ward')
                ward.fit(X_train)
                X_red1= ward.transform(X_train_1)
                X_red2= ward.transform(X_train_2)
                X_v1= ward.transform(X_val_1)
                X_v2= ward.transform(X_val_2)
                X_red= np.concatenate((X_red1,X_red2),axis=1)
                X_val= np.concatenate((X_v1,X_v2),axis=1)
                liste_ward_label.append(np.ravel(ward.labels_))
                L= [0.0001,0.001,0.01,0.1,1,10,100,120,150,170,200,220,250,300,350,370,390,400,420,450,470,500,550,570,600,650,670,700,720,750,800,820,850,900,1000,1200,1500,2000,3000,5000] 
                
                
                for l in L:
                    mod = model.set_params(C = l)
                    mod.fit(X_red,y_train_1)
                    predi= mod.predict(X_val)
                    liste_r2.append(r2_score(y_val_1,predi))
                    liste_coef.append(mod.coef_)
                    intercept.append(mod.intercept_)
                    

            #PART3: Select the best weight with the best performance on validation set
            Best = max(liste_r2) 
            Best_parcels=liste_r2.index(Best)
            coef = liste_coef[Best_parcels]
            intercp.append(intercept[Best_parcels])
            labels= liste_ward_label[int(Best_parcels/len(L))]
            n_parcels= int(coef.shape[0]/2)
            parcelles.append(n_parcels)
            n_voxels = len(labels)
            #save_clustering(labels,j,self.fold,n_parcels)   

            #Return to voxel space
            incidence = coo_matrix(
                    (np.ones(n_voxels), (labels, np.arange(n_voxels))),
                    shape=(n_parcels, n_voxels), dtype=np.float32).tocsc()
            
            inv_sum_col = dia_matrix(
                    (np.array(1. / incidence.sum(axis=1)).squeeze(), 0),
                    shape=(n_parcels, n_parcels))    
            incidence = inv_sum_col * incidence
            Beta_1 = coef[:n_parcels] * incidence     #weight maps for the first modality 
            Beta_2= coef[n_parcels:] * incidence       #weight maps for the second modality

            Beta=np.concatenate((Beta_1,Beta_2),axis=0)
            liste_w_aprox.append(Beta)
            
            
            #PART 4: Saving results
            if ((j+1) % 10) ==0 :
                W= np.array(liste_w_aprox).reshape(j+1,340012)
                it= np.mean(intercp)
                W_bagg= np.array(mean_weights(W))
                pickle.dump(W_bagg, open('%s/Weight/%s_timestep_%s_fold_Beta'%(OUtput,j+1,self.fold +1), 'wb'))
                pickle.dump(it, open('%s/Intercept/%s_timestep_%s_fold_intercept'%(OUtput,j+1,self.fold +1), 'wb'))
                pickle.dump(parcelles, open('%s/FINAL/parcels' %OUtput, 'wb'))
    
    
        
            

def cross_val_compute(model,X1,X2,Outcome,fold,ftrain, test,K):
    """
    Parameters
    ----------
    model : Regression model
    X : 
    y : Outcome (Gose)
    b: number of parcellation
    fold: index of current iteration of external cross validation
    Yields
    -------
    weight, r2_score_ensemble, r2 on validation
    
    """

    X_trai_1, X_tes_1, y_trai, y_tes1 = X1[ftrain], X1[test], Outcome[ftrain], Outcome[test]
    X_trai_2, X_tes_2 = X2[ftrain], X2[test]
    X_tes = np.concatenate((X_tes_1,X_tes_2),axis=0)
    y_tes = np.concatenate((y_tes1 ,y_tes1),axis=0)
    frem =  FrEM(fold=fold, memory=None,  n_jobs=1)
    pickle.dump(X_tes, open('%s/Folds/%s_fold_X_test_mv'%(OUtput,fold+1), 'wb'))
    pickle.dump(y_tes, open('%s/Folds/%s_fold_y_test_mv'%(OUtput,fold+1), 'wb'))
    frem.fit(model,X_trai_1,X_trai_2,y_trai)



def run(model,X1,X2,Outcome):
    
    Ftrain_idx = pickle.load(open('INDEX/Fulltrain_idx', 'rb'))
    test_idx = pickle.load(open('INDEX/Test_idx', 'rb'))
    list_test_index= np.hstack(test_idx)
    pickle.dump(list_test_index, open('%s/FINAL/Test' %OUtput, 'wb'))   #Save all index of Test

    jobs= -1
    if jobs == -1:
        print("Parallel Running")
    print("This step can take few minutes")
    ret= joblib.Parallel(n_jobs=jobs)(joblib.delayed(
            cache(cross_val_compute, memory=Memory(cachedir=None)))
          (model,X1,X2,Outcome,fold,ftrain,test,len(Ftrain_idx))
              for ftrain,test,fold in zip(Ftrain_idx,test_idx,range(len(Ftrain_idx))))
    
    
def recupere_resultat(K,t):

    predi=[]
    y_test=[]
    Weight=[]
    Weight_2=[]
    X_test=[]
    score_folds=[]
    for i in range(K):

        weight = pickle.load(open('%s/Weight/%s_timestep_%s_fold_Beta'%(OUtput,t,i+1), 'rb')) 
        intercept = pickle.load(open('%s/Intercept/%s_timestep_%s_fold_intercept'%(OUtput,t,i+1), 'rb'))
        X_tes = pickle.load(open('%s/Folds/%s_fold_X_test_mv'%(OUtput,i+1), 'rb'))
        l = int(len(X_tes)/2)
        x1 = X_tes[:l,:]
        x2 = X_tes[l:,:]
        X_tes=np.concatenate((x1,x2),axis=1)
        prediction = X_tes @ weight.T + intercept
        y_tes = pickle.load(open('%s/Folds/%s_fold_y_test_mv'%(OUtput,i+1), 'rb'))
        y_tes = y_tes[:l]
        predi.append(prediction)
        y_test.append(y_tes)
        Weight.append(weight[:,:170006])
        Weight_2.append(weight[:,170006:])
        X_test.append(X_tes)

    W_final=np.array(Weight).reshape(K,170006)
    ww =mean_weights(W_final)
    neg = nifti_masker.inverse_transform(ww)
    nb.save(neg, "%s/FINAL/%s_timestep_weight_map_1.nii.gz" %(OUtput,t))
    W_final_2=np.array(Weight_2).reshape(K,170006)
    www =mean_weights(W_final_2)
    neg = nifti_masker.inverse_transform(www)
    nb.save(neg, "%s/FINAL/%s_timestep_weight_map_2.nii.gz" %(OUtput,t))
    y_true =np.hstack(y_test)
    X_all =np.vstack(X_test)
    predict=np.vstack(predi)
    sc= r2_score(y_true,predict)
    y_true=y_true.reshape(-1,1)
    regr = LinearSVR()
    a=regr.fit(y_true,predict)
    droite = a.coef_ * y_true + a.intercept_ 
    plt.scatter(y_true,predict)
    plt.plot(y_true,droite,c='r',label="regression line")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=4,label="D: y = x")

    plt.xlabel("True Gose score")
    plt.ylabel("Predicted Gose score")
    plt.title("score r2")
    plt.legend()
    plt.savefig("%s/%s_parcellation_r2_correlation.png" %(OUtput,t))
    
    pickle.dump(predict, open('%s/FINAL/%s_timestep_predict'%(OUtput,t), 'wb'))   
    pickle.dump(sc, open('%s/FINAL/%s_timestep_r2_score'%(OUtput,t), 'wb'))   
    pickle.dump(Weight, open('%s/FINAL/%s_timestep_Weight_folds'%(OUtput,t), 'wb')) 
    pickle.dump(y_true, open('%s/FINAL/y_true'%(OUtput,t), 'wb')) 


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

    if 'FA' and 'MD' in L:
        run(LinearSVR(),X1_normalized,X2_normalized,Outcome)   
    elif 'FA' and 'L1' in L:
        run(LinearSVR(),X1_normalized,X3_normalized,Outcome)   
    elif 'FA' and 'Lt' in L:
        run(LinearSVR(),X1_normalized,X4_normalized,Outcome)   
    elif 'MD' and 'L1' in L:
        run(LinearSVR(),X2_normalized,X3_normalized,Outcome)   
    elif 'MD' and 'Lt' in L:
        run(LinearSVR(),X2_normalized,X4_normalized,Outcome)  
    elif 'L1' and 'Lt' in L:
        run(LinearSVR(),X3_normalized,X4_normalized,Outcome)   
  
    recupere_resultat(10,50) 

