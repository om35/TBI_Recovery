"""
TBI_Recovery:  MeCA Team - Institut de Neurosciences de la Timone
Author:
    Mohamed Ouerfelli  & Sylvain Takerkart
"""
import os
import numpy as np
import sys
import nibabel as nb
import time
import matplotlib.pyplot as plt
import pandas as pd
from nilearn import image
from nilearn.input_data import NiftiMasker
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR
from sklearn.cluster import FeatureAgglomeration 
from sklearn.metrics import r2_score
from nilearn.input_data import NiftiLabelsMasker
from sklearn.preprocessing import normalize


import pickle
def warn(*args, **kwargs):
    pass
import warnings

warnings.warn = warn


os.path.join(os.path.dirname(__file__))
if not os.path.exists("OUTPUT_atlas"):
    os.makedirs("OUTPUT_atlas")

#########################################################"
#define path of data (NIFTI Image of BrainTale)
label_path = 'TC_Lionel.xls'
#Define Atlas 
atlas_filename = 'JHU-ICBM-labels-1mm.nii.gz'  

maskerr = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                           memory='nilearn_cache',verbose=5)
                           
                
def read_nifti(reference,path):
    
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


def read_labels():
    """"
    Input : file.xls
    Output : Outcome , ID_Patient
    This function allows us to read the vector Y (Outcome) with some constraints : 
        -Exclude patients who have "Excluded" in observations column
    """
    data = pd.read_excel(label_path)
    outcome=[]
    ID_PATIENT = []
    for i in range (data.shape[0]):
        a= str(data['Observations'][i])
        b= str(data['Outcome'][i])
        c= str(data['DISPO'][i])
        if a.startswith('EXC') or a.startswith('exc')  or a.startswith('Exc') :
            continue
        elif b != 'nan' and b!='-3' and c == 'YES':
            outcome.append(data['Outcome'][i])
            ID_PATIENT.append(data['Patient ID'][i])
    label = np.array(outcome).reshape(len(outcome), 1)   #Convert Outcome to  vector (array of shape (len(Outcome),1))
    idP = np.array(ID_PATIENT).reshape(len(ID_PATIENT), 1)     #Convert ID_patient to vector (array of shape (len(ID_patient),1))
    return label , idP



#Fixed Parcellation
def extract(ID_patient,path): 
    
    image_list =[]
    for element in os.listdir(path):  # loop into all patients
        if element in ID_patient :
            img = read_nifti(element,path)
            image_list.append(img)         #Contains all images 3D
    zz=maskerr.fit_transform(image_list)
    
    return zz



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
    """
    coef = []
    for i in range(weights.shape[1]):
        coef.append(np.mean(weights[:,i]))  
    W = np.array(coef).reshape(1,-1)
    return W



def run_simplest_version(model,X,y):
    train_idx=pickle.load(open('INDEX/Fulltrain_idx', 'rb'))
    test_idx = pickle.load(open('INDEX/Test_idx', 'rb'))
    scores=[]
    coef=[]
    for train,test in zip(train_idx,test_idx):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]      
        L= [0.01,0.1,1,10,100,150,170,200,250,300,350,370,390,400,420,450,470,500,550,570,600,650,670,700,750,800,820,850,900,1000,1200,1500,2000,5000]
        liste_r2=[]
        liste_coef=[]
        for l in L:
            mod = model.set_params(C = l)
            mod.fit(X_train,y_train)
            predict = mod.predict(X_test) 
            r2=r2_score(y_test,predict)
            liste_r2.append(r2)
            liste_coef.append(mod.coef_)
        Best_R2 = np.max(liste_r2) 
        Best_index =liste_r2.index(Best_R2)
        Best_coef= liste_coef[Best_index]
        scores.append(Best_R2)
        coef.append(Best_coef)
    
    weight= np.vstack(coef)
    to_voxel= mean_weights(weight) 
    neg= maskerr.inverse_transform(to_voxel) 
    nb.save(neg, "OUTPUT_atlas/weight_map")
    pickle.dump(scores, open('OUTPUT_atlas/R2_sc', 'wb'))
    pickle.dump(coef, open('OUTPUT_atlas/weight', 'wb'))     


if __name__ == "__main__":
    path = sys.argv[1]
    warnings.warn = warn
    Outcome , ID_patient = read_labels()  #Read Outcome and ID_Patient 
    if 'L1' in path : 
        X = extract(ID_patient,'L1_view')
    if 'MD' in path : 
        X = extract(ID_patient,'MD_view')
    if 'Lt' in path : 
        X = extract(ID_patient,'Lt_view')    
    if 'FA' in path : 
        X = extract(ID_patient,'FA_view')
    run_simplest_version(LinearSVR(),X,Outcome)  


 
