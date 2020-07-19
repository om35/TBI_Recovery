# TBI_Recovery

## Examples to run code : 
### Atlas based method :  
$ python atlas.py ['FA']

### Fixed data driven parcellation with Ward :
$ python Fixed_data_driven.py ['FA']

### Randomized Ensemble based Regression model:
$ python mono_vue.py ['FA']


### Multi-view Learning - Randomized Ensemble based Regression model :  
$ python fichier.py ['modality_1','modality_2']
* Early integration :  $ python multi_precoce.py  ['FA','MD']
* Early integration vstack : $ python multi_precoce_2.py ['FA','L1']
* Intermediate integration : $ python multi_intermediaire.py ['FA','Lt']
* Late integration : $ python multi_tardive.py ['L1','MD']
