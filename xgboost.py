# -*- coding: utf-8 -*-
"""
This script is for the Kaggle project for Ames, Iowa house price data.
Author: Jason R. Dunavin\
Last revision: 20170115 
"""

### Imports
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import model_selection
import matplotlib.pylab as plt

### Read in the data
train = pd.read_csv("./Data/train.csv", header=0)
test = pd.read_csv("./Data/test.csv", header=0)

### Clean up the data

# Fill in na's with 0
train = train.fillna(0)
test = test.fillna(0)

# Separate the features into categorical, ordered, and numeric
features = [x for x in train.columns if x not in ['id','SalePrice']]
cat_features = ['MSSubClass', 'MSZoning', 'Street','Alley', 
'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
'HouseStyle','OverallQual','OverallCond',
'Foundation', 'BsmtFinType1','BsmtFinType2', 
'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
'Heating','CentralAir','Electrical','BsmtFullBath',
'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
'TotRmsAbvGrd','Functional','Fireplaces','GarageType',
'GarageFinish','GarageCars','PavedDrive',
'Fence','MiscFeature','SaleType','SaleCondition']
ord_cat_features = ['ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure',
                   'HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']
num_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2',
'BsmtUnfSF','TotalBsmtSF', 'LowQualFinSF','1stFlrSF','2ndFlrSF',
'GrLivArea', 'YearRemodAdd', 'YearBuilt','GarageYrBlt','GarageArea','WoodDeckSF',
'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold']

# Prepare to encode ordered variables
# glue data sets together
train_test = pd.concat((train[features], test[features])).reset_index(drop=True)
# Convert categoricals to codes
for c in range(len(cat_features)):
    train_test[cat_features[c]] = train_test[cat_features[c]].astype('category').cat.codes

    #Define a generic function using Pandas replace function
def coding(col, codeDict):
  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded
  
train_test['ExterQual'] = coding(train_test['ExterQual'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1})
train_test['ExterCond'] = coding(train_test['ExterCond'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1})
train_test['BsmtQual'] = coding(train_test['BsmtQual'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1, 0:0})
train_test['BsmtCond'] = coding(train_test['BsmtCond'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1, 0:0})
train_test['BsmtExposure'] = coding(train_test['BsmtExposure'], {'Gd':4,'Av':3,'Mn':2,'No':1, 0:0})
train_test['HeatingQC'] = coding(train_test['HeatingQC'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1})
train_test['KitchenQual'] = coding(train_test['KitchenQual'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1})
train_test['FireplaceQu'] = coding(train_test['FireplaceQu'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1, 0:0})
train_test['GarageQual'] = coding(train_test['GarageQual'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1, 0:0})
train_test['GarageCond'] = coding(train_test['GarageCond'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1, 0:0})
train_test['PoolQC'] = coding(train_test['PoolQC'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1, 0:0})

# Re-split the training and test sete again
rows_train = len(train)
rows_test = len(test)
trainX = train_test.iloc[:rows_train,:]
testX = train_test.iloc[rows_train:,:]

# Transform neigborhoods
def transform_nb(x):
    if x in ("NoRidge", "NridgHt", "StoneBr"):
        return 5 #Over 250
    elif x in ('CollgCr','Veenker','Crawfor','Somerst','Timber','ClearCr'):
        return 4 #200-250
    elif x in ('Mitchel','NWAmes','SawyerW','Gilbert','Blmngtn','SWISU', 'Blueste'):
        return 3 #150-200
    elif x in ('OldTown','BrkSide','Sawyer','NAmes','IDOTRR','Edwards','BrDale', 'NPkVill'):
        return 2 #100 - 150
    elif x in ('MeadowV'):
        return 1
    else:
        return 9 # Catch mistakes

train['NbdClass'] = train['Neighborhood'].apply(transform_nb)
test['NbdClass'] = test['Neighborhood'].apply(transform_nb)
trainX['NbdClass'] = train['NbdClass']
z = test.loc[:,'NbdClass']
z=np.asarray(z)
testX.loc[:,'NbdClass'] = z

### MORE FEATURE ENGINEERING           
