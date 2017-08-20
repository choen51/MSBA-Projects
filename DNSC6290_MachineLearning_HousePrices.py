# -*- coding: utf-8 -*-
"""
Machine Learning(2017 Fall) Group Project - Predicting house prices 

@author: Team SDR (Soomin, Rongchu, Dan)
"""

import os
import time
import re

import pandas as pd
import numpy as np
#from sklearn.model_selection import cross_val_score, train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
#from sklearn.metrics import mean_squared_error, make_scorer
#from scipy.stats import skew
#from IPython.display import display
#import matplotlib.pyplot as plt
#import seaborn as sns

import h2o
from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator 
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch 
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

h2o.init(max_mem_size='12G') # give h2o as much memory as possible
h2o.show_progress() # turn off h2o progress bars

pd.set_option('display.float_format', lambda x: '%.3f' % x)


## define all the functions that will be used later ----------------------------------------------------

def get_type_lists(frame, rejects):

    """Creates lists of numeric and categorical variables.
    :param frame: The frame from which to determine types.
    :param rejects: Variable names not to be included in returned lists.
    :return: Tuple of lists for numeric and categorical variables in the frame.
    """
    
    nums, cats = [], []
    for key, val in frame.types.items():
        if key not in rejects:
            if val == 'enum':
                cats.append(key)
            else: 
                nums.append(key)
                
    print('Numeric =', nums)                
    print()
    print('Categorical =', cats)
    
    return nums, cats


def get_type_lists2(pandas_frame, rejects):
# Works the same for pandas dataframe as how the above 'get_type_lists' function does  
    categorical_features = pandas_frame.select_dtypes(include = ["object"]).columns
    numerical_features = pandas_frame.select_dtypes(exclude = ["object"]).columns
    #numerical_features = numerical_features.drop('SalePrice')
    numerical_features = numerical_features.drop(rejects)
                
    print('Numeric =', numerical_features)                
    print()
    print('Categorical =', categorical_features)
    
    return numerical_features, categorical_features


def glm_grid(X, y, train, valid):
    
    """ Wrapper function for penalized GLM with alpha and lambda search.
    
    :param X: List of inputs.
    :param y: Name of target variable.
    :param train: Name of training H2OFrame.
    :param valid: Name of validation H2OFrame.
    :return: Best H2Omodel from H2OGeneralizedLinearEstimator

    """
    
    alpha_opts = [0.001, 0.008, 0.009, 0.01, 0.01, 0.02, 0.03, 0.04, 0.8, 0.85] # always keep some L2
    hyper_parameters = {'alpha': alpha_opts}

    # initialize grid search
    grid = H2OGridSearch(
        H2OGeneralizedLinearEstimator(
            family="gaussian",
            standardize=True,
            lambda_search=True,
            seed=12345),
        hyper_params=hyper_parameters)
    
    # train grid
    grid.train(y=y,
               x=X, 
               training_frame=train,
               validation_frame=valid)

    # show grid search results
    print(grid.show())

    best = grid.get_grid()[0]
    print(best)
    
    # plot top frame values
    yhat_frame = valid.cbind(best.predict(valid))
    print(yhat_frame[0:10, [y, 'predict']])

    # plot sorted predictions
    yhat_frame_df = yhat_frame[[y, 'predict']].as_data_frame()
    yhat_frame_df.sort_values(by='predict', inplace=True)
    yhat_frame_df.reset_index(inplace=True, drop=True)
   # _ = yhat_frame_df.plot(title='Ranked Predictions Plot')
    
    # select best model
    return best



def feature_combiner(pandas_frame, nums):
    
    """ Combines numeric features using simple arithmatic operations.
    
    :param pandas_frame: Training frame from which to generate features and onto which generated 
                           feeatures will be cbound.
    :param test_frame: Test frame from which to generate features and onto which generated 
                       feeatures will be cbound.
    :param nums: List of original numeric features from which to generate combined features.
    
    """
    total = len(nums)
    train_df = pandas_frame
    for i, col_i in enumerate(nums):
        print('Combining: ' + col_i + ' (' + str(i+1) + '/' + str(total) + ') ...')        
        for j, col_j in enumerate(nums):
            
            # don't repeat (i*j = j*i)
            if i < j:
                
                # convert to pandas
                col_i_train_df = train_df[col_i]
                col_j_train_df = train_df[col_j]
 
                # multiply, convert back to h2o
                train_df[str(col_i + '|' + col_j)] = col_i_train_df.values*col_j_train_df.values
    print('Done.')
    print()
    
    return train_df


def gen_submission(preds, test):

    """ Generates submission file for Kaggle House Prices contest.
    
    :param preds: Column vector of predictions.
    :param test: Test data.
    
    """
    
    # create time stamp
    time_stamp = re.sub('[: ]', '_', time.asctime())

    # create predictions column
    sub = test['Id'].cbind(preds.exp())
    sub.columns = ['Id', 'SalePrice']
    
    # save file for submission
    sub_fname = str(time_stamp) + '.csv'
    h2o.download_csv(sub, sub_fname)
    
    
def gen_submission_glm(model, test):

    """ Generates submission file for Kaggle House Prices contest.
    
    :param model: Model with which to score test data.
    :param test: Test data.
    
    """
    
    # create time stamp
    time_stamp = re.sub('[: ]', '_', time.asctime())

    # create predictions column
    sub = test['Id'].cbind(model.predict(test).exp())
    sub.columns = ['Id', 'SalePrice']
    
    # save file for submission
    sub_fname = str(time_stamp) + '.csv'
    h2o.download_csv(sub, sub_fname)


def pred_blender(dir_, files):
    
    """ Performs simple blending of prediction files. 
    
    :param dir_: Directory in which files to be read are stored.
    :param files: List of prediction files to be blended.
    
    """
    
    # read predictions in files list and cbind
    for i, file in enumerate(files):
        if i == 0:
            df = pd.read_csv(dir_ + os.sep + file).drop('SalePrice', axis=1)
        col = pd.read_csv(dir_ + os.sep + file).drop('Id', axis=1)
        col.columns = ['SalePrice' + str(i)]
        df = pd.concat([df, col], axis=1)
        
    # create mean prediction    
    df['mean'] = df.iloc[:, 1:].mean(axis=1)
    print(df.head())
        
    # create time stamp
    time_stamp = re.sub('[: ]', '_', time.asctime())        
        
    # write new submission file    
    df = df[['Id', 'mean']]
    df.columns = ['Id', 'SalePrice']
    
    # save file for submission
    sub_fname = str(time_stamp) + '.csv'
    df.to_csv(sub_fname, index=False)
    

## functions defined above --------------------------------------------------------------------------------------------

'''
(1) Importing data
'''
# Get data
pandas_train = pd.read_csv('train.csv')
pandas_test = pd.read_csv('test.csv')
pandas_train.shape
pandas_test.shape

# Log transform the target for Kaggle scoring
pandas_train.SalePrice = np.log1p(pandas_train.SalePrice)
y = pandas_train.SalePrice



'''
(2) Preprocessing train data
'''
## check the missing value 
missing_df = pandas_train.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['missing_ratio'] = missing_df['missing_count'] / pandas_train.shape[0]
missing_df.ix[missing_df['missing_ratio']>0.50]
    #--> we can see that the missing value issue in our data is not sever so we can impute it using different method based on the characteristic of each input variables

## process each variable
# Lotfrontage
    # We think LotFrontage is related to Neighborhood based on correlation and the boxplots we checked. 
    # Therefore, we imputed the missing LotFrontage by using the Neighbourhood medians.
temp = pandas_train.groupby('Neighborhood', as_index=False)['LotFrontage'].median()
temp = temp.rename(columns={"LotFrontage":"LotFrontage2"})
pandas_train = pd.merge(pandas_train, temp, how='left', on='Neighborhood')
pandas_train['LotFrontage'][pandas_train['LotFrontage'].isnull()] = pandas_train['LotFrontage2'][pandas_train['LotFrontage'].isnull()]
pandas_train = pandas_train.drop('LotFrontage2', axis=1)

# Alley
    #impute the Alley variables using the most fequent value, which is none
pandas_train["Alley"].fillna("None", inplace=True)

# MasVnrType, MasVnrArea
    #same thing for these two variables
pandas_train['MasVnrType'].fillna(pandas_train['MasVnrType'].value_counts().index[0],inplace=True)
pandas_train['MasVnrArea'].fillna(pandas_train['MasVnrArea'].mode()[0],inplace=True)

# Basement related
pandas_train["BsmtQual"].fillna("None", inplace=True)
pandas_train["BsmtCond"].fillna("None", inplace=True)
pandas_train["BsmtExposure"].fillna("None", inplace=True)
pandas_train["BsmtFinType1"].fillna("None", inplace=True)
pandas_train["BsmtFinSF1"].fillna(0, inplace=True)
pandas_train["BsmtFinType2"].fillna("None", inplace=True)
pandas_train["BsmtFinSF2"].fillna(0, inplace=True)
pandas_train["BsmtUnfSF"].fillna(0, inplace=True)

# Electrical
pandas_train["Electrical"].fillna("SBrkr", inplace=True)

# FireplaceQu
pandas_train["FireplaceQu"].fillna("None", inplace=True)

# Garage related
pandas_train["GarageType"].fillna("None", inplace=True)
pandas_train["GarageQual"].fillna("None", inplace=True)
pandas_train["GarageCond"].fillna("None", inplace=True)
pandas_train["GarageFinish"].fillna("None", inplace=True)
pandas_train["GarageCars"].fillna(0, inplace=True)
pandas_train["GarageArea"].fillna(0, inplace=True)

# GarageYrBlt Binning
minval = pandas_train['GarageYrBlt'].min()
maxval = pandas_train['GarageYrBlt'].max()+1
binlist=[0,minval,1920,1940,1960,1980,2000,maxval]
pandas_train['GarageYrBlt'].fillna(0,inplace=True)
#pandas_train['GarageYrBltBins'] = pd.cut(pandas_train['GarageYrBlt'],binlist,include_lowest=True,right=False)

# PoolQC
pandas_train["PoolQC"].fillna("None", inplace=True)

# Fence, MiscFeature
pandas_train["Fence"].fillna("None", inplace=True)
pandas_train["MiscFeature"].fillna("None", inplace=True)

# ------------------------------------------------------
def show_missing(pandas_frame):
    missing = pandas_frame.columns[pandas_frame.isnull().any()].tolist()
    return missing

show_missing(pandas_train)

### Set some numerical predictors that are not truly numerical as categorical 
pandas_train = pandas_train.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })

# Encode some categorical features as ordered numbers when there is information in the order
pandas_train = pandas_train.replace({
                        "Alley" : {"Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"None" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       # "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"None" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoneSeWa" : 2, "NoneSewr" : 3, "AllPub" : 4}}
                     )

# Differentiate numerical variables (minus ID and the target) and categorical variables 
exclude = ['Id','SalePrice']
nums, cats =  get_type_lists2(pandas_train, exclude)
print("Numerical variables : " + str(len(nums)))
print("Categorical variables : " + str(len(cats)))
train_num = pandas_train[nums]
train_cat = pandas_train[cats]

# Handle remaining missing values for numerical variables by using median as replacement
print("NAs for numerical variables in train : " + str(train_num.isnull().values.sum()))
train_num = train_num.fillna(train_num.median())
print("Remaining NAs for numerical variables in train : " + str(train_num.isnull().values.sum()))


## Encoding
# Create dummy features for categorical values using one-hot encoding
print("NAs for categorical variables in train : " + str(train_cat.isnull().values.sum()))
train_cat = pd.get_dummies(train_cat)
print("Remaining NAs for categorical variables in train : " + str(train_cat.isnull().values.sum()))

# Merge categorical and numerical variables
train = pd.concat([train_num, train_cat, pandas_train[['Id','SalePrice']]], axis = 1)
train.shape

# Gets dombined numeric features, multiplied each others
train = feature_combiner(train, nums)
train.shape 

nums, cats =  get_type_lists2(train, exclude)


'''
(3) Preprocessing test data
'''
# Preprocess test data in the same way for the train data

# Lotfrontage
temp = pandas_test.groupby('Neighborhood', as_index=False)['LotFrontage'].median()
temp = temp.rename(columns={"LotFrontage":"LotFrontage2"})
pandas_test = pd.merge(pandas_test, temp, how='left', on='Neighborhood')
pandas_test['LotFrontage'][pandas_test['LotFrontage'].isnull()] = pandas_test['LotFrontage2'][pandas_test['LotFrontage'].isnull()]
pandas_test = pandas_test.drop('LotFrontage2', axis=1)

# Alley
pandas_test["Alley"].fillna("None", inplace=True)

# MasVnrType, MasVnrArea
pandas_test['MasVnrType'].fillna(pandas_test['MasVnrType'].value_counts().index[0],inplace=True)
pandas_test['MasVnrArea'].fillna(pandas_test['MasVnrArea'].mode()[0],inplace=True)

# Basement related
pandas_test["BsmtQual"].fillna("None", inplace=True)
pandas_test["BsmtCond"].fillna("None", inplace=True)
pandas_test["BsmtExposure"].fillna("None", inplace=True)
pandas_test["BsmtFinType1"].fillna("None", inplace=True)
pandas_test["BsmtFinSF1"].fillna(0, inplace=True)
pandas_test["BsmtFinType2"].fillna("None", inplace=True)
pandas_test["BsmtFinSF2"].fillna(0, inplace=True)
pandas_test["BsmtUnfSF"].fillna(0, inplace=True)

# Electrical
pandas_test["Electrical"].fillna("SBrkr", inplace=True)

# FireplaceQu
pandas_test["FireplaceQu"].fillna("None", inplace=True)

# Garage related
pandas_test["GarageType"].fillna("None", inplace=True)
pandas_test["GarageQual"].fillna("None", inplace=True)
pandas_test["GarageCond"].fillna("None", inplace=True)
pandas_test["GarageFinish"].fillna("None", inplace=True)
pandas_test["GarageCars"].fillna(0, inplace=True)
pandas_test["GarageArea"].fillna(0, inplace=True)

# GarageYrBlt Binning
minval = pandas_test['GarageYrBlt'].min()
maxval = pandas_test['GarageYrBlt'].max()+1
binlist=[0,minval,1920,1940,1960,1980,2000,maxval]
pandas_test['GarageYrBlt'].fillna(0,inplace=True)
#pandas_test['GarageYrBltBins'] = pd.cut(pandas_test['GarageYrBlt'],binlist,include_lowest=True,right=False)

# PoolQC
pandas_test["PoolQC"].fillna("None", inplace=True)

# Fence, MiscFeature
pandas_test["Fence"].fillna("None", inplace=True)
pandas_test["MiscFeature"].fillna("None", inplace=True)


# ------------------------------------------------------

show_missing(pandas_test)

## Fill remaining NAs
# MSZoning
pd.crosstab(index=pandas_test['Neighborhood'], columns=pandas_test['MSZoning'])
pandas_test[['Neighborhood', 'MSZoning']][pandas_test['MSZoning'].isnull()]
pandas_test.at[[455,756,790], 'MSZoning'] = 'RM'
pandas_test.at[1444, 'MSZoning'] = 'RL'
# Utilities : NA most likely means all public utilities
pandas_test.loc[:, "Utilities"] = pandas_test.loc[:, "Utilities"].fillna("AllPub")
# Exterior1st
pandas_test.loc[:, 'Exterior1st' ] = pandas_test.loc[:, 'Exterior1st'].fillna("Plywood")
pandas_test.loc[:, 'Exterior2nd' ] = pandas_test.loc[:, 'Exterior2nd'].fillna("Wd Shng")
# TotalBsmtSF
pandas_test['TotalBsmtSF'].fillna(0, inplace=True)
pandas_test['BsmtFullBath'].fillna(0, inplace=True)
pandas_test['BsmtHalfBath'].fillna(0, inplace=True)
# KitchenQual : remodeled in 1950..  should be 'TA' (actually TA are the most frequent for houses remodeled around 1950)
pandas_test['KitchenQual'].fillna('TA', inplace=True)
# Functional : data description says NA means typical
pandas_test['Functional'].fillna('Typ', inplace=True)
# SaleType : sold in 2007, but built in 1958, so not 'NEW',  'WD' is common 
pandas_test['SaleType'].fillna('WD', inplace=True)

show_missing(pandas_test)


pandas_test = pandas_test.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })

pandas_test = pandas_test.replace({
                        "Alley" : {"Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"None" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       # "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"None" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoneSeWa" : 2, "NoneSewr" : 3, "AllPub" : 4}}
                     )


# Differentiate numerical variables (minus ID and the target) and categorical variables 
exclude = ['Id']
nums, cats =  get_type_lists2(pandas_test, exclude)
print("Numerical features : " + str(len(nums)))
print("Categorical features : " + str(len(cats)))
test_num = pandas_test[nums]
test_cat = pandas_test[cats]

# Handle remaining missing values for numerical variables by using median as replacement
print("NAs for numerical variables in train : " + str(test_num.isnull().values.sum()))
test_num = test_num.fillna(test_num.median())
print("Remaining NAs for numerical variables in train : " + str(test_num.isnull().values.sum()))


## Encoding
# Create dummy features for categorical values using one-hot encoding
print("NAs for categorical variables in train : " + str(test_cat.isnull().values.sum()))
test_cat = pd.get_dummies(test_cat)
print("Remaining NAs for categorical variables in train : " + str(test_cat.isnull().values.sum()))

# Merge categorical and numerical variables
test = pd.concat([test_num, test_cat, pandas_test[['Id']]], axis = 1)
test.shape

# Feature combine here
test = feature_combiner(test, nums)
test.shape 


'''
(4) Adjusting train and test data, and spliting frames before modeling
'''
# now remove columns in encoded test not in encoded train, and vice and versa
# (they different b/c of different levels in variables)
train_diff_cols = list(set(train.columns) - set(test.columns))
train_diff_cols.remove('SalePrice')
train_diff_cols2 = list(set(test.columns) - set(train.columns))
train_diff_cols
len(train_diff_cols2)

train.drop(train_diff_cols, axis=1, inplace=True)
test.drop(train_diff_cols2, axis=1, inplace=True)

trainset, validset = h2o.H2OFrame(train).split_frame([0.7], seed=12345) #Convert back to H2O frame and split the frames
print(trainset.shape)
print(validset.shape)
finalnums, finalcats =  get_type_lists(trainset, exclude)
variables = finalnums+finalcats

half_train, other_half_train = trainset.split_frame([0.5], seed=12345)
half_valid, other_half_valid = validset.split_frame([0.5], seed=12345)
print(half_train.shape)
print(half_valid.shape)
print(other_half_train.shape)
print(other_half_valid.shape)

### Create testset and predict for the testset and generate submission file
testset = h2o.H2OFrame(test) #Convert back to H2O frame 
dummy_col = np.random.rand(testset.shape[0])
testset = testset.cbind(h2o.H2OFrame(dummy_col))
cols = testset.columns
cols[-1] = 'SalePrice'
testset.columns = cols

print(testset.shape)
print(trainset.shape)


'''
(5) Running Models and predicting with the models
'''
###### GLM 
    # standardize=True,lambda_search=True, changed alpha depending on the results
glm0_0 = glm_grid(variables, 'SalePrice',  half_train, half_valid)
glm0_1 = glm_grid(variables, 'SalePrice',  other_half_train, other_half_valid)
glm0_2 = glm_grid(variables, 'SalePrice',  trainset, validset) #(Train RMSE: 0.10107719088812989, Valid RMSE: 0.11131002931764424)

pred_blender('', 
             ['Fri_Jun_30_15_12_19_2017.csv',
              'Fri_Jun_30_15_12_17_2017.csv',
              'Fri_Jun_30_15_12_18_2017.csv'])


###### Random Forest (Valid RMSE: 0.12620064722709262)
# initialize rf model
rf_model1 = H2ORandomForestEstimator(
    ntrees=10000,                    
    max_depth=20, 
    col_sample_rate_per_tree=0.1,
    sample_rate=0.8,
    stopping_rounds=50,
    score_each_iteration=True,
    nfolds=3,
    keep_cross_validation_predictions=True,
    seed=12345)           

# train rf model
rf_model1.train(
    x=variables,
    y='SalePrice',
    training_frame=trainset,
    validation_frame=validset)

# print model information
print(rf_model1)

rf_preds1_val = rf_model1.predict(validset)
#ranked_preds_plot('SalePrice', validset, rf_preds1_val) # valid RMSE not so hot ...
rf_preds1_test = rf_model1.predict(testset)
#gen_submission(rf_preds1_test)


##### Extremely random trees model (Valid RMSE: 0.12527112506898552)
# initialize extra trees model
ert_model1 = H2ORandomForestEstimator(
    ntrees=10000,                    
    max_depth=50, 
    col_sample_rate_per_tree=0.1,
    sample_rate=0.8,
    stopping_rounds=50,
    score_each_iteration=True,
    nfolds=3,
    keep_cross_validation_predictions=True,
    seed=12345,
    histogram_type='random') # <- this is what makes it ERT instead of RF

# train ert model
ert_model1.train(
    x=variables,
    y='SalePrice',
    training_frame=trainset,
    validation_frame=validset)

# print model information/create submission
print(ert_model1)
ert_preds1_val = ert_model1.predict(validset)
#ranked_preds_plot('SalePrice', validset, ert_preds1_val) # valid RMSE not so hot ...
ert_preds1_test = ert_model1.predict(testset)
#gen_submission(ert_preds1_test)


##### GBM (Valid RMSE: 0.12028460858594642, Train RMSE: 0.08066981974592213)
# initialize H2O GBM
h2o_gbm_model = H2OGradientBoostingEstimator(
    ntrees = 10000,
    learn_rate = 0.005,
    sample_rate = 0.1, 
    col_sample_rate = 0.8,
    max_depth = 5,
    nfolds = 3,
    keep_cross_validation_predictions=True,
    stopping_rounds = 10,
    seed = 12345)

# execute training
h2o_gbm_model.train(x=variables,
                    y='SalePrice',
                    training_frame=trainset,
                    validation_frame=validset)

# print model information/create submission
print(h2o_gbm_model)
h2o_gbm_preds1_val = h2o_gbm_model.predict(validset)
#ranked_preds_plot('SalePrice', valid, h2o_gbm_preds1_val) # better validation error
h2o_gbm_preds1_test = h2o_gbm_model.predict(testset)
#gen_submission(h2o_gbm_preds1_test)


##### Stacked Ensembles (Valid RMSE: 0.11681957554638625, Train RMSE: 0.053761172832335766)
stack = H2OStackedEnsembleEstimator(training_frame=trainset, 
                                    validation_frame=validset, 
                                    base_models=[rf_model1, ert_model1, 
                                                 h2o_gbm_model])
stack.train(x=variables,
            y='SalePrice',
            training_frame=trainset,
            validation_frame=validset)

# print model information/create submission
print(stack)
stack_preds1_val = stack.predict(validset)
#ranked_preds_plot('SalePrice', validset, stack_preds1_val) 
stack_preds1_test = stack.predict(testset)
#gen_submission(stack_preds1_test, testset) 

