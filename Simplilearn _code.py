
# coding: utf-8

# In[178]:

import numpy as np
import pandas as pd
from sklearn import preprocessing 
from sklearn.decomposition import FactorAnalysis
from numpy import loadtxt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[23]:

df=pd.read_csv("C:/Users/lohit/Desktop/Simplilearn/train.csv")


# In[24]:

# Checking the dataframe and basic statistics such as varianc and mean
# All numeric variable/non-nominal
df.describe().transpose()


# In[25]:

df.var(axis=0)
enum_df = df.select_dtypes(include=['object'])
num_df = df.select_dtypes(exclude=['object'])


# In[26]:

# coulmns with zero variance
num_df = num_df.loc[:, num_df.var() == 0.0]


# In[27]:

#coulmns with zero variance
list(num_df.columns)


# In[30]:

#Ques 1 Droping column with variance = 0
df =df.drop(['X11',
 'X93',
 'X107',
 'X233',
 'X235',
 'X268',
 'X289',
 'X290',
 'X293',
 'X297',
 'X330',
 'X347']
, axis=1)
df.describe().transpose()


# In[37]:

#Ques20
np.logical_not(df.isnull()).sum()
# Checking no of Null Values
df.isnull().sum(axis=0)


# In[38]:

# Null Values across rows
df.isnull().sum(axis=1)


# In[39]:

# drop rows with missing values
#Ques2
df.dropna(inplace=True)
# summarize the number of rows and columns in the dataset
print(df.shape)
# There is no missing values
# No row is deleted


# In[40]:

# Check No of Unique ID
df['ID'].nunique()
# All are uniue IDs, so not replacing any ID


# In[ ]:

# Checking Unique Values in Object Column( Categorical/Nominal)
enum_df
enum_df.nunique()


# In[53]:

# Loading Test Data, so that Label Encoding and Dimension reduction of similar labels and factors
# Will Use PCA or Factor Analysis
dft=pd.read_csv("C:/Users/lohit/Desktop/Simplilearn/test.csv")


# In[54]:

dft.shape


# In[55]:

# Checking the dataframe and basic statistics such as varianc and mean
# All numeric variable/non-nominal
dft.describe().transpose()


# In[56]:

# Removing same variable which we dropped from Train data
dft =dft.drop(['X11', 'X93', 'X107','X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347'], axis=1)


# In[59]:

# Train Test numeric and catgorical column split
enum_df = df.select_dtypes(include=['object'])
num_df = df.select_dtypes(exclude=['object'])
enum_dft = dft.select_dtypes(include=['object'])
num_dft = dft.select_dtypes(exclude=['object'])


# In[61]:

# Checking Unique
enum_df.nunique()
enum_dft.nunique()


# In[73]:

# Train Target Variable - Seperating from Train Data to perform Dimension Reduction
num_df_y=num_df['y']
# Droping ID and y from train and ID from test.
num_df=num_df.drop(['ID','y'],axis=1)


# In[74]:

# Droping  ID from test
num_dft=num_dft.drop(['ID'],axis=1)
# Now All Num and Cat variables are having same columns 
# Train (first 4209 rows) and test next (next 4209 rows)


# In[84]:

# Now appending categorical data and applying label encoder
enum = enum_df.append(enum_dft)
# Appending Numeric data
num= num_df.append(num_dft)


# In[95]:

# Applying Label encoder on categorical data
# From sklearn using one hot key encoder
#Ques3 
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for i in range(0,enum.shape[1]):
    if enum.dtypes[i]=='object':
        enum[enum.columns[i]] = le.fit_transform(enum[enum.columns[i]])


# In[108]:

df_new = pd.concat([enum,num], axis=1)
df_new


# In[121]:

# Ques 4 Dimensionality reduction
# Applying Dimension reduction
# Starting with Factor Analysis using Varimax rotation
# Considering Underoot(n) as no of factor for primary check , will select based on the varaince installed
transformer = FactorAnalysis(n_components=20, random_state=0)
df_transformed = transformer.fit_transform(df_new)
# Checking shape , dimension reduced to 20 from 364
df_transformed.shape


# In[136]:

# Converting array to Dataframe
df_transformed=pd.DataFrame(df_transformed)


# In[137]:

df_train= df_transformed.loc[0:4208,:]


# In[138]:

df_test= df_transformed.loc[4209:,:]


# In[144]:

# Merging Y ( Dependent Variable) with df_train to build model
df_train_final=pd.concat([num_df_y,df_train], axis=1)


# In[145]:

# checking the dataframe
df_train_final


# In[174]:




# In[154]:

# Train Y and X
# num_df_y  , df_train
# Converting data into matrix for XG Boost
data_dmatrix = xgb.DMatrix(data=df_train,label=num_df_y)


# In[155]:

#train dev set split for pretesting model keeping 20 % for development
seed = 7
test_size = 0.2
X_train, X_dev, y_train, y_dev= train_test_split(df_train, num_df_y, test_size=test_size, random_state=seed)


# In[157]:

# setting parameters as standards
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
# Can tune based on accuracy - RMSE in regression or Precision/recall in classification


# In[158]:

# Training Model
xg_reg.fit(X_train,y_train)


# In[159]:

preds = xg_reg.predict(X_dev)


# In[162]:

# Printing RMSE
rmse = np.sqrt(mean_squared_error(y_dev, preds))
print("RMSE: %f" % (rmse))


# In[165]:

# k-fold Cross Validation using XGBoost
#In order to build more robust models, it is common to do a k-fold cross validation where all the entries in the original training dataset are used for both training as well as validation


# In[166]:

params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)


# In[167]:

cv_results.head()


# In[168]:

# Extract and print the final boosting round metric.
print((cv_results["test-rmse-mean"]).tail(1))


# In[169]:

xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)


# In[177]:

#Plotting the feature imporatnce with the matplotlib library:
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()


# In[182]:

# Predicting y on test data
data_test = xgb.DMatrix(data=df_test)
test_pred = xg_reg.predict(data_test)


# In[194]:

test_pred = pd.DataFrame(test_pred)


# In[195]:

# Ques 5
# output result on test data
test_pred


# In[193]:

# Building Final test df data with predicted y
dft_X=pd.read_csv("C:/Users/lohit/Desktop/Simplilearn/test.csv")


# In[202]:

dft_X_ID=dft_X['ID']
dft_X=dft_X.drop(['ID'],axis=1)


# In[203]:

test_data = pd.concat([dft_X_ID,test_pred,dft_X], axis=1)


# In[211]:

test_data.rename(index=str, columns={0: "Y"})
# Output file
test_data.to_csv("C:/Users/lohit/Desktop/Simplilearn/test_result.csv")


# In[ ]:



