#!/usr/bin/env python
# coding: utf-8

# ## Dragon Real Estate Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


# for plotting histogram
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))


# In[ ]:





# ## Train-Test splitting

# In[9]:


# for Learning purpose
import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[10]:


#train_set, test_set = split_train_test(housing,0.2)


# In[11]:


#print(f"Rows in train set: {len(train_set)}\nRows in test set:{len(test_set)}\n")


# In[12]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set:{len(test_set)}\n")


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[14]:


strat_test_set.info()


# In[15]:


strat_test_set['CHAS'].value_counts()


# In[16]:


strat_train_set['CHAS'].value_counts()


# In[17]:


#95/7


# In[18]:


#376/28


# In[19]:


housing = strat_train_set.copy()


# ## looking for correlation

# In[20]:


corr_matrix = housing.corr()


# In[21]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[22]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes],figsize = (12,8))


# In[23]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)


# ## trying out Attribute combinations

# In[24]:


housing["TAXRM"] = housing['TAX']/housing['RM']


# In[25]:


housing.head()


# In[26]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[27]:


housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# In[28]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# ## Missing Attributes

# In[29]:


#to take care of missing attributes, you have three option:
# 1. Get rid of the missing data points
 #2. Get rid of the whole attribute
 #3. Set the value to some value(0, mean or median)


# In[30]:


a=housing.dropna(subset=["RM"]) #option 1
a.shape
#note that the original housing dataframe will remain unchanged


# In[31]:


housing.drop("RM", axis=1).shape #option2
# note that there is no RM colum and  also note that the original housing dataframe will remain unchanged


# In[32]:


median = housing["RM"].median() # Compute median for option 3


# In[33]:


housing["RM"].fillna(median)
#note that the original housing dataframe will remain unchanged


# In[34]:


housing.shape


# In[35]:


housing.describe() # before we started filling missing attributes


# In[36]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[37]:


imputer.statistics_


# In[38]:


X = imputer.transform(housing)


# In[39]:


housing_tr = pd.DataFrame(X, columns=housing.columns)


# In[40]:


housing_tr.describe()


# ## Scikit-learn Design

# Primarily, three types of objects
# 1. Estimators - It estimates some parameter based on a dataset. Eg. imputer .It has a fit method and transform method.Fit method - Fit the dataset and calculates internal parameters
# 
# 2. Transformers - transform method takes input and returns output based on the learning form fit(). It also has a convenience function called fit_transform()which fits and then transforms.
# 
# 3. Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions. It also gives score() function which will evaluate the prediction.

# ## Feature Scaling

# primarily, two types of feature scaling methods:
# 1. Min-max sacling (Normalization)
#    (value-min)/(max-min)
#    Sklearn provides a class called MinMaxScaler for this
#   
# 2. Standardization
#    (value- Mean)/std
#    Sklearn provides a class called StandardScalar for this

# ## Creating a Pipline

# In[41]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    #...........add as many as you want in your pipline
    ('std_scalar', StandardScaler()),
])


# In[42]:


housing_num_tr = my_pipeline.fit_transform(housing_tr)


# In[43]:


housing_num_tr


# In[44]:


housing_num_tr.shape


# ## selecting a desired model for Dragon Real Estates

# In[45]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[46]:


some_data = housing.iloc[:5]


# In[47]:


some_labels = housing_labels.iloc[:5]


# In[48]:


prepared_data = my_pipeline.transform(some_data)


# In[49]:


model.predict(prepared_data)


# In[50]:


list(some_labels)


# ## Evaluating the model

# In[51]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[52]:


rmse


# ## Using better evaluation technique - Cross Validation

# In[53]:


#1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,housing_num_tr,housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[54]:


rmse_scores


# In[55]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[56]:


print_scores(rmse_scores)


# ## Saving the model

# In[59]:


from joblib import dump, load
dump(model, 'Dragon.joblib')


# ## Testing the model on test data

# In[66]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
#print(final_predictions, list(Y_test))


# In[62]:


final_rmse


# In[68]:


prepared_data[0]


# ## Using the model

# In[69]:


from joblib import dump, load
import numpy as np
model = load('Dragon.joblib')
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24141041, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(features)


# In[ ]:




