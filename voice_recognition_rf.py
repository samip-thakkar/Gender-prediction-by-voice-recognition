# -*- coding: utf-8 -*-
"""

@author: Samip
"""

#Ignore the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import missingno as msno

#Configure the style
style.use('fivethirtyeight')
sns.set(style = 'whitegrid', color_codes = True)

#Import algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Import model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV

#Load the dataset
df = pd.read_csv('voice.csv')

""" The idea is to identify a voice as male or female, based upon the acoustic properties of the voice and speech. 
The dataset consists of 3,168 recorded voice samples, collected from male and female speakers."""

#Check for null values
print("Checking null values: ", df.isnull().any())

"""We will perform univariate analysis for the outlier detection. Other than plotting a boxplot/histogram for each column, we create a small
utility function that tells no. of remaing observations in features, if we remove its outliers."""

"""Here 1.5 Interquartile Range rule is used. If the observation is less than ‘first quartile — 1.5 IQR’ or greater than 'third quartile +1.5 IQR' is an outlier."""

#Function to calculate the limit/ range
def calc_limits(features):
    q1, q3 = df[features].quantile([0.25, 0.75])
    iqr = q3 - q1
    rnge = iqr * 1.5
    return (q1 - rnge, q3 + rnge)

#Plot function
def plot(features):
    fig, axes = plt.subplots(1, 2)
    sns.boxplot(data = df, x = features, ax = axes[0])
    sns.distplot(a = df[features], ax = axes[1], color = 'red')
    fig.set_size_inches(15, 5)
    lower, upper = calc_limits(features)
    l = [df[features] for i in df[features] if i > lower and i < upper]
    print("Number of data points remaining if outliers removed: ", len(l))
    
#Plotting first feature: meanfreq
plot('meanfreq')

"""Inferences made from the result:
1. We have some outliers wrt our rule (represented by dots). Removing these outliers leaves us with 3104 values.
2. From displot, we can say that the distribution is little bit skewed, and hence we need to normalize the data.
3. Left tail distribution has more outliers to Q1 than right of Q3."""

"Plotting and checking for target class."""

sns.countplot(data = df, x = 'label')
print(df['label'].value_counts())

"""We have perfectly balanced target feature as we have same number of males and females.
Now, we will perform a bivariate analysis to find the correlation between different features."""

temp = []
for i in df['label']:
    if i == 'male':
        temp.append(0)
    else:
        temp.append(1)
df['label'] = temp

#Correlation matrix
corr_mat = df[:].corr()
mask = np.array(corr_mat)
mask[(np.tril_indices_from(mask))] = False
fig = plt.gcf()     #gcf: Get Current Figure
fig.set_size_inches(23, 9)
sns.heatmap(data = corr_mat, mask = mask, square = True, annot = True, cbar = True)

"""Inferences made from the plot is that we can drop the columns which are hhighly correlated to each other, 
in order to remove the redundancy. We can use PCA for dimensionality reduction in feature space."""

df.drop('centroid', axis = 1, inplace = True)


#Outlier treatment
"""From univeriate analysis, we came to know about some of the outliers. We can either remove the corresponding data points or we can 
replace these values with any statistical value (like median, which are robust to the outliers)."""

#Removing the outliers
for col in df.columns:
    lower, upper = calc_limits(col)
    df = df[(df[col] > lower) & (df[col] < upper)]
print("New data shape: ", df.shape)


#Feature Engineering
temp_df = df.copy()
temp_df.drop(['skew', 'kurt', 'mindom', 'maxdom'], axis = 1, inplace = True)

"""Now, we will create statistical entities: meanfreq, median and mode with relation 3med = 2mod + mean."""
temp_df['meanfreq'] = temp_df['meanfreq'].apply(lambda x: x * 2)
temp_df['median'] = temp_df['meanfreq'] + temp_df['mode']
temp_df['median'] = temp_df['median'].apply(lambda x: x / 3)

sns.boxplot(data = temp_df, x = 'label', y = 'median')

"""Now, we will add new feature to measure the skewness. Hear, we use Karl Pearson Coefficient: coefficient = (Mean - Mode) / StandardDeviation."""
temp_df['pear_skew'] = temp_df['meanfreq'] - temp_df['mode']
temp_df['pear_skew'] = temp_df['pear_skew'] / temp_df['sd']
sns.boxplot(data = temp_df, x = 'label', y = 'pear_skew')

#Normalizing the data
ss = StandardScaler()
scaled_df = ss.fit_transform(temp_df.drop('label', axis = 1))
X = scaled_df
y = df['label'].as_matrix()

#Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


#Model building. We will built model using Decision Tree and Random Forest classifier and compare both results
models = [RandomForestClassifier(), DecisionTreeClassifier()]
model_names = ['RandomForest', 'DecisionTree']

acc = []
d = {}
for model in range(len(models)):
    clf = models[model]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc.append(accuracy_score(y_pred, y_test))

d = {'Algorithm': model_names, 'Accuracy': acc}
acc_df = pd.DataFrame(d)
print(acc_df)

#Plot the accuracies
sns.barplot(x = 'Accuracy', y = 'Algorithm', data = acc_df)
"""As expected, Random Forest accuracy was more than Decision Tree Classifier."""

#Parameter tuning with GridSearchcv (It implements fit and score method, also others if implemented).

"""Parameter tuning with GridSearchCV for Random Forest."""
param_grid = {'n_estimators': [200, 500], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth' : [4,5,6,7,8], 'criterion' :['gini', 'entropy']}
CV_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, scoring='accuracy', cv= 5)
CV_rf.fit(X_train, y_train) 
print("Best score : ",CV_rf.best_score_)
print("Best Parameters : ",CV_rf.best_params_)
print("Precision Score : ", precision_score(CV_rf.predict(X_test),y_test))   

"""Checking importance of each feature using plot."""
df1 = pd.DataFrame.from_records(X_train)     
tmp = pd.DataFrame({'Feature': df1.columns, 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()