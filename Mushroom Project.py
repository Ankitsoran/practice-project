#!/usr/bin/env python
# coding: utf-8

# In[101]:


get_ipython().system('pip install xgboost')


# In[129]:


import os 
import numpy as np 
import pandas as pd 
import warnings
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection  import cross_val_score as CVS
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
sns.set(context='notebook', style='whitegrid', palette='Dark2', font='sans-serif', font_scale=1, color_codes=False, rc=None)
pd.set_option('display.max_columns', None)

warnings.filterwarnings("always")
warnings.filterwarnings("ignore")


# In[103]:


# Mushroom_path = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/mushrooms.csv"
Mushroom_df = pd.read_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/mushrooms.csv")
"Data Shape (row, col): {}".format(Mushroom_df.shape)
Mushroom_df.info()


# In[130]:


M_cols = Mushroom_df.columns.to_list()
print("Value Distribution:\n")
for col in M_cols:
    print(col,"\n",Mushroom_df[col].value_counts(),"\n\n") 


# In[137]:


total = float(len(Mushroom_df[M_cols[1]]))
plt.figure(figsize=(6,6))
sns.set(style="whitegrid")
ax = sns.countplot(Mushroom_df[M_cols[1]])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format(height/total),ha="left")
plt.title("Plot", fontsize = 20)
plt.show()


# In[141]:


for col in M_cols[1:]:
    plt.figure(figsize=(10,4))
    sns.countplot(x=col , data=Mushroom_df ,palette='rainbow')
    plt.title(col, fontsize=14)
    plt.show()
    print("% of total:")
    print(round((Mushroom_df[col].value_counts()/Mushroom_df.shape[0]),4)*100)


# In[139]:


for col in M_cols[1:]: 
    plt.figure(figsize=(30,20))
    plt.subplot(2,3,1)
    sns.countplot(x=col ,hue='class', data=Mushroom_df ,palette='Paired')
    plt.xlabel(col, fontsize=20)
    plt.legend(loc='upper left')


# In[108]:


[ pd.pivot_table(Mushroom_df, index=[col,"class"], aggfunc = {col:np.count_nonzero}) for col in M_cols[1:]]


# In[109]:


Del_cols = ['class','gill-attachment', 'stalk-shape', 'stalk-surface-below-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number'] 
X =Mushroom_df.copy()
for col in Del_cols:
    X = X.drop(col, axis = 1)
X=pd.get_dummies(X,columns=X.columns,drop_first=True)
X.head(2)


# In[110]:


y = Mushroom_df['class']
Encoder_y=LabelEncoder()
y = Encoder_y.fit_transform(y)
y


# In[111]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.75, random_state = 101)


# In[112]:


def cross_val(model_name,model,X,y,cv):
    scores = CVS(model, X, y, cv=cv)
    print(f'{model_name} Scores:')
    for i in scores:
        print(round(i,2))
    print(f'Average {model_name} score: {round(scores.mean(),4)}')

index = ['Valid1','Valid2','Valid3','Valid4','Valid5']
scoring = ['accuracy','precision', 'recall','f1']


# In[113]:


LR = LogisticRegression()
#cross validation
LR_report = pd.DataFrame(cross_validate(LR, X, y, scoring=scoring))
LR_report = LR_report.iloc[:,2:]
#naming the columns
LR_report.columns = scoring
#naming the index
LR_report.index = index

model_1 = []#final result collection

print("Clasification report:\n")#Classification report
print(LR_report,"\n\n")
print("Mean Scores:\n")
for score in scoring:
    print(f"\t{score}: {round(LR_report[score].mean(),4)}")
    model_1.append(round(LR_report[score].mean(),4))


# In[114]:


LR.fit(X_train, y_train)#training
y_predict = LR.predict(X_test)#predicting
Logistic_Regression=pd.DataFrame({'y_test':y_test,'prediction':y_predict})#camparison
Logistic_Regression.to_csv("Logistic Regression.csv")#export


# In[143]:


#model
DT = DecisionTreeClassifier()
#cross validation
DT_report = pd.DataFrame(cross_validate(DT, X, y, scoring=scoring))
#getting required score methods
DT_report = DT_report.iloc[:,2:]
#naming the columns
DT_report.columns = scoring
#naming the index
DT_report.index = index

#final result collection
model_2 = []

#Classification report
print("Clasification report:\n")
print(DT_report,"\n\n")
print("Mean Scores:\n")
for score in scoring:
    print(f"\t{score}: {round(DT_report[score].mean(),4)}")
    model_2.append(round(DT_report[score].mean(),4))


# In[125]:


DT.fit(X_train, y_train)#training
y_predict = DT.predict(X_test)#predicting
Decision_Tree=pd.DataFrame({'y_test':y_test,'prediction':y_predict})#df for camparison
Decision_Tree.to_csv("Decision Tree.csv")


# In[117]:


RF = RandomForestClassifier()
#cross validation
RF_report = pd.DataFrame(cross_validate(RF, X, y, scoring=scoring))
#getting required score methods
RF_report = RF_report.iloc[:,2:]
#naming the columns
RF_report.columns = scoring
#naming the index
RF_report.index = index

#final result collection
model_3 = []

#Classification report
print("Clasification report:\n")
print(RF_report,"\n\n")
print("Mean Scores:\n")
for score in scoring:
    print(f"\t{score}: {round(RF_report[score].mean(),4)}")
    model_3.append(round(RF_report[score].mean(),4))


# In[118]:


#sample prediction for csv file
RF.fit(X_train, y_train)#training
y_predict = RF.predict(X_test)#predicting
Random_Forest = pd.DataFrame({'y_test':y_test,'prediction':y_predict})#df for camparison
Random_Forest.to_csv("Random Forest.csv")#export to csv


# In[119]:


XGB = XGBClassifier()
#cross validation
XGB_report = pd.DataFrame(cross_validate(XGB, X, y, scoring=scoring))
#getting required score methods
XGB_report = XGB_report.iloc[:,2:]
#naming the columns
XGB_report.columns = scoring
#naming the index
XGB_report.index = index

#final result collection
model_4 = []
#Classification report
print("Clasification report:\n")
print(XGB_report,"\n\n")
print("Mean Scores:\n")
for score in scoring:
    print(f"\t{score}: {round(XGB_report[score].mean(),4)}")
    model_4.append(round(XGB_report[score].mean(),4))


# In[120]:


XGB.fit(X_train, y_train)#training
y_predict = XGB.predict(X_test)#predicting
XGBoost=pd.DataFrame({'y_test':y_test,'prediction':y_predict})#df for camparison
XGBoost.to_csv("XGBoost.csv")


# In[122]:


SVM = SVC(kernel = 'linear')
#cross validation
SVM_report = pd.DataFrame(cross_validate(SVM, X, y, scoring=scoring))
#getting required score methods
SVM_report = SVM_report.iloc[:,2:]
#naming the columns
SVM_report.columns = scoring
#naming the index
SVM_report.index = index

#final result collection
model_5 = []
#Classification report
print("Clasification report:\n")
print(SVM_report,"\n\n")
print("Mean Scores:\n")
for score in scoring:
    print(f"\t{score}: {round(SVM_report[score].mean(),4)}")
    model_5.append(round(SVM_report[score].mean(),4))
    


# In[124]:


Models = pd.DataFrame({
    'Logistic Regression': np.array(model_1),
    'Decision Tree': np.array(model_2),
    'Random Forest': np.array(model_3),
    'XGBoost': np.array(model_4),
    'Support Vector Machines': np.array(model_5)})
Models.index = scoring
#Models.sort_values(by='Score', ascending=False)
Models


# In[ ]:




