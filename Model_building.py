############################# MODEL BUILDING


#####Importing libraries

import numpy as np # linear algebra
import pandas as pd # data processing 
import os
import io
import itertools
import warnings
import math  # for missing values 
import matplotlib.pyplot as plt #visualization
from PIL import  Image
%matplotlib inline
import seaborn as sns #visualization
warnings.filterwarnings("ignore")
import plotly.offline as py #visualization
py.init_notebook_mode(connected=True) #visualization
import plotly.graph_objs as go #visualization
import plotly.tools as tls #visualization
import plotly.figure_factory as ff #visualization

from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # encode categorical variables 
from sklearn.preprocessing import StandardScaler  # standardize data
from sklearn.model_selection import train_test_split  # split between train and test split
from xgboost import XGBClassifier  # classification xgboost
import xgboost as xgb # xgboost 
from yellowbrick.classifier import DiscriminationThreshold  # find best threshold for classification
from imblearn.over_sampling import SMOTE   # unbalanced dataset
from imblearn.over_sampling import SMOTENC  # unbalanced dataset, deals with categorical variables
from sklearn.model_selection import cross_val_score  # assess performance, avoid overfitting 
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report, roc_auc_score,roc_curve,scorer,f1_score, precision_score,recall_score  # performance metrics
import statsmodels.api as sm
from sklearn.feature_selection import RFECV, RFE  # feature selection 
from xgboost import plot_tree  # plot Tree XGBoost



# Import dataset 
telcom = pd.read_csv('Telco_customer_churn.csv')
dataset = pd.read_csv('Telco_customer_churn.csv')



########## DATA MANIPULATION 

# Identify missing values 
w=float('nan')
math.isnan(w)
telcom.isna().sum()


# Missing values approach
telcom[telcom['TotalCharges']== " "] # 11 missing values for the Total charges column
telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ",np.nan) # replace spaces with null values in TotalCharges column
telcom = telcom[telcom["TotalCharges"].notnull()] # drop these observations 
telcom = telcom.reset_index()[telcom.columns] # reset index 
telcom["TotalCharges"] = telcom["TotalCharges"].astype(float) #convert to float type


# Replace 'No internet service' to 'No' for the following columns. If they don't have internet, they don't have access to these services. 
replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport','StreamingTV', 'StreamingMovies']           
for i in replace_cols : 
    telcom[i]  = telcom[i].replace({'No internet service' : 'No'})
    

# Replace 'No phone service' to 'No' for the MultipleLines columns. 
telcom['MultipleLines'] = telcom['MultipleLines'].replace({'No phone service':'No'})


# Modify Senior Citizen binary dummy variable. Want Yes or No instead of 0,1; only because other dummies are in this format. Easier to deal with. 
telcom["SeniorCitizen"] = telcom["SeniorCitizen"].replace({1:"Yes",0:"No"})


# Separating churn and non churn customers. 
churn     = telcom[telcom["Churn"] == "Yes"]
not_churn = telcom[telcom["Churn"] == "No"]


# Customer id col
Id_col     = ['customerID']
# Target columns
target_col = ["Churn"]
# Categorical columns
categorical_features = [0,1,2,3,5,6,7,8,9,10,11,12,13,16,17,18,19,20,21,22,23,24,25]
cat_cols   = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
# Continuous columns
num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]
# Binary columns with 2 values
bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
# Multiple categorical columns (more than 2 values)
multi_cols = [i for i in cat_cols if i not in bin_cols]




##########  Feature engineering part



# Label encoding for Binary columns
le = LabelEncoder()
for i in bin_cols :
    telcom[i] = le.fit_transform(telcom[i])
  
    
# Encode categorical variables. 
telcom = pd.get_dummies(data = telcom,columns = multi_cols )
# No need to delete some dummies to avoid the dummy variable trap, Python handles it. 


# Define dependent and independent variables 
X = telcom.iloc[:, 1: 32]  # delete customer ID from X because won't use it to predict churn. 
X = X.drop(['Churn'], axis = 1) # create dataframe with only independent variables. 
X = X.astype(float)   # keep a dataframe but where all columns are floats. 
Z = X.values  # Save X as an array under the name Z
y = telcom['Churn'].values   # dependent variables, as array. 
features = [i for i in telcom.columns if i not in Id_col + target_col] # Store the name of all variables. 


"""
NOT NEEDED FOR XGBOOST

#Scale Numerical variables
std = StandardScaler()
X[num_cols] = std.fit_transform(X[num_cols])
"""

# Split the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(Z, y, test_size = 0.2, random_state = 44)




##########   Build model 


# Fit XGBoost to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


# Apply k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print('cross val: ', accuracies.mean()) # 0.8012494179982639
print('cross val std: ', accuracies.std()) # 0.012701147404560334


# Predict the Test set results
y_pred = classifier.predict(X_test)
y_proba = classifier.predict_proba(X_test)   # illustrates confidence about predictions. Useful for interpretations. 


# Make the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels = [1, 0])
print(cm) # [211 170]
          #[ 98 928]


# Additional metrics to measure performance 
print (classification_report(y_test,y_pred))
print (" Accuracy score: " , accuracy_score(y_test,y_pred))  #0.8095238095238095


# roc_auc_score
model_roc_auc = roc_auc_score(y_test,y_pred) 
print ('AUC: ', model_roc_auc) # 0.7291446025387177
fpr,tpr,thresholds = roc_curve(y_test,y_proba[:,1])  # will be used to find optimal threshold point 
    

# Plot confusion matrix
trace1 = go.Heatmap(z = cm ,
                    x = ["Not churn","Churn"],
                    y = ["Not churn","Churn"],
                    showscale  = False,colorscale = "Picnic",
                    name = "matrix")
    

# Plot roc curve
trace2 = go.Scatter(x = fpr,y = tpr,
                    name = "Roc : " + str(model_roc_auc),
                    line = dict(color = ('rgb(22, 96, 167)'),width = 2))

trace3 = go.Scatter(x = [0,1],y=[0,1],
                    line = dict(color = ('rgb(205, 12, 24)'),width = 2,
                    dash = 'dot'))
  

# Feature importance 
coefficients  = pd.DataFrame(classifier.feature_importances_)
column_df     = pd.DataFrame(features)
coef_sumry    = pd.merge(coefficients,column_df,left_index= True,right_index= True, how = "left")
coef_sumry.columns = ["coefficients","features"]
coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)
 
# Plot feature importance
trace4 = go.Bar(x = coef_sumry["features"],y = coef_sumry["coefficients"],
                name = "coefficients",
                marker = dict(color = coef_sumry["coefficients"],
                              colorscale = "Picnic",
                              line = dict(width = .6,color = "black")))
    

# Subplots - layout plots created above 
fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                        subplot_titles=('Confusion Matrix',
                        'Receiver operating characteristic',
                        'Feature Importances'))

fig.append_trace(trace1,1,1)
fig.append_trace(trace2,1,2)
fig.append_trace(trace3,1,2)
fig.append_trace(trace4,2,1)
    
fig['layout'].update(showlegend=False, title="Model performance" ,
                     autosize = False,height = 900,width = 800,
                     plot_bgcolor = 'rgba(240,240,240, 0.95)',
                     paper_bgcolor = 'rgba(240,240,240, 0.95)',
                     margin = dict(b = 195))
fig["layout"]["xaxis2"].update(dict(title = "false positive rate"))
fig["layout"]["yaxis2"].update(dict(title = "true positive rate"))
fig["layout"]["xaxis3"].update(dict(showgrid = True,tickfont = dict(size = 10),tickangle = 90))
py.iplot(fig)
    


# Usually, we classify an object to a class if the probabily of belonging to this class is above 0.5
# However, this threshold can be adjusted and this function allows to find the optimal value given some metrics (recall, precison, f1, queue rate)
# Find optimal threshold  
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
# The optimal threshold is 0.30. 

# Better method. 
visualizer = DiscriminationThreshold(classifier)
visualizer.fit(X_train,y_train)
visualizer.poof()

        


##########################################   IMPROVEMENTS 




# Implement SMOTE 
"""
# SMOTE is not appropriate as it does not deal with dummy variables. 
sm = SMOTE()
X_smote, y_smote = sm.fit_resample(X_train, y_train) 
# Use SMOTENC instead, which does. 
"""
sm = SMOTENC(random_state=40, categorical_features=categorical_features)
X_smote, y_smote = sm.fit_resample(X_train, y_train)


# Train the model on the resampled training dataset. No more unbalanced classes. 
X_train = X_smote
y_train = y_smote


# Check distribution, set according to the train/test split performed earlier. 
unique, counts = np.unique(y_train, return_counts=True)
print('training set: ', dict(zip(unique, counts)))
# There are 4137 observations of each class (churn and non churn) in the training set! 
unique, counts = np.unique(y_test, return_counts=True)
print('test set: ', dict(zip(unique, counts)))
# In the test set, SMOTE is not applied, which leaves 1026 non churners and 381 churners. 
unique, counts = np.unique(y, return_counts=True)
print('original dataset: ', dict(zip(unique, counts)))
# In total we have our 5163 non churners! Only number of churners has been increased (from 1869 to 4518)


# Fit XGBoost to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
 

# Recursive Feature elimination (RFE), only informative.  
rfe = RFE(classifier, n_features_to_select = 1)
X_opt2 = rfe.fit_transform(X_train, y_train)
print(rfe.ranking_)
rfe_rank = rfe.ranking_
for i in range(0, len(features)): 
    print rfe_rank[i], features[i]
# Obtain a ranking of feature importance


# RFECV : Recursive feature selection 
rfecv = RFECV(classifier, cv = 5, scoring = 'roc_auc')
X_opt = rfecv.fit_transform(X_train, y_train)
print(rfecv.support_)   # summarize selection 
print(rfecv.n_features_)
# No feature should be deleted! 
# From data vizualisation, we could have deleted 3 features: gender, DeviceProtection and DSL.


# Find optimal threshold  
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
# Better method. 
visualizer = DiscriminationThreshold(classifier)
visualizer.fit(X_train,y_train)
visualizer.poof()
# With SMOTE, the threshold is back to normal (approximatiely 0.5, exactly 0.52) so we won't set a particular threshold when building the model. 


# Parameter tuning 
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'learning_rate':[0.15, 0.2], 'n_estimators':[150], 'max_depth': [4,3], 'min_child_weight':[0.6], 'gamma':[0], 'subsample':[1], 'subsample':[1], 'colsample_bytree':[1], 'reg_alpha':[0] }] 
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'roc_auc',
                           cv = 5)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_accuracy, best_parameters)
# Only tiny improvements in performance 





##################### FINAL MODEL




# Fit tuned XGBoost to the Training set
classifier = XGBClassifier( n_estimators=150, learning_rate=0.15, max_depth=3, min_child_weight=0.6,
                            colsample_bytree=1, subsample=1, reg_alpha=0, gamma=0,
                            booster='gbtree', objective='binary:logistic') 
classifier.fit(X_train, y_train)


# Predict the Test set results
y_pred = classifier.predict(X_test)
y_proba = classifier.predict_proba(X_test)   # illustrates confidence about predictions. Useful for interpretations. 


# Make the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels = [1, 0])
print(cm) # [[271 110]
          # [204 822]]


# Additional metrics to measure performance 
print (classification_report(y_test,y_pred))  
print (" Accuracy score test set : " , accuracy_score(y_test,y_pred))  # 0.7768301350390903


# roc_auc_score
model_roc_auc = roc_auc_score(y_test,y_pred) 
print ('AUC test set: ', model_roc_auc)  # 0.75622783994106
fpr,tpr,thresholds = roc_curve(y_test,y_proba[:,1])  # will be used to find optimal threshold point 
    

# Plot confusion matrix
trace1 = go.Heatmap(z = cm ,
                    x = ["Not churn","Churn"],
                    y = ["Not churn","Churn"],
                    showscale  = False,colorscale = "Picnic",
                    name = "matrix")
    

# Plot roc curve
trace2 = go.Scatter(x = fpr,y = tpr,
                    name = "Roc : " + str(model_roc_auc),
                    line = dict(color = ('rgb(22, 96, 167)'),width = 2))

trace3 = go.Scatter(x = [0,1],y=[0,1],
                    line = dict(color = ('rgb(205, 12, 24)'),width = 2,
                    dash = 'dot'))
  

# Feature importance 
coefficients  = pd.DataFrame(classifier.feature_importances_)
column_df     = pd.DataFrame(features)
coef_sumry    = pd.merge(coefficients,column_df,left_index= True,right_index= True, how = "left")
coef_sumry.columns = ["coefficients","features"]
coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)
 

# Plot feature importance
trace4 = go.Bar(x = coef_sumry["features"],y = coef_sumry["coefficients"],
                name = "coefficients",
                marker = dict(color = coef_sumry["coefficients"],
                              colorscale = "Picnic",
                              line = dict(width = .6,color = "black")))
    
# Subplots - layout plots created above 
fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                        subplot_titles=('Confusion Matrix',
                        'Receiver operating characteristic',
                        'Feature Importances'))

fig.append_trace(trace1,1,1)
fig.append_trace(trace2,1,2)
fig.append_trace(trace3,1,2)
fig.append_trace(trace4,2,1)
    
fig['layout'].update(showlegend=False, title="Model performance" ,
                     autosize = False,height = 900,width = 800,
                     plot_bgcolor = 'rgba(240,240,240, 0.95)',
                     paper_bgcolor = 'rgba(240,240,240, 0.95)',
                     margin = dict(b = 195))
fig["layout"]["xaxis2"].update(dict(title = "false positive rate"))
fig["layout"]["yaxis2"].update(dict(title = "true positive rate"))
fig["layout"]["xaxis3"].update(dict(showgrid = True,tickfont = dict(size = 10),tickangle = 90))
py.iplot(fig)
    



###### Plot tree XGBoost - The following is analysed in the interpretability part, as feature contribution

# This is the first tree plotted, which is built on the training set. Probably focus instead on Decision Tree that approximates XGBoost (Global surrogate model). 
plot_tree(classifier, num_trees=0)
fig = plt.gcf()
fig.set_size_inches(100, 50)
fig.savefig('tree.png')
# Interpretation of this tree. Features used are ‘Contract_Month-to-month', MonthlyCharges, 'PaymentMethod_Electronic check’,  'Contract_One year’, 'tenure'
# They are all among the most relevant features. 

# Plot another tree to see the difference. 
plot_tree(classifier, num_trees=5)
fig = plt.gcf()
fig.set_size_inches(150, 100)
fig.savefig('tree2.png')
# Not many differences, still uses ‘Contract_Month-to-month', 'MonthlyCharges', 'Contract_One year’, 'tenure' and additionally ncludes 'PaperlessBilling’ and 'Dependents'
# 2 new variables, also very important are now considered. 

# PLot another tree, this time much weaker. 
plot_tree(classifier, num_trees=70)
fig = plt.gcf()
fig.set_size_inches(150, 100)
fig.savefig('tree3.png')
# MultipleLines, TotalCharges, Monthly Charges, Credit Card. 
# When we vizualize the last trees built, they are smaller and yield smaller leaf value. 

# No names for the vizualised tree. Correspondance bewteen 'fi' and variable names is found here.  
dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.feature_names 
features 


##### Other feature importance graph, with figures but no names as well. 
xgb.plot_importance(classifier, color='red')
plt.title('importance', fontsize = 20)
plt.yticks(fontsize = 10)
plt.ylabel('features', fontsize = 20)




