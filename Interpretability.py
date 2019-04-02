# Alexandre Duval
# Master Research Project 
########################## INTERPRETATION METHODS 

#Import libraries
import numpy as np # linear algebra
import pandas as pd # data processing 
import os
import io
import itertools
import warnings
import math  # for missing values 
import matplotlib.pyplot as plt#visualization
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
from xgboost import XGBClassifier  # model 
import xgboost as xgb
from imblearn.over_sampling import SMOTENC  # unbalanced dataset, deals with categorical variables
from sklearn.model_selection import cross_val_score  # assess performance, avoid overfitting 
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report, roc_auc_score,roc_curve,scorer,f1_score, precision_score,recall_score  # performance metrics
import statsmodels.api as sm
from xgboost import plot_tree  # plot Tree XGBoost



# Import dataset 
telcom = pd.read_csv('Telco_customer_churn.csv')
dataset = pd.read_csv('Telco_customer_churn.csv')



############# DATA MANIPULATION 


## Identify missing values 
w=float('nan')
math.isnan(w)
telcom.isna().sum()

# Missing values 
telcom[telcom['TotalCharges']== " "] # 11 missing values for the Total charges column
telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ",np.nan) # replace spaces with null values in TotalCharges column
telcom = telcom[telcom["TotalCharges"].notnull()] # drop these observations 
telcom = telcom.reset_index()[telcom.columns] # reset index 
telcom["TotalCharges"] = telcom["TotalCharges"].astype(float) #convert to float type

#Replace 'No internet service' to 'No' for the following columns. If they don't have internet, they don't have access to these services. 
replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport','StreamingTV', 'StreamingMovies']           
for i in replace_cols : 
    telcom[i]  = telcom[i].replace({'No internet service' : 'No'})
    
# Replace 'No phone service' to 'No' for the MultipleLines columns. 
telcom['MultipleLines'] = telcom['MultipleLines'].replace({'No phone service':'No'})
    
# Modify Senior Citizen binary dummy variable. Want Yes or No instead of 0,1; only because other dummies are in this format. Easier to deal with. 
telcom["SeniorCitizen"] = telcom["SeniorCitizen"].replace({1:"Yes",0:"No"})
                                      
#Separating churn and non churn customers. 
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
#numerical columns
num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]
#Binary columns with 2 values
bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
#Columns more than 2 values
multi_cols = [i for i in cat_cols if i not in bin_cols]




################ Feature engineering part


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


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(Z, y, test_size = 0.2, random_state=44)



# Implement SMOTE 
sm = SMOTENC(random_state=40, categorical_features=categorical_features)
X_smote, y_smote = sm.fit_resample(X_train, y_train)

# Train the model on the resampled training dataset. No more unbalanced classes. 
X_train = X_smote
y_train = y_smote





####### MODEL IMPLEMENTATION


# Fitting tuned XGBoost to the Training set
classifier = XGBClassifier( n_estimators=150, learning_rate=0.15, max_depth=3, min_child_weight=0.6,
                            colsample_bytree=1, subsample=1, reg_alpha=0, gamma=0,
                            booster='gbtree', objective='binary:logistic') 
classifier.fit(X_train, y_train)


# Predicting the Test set and Training set results. Also store predicted probabilities used to classify. 
y_pred = classifier.predict(X_test)
y_pred2 = classifier.predict(X_train)
y_proba = classifier.predict_proba(X_test)   # illustrates confidence about predictions. Useful for interpretations. 
y_proba2 = classifier.predict_proba(X_train)  # for ICE,PDP
y_proba3 = y_proba2[:,1]  # only proba of churners 
#y_proba4 = pd.DataFrame(y_proba3)


# Need a dataframe as input. 
X_train2 = pd.DataFrame(X_train, columns = features)
X_test2 = pd.DataFrame(X_test, columns = features)
y_test2 = pd.DataFrame(y_test, columns = ['churn'])







################## INTERPRETABILITY PART  




### Main manipulations that need to be done to implement correctly future interpretation methods 


# Feature_names are not kept when fitting XGBoost to the data. We need to proceed to a special manipulation. This will be useful for another feature_importance graph and XGBoost tree plots. 
# Use Dmatrix and xgb.train() to get true feature names in XGBClassifier. Actually, can't add feature names for classification with xgboost so we do it for regression, meaning the redicted probability of churn
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_train, feature_names=features)
xgb_params = {'n_estimators':150, 'learning_rate':0.15, 'max_depth':3, 'min_child_weight':1, 'colsample_bytree':1, 'subsample':1, 'reg_alpha':0, 'gamma':0, 'booster':'gbtree', 'objective':'binary:logistic'}
model = xgb.train(xgb_params, dtrain) 
model_ypred = model.predict(dtest) 


# For ICE & PDP, we need a unique column with churn probabilities of customers instead of original classification. 
# So we create an XGBoost Regressor similarly to the XGBClassifier (same tuning, SMOTE, RFE, split train/test...)
# The results obtained are identical to the 'predict_proba' function of the XGBClassifier for churn = 1, which is exactly what we wanted. 
regressor = xgb.XGBRegressor ( n_estimators=150, learning_rate=0.15, max_depth=3, min_child_weight=1,
                            colsample_bytree=1, subsample=1, reg_alpha=0, gamma=0,
                            booster='gbtree', objective='binary:logistic')
regressor.fit(X_train, y_train)
y2_pred = regressor.predict(X_test)
y2_pred2 = regressor.predict(X_train)  # same as proba_3 (see XGBClassifier above)



# Finally, for a piece of the Partial Dependence Plots analysis, I need a Base Gradient Boosting method (GBM) to make 'PDPbox' work due to Python's current limitations. 
from sklearn.ensemble import GradientBoostingClassifier
model2 = GradientBoostingClassifier(n_estimators=150, max_depth=3,learning_rate=0.1, loss='deviance',random_state=1)
model2.fit(X_train, y_train)








########## Feature contribution 


### Plot trees of XGBoost 


# Feature names are too long so we take the following correspondence in the tree plots. 
dtrain_bis = xgb.DMatrix(X_train, label=y_train)
for i in range(0, len(features)): 
    print dtrain_bis.feature_names[i], features[i]


# Plot the first tree used by XGBoost. (It is built on the training set) 
plot_tree(classifier, num_trees=0)
fig = plt.gcf()
fig.set_size_inches(150, 100)
#fig.savefig('tree.png')
# Features used are Contract_Month-to-month / InternetService_Fiber optic, MonthlyCharges / Contract_Two year (x2), tenure, OnlineSecurity

# Compute the probability of churn for a single leaf of this particulartree. It involves taking the sigmoid function of the leaf output. 
1/(1+np.exp(-1* 0.187949)) # 0.49796073617491465


# Plot the second tree used by XGBoost. 
plot_tree(classifier, num_trees=1)
fig = plt.gcf()
fig.set_size_inches(150, 100)
#fig.savefig('tree1.png')
# Not many differences with 1st tree, Contract_Month-to-month / MonthlyCharges, PaperlessBilling / PaymentMethod_Electronic check, Contract_Two year, tenure, OnlineSecurity


# Plot a few other trees to get a good overview of the overall structure of the model. 
# You only need to change the 'num_trees' parameter to get another one. 







########## Feature importance 



### 'Weight' feature importance - Mean Decrease in Impurity (MDI)

# Use the 'feature_importances_' function of XGBClassifier
coefficients  = pd.DataFrame(classifier.feature_importances_)
column_df     = pd.DataFrame(features)
coef_sumry    = pd.merge(coefficients,column_df,left_index= True,right_index= True, how = "left")
coef_sumry.columns = ["coefficients","features"]
coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)
 
# Plot feature importance - taken from data vizualisation 
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

fig.append_trace(trace4,2,1)

# Plot 
fig['layout'].update(showlegend=False, title="Model performance" ,
                     autosize = False,height = 900,width = 800,
                     plot_bgcolor = 'rgba(240,240,240, 0.95)',
                     paper_bgcolor = 'rgba(240,240,240, 0.95)',
                     margin = dict(b = 195))
fig["layout"]["xaxis2"].update(dict(title = "false positive rate"))
fig["layout"]["yaxis2"].update(dict(title = "true positive rate"))
fig["layout"]["xaxis3"].update(dict(showgrid = True,tickfont = dict(size = 10),tickangle = 90))
py.iplot(fig)


# 'weight' feature importance 
xgb.plot_importance(classifier , importance_type='weight')
plt.title('importance', fontsize = 20)
plt.yticks(fontsize = 10)
plt.ylabel('features', fontsize = 20)

# This method give results that are closely related to what was observed in the previous part when we were having a look at several trees. 
# A big advantage is given to continuous features that are used a lot more to make a split in trees construction since they offer more split possibilities.  



### Feature Importance type = 'gain'
plt.figure(figsize = (7,8))
ax = plt.subplot()
xgb.plot_importance(classifier, importance_type='gain', ax=ax, height= 0.3, show_values=False), 
plt.title('importance', fontsize = 20)
plt.yticks(fontsize = 12)
plt.ylabel('features', fontsize = 20)

# Probably the best method. Most widely used. 
# Importance of Contract month to month is maybe a bit over-evaluated. The one of MonthlyCharges a bit under-evaluated. 



### Feature Importance type = 'cover'
plt.figure(figsize = (7,8))
ax = plt.subplot()
xgb.plot_importance(classifier, importance_type='cover', ax=ax, height= 0.3, show_values=False), 
plt.title('importance', fontsize = 20)
plt.yticks(fontsize = 12)
plt.ylabel('features', fontsize = 20)

# More balanced results - its use is not extremely spread.



### Permutation feature importance - (or Mean Decrease Accuracy)

# Import special library, designed for interpretation tasks
import eli5
from eli5.sklearn import PermutationImportance

#Fit and see permutation importance on our training data
perm_train = PermutationImportance(classifier)
perm_train.fit(X_train, y_train)
eli5.explain_weights_df(perm_train, feature_names=features)

#Fit and see permutation importance on our test data
perm_test = PermutationImportance(classifier)
perm_test.fit(X_test, y_test)
eli5.explain_weights_df(perm_test, feature_names=features)

# For this method, it is not clear on what set it should be applied. 
# In both cases, we can observe a table where features are ranked according to their importance. 
# The output takes the form of a weight (along with a standard deviation measure)



# Results vary according to the method used. Take into account limitations / biases of each method 
# Combinining them allows to get a more objective view of true feature importance, which is a great explanation factor. 
# Cross compare with deductions made during the data visualisation phase, where each feature's impact on churn was more or less assessed. 





########## Feature interaction - Friedman's H-statistics 



# Import library 
from sklearn_gbmi import *

# Between any pairs of features 
# h_all_pairs(classifier, X_train)

# Does not work in Python except for BaseGradientBoosting (GBM). 

# Between two features. Add a feature_name or column_index instead of features 
h(model2, X_train, [1,2])

# Between all pairs, with names
h_all_pairs(model2, X_train2.iloc[range(int(8274))])

# Same but without names
h_all_pairs(model2, X_train)




########## Global Surrogate Model (Decision Tree)


# Import libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score


# We approximate the predictions of our XGBClassifier by a CART Decision Tree. 
# We train it on the training dataset (X_train, y_pred) where y_pred are the predictions of our true XGBClassifier model. 

# As a result, we have 3863 predicted non churners and 4411 churners since we use predictions of the black box. 
# Note that using true value, we would have 4137 of each class. 
y_pred2[y_pred2==0].shape # non churners
y_pred2[y_pred2==1].shape # churners

# Fit the Decision Tree, which was tuned using GridSearchCV
classifier2 = DecisionTreeClassifier(criterion = 'entropy', max_depth = 7, max_features = None, min_samples_leaf = 0.0005, min_samples_split = 0.008)  #min_impurity_decrease = 
classifier2.fit(X_train, y_pred2)

#Predict the training results and extend to test set. Compare. 
DT_y_pred2 = classifier2.predict(X_train)
DT_y_pred = classifier2.predict(X_test)

# Evaluate performance using R^2. Confusion matrix. 
tree_performance = r2_score(DT_y_pred2 , y_pred2)
tree_performance2 = roc_auc_score(DT_y_pred2, y_pred2)
tree_perf = r2_score(DT_y_pred , y_pred)
tree_perf2 = roc_auc_score(DT_y_pred , y_pred)  
tree_p = accuracy_score(DT_y_pred2 , y_pred2) 
tree_p2 = accuracy_score(DT_y_pred , y_pred)           
print('Training r2: '+ str(tree_performance), 'Test r2 : ' + str (tree_perf))
print('Training AUC: '+  str(tree_performance2) ,'Test AUC: ' + str(tree_perf2))
print('Training accuracy: ' + str(tree_p), 'Test accuracy : ' + str (tree_p2))
# This (global) surrogate model is a good approximation of the black box. 
# The R^2 equals approximately 0.72 and the AUC is excellent (around 0.9) for the training set. For the test set, we obtain respectively 0.63 and 0.89. 
# If R^2 was moderatly higher, then we could have used a Decision Tree instead of XGBoost. If it was lower, global surrogate model would not have been great. 
# R^2 is the most important measure, even for classification tasks, to see if the surrogate model is a good approximation. 

# Training set confusion matrix 
cm = confusion_matrix(DT_y_pred2, y_pred2, labels = [1, 0])
print(cm)
# Test set confusion matrix 
cm2 = confusion_matrix(DT_y_pred, y_pred, labels = [1, 0])
print(cm2)
# Excellent results. Good surrogate model. 
# Notice class unbalance for test set and SMOTE's effect on training set. 


# Plot Decision Tree
"""
from sklearn.tree import export_graphviz
import graphviz
dot_data = export_graphviz(classifier2, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("DecisionTree") 
# We could not make it much smaller without sacrificing quite a lot of performance. We already limited its size! 
# This means that it would not have been such a good global surrogate model. 
# If we decrease max_depth to 6, we lose 0.1 point of R^2 in both the training and test set. 
"""

# We could use some interpretable methods on this decision tree. 
# Since this approach is not really rigorous, I will not use it too much to get additional information about the original's model functioning, but note that this could be done. 
# I just have a quick look at feature importance

# Feature importance 
DT_feat_imp = classifier2.feature_importances_
# Graph of feature importance 
k = list(range(0,26))
plt.plot(k, DT_feat_imp)   
# Get numerical feature importances. 
DT_importances = list(classifier2.feature_importances_)
DT_feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features, DT_importances)]
DT_feature_importances = sorted(DT_feature_importances, key = lambda x: x[1], reverse = True)
# print('Variable: {:20} Importance: {}'.format(*pair)) for pair in DT_feature_importances ONLY PYTHON3
# You can compare these results with XGBoost's individual trees and XGBoost's feature importance output. 










############### Individual Conditional Expectations (ICE) & Partial Dependence Plots (PDP)



# Import libraries 
from pycebox.ice import ice, pdp, ice_plot


# Define two important concepts in order to center ICEs and plot only a fraction of them. 
def _get_quantiles(x):
    return np.greater.outer(x, x).sum(axis=1) / x.size

def ICE_plot(ice_data, frac_to_plot=1.,
             plot_points=False, point_kwargs=None,
             x_quantile=False, plot_pdp=False,
             centered=False, centered_quantile=0.,
             color_by=None, cmap=None,
             ax=None, pdp_kwargs=None, **kwargs):
    """
    Plot the ICE curves
    :param ice_data: the ICE data generated by :func:`pycebox.ice.ice`
    :type ice_data: ``pandas`` ``DataFrame``
    :param frac_to_plot: the fraction of ICE curves to plot.  If less than one,
        randomly samples columns of ``ice_data`` to plot.
    :type frac_to_plot: ``float``
    :param plot_points: whether or not to plot the original data points on the
        ICE curves.  In this case, ``point_kwargs`` is passed as keyword
        arguments to plot.
    :type plot_points: ``bool``
    :param x_quantile: if ``True``, the plotted x-coordinates are the quantiles of
        ``ice_data.index``
    :type x_quantile: ``bool``
    :param plot_pdp: if ``True``, plot the partial depdendence plot.  In this
        case, ``pdp_kwargs`` is passed as keyword arguments to ``plot``.
    :param centered: if ``True``, each ICE curve is centered to zero at the
        percentile closest to ``centered_quantile``.
    :type centered: ``bool``
    :param color_by:  If a string, color the ICE curve by that level of the
        column index.
        If callable, color the ICE curve by its return value when applied to a
        ``DataFrame`` of the column index of ``ice_data``
    :type color_by: ``None``, ``str``, or callable
    :param cmap:
    :type cmap: ``matplotlib`` ``Colormap``
    :param ax: the ``Axes`` on which to plot the ICE curves
    :type ax: ``None`` or ``matplotlib`` ``Axes``
    Other keyword arguments are passed to ``plot``
    """
    if not ice_data.index.is_monotonic_increasing:
        ice_data = ice_data.sort_index()

    if centered:
        quantiles = _get_quantiles(ice_data.index)
        centered_quantile_iloc = np.abs(quantiles - centered_quantile).argmin()
        ice_data = ice_data - ice_data.iloc[centered_quantile_iloc]

    if frac_to_plot < 1.:
        n_cols = ice_data.shape[1]
        icols = np.random.choice(n_cols, size= int(frac_to_plot * n_cols), replace=False)
        plot_ice_data = ice_data.iloc[:, icols]
    else:
        plot_ice_data = ice_data


    if x_quantile:
        x = _get_quantiles(ice_data.index)
    else:
        x = ice_data.index

    if plot_points:
        point_x_ilocs = _get_point_x_ilocs(plot_ice_data.index, plot_ice_data.columns)
        point_x = x[point_x_ilocs]
        point_y = plot_ice_data.values[point_x_ilocs, np.arange(point_x_ilocs.size)]

    if ax is None:
        _, ax = plt.subplots()

    if color_by is not None:
        if isinstance(color_by, six.string_types):
            colors_raw = plot_ice_data.columns.get_level_values(color_by).values
        elif hasattr(color_by, '__call__'):
            col_df = pd.DataFrame(list(plot_ice_data.columns.values), columns=plot_ice_data.columns.names)
            colors_raw = color_by(col_df)
        else:
            raise ValueError('color_by must be a string or function')

        norm = colors.Normalize(colors_raw.min(), colors_raw.max())
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        for color_raw, (_, ice_curve) in zip(colors_raw, plot_ice_data.iteritems()):
            c = m.to_rgba(color_raw)
            ax.plot(x, ice_curve, c=c, zorder=0, **kwargs)
    else:
        ax.plot(x, plot_ice_data, zorder=0, **kwargs)

    if plot_points:
        ax.scatter(point_x, point_y, zorder=10, **(point_kwargs or {}))

    if plot_pdp:
        pdp_kwargs = pdp_kwargs or {}
        pdp_data = pdp(ice_data)
        ax.plot(x, pdp_data, **pdp_kwargs)

    return ax



# For the variable TENURE, create its ICE function
# As mentionned earlier, we are using the predicted probability that a customer will churn as target variable. 
ice_tenure = ice(X_train2, 'tenure', regressor.predict, num_grid_points=72) # Here 72 grid points because tenure can take 72 values. The more grid points, the more accurate it would be.  
ice_tenure.head() # Each column corresponds to a datapoint

# Data points plots and ICE plots. Run 2 parts together to obtain both plots. 
fig, (data_ax, ice_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 6))
data_ax.scatter(X_train2.tenure, y2_pred2, c='k', alpha=0.5);
data_ax.set_xlabel('tenure$');
data_ax.set_ylabel('$churn$');
data_ax.set_title('Data');
# This is a first version of ICE. It is too crowded and uncentered, which we will try to modify. 
ice_plot(ice_tenure, ax=ice_ax, plot_points=False, linewidth=0.2, plot_pdp = True); # frac_to_plot = 0.1  ;  the fraction of ICE curves to plot.
ice_ax.set_xlabel('$tenure$');
ice_ax.set_ylabel('$churn$');
ice_ax.set_title('ICE curves');

# Play with the following parameters of ice and ice_plot: num_grid_points, frac_to_plot, centered and plot_points. 

# New centered-ICE plots, still for tenure and with only a fraction of the total number of instances being considered. 
ICE_plot(ice_tenure, plot_points=False, linewidth=0.2, plot_pdp = True, frac_to_plot = 0.1)
plt.title('Uncentered ICE for a fraction of instances')
ICE_plot(ice_tenure, plot_points=False, linewidth=0.2, plot_pdp = True, frac_to_plot = 0.1, centered=True)
plt.title('Centered ICE for a fraction of instances')
plt.show()

# PDP for 'tenure': created manually from definition of 'pycebox.ice.pdp'
plt.plot(ice_tenure.index, ice_tenure.mean(axis=1))
plt.xlabel('tenure')
plt.ylabel('churn')



#### MONTHLY CHARGES
ice_MC = ice(X_train2, 'MonthlyCharges', regressor.predict, num_grid_points=70)
# Data points plots and ICE plots. Plot together
fig, (data_ax, ice_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 6))
data_ax.scatter(X_train2.MonthlyCharges, y2_pred2, c='k', alpha=0.5);
data_ax.set_xlabel('MonthlyCharges$');
data_ax.set_ylabel('$churn$');
data_ax.set_title('Data');
# This is a first version of ICE. It is too crowded and uncentered, which we will try to modify. 
ice_plot(ice_MC, ax=ice_ax, plot_points=False, linewidth=0.2, plot_pdp = True); # frac_to_plot = 0.1  ;  the fraction of ICE curves to plot.
ice_ax.set_xlabel('$MonthlyCharges$');
ice_ax.set_ylabel('$churn$');
ice_ax.set_title('ICE curves');
ICE_plot(ice_MC, plot_points=False, linewidth=0.2, plot_pdp = True, frac_to_plot = 0.1)
ICE_plot(ice_MC, plot_points=False, linewidth=0.2, plot_pdp = True, frac_to_plot = 0.1, centered=True)
plt.show()
# PDP for 'tenure': created manually from definition of 'pycebox.ice.pdp'
plt.plot(ice_MC.index, ice_MC.mean(axis=1))
plt.xlabel('MonthlyCharges')
plt.ylabel('churn')
plt.title('PDP for Monthly')


# Do it for whatever variable you wish
name = 'PaperlessBilling'

#### MONTHLY CHARGES
ice_var = ice(X_train2, name, regressor.predict, num_grid_points=70)
# Data points plots and ICE plots. Plot together
fig, (data_ax, ice_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 6))
data_ax.scatter(X_train2[name], y2_pred2, c='k', alpha=0.5);
data_ax.set_xlabel(name);
data_ax.set_ylabel('$churn$');
data_ax.set_title('Data');
# This is a first version of ICE. It is too crowded and uncentered, which we will try to modify. 
ice_plot(ice_var, ax=ice_ax, plot_points=False, linewidth=0.2, plot_pdp = True); # frac_to_plot = 0.1  ;  the fraction of ICE curves to plot.
ice_ax.set_xlabel(name);
ice_ax.set_ylabel('$churn$');
ice_ax.set_title('ICE curves');
ICE_plot(ice_var, plot_points=False, linewidth=0.2, plot_pdp = True, frac_to_plot = 0.1)
ICE_plot(ice_var, plot_points=False, linewidth=0.2, plot_pdp = True, frac_to_plot = 0.1, centered=True)
plt.show()
# PDP for 'tenure': created manually from definition of 'pycebox.ice.pdp'
plt.plot(ice_var.index, ice_var.mean(axis=1))
plt.xlabel(name)
plt.ylabel('churn')
plt.title('PDP')



# SEE these websites for further improvements: 
# Coulour, linewidth, 2-dimensional PDP, little bars showing gridpoints....
# http://savvastjortjoglou.com/intrepretable-machine-learning-nfl-combine.html#Centered-ICE-Plots
# https://github.com/SauceCat/PDPbox/blob/master/tutorials/pdpbox_binary_classification.ipynb




######  Further analysis on PDP ; other way to proceed 


# Import libraries
from pdpbox import pdp, get_dataset, info_plots
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence


# New options: PDP of multiple categorical features and 2 dimensional PDPs (at the end)
# Limit: use a GBM model because not available for classifier version of XGBoost. 



### Redo PDP of all covariates using another method. 
# Visualisation is not great.

# Plot PDP using a GBM model, giving similar performances to the XGBoost model. 
for i in range(0, len(telcom.columns)-2): # CustomerID and Churn are not included in X_train
    feat = [i]
    fig, axs = plot_partial_dependence(model2, X_train, feat ,feature_names= features,n_jobs=3, grid_resolution=50)


### Continuous variables 

# For the variable tenure (cts)  
pdp_tenure = pdp.pdp_isolate(model= model2, dataset=X_train2, model_features=features, feature='tenure', num_grid_points=20)
fig, axes = pdp.pdp_plot(pdp_tenure, 'tenure', plot_pts_dist=True) # plot
# Same graph -  Modified parameters _ with some ICE
fig, axes = pdp.pdp_plot(pdp_tenure, 'tenure',  center=True, plot_pts_dist= True, plot_lines= True, frac_to_plot=0.2) # plot

# MonthlyCharges
pdp_MC = pdp.pdp_isolate(model= model2, dataset=X_train2, model_features=features, feature='MonthlyCharges', num_grid_points=20)
fig, axes = pdp.pdp_plot(pdp_MC, 'MonthlyCharges',  center=True, plot_pts_dist= True, plot_lines= True, frac_to_plot=0.2) # plot

# TotalCharges
pdp_TC = pdp.pdp_isolate(model= model2, dataset=X_train2, model_features=features, feature='TotalCharges', num_grid_points=20)
fig, axes = pdp.pdp_plot(pdp_TC, 'TotalCharges',  center=True, plot_pts_dist= True, plot_lines= True, frac_to_plot=0.2) # plot



### Dummy variables 

## For PaperlessBilling (dummy variable)
pdp_PaperlessBilling = pdp.pdp_isolate(model=model2, dataset=X_train2,  model_features= features, feature='PaperlessBilling' , num_grid_points=2)
fig, axes = pdp.pdp_plot(pdp_PaperlessBilling, 'PaperlessBilling') # plot
_ = axes['pdp_ax'].set_xticklabels(['Yes', 'No'])

"""
# For all dummy variables 
import copy
features2  = copy.copy(features)
features2.remove('tenure')
features2.remove('MonthlyCharges')
features2.remove('TotalCharges')
for name in features2: 
    pdp_name = pdp.pdp_isolate(model=model2, dataset=X_train2,  model_features= features, feature= name , num_grid_points=2)
    fig, axes = pdp.pdp_plot(pdp_name, name) # plot
    _ = axes['pdp_ax'].set_xticklabels(['Yes', 'No'])
"""


### Multiple categorical variables

## For PaymentMethod  
pdp_PaymentMethod = pdp.pdp_isolate( model2, dataset=X_train2,  model_features= features, feature=[ 'PaymentMethod_Bank transfer (automatic)','PaymentMethod_Credit card (automatic)','PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'], num_grid_points=2)
fig, axes = pdp.pdp_plot(pdp_PaymentMethod, 'PaymentMethod' ,plot_pts_dist=True)
# Variant 
fig, axes = pdp.pdp_plot(pdp_PaymentMethod, 'PaymentMethod', center=True, plot_lines=True, frac_to_plot=0.2, plot_pts_dist=True)
# Interesting result only for this variable. Suggests that only Electronic Payment is truly important, credit card in a smaller extent.
# We could keep solely these two dummies and eliminate the others. This is deduced using this plot + PDP/ICE analysis of dummies, made above. 

"""
## For InternetService - multiple categorical variable 
pdp_InternetService = pdp.pdp_isolate( model2, dataset=X_train2,  model_features= features, feature=['InternetService_DSL','InternetService_Fiber optic', 'InternetService_No'], num_grid_points=2)
fig, axes = pdp.pdp_plot(pdp_InternetService, 'InternetService' ,plot_pts_dist=True)
# Variant 
fig, axes = pdp.pdp_plot(pdp_InternetService, 'InternetService', center=True, plot_lines=True, frac_to_plot=0.2, plot_pts_dist=True)

## For Contract - multiple categorical variable 
pdp_Contract = pdp.pdp_isolate( model2, dataset=X_train2,  model_features= features, feature=['Contract_Two year', 'Contract_One year', 'Contract_Month-to-month'], num_grid_points=2)
fig, axes = pdp.pdp_plot(pdp_Contract, 'Contract' ,plot_pts_dist=True)
# Variant 
fig, axes = pdp.pdp_plot(pdp_Contract, 'Contract', center=True, plot_lines=True, frac_to_plot=0.2, plot_pts_dist=True)
"""


### 2 dimensional PDP. 

# Capture interaction between two variables - especially useful for continuous ones
inter1 = pdp.pdp_interact(model2, dataset=X_train2,  model_features= features, features=['tenure', 'MonthlyCharges'])
fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['tenure','MOnthlyCharges'], plot_type='contour', x_quantile=True, plot_pdp=True)

"""
inter2 = pdp.pdp_interact(model2, dataset=X_train2,  model_features= features, features=['MonthlyCharges', 'TotalCharges'])
fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter2, feature_names=['MonthlyCharges','TotalCharges'], plot_type='contour', x_quantile=True, plot_pdp=True)

inter3 = pdp.pdp_interact(model2, dataset=X_train2,  model_features= features, features=['tenure', 'TotalCharges'])
fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter3, feature_names=['tenure','TotalCharges'], plot_type='contour', x_quantile=True, plot_pdp=True)


inter2 = pdp.pdp_interact(model2, dataset=X_train2,  model_features= features, features=['tenure', 'InternetService_Fiber optic'])
fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter2, feature_names=['tenure', 'InternetService_Fiber optic'], plot_type='contour', x_quantile=True)
fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter2, feature_names=['tenure', 'InternetService_Fiber optic'], plot_type='contour', x_quantile=True, plot_pdp=True)
 """

# Look at this resource to use XGBClassifier to plot PDP and not a surrogate model 
# https://resources.oreilly.com/oriole/interpretable-machine-learning-with-python-xgboost-and-h2o/blob/master/xgboost_pdp_ice.ipynb









########## Local Surrogate Model (LIME)



# Import libraries 
import lime
from lime.lime_tabular import LimeTabularExplainer


# Create Lime Explainer for tabular data, classification task
explainer = LimeTabularExplainer(X_train, mode='classification', class_names = ['NOT churn', 'churn'],
                                 feature_names=features, 
                                 categorical_names = cat_cols,  
                                 categorical_features = [0,1,2,3,5,6,7,8,9,10,11,12,13,16,17,18,19,20,21,22,23,24,25],
                                 discretize_continuous= True)



### Explain a particular instance 
i = 10  # instance studied
expl = explainer.explain_instance(X_train2.iloc[i,:].values,  # the index of the instance we want to explaine
                                 classifier.predict_proba, 
                                 num_features= len(features), # get all features   
                                 top_labels = 0)  # only most relevant features
                        
# Vizualisation of each feature's contribution 
expl.show_in_notebook(show_table=True, show_all=False)
# Note that the row we are explaining is displayed on the right side, in table format. More precisely, only the features used in the explanation are displayed.
# The left-most numbers reflect the predictions of the classifier used. 
# The central part reveals the average influence of that particular feature value in the final predictions. 
# These two sets of numbers should indeed convey similar information but they do not need to be exactly the same.


# Checking predictions of lime (cf true predictions)
print('True value of the observation (churn = 1)', y_train[i])
# Prediction of my model 
print('Predicted proba that customer will churn, with true model', y_proba2[i,1])

# The prediction made by LIME
print('Predicted proba that customer will churn, with LIME', expl.local_pred[0])  
expl.intercept   # bias term for the local explanation


# A plot of the weights for each feature - results already obtained, simply another form. 
expl.as_pyplot_figure();



### Provide more selective explanations 

# Use feature importance to look mainly at most relevant features of the dataset! 
# Since trees are short, only the features that are often used have an influence on the ouput! 

# If we want to be more concise/selective in our explanations. 
exp = explainer.explain_instance(X_train2.iloc[i,:].values,  # the index of the instance we want to explain. 
                                 classifier.predict_proba,
                                 num_features = 10, 
                                 top_labels =1)  # only most relevant features
   
# Visualisation
exp.show_in_notebook(show_table=True, show_all=False)  

# Checking predictions of lime (cf true predictions)
print('True value of the observation (churn = 1)', y_train[i])
# Prediction of my model 
print('Predicted proba that customer will churn, with true model', y_proba2[i,1])
# The prediction made by LIME
print('Predicted proba that customer will churn, with LIME', exp.local_pred[0])  
exp.intercept   # bias term for the local explanation



### SP_LIME - Submodular pick.

# Import
import warnings
from lime import submodular_pick

# SP-LIME returns explanations (on a sample set) to provide a non redundant global decision boundary of original model
sp_obj = submodular_pick.SubmodularPick(explainer, X_train, classifier.predict_proba, num_features=10, num_exps_desired=5)  # Not used sample_size= ? param. 
[exp.show_in_notebook(show_table=True, show_all=False) for exp in sp_obj.sp_explanations]; # visualise the 5 explanations selected by SP-LIME to provide a global understanding od the model. 
# [exp.as_pyplot_figure() for exp in sp_obj.sp_explanations];  # Other visualisation possibility 



# I can explain the model predictions for the test set in a similar way. 
# When applied on training set, we focus on the inner workings of the model to gain an understanding of it or to improve it. 
# When applied on test set, model is already understood and diffused. It could be used in a multitude of different ways by the company to reduce customer churn. 



### TEST SET 


# If we want to be more concise/selective in our explanations. 
exp_test = explainer.explain_instance(X_test2.iloc[i,:].values,  # the index of the instance we want to explain. 
                                 classifier.predict_proba,
                                 num_features = 10, 
                                 top_labels =1)  # only most relevant features
   
# Visualisation
exp_test.show_in_notebook(show_table=True, show_all=False)  


# Careful, computationnally expensive
"""
# Now look at all instances of the test set. Using "apply", we can get the predictions for all instances and judge how good LIME is overall. 
lime_expl = X_test2.apply(explainer.explain_instance, 
                          predict_fn=classifier.predict_proba, 
                          num_features= len(features),
                          axis=1)


# Checking predictions of lime (cf true predictions)
print('True value of the observation (churn = 1)', y_test[i])
# Prediction of my model 
print('Predicted proba that customer will churn, with true model', y_proba[i,1])

# The prediction made by LIME
print('Predicted proba that customer will churn, with LIME', lime_expl.local_pred[0])  
lime_expl.intercept  # bias term for the local explanation


# Double check that the local predictions from our surrogate models match our actual predictions. 
# We can judge the local prediction by looking at either the root-mean-squared error or the R
from sklearn.metrics import mean_squared_error, r2_score
# get all the lime predictions
lime_pred = lime_expl.apply(lambda x: x.local_pred[0])
# RMSE of lime pred
mean_squared_error(y_pred, lime_pred)**0.5
# r^2 of lime predictions
r2_score(y_pred, lime_pred)       
"""

 







########## Shapley value (SHAP)


# Import library
import shap

# load JS in order to use some of the plotting functions from the shap package in the notebook
shap.initjs()

# Similarly to LIME, I can explain model prediction for the training set or test set depending on the utility required. 
# In most cases it will be more useful on the test set, especially to answer the 'right to explanation' demand.
# Since it is not really the case here and since we are principally concerned with the building phase of the model for now, we focus on the training set.

# Explain the model's predictions using SHAP values
shap_explainer = shap.TreeExplainer(classifier)
shap_values = shap_explainer.shap_values(X_train)

# Store shap values as probabilities (of customer churn)
from scipy.special import expit
proba_shap_values = expit(shap_values)



### Focus on a single instance 

# Exact contribution of each feature to the output of instance i. 
i = 10 
for j in range(0, len(features)):
    contribution = (proba_shap_values[i,j]- sigmoid(shap_explainer.expected_value))
    print features[j], (X_train2[features[j]][i]) , 'contribution is '+ str(round(contribution,3))

# Sort by size of contribution
dict = {}
for j in range(0, len(features)):
    contribution = (proba_shap_values[i,j]- sigmoid(shap_explainer.expected_value))
    dict[features[j]] = round(contribution,3)
# Descending order
for k,v in sorted(dict.items() , key=lambda t : t[1] , reverse=True):
    print k, X_train2[k][i],'contribution: '+ str(v)
    # print k,v
"""
# Ascending order
for key, value in sorted(dict.iteritems(), key=lambda (k,v): (v,k)):
  print "%s: %s" % (key, value)
"""

# Visualize the prediction's explanation for the 10th individual. 
shap.force_plot(shap_explainer.expected_value, shap_values[i,:], X_train2.iloc[i,:])

# Let's express SHAP values as probabilities
shap.force_plot(shap_explainer.expected_value, shap_values[i,:], X_train2.iloc[i,:], link = 'logit')

# The above explanation shows features each contributing to push the model output from the base value (the average model output over the training dataset we passed, mean output of trees) to the model output. 
# Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue



### Global approach 
# Dependence plots, interaction effects, feature importance 

# Visualize the training set predictions' explanations
# If we take many explanations such as the one shown above, rotate them 90 degrees, and then stack them horizontally, we can see explanations for an entire dataset.
shap.force_plot(shap_explainer.expected_value, shap_values, X_train2)
# Especially useful for continuous variabels! PLay with it, many possibilities. 


# Create a SHAP dependence plot to show the effect of a single feature across the whole dataset
shap.dependence_plot("tenure", shap_values, X_train2, interaction_index= None)
shap.dependence_plot("tenure", proba_shap_values, X_train2, interaction_index= None)  # as proba

# Same but including interaction effects, meaning without extra dependence with another feature.
shap.dependence_plot("tenure", shap_values, X_train2)
shap.dependence_plot("tenure", proba_shap_values, X_train2) # as proba


# Summarize the effects of all the features. Can also tell if effects are heterogenous using this plot. 
shap.summary_plot(shap_values, X_train2)
shap.summary_plot(proba_shap_values, X_train2) # as proba
# This plot sorts features by the sum of SHAP value magnitudes over all samples, and uses SHAP values to show the distribution of the impacts each feature has on the model output.


# Feature importance - plot the SHAP values of every feature for all samples. 
# Take the mean absolute value of the SHAP values for each feature to get a standard bar plot
shap.summary_plot(shap_values, X_train2, plot_type="bar")
# As proba, not significant 








