This file provides and investigates some key results of the Model Building code.
Most of the content below is also included in the paper. If you have already read it, including Appendix C, you may find this document repetitive. 


## Data manipulation part 

This dataset contains 3 continuous features, 7 binary dummy variables and 11 categorical variables (not binary). Here is a quick description of them. 

*customerID* - Customer ID \
*gender* - Whether the customer is a male or a female \
*SeniorCitizen* - Whether the customer is a senior citizen or not (1, 0) \
*Partner* - Whether the customer has a partner or not (Yes, No)\
*Dependents* - Whether the customer has dependents or not (Yes, No) \
*tenure* - Number of months the customer has stayed with the company \
*PhoneService* - Whether the customer has a phone service or not (Yes, No)\
*MultipleLines* - Whether the customer has multiple lines or not (Yes, No, No phone service)\
*InternetService* - Customer’s internet service provider (DSL, Fiber optic, No) \
*OnlineSecurity* - Whether the customer has online security or not (Yes, No, No internet service) \
*OnlineBackup* - Whether the customer has online backup or not (Yes, No, No internet service)\
*DeviceProtection* - Whether the customer has device protection or not (Yes, No, No internet service)\
*TechSupport* - Whether the customer has tech support or not (Yes, No, No internet service)\
*StreamingTV* - Whether the customer has streaming TV or not (Yes, No, No internet service) \
*StreamingMovies* - Whether the customer has streaming movies or not (Yes, No, No internet service)\
*Contract* - The contract term of the customer (Month-to-month, One year, Two year)\
*PaperlessBilling* - Whether the customer has paperless billing or not (Yes, No) \
*PaymentMethod* - The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)) \
*MonthlyCharges* - The amount charged to the customer monthly \
*TotalCharges* - The total amount charged to the customer \
*Churn* - Whether the customer churned or not (Yes or No)

Note that the original data was slightly modifed to facilitate its comprehension while visualising it; without real consequences. In particular, I replaced the event 'No internet service' by 'No' for the following columns: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV and StreamingMovies. It follows because they all are conditioned on the categorical feature InternetService, where the category 'No' represents a small proportion of the customer base. Similarly, we change the value 'No phone service' to 'No' in the MultipleLines column. Lastly, each of the three multiple categorical variables were replaced by several dummy variables. 

There are only 11 missing values, all of them for the TotalCharges feature. They affect exclusively customers with zero tenure, one-year or two-years contract and monthly charges strictly greater than 0. It is possible that these consumers have never paid what they owe the company. These observations are considered to be involuntary churn and therefore, as mentioned earlier, will be excluded from the analysis. No outliers are found (for continuous variables), meaning that the quasi totality of the dataset is considered. 



## Unbalanced repartition of the dataset

The main difficulty faced in this phase is the unbalanced repartition of the dataset. From the data visualisation part, there are indeed approximately 73 \% of non-churners. Hence, if the model was predicting that every customer would stay, which is wrong, it would still achieve a 0.73 accuracy. Although it is not that accentuated, and given that the positive class defines churners, there exists many more Type II errors (you predict that a customer will stay while he actually churns, called False Negative) than Type I errors (you predict the customer will churn while he actually does not, also called False Positive). Refer to the confusion matrix below. This implies that the telecom company fails to identify many customers that are about to churn, which is highly undesirable because its purpose is to retain existing customers. To limit this type of error, one solution would be to set a different discrimination threshold (minimal probability at which the positive class (churn) is chosen over the negative class. By default, it is equal to 0.5). From the coding part, a threshold of 0.36 would be optimal and would yield better results. This means that if a customer is likely to churn with probability higher or equal to 0.36 according to our model, it is classified in the churn category. Another solution, which we prefer, is to use SMOTE to oversample the minority class of our training dataset and train the model on this new data. This approach is detailed below. SMOTE, by increasing the percentage of churn instances to 50%, augments the number of Type I errors and reduces the amount of Type II errors while maintaining a similar performance. The new confusion matrix reveals the trade off operated between False Negative and False Positive errors, which is exactly what we want. In this case, the optimal threshold is approximately back to 0.5. We are not interested in further reducing the number of Type II errors via a smaller threshold not only because it will hurt performance but also because we cannot tolerate too many Type I errors. Although, we do not have enough information on that, the firm might try to retain churners via some promotional offers. Making a large number of those to customers that were not about to leave the company would cause losses to the company. Modifying the proportion of Type I/II errors further therefore depends on the firm's strategy regarding churn customers.

TABLE (2 confusion matrices)

Comparing these two confusion matrices is very interesting as it illustrates the trade off between false negative and false positive errors that we just mentioned. While performance metrics are very similar for both models, our predictions for the final model are a bit less accurate for non-churners but more accurate for churners, compared to the initial model. 


## SMOTE 

If you want to understand exactly what SMOTE does, I summarize its pseudo code below.

1. Split between training and test set; focus solely on the training set.
2. Randomly pick a point from the minority class (churn).
3. Compute the k-nearest neighbours for this point (for some pre-specified k).
4. Introduce synthetic examples along the line segments joining any/all of the k minority class nearest neighbours.
5. Depending upon the amount of over-sampling required, neighbours from the k-nearest neighbours are randomly chosen.
6. The test set is left unchanged
  
In fact, we use a slightly different version of SMOTE, called SMOTENC (NC for Numerical Continuous), which simply enables us to deal with both categorical and numerical variables. Otherwise the newly created samples would attribute new numeric values to our categorical features. \
Note that we cannot cross validate after using SMOTE. Indeed, the results would not be relevant as some newly created instances sent to the validation set will have extremely similar instances in the training set, which induces the presence of overfitting. 


## Feature selection 

Concerning feature selection, no new attribute was added and no feature was deleted by RFECV. RFECV is similar to RFE but suppresses features automatically in a recursive fashion through a cross-validated selection of the best number of features, with respect to the Area Under the Curve metric.\
Recursive feature elimination (RFE) is a feature selection method that fits a model and removes recursively the weakest feature, which is defined by its importance in the model. 

The list below provides a ranking of the features via RFE in the final model, from the most important to the least. In parenthesis is their ranking in the initial (most basic) model. 

MonthlyCharges (2) \
TotalCharges (1) \
tenure (3)\
PaymentMethod\_Electronic check (7)\
PaymentMethod\_Mailed check (17)\
PaymentMethod\_Bank transfer (automatic) (23) \
PaymentMethod\_Credit card (automatic) (20)\
MultipleLines (15)\
PaperlessBilling (6)\
TechSupport (4)\
SeniorCitizen (16)\
Contract\_Month-to-month (8)\
Contract\_Two year (11)\
gender (19)\
OnlineSecurity (10)\
InternetService\_No (9)\
OnlineBackup' (21)\
InternetService\_Fiber optic (13)\
Dependents (22)\
StreamingMovies (12)\
PhoneService (18)\
DeviceProtection (26)\
StreamingTV (14)\
InternetService\_DSL (24)\
Partner (25)\
Contract\_One year (5)\



## Results

The final XGBoost model uses SMOTE, has only relevant features and has been tuned following the recommandations of <https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/>. Note that it contains 150 trees of maximal depth 3. 

Since we have used the oversampling technique SMOTE to deal with our unbalance dataset, cross validation results become meaningless for the training test. For the test set, the Area Under the Curve is 0.756 and the accuracy score 0.776. These results are satisfying although we could have hoped for an even better performance for this kind of task. Despite an undeniable improvement caused by the use SMOTE, our model still struggles a bit to spot customers who churn, which appears clearly in the confusion matrices above. 

The most important fact to bear in mind from this subsection is that given its good performance, the model can be used by the company, which makes the interpretability analysis meaningful.  

You can refer to the coding part to get more details on how this model was derived and on the results obtained. 
