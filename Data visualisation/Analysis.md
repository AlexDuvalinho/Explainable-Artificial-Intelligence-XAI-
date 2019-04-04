# DATA VISUALISATION 

These conclusions were all drawn from the graphs plotted using the avaiable Pyhton code. 


## Variables
3 continuous variables, 7 binary dummy variables, 11 categorical variables

customerID - Customer ID\
gender - Whether the customer is a male or a female\
SeniorCitizen - Whether the customer is a senior citizen or not (1, 0)\
Partner - Whether the customer has a partner or not (Yes, No)\
Dependents - Whether the customer has dependents or not (Yes, No)\
tenure - Number of months the customer has stayed with the company\
PhoneService - Whether the customer has a phone service or not (Yes, No)\
MultipleLines - Whether the customer has multiple lines or not (Yes, No, No phone service)\
InternetService - Customer’s internet service provider (DSL, Fiber optic, No)\
OnlineSecurity - Whether the customer has online security or not (Yes, No, No internet service)\
OnlineBackup - Whether the customer has online backup or not (Yes, No, No internet service)\
DeviceProtection - Whether the customer has device protection or not (Yes, No, No internet service)\
TechSupport - Whether the customer has tech support or not (Yes, No, No internet service)\
StreamingTV - Whether the customer has streaming TV or not (Yes, No, No internet service)\
StreamingMovies - Whether the customer has streaming movies or not (Yes, No, No internet service)\
Contract - The contract term of the customer (Month-to-month, One year, Two year)\
PaperlessBilling - Whether the customer has paperless billing or not (Yes, No)\
PaymentMethod - The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))\
MonthlyCharges - The amount charged to the customer monthly\
TotalCharges - The total amount charged to the customer\
Churn - Whether the customer churned or not (Yes or No)


## Data Manipulation

There are only 11 missing values, all of them for the TotalCharges feature. They affect exclusively customers with zero tenure, one-year or two-years contract and monthly charges strictly greater than 0. It is possible that these consumers have never paid what they owe the company. These observations are considered to be involuntary churn and therefore, as mentioned earlier, will be excluded from the analysis. 

We slightly modify the original data in order to facilitate its comprehension while visualising it; without real consequences. In particular, we replace the event ‘No internet service’ by ‘No’ for the following columns: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV and StreamingMovies. It follows because they all are conditioned on the categorical feature InternetService, where the category ‘No’ represents a small proportion of the customer base. Similarly, we change the value ‘No phone service’ to ‘No’ in the MultipleLines column. This paper also creates a categorical variable for tenure, where its values are regrouped by years (same as contract). Finally, it checks and do not spot any outliers when regarding solely continuous variables. 


## Visualisation and interpretation 

Churn columns tells us about the number of customers who left within the last month: around 27%. We thus have a binary classification problem with an unbalanced target. 

### Continuous variables 

The probability density distribution of the three continuous variables, namely tenure, MonthlyCharges and TotalCharges, can be estimated using the seaborn kdeplot function. \
From the plots, we can conclude that: 
* Recent clients are more likely to churn 
* Clients with higher monthly charges are also more likely to churn 
* tenure and MonthlyCharges are probably important features whereas TotalCharges alone does not seem to be. 


### Categorical variables 


#### Demography

*gender* does not appear useful: the churn percentage is almost equal for males and females. \
*SeniorCitizen* represents only 16% of customers but has a much higher churn rate, that is 42% against 23% for non-senior customers. \
*partner*: customers without partners are more likely to churn. \
*Dependents*: similarly for customers without dependents.


#### Services

Let's now look at the two main services that customers are using, namely phone and internet. Note that the later includes additional options such as OnlineBackup or OnlineSecurity. 

*PhoneService* does not appear useful. Note that only few customers don’t have phone service. (9%) 

*MultipleLines*\
For consumers, having multiple phone lines has practically no correlation with attrition; maybe a slightly higher churn rate.
This could be due to the fact that having multiple lines increases monthly charges.\
It is not correlated with tenure, dependents or partners. Nevertheless, it is clearly visible that churners having multiple phone lines are principally new customers.
 
*InternetService*\
Very informative feature. 3 categories (DSL, Fiber optic, No) \
Clients without internet have a very low churn rate. They represent 22% of the dataset, where only 2% churn and 20% does not. \
Customers with fiber optic are very likely to churn. 43% of customers in total, including 18% that churn and 25% that don’t. \
Those with DSL connection usually don’t churn. (27% don’t, 7% do) \
It's interesting how customers with DSL (slower connection) and higher charges are less probable to churn. \
Relation to numerical features: using Fiber optic yields much higher monthly charges (around 90) than DSL (around 55) and no Internet (20). 

Concerning the six additional services (for customers without/with internet): \
*StreamingTV* and *StreamingMovies* are not predictive for churn. Used by 50% of customer base.\
Customers subscribed to *OnlineSecurity*, *OnlineBackup*, *DeviceProtection*, *TechSupport*, are less likely to churn. An important proportion of customers take each service but less than half.\
They all are seriously correlated between each other and with monthly charges, which makes sense. 


#### Account info:  (4-5 graphs + for all, see distribution binary var wrt both churn and non-churn )

*Contract*\
A larger percent of customers with monthly subscription have left compared to customers with one year contract. This is even more accentuated compared to two-years contract, which is quite logic.\
One-year and two-year contracts probably have contractual fines and therefore customers have to wait until the end of the contract to churn. A time-series dataset (survival analysis) would be better to understand this kind of behaviour.\

*PaymentMethod*\
It is spit into four categories: electronic check, mail check, credit card and bank transfer. \
The preferred payment method is electronic check with approximately 35% of customers. It has a very high churn rate compared to the other methods, which are used by an equal amount of individuals and all present a similar churn rate.\
There is no special correlation with contract and tenure. Longer contracts or higher tenure do not favour a particular type of payment.\

*PaperlessBilling*\
Customers with paperless billing are more likely to churn, probably because it is easier to do and because it is correlated with smaller time-length contracts. 




### Correlation matrix

We simply notice a decent correlation between Contract and all additional services (linked to Internet). This is not too surprising. Individuals that subscribe to this kind of offer (OnlineBackup, OnlineSecurity or StreamingTV) are usually long-term customers, meaning having one-year or two-year contracts. It is rare to subscribe to this type of offer only for a month.\
When multiple category categorical variables are one-hot-encoded, this correlation disappears.\
However, a strong correlation between the services remains.

Correlation matrix \ 
Nothing really new to note. There does not seem to be any strong correlation, else than between 
* tenure, TotalCharges, MonthlyCharges
* MonthlyCharges and FiberOptic / No_internet
* MultipleLines with FiberOptic (positively) and DSL (-) (in a smaller extent).
* all additional services (in a smaller extent).


### Summary statistics 

Distribution of binary variables\
We have figures here and not proportion, which makes us notice the unbalanced repartition of the dataset. \
On the first radar, distribution of variables concerns solely customers who don't churn (0s and 1s). The other is similar but for churners. Comparing them allows to notice the variables whose distribution strongly changes from churners to non-churners. These variables will probably be the most helpful in our classification task. 

Big differences between the two radars are visible for FiberOptic, Month-to-month (Contract) and ElectronicCheck (PaymentMethod) \
In a smaller extent: TechSupport and PaperlessBilling 

This pretty much summarises what we found above and what we think the important variables will be. It excludes tenure and MonthlyCharges, which also seem important, but continuous. 
