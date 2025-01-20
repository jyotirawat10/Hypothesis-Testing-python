#!/usr/bin/env python
# coding: utf-8

# # Business problem 1

# In[1]:


import numpy as np
import pandas as pd


# In[26]:


loans_data = pd.read_csv(r'C:\Users\jyoti rawat\Downloads\Basic Statistics - Hypothesis Testing\4. Basic Statistics - Hypothesis Testing\LoansData.csv')


# In[27]:


loans_data.head()


# In[28]:


loans_data.columns = loans_data.columns.str.replace('.' ,'').str.lower()
loans_data.columns


# In[29]:


loans_data.dtypes


# In[30]:


loans_data['interestrate'] = loans_data.interestrate.str.replace('%' ,'')


# In[31]:


loans_data['loanlength'] = loans_data.loanlength.str.replace('months' ,'')
loans_data['debttoincomeratio'] = loans_data.debttoincomeratio.str.replace('%' ,'')


# In[32]:


loans_data.head()


# In[51]:


loans_data['employmentlength'] = loans_data.employmentlength.str.replace('<' ,'').str.replace(' ','').str.replace('years','').str.replace('year','')
loans_data['employmentlength'] = loans_data.employmentlength.str.replace('s' ,'')
loans_data['employmentlength'] = loans_data.employmentlength.str.replace('+' ,'')


# In[52]:


loans_data.head()


# In[57]:


##changing data types
loans_data['interestrate'] = loans_data['interestrate'].astype('float64')
loans_data['loanlength'] = loans_data['loanlength'].astype('int64')
loans_data['debtAtoincomeratio'] = loans_data['debttoincomeratio'].astype('float64')
loans_data['employmentlength'] = loans_data['employmentlength'].astype('float64')


# In[123]:


##ficorange: Consider splitting into two columns (fico_lower and fico_upper) as int64.


loans_data[['ficolower' ,'ficoupper']] = loans_data['ficorange'].str.split('-' , expand = True)
loans_data.head()


# In[85]:



loans_data['ficolower'] = loans_data['ficolower'].astype('float64')
loans_data['ficoupper'] = loans_data['ficoupper'].astype('float64')


# # hypothesis testing 
# 
# 
# ## a. Intrest rate is varied for different loan amounts (Less intrest charged for high loan amounts
# 

# In[9]:


import scipy.stats as stats
from scipy.stats import pearsonr


# In[91]:


## H0:there is no relationship between interest rate and loan amount

## Ha: there is relationship between interest rate and loan amount

## confidence interval - 95%
## p value - 5%

## as both are continuous column we will conduct pearsoncorrelation


# In[92]:


loans_data_cleaned = loans_data[['amountfundedbyinvestors', 'interestrate']].replace([np.inf, -np.inf], np.nan).dropna()
pearsonr(loans_data_cleaned['amountfundedbyinvestors'], loans_data_cleaned['interestrate'])


# In[93]:


##since p value is lesser than 0.05 we will reject null hypothesis


# b. Loan length is directly effecting intrest rate.
# 

# In[95]:


## H0:there is no relationship between interest rate and loan length

## Ha: there is relationship between interest rate and loan length

## confidence interval - 95%
## p value - 5%

## as both are continuous column we will conduct pearsoncorrelation


# In[96]:


loans_data_cleaned = loans_data[['loanlength', 'interestrate']].replace([np.inf, -np.inf], np.nan).dropna()
pearsonr(loans_data_cleaned['loanlength'], loans_data_cleaned['interestrate'])


# In[97]:


##since p value is lesser than 0.05 we will reject null hypothesis


# c. Inrest rate varies for different purpose of loans

# In[98]:


## H0:interest rate do not vary with loan purpose

## Ha:interest rate varies with loan purpose

## confidence interval - 95%
## p value - 5%

## we will conduct anova


# In[118]:


loans_data['interestrate'] = loans_data['interestrate'].fillna(0)


# In[119]:


debt_consolidation = loans_data.loc[loans_data.loanpurpose == 'debt_consolidation' , 'interestrate']
credit_card = loans_data.loc[loans_data.loanpurpose == 'credit_card' , 'interestrate']
other = loans_data.loc[loans_data.loanpurpose == 'other' , 'interestrate']
moving = loans_data.loc[loans_data.loanpurpose == 'moving' , 'interestrate']
car = loans_data.loc[loans_data.loanpurpose == 'car' , 'interestrate']
vacation = loans_data.loc[loans_data.loanpurpose == 'vacation' , 'interestrate']
home_improvement = loans_data.loc[loans_data.loanpurpose == 'home_improvement' , 'interestrate']
house = loans_data.loc[loans_data.loanpurpose == 'house' , 'interestrate']
major_purchase = loans_data.loc[loans_data.loanpurpose == 'major_purchase' , 'interestrate']
educational = loans_data.loc[loans_data.loanpurpose == 'educational' , 'interestrate']
medical = loans_data.loc[loans_data.loanpurpose == 'medical' , 'interestrate']
wedding = loans_data.loc[loans_data.loanpurpose == 'wedding' , 'interestrate']
small_business = loans_data.loc[loans_data.loanpurpose == 'small_business' , 'interestrate']
renewable_energy = loans_data.loc[loans_data.loanpurpose == 'renewable_energy' , 'interestrate']


# In[120]:


stats.f_oneway(debt_consolidation, credit_card, other, moving, car,
       vacation, home_improvement, house, major_purchase,
       educational, medical, wedding, small_business,
       renewable_energy)


# In[121]:


##since p value is lesser than 0.05 we will reject null hypothesis


# d. There is relationship between FICO scores and Home Ownership. It means that, People 
# with owning home will have high FICO scores

# In[122]:


## H0:fico score do not vary with home ownership

## Ha:fico score varies with home ownership

## confidence interval - 95%
## p value - 5%

## we will conduct anova


# In[125]:


loans_data['homeownership'] = loans_data['homeownership'].fillna('NONE')


# In[132]:


loans_data['ficoupper'] = loans_data['ficoupper'].fillna(0)


# In[133]:


mortgage = loans_data.loc[loans_data.homeownership == 'MORTGAGE' , 'ficoupper']
rent = loans_data.loc[loans_data.homeownership == 'RENT' , 'ficoupper']

own = loans_data.loc[loans_data.homeownership == 'OWN' , 'ficoupper']

other = loans_data.loc[loans_data.homeownership == 'OTHER' , 'ficoupper']

none = loans_data.loc[loans_data.homeownership == 'NONE' , 'ficoupper']


# In[134]:


stats.f_oneway(mortgage , rent,own, other,none)


# In[2]:


##since p value is lesser than 0.05 we will reject null hypothesis


# # Business Problem 2

# We would like to assess if there is any difference in the average price quotes provided by Mary and Barry.

# In[3]:


price_quotes = pd.read_csv(r'C:\users\jyoti rawat\Downloads\Basic Statistics - Hypothesis Testing\4. Basic Statistics - Hypothesis Testing\Price_Quotes.csv')


# In[8]:


## H0: there is no difference in the average price quotes provided by Mary and Barry.

## Ha:there is difference in the average price quotes provided by Mary and Barry.

## confidence interval - 95%
## p value - 5%

## we will conduct t test


# In[7]:


barry_price =  price_quotes['Barry_Price']
mary_price =  price_quotes['Mary_Price']


# In[10]:


stats.ttest_rel(barry_price, mary_price)


# In[11]:


##since p value is less than 0.05 we will reject null hypothesis


# # Business Problem 3

# Determine what effect, if any, the reengineering effort had on the 
# incidence behavioral problems and staff turnover. i.e To determine if the reengineering effort
# changed the critical incidence rate. Isthere evidence that the critical incidence rate
# improved?

# In[12]:


treatment = pd.read_csv(r'C:\users\jyoti rawat\Downloads\Basic Statistics - Hypothesis Testing\4. Basic Statistics - Hypothesis Testing\Treatment_Facility.csv')


# In[19]:


## H0: there is no difference in the critical incidence rate before and after reengineering effort

## Ha:there is difference in the critical incidence rate before and after reengineering effort

## confidence interval - 95%
## p value - 5%

## we will conduct t test


# In[17]:


prior_ci = treatment.loc[treatment['Reengineer'] == 'Prior' , 'CI']
post_ci = treatment.loc[treatment['Reengineer'] == 'Post' , 'CI']


# In[18]:


stats.ttest_ind(prior_ci, post_ci)


# In[20]:


##since p value is greater than 0.05 we will accept null hypothesis


# # Business problem 4

# We will focus on the prioritization system. If the system is working, then
# high priority jobs, on average, should be completed more quickly than medium priority jobs,
# and medium priority jobs should be completed more quickly than low priority jobs. Use the
# data provided to determine whether thisis, in fact, occurring

# In[25]:


priority = pd.read_csv(r'C:\users\jyoti rawat\Downloads\Basic Statistics - Hypothesis Testing\4. Basic Statistics - Hypothesis Testing\Priority_Assessment.csv')
priority.head()


# In[24]:


## H0: there is no difference in the mean no of days to complete high, medium, and low priority jobs.

## Ha:there is difference in the mean no of days to complete high, medium, and low priority jobs.
## confidence interval - 95%
## p value - 5%

## we will conduct anova test as more than 2 variables


# In[35]:


high = priority.loc[priority['Priority'] == 'High' , 'Days']
medium = priority.loc[priority['Priority'] == 'Medium' , 'Days']
low = priority.loc[priority['Priority'] == 'Low' , 'Days']


# In[102]:


stats.f_oneway(high,medium,low)


# In[103]:


##since p value is greater than 0.05 we will accept null hypothesis


# # Business Problem 5

# In[104]:


films = pd.read_csv(r'C:\users\jyoti rawat\Downloads\Basic Statistics - Hypothesis Testing\4. Basic Statistics - Hypothesis Testing\Films.csv')


# In[105]:


films


# In[106]:


films['Gender'] = films['Gender'].str.replace('1' ,'Male').str.replace('2' , 'Female')
films.head()


# In[107]:


films['Marital_Status'] = films['Marital_Status'].str.replace('1' ,'Married').str.replace('2' , 'Single')
films.Marital_Status.unique()


# In[108]:


films['Marital_Status'] = films['Marital_Status'].str.replace('Slngle' ,'Single')
films.Marital_Status.unique()


# In[109]:



satisfaction_columns = ['Sinage', 'Parking', 'Clean', 'Overall']


films['Overall_Satisfaction'] = films[satisfaction_columns].mean(axis=1)


overall_satisfaction = films['Overall_Satisfaction'].mean()
overall_satisfaction


# In[110]:


## it represents the good satisfaction level


# What factors are linked to satisfaction?

# In[111]:


films.head(2)


# In[112]:



correlations = films[['Overall_Satisfaction', 'Age', 'Income']].corr()

print(correlations)


# In[113]:


films['Overall_Satisfaction'] = films['Overall_Satisfaction'].fillna(0)


# In[114]:




# Gender: Comparing satisfaction scores for males and females
male_satisfaction = films.loc[films['Gender'] == 'Male' , 'Overall_Satisfaction' ]
female_satisfaction = films.loc[films['Gender'] == 'Female' , 'Overall_Satisfaction' ]

stats.ttest_ind(male_satisfaction ,female_satisfaction)


# In[115]:


## since p value is greater we will accept null i.e there is no difference in satisfaction scores for male and female


# In[116]:


# Marital Status: Compare satisfaction scores for married and single 
married_satisfaction = films.loc[films['Marital_Status'] == 'Married' , 'Overall_Satisfaction' ]
single_satisfaction = films.loc[films['Marital_Status'] == 'Single' , 'Overall_Satisfaction' ]

stats.ttest_ind(married_satisfaction,single_satisfaction)


# In[117]:


## since p value is greater we will accept null i.e there is no difference in satisfaction scores for married and single


# What is the demographic profile of Film on the Rocks patrons?
# 

# In[118]:


gender_distribution =films['Gender'].value_counts(normalize = True) * 100
gender_distribution


# In[119]:


marital_distribution =films['Marital_Status'].value_counts(normalize = True) * 100
marital_distribution


# In[123]:


films.Income.unique()


# In[122]:


age_distribution =films['Age'].value_counts(normalize = True) * 100
age_distribution


# In[124]:


income_distribution =films['Income'].value_counts(normalize = True) * 100
income_distribution


# In[125]:


hearabout_distribution =films['Hear_About'].value_counts(normalize = True) * 100
hearabout_distribution


# In[127]:


films['Hear_About'].str.split(',', expand=True).stack().reset_index(drop=True).value_counts(normalize = True) * 100


#  In what media outlet(s) should the film series be advertised?

# In[128]:


## word of mouth and websites are the best ways to advertise


# In[ ]:




