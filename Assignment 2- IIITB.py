import numpy as np
import pandas as pd
#import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import iplot

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',40)

#Reading loan dataset
df = pd.read_csv('loan.csv',encoding = "ISO-8859-1")
#print(df.describe)

#find missing values
missing=round(100*(df.isnull().sum()/len(df.index)),2)
#print(missing)

#As most of the columns have missing values, lets remove columns with missing values>=50%
missing_col=list(missing[missing>=50].index)
#print(len(missing_col))

#Drop these 57 columns
print("Actual shape",df.shape)
df=df.drop(missing_col,axis=1)
print("Shape after removing missing values>=50%",df.shape)

#Further looking into remaining columns with missing values
missing_remaining=round(100*(df.isnull().sum()/len(df.index)),2)
print(missing_remaining)

#Remove desc column
df=df.drop('desc',axis=1)

#lets find the number of unique values in remaining columns
print(len(df['emp_title'].unique()))
print(len(df.pub_rec_bankruptcies.unique()))
print(len(df.chargeoff_within_12_mths.unique()))
print(len(df.collections_12_mths_ex_med.unique()))
print(len(df.last_pymnt_d.unique()))
print(len(df.revol_util.unique()))
print(len(df.title.unique()))
print(len(df['emp_length'].unique()))
print(len(df['tax_liens'].unique()))
print(len(df['last_credit_pull_d'].unique()))

#identifying the unique values now in above columns

print(df.pub_rec_bankruptcies.unique(),"\n")
print(df.chargeoff_within_12_mths.unique(),"\n")
print(df.collections_12_mths_ex_med.unique(),"\n")
print(df.revol_util.unique(),"\n")
print(df['tax_liens'].unique(),"\n")

#removing columns have 0,1 or nan values as they might not impact the analysis much
drop_col=['pub_rec_bankruptcies','chargeoff_within_12_mths','collections_12_mths_ex_med','tax_liens']
df=df.drop(drop_col,axis=1)
#print(df.shape)

#checking missing % again
missing_remaining_2=round(100*(df.isnull().sum()/len(df.index)),2)
print(missing_remaining_2)

#The columns emp_title and emp_length have 6.19% and 2.71% missing values.
##These columns have information about the customer/borrower like their job title and their employment length in years.

#Lets drop missing values of the remaining columns as there is minimal amount of missing values
df=df[~df.emp_title.isnull()]
df=df[~df.emp_length.isnull()]
df=df[~df.title.isnull()]
df=df[~df.revol_util.isnull()]
df=df[~df.last_pymnt_d.isnull()]
df=df[~df.last_credit_pull_d.isnull()]
print(df.shape)
a=round(100*(df.isnull().sum()/len(df.index)),2)
print(a)

#Now our data is cleaned and ready for deep analysis
#As a sanity check, lets save this into another file, cleaned.csv for further analysis
df.to_csv('cleaned.csv',encoding='ISO-8859-1', index=False)
cleaned_data = pd.read_csv('cleaned.csv',encoding='utf-8')

#To print unique values in all columns in the data set
print(cleaned_data.nunique().sort_values())

#Refer to data dictionary and drop unnecessary columns
dropped_cols =['id','member_id','funded_amnt','pymnt_plan','url','zip_code','initial_list_status','policy_code','application_type','acc_now_delinq','delinq_amnt','pub_rec','title','verification_status','delinq_2yrs','inq_last_6mths','open_acc','revol_bal',
               'revol_util','total_acc','out_prncp','out_prncp_inv','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee']
cleaned_data= cleaned_data.drop(dropped_cols,axis=1)
print(cleaned_data.shape)

datatypeseries=cleaned_data.dtypes
#print(datatypeseries)

#converting date/time columns that are of type object into datetime data type
##date time colums
date_time_cols=['issue_d','earliest_cr_line','last_pymnt_d','last_credit_pull_d']
#print(cleaned_data[date_time_cols].info())

#converting to datetime data type
cleaned_data.issue_d = pd.to_datetime(cleaned_data.issue_d, format='%b-%y')
cleaned_data.earliest_cr_line = pd.to_datetime(cleaned_data.earliest_cr_line, format='%b-%y')
cleaned_data.last_pymnt_d = pd.to_datetime(cleaned_data.last_pymnt_d, format='%b-%y')
cleaned_data.last_credit_pull_d = pd.to_datetime(cleaned_data.last_credit_pull_d, format='%b-%y')
print(cleaned_data[date_time_cols].info())

#Dropping duplicate rows if any
cleaned_data=cleaned_data.drop_duplicates()
#print(cleaned_data.shape)
#print(cleaned_data.shape)
#print(cleaned_data.head(5))

cleaned_data['int_rate'] = cleaned_data['int_rate'].str.strip('%').astype('float')
#cleaned_data['revol_util'] = cleaned_data['revol_util'].str.strip('%').astype('float')
cleaned_data[['int_rate']].info()

index_names = cleaned_data[ cleaned_data['loan_status'] == 'Current' ].index
#print(len(index_names))
index_names1 = cleaned_data[ cleaned_data['emp_length'] == 'n/a' ].index

cleaned_data.drop(index_names,inplace=True)
cleaned_data.drop(index_names1,inplace=True)

emp_length_dict = {
    '< 1 year' : 0,
    '1 year' : 1,
    '2 years' : 2,
    '3 years' : 3,
    '4 years' : 4,
    '5 years' : 5,
    '6 years' : 6,
    '7 years' : 7,
    '8 years' : 8,
    '9 years' : 9,
    '10+ years' : 10
}

cleaned_data = cleaned_data.replace({"emp_length": emp_length_dict })
cleaned_data.emp_length.value_counts()
#print(cleaned_data.head(5))

cleaned_data['earliest_cr_line_month'] = cleaned_data['earliest_cr_line'].dt.month
cleaned_data['earliest_cr_line_year'] = cleaned_data['earliest_cr_line'].dt.year

cleaned_data['issue_d_month']=cleaned_data['issue_d'].dt.month
cleaned_data['issue_d_year']=cleaned_data['issue_d'].dt.year

#print(cleaned_data.head(5))

#Saving the refined data into final csv file for further analysis
cleaned_data.to_csv('final.csv',encoding='ISO-8859-1', index=False)
final_data = pd.read_csv('final.csv')
print(final_data.shape)

#Data Analysis
print(final_data.head(5))
print(final_data.loan_status.value_counts())

plt.figure(figsize=(10,6))
ax = sns.boxplot(y='int_rate', x='term', data =final_data,palette="tab10")
ax.set_title('Term of loan vs Interest Rate',fontsize=15,color='w')
ax.set_ylabel('Interest Rate',fontsize=14,color = 'w')
ax.set_xlabel('Term of loan',fontsize=14,color = 'w')

final_data['funded_amnt_inv']=final_data['funded_amnt_inv'].astype('int64')
#fig = plt.subplots(figsize=(16,5))
fig = plt.subplots(figsize=(16,5))
loan_amount = final_data["loan_amnt"].values
#investor_funds = final_data["funded_amnt_inv"].values

sns.distplot(loan_amount, color="#F7522F")
plt.title("Loan Applied by the Borrower", fontsize=14)
#sns.distplot(investor_funds, ax=ax[1], color="#2EAD46")
#ax[1].set_title("Total committed by Investors", fontsize=14)

plt.figure(figsize=(12,8))
sns.barplot('issue_d_year', 'loan_amnt', data=final_data, palette='tab10')
plt.title('Issuance of Loans', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average loan amount issued', fontsize=14)
#plt.show()

#Info on Loan conditions by loan status
f, ax = plt.subplots(1,2, figsize=(16,8))

colors = ["#3791D7", "#D72626"]
labels ="Fully Paid", "Charged Off"

plt.suptitle('Information on Loan Conditions', fontsize=20)

final_data["loan_status"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors,
                                             labels=labels, fontsize=12, startangle=70)

# ax[0].set_title('State of Loan', fontsize=16)
ax[0].set_ylabel('% of Condition of Loans', fontsize=14)

# sns.countplot('loan_condition', data=df, ax=ax[1], palette=colors)
# ax[1].set_title('Condition of Loans', fontsize=20)
# ax[1].set_xticklabels(['Good', 'Bad'], rotation='horizontal')
palette = ["#3791D7", "#E01E1B"]

sns.barplot(x="issue_d_year", y="loan_amnt", hue="loan_status", data=final_data, palette=palette, estimator=lambda x: len(x) / len(df) * 100)
ax[1].set(ylabel="(%)")
#plt.show()

#Plotting by term
#plt.show()

# Plotting by states

# Grouping by our metrics
# First Plotly Graph (We evaluate the operative side of the business)
by_loan_amount = final_data.groupby(['addr_state'], as_index=False).loan_amnt.sum()
by_interest_rate = final_data.groupby(['addr_state'], as_index=False).int_rate.mean()
by_income = final_data.groupby(['addr_state'], as_index=False).annual_inc.mean()
by_dti=final_data.groupby(['addr_state'], as_index=False).dti.mean()

# Take the values to a list for visualization purposes.
states = by_loan_amount['addr_state'].values.tolist()
average_loan_amounts = by_loan_amount['loan_amnt'].values.tolist()
average_interest_rates = by_interest_rate['int_rate'].values.tolist()
average_annual_income = by_income['annual_inc'].values.tolist()
average_dti = by_dti['dti'].values.tolist()


from collections import OrderedDict

# Figure Number 1 (Perspective for the Business Operations)
metrics_data = OrderedDict([('addr_state', states),
                            ('loan_amnt_issued', average_loan_amounts),
                            ('int_rate', average_interest_rates),
                            ('annual_inc', average_annual_income),
                            ('dti', average_dti)])

metrics_df = pd.DataFrame.from_dict(metrics_data)
metrics_df = metrics_df.round(decimals=2)

print(metrics_df.head(51))
print(metrics_df.sort_values(by="loan_amnt_issued",ascending=False).head(10))
print(metrics_df.sort_values(by="dti",ascending=False).head(10))
print(metrics_df.sort_values(by="dti").head(10))

final_data['income_category'] = np.nan
lst = [final_data]

for col in lst:
    col.loc[col['annual_inc'] <= 100000, 'income_category'] = 'Low'
    col.loc[(col['annual_inc'] > 100000) & (col['annual_inc'] <= 200000), 'income_category'] = 'Medium'
    col.loc[col['annual_inc'] > 200000, 'income_category'] = 'High'
print(final_data.head(5))

lst = [final_data]
final_data['loan_condition_int'] = np.nan

for col in lst:
    col.loc[final_data['loan_status'] == 'Fully Paid', 'loan_condition_int'] = 0  # Negative (Bad Loan)
    col.loc[final_data['loan_status'] == 'Charged Off', 'loan_condition_int'] = 1  # Positive (Good Loan)

# Convert from float to int the column (This is our label)
final_data['loan_condition_int'] = final_data['loan_condition_int'].astype(int)
#final_data['income_category'] = final_data['income_category'].astype(str)

print(final_data.dtypes)

fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(nrows=2, ncols=2, figsize=(14,6))

# Change the Palette types

sns.violinplot(x="income_category", y="loan_amnt", data=final_data, palette="Set2", ax=ax1 )
sns.violinplot(x="income_category", y="loan_condition_int", data=final_data, palette="Set2", ax=ax2)
sns.boxplot(x="income_category", y="emp_length", data=final_data, palette="Set2", ax=ax3)
sns.boxplot(x="income_category", y="int_rate", data=final_data, palette="Set2", ax=ax4)
#plt.show()


fig = plt.figure(figsize=(16,12))

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(212)

cmap = plt.cm.coolwarm_r

loans_by_region = final_data.groupby(['grade', 'loan_status']).size()
loans_by_region.unstack().plot(kind='bar', stacked=True, colormap=cmap, ax=ax1, grid=False)
ax1.set_title('Type of Loans by Grade', fontsize=14)

loans_by_grade = final_data.groupby(['sub_grade', 'loan_status']).size()
loans_by_grade.unstack().plot(kind='bar', stacked=True, colormap=cmap, ax=ax2, grid=False)
ax2.set_title('Type of Loans by Sub-Grade', fontsize=14)

by_interest = final_data.groupby(['issue_d_year', 'loan_status']).int_rate.mean()
by_interest.unstack().plot(ax=ax3, colormap=cmap)
ax3.set_title('Average Interest rate by Loan Condition', fontsize=14)
ax3.set_ylabel('Interest Rate (%)', fontsize=12)
#plt.show()

#Condition of Loans and Purpose

final_data['purpose'].value_counts()
plt.figure(figsize=(10,6))
ax = sns.countplot(y='purpose', hue='loan_status', data =final_data,palette="tab10")
ax.set_title('Condition of Loan by Purpose',fontsize=15,color='w')
ax.set_ylabel('% of Loan',fontsize=14,color = 'w')
ax.set_xlabel('Loan Purpose',fontsize=14,color = 'w')

check=final_data.pivot_table(index='term',
               columns='loan_status',
               aggfunc='size',
               fill_value=0)
print(check)

plt.show()