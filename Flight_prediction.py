#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ## Data Collection

# In[3]:


train_data = pd.read_excel(r"C:\Users\BHS7BAN\Desktop\Python/Data_Train.xlsx")


# In[4]:


train_data.head(3)


# In[5]:


train_data.tail(3)


# ## Data Cleaning

# In[6]:


train_data.info()


# In[7]:


train_data.isnull()  #boolen dataframe - will only retuen True/False


# In[8]:


train_data.isnull().sum()


# In[9]:


train_data['Route'].isnull()


# In[10]:


train_data[train_data['Route'].isnull()]


# In[11]:


train_data['Total_Stops'].isnull()


# In[12]:


train_data[train_data['Total_Stops'].isnull()]  


# In[13]:


#since the there is only 1 ror with 2 NaN values we can drop the row
train_data.dropna(inplace=True)

#inplace parameter is use to update the train_data variable also 


# In[14]:


#checking
train_data.isnull().sum()


# In[15]:


train_data.dtypes  
# object - string
#int - integer 


# #### Let's perfrom Pre-processing & extract Derived attributes from "Date_of_Journey" 

# In[16]:


#we will 1st make a copy od main data and perfrom all pre-procesing in it 
data = train_data.copy()


# In[17]:


data.head(3)


# In[18]:


data.columns


# In[19]:


#now we have to convert the data types since Ml can only work in structed data like date, time, number, vectors, etc..
#let's make a function to change the data type to 'datetime'

def change_into_datetime(col):
    data[col] = pd.to_datetime(data[col])


# In[20]:


#if any error come then to ignore it let's make a warning 

import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[21]:


data.columns


# In[22]:


for col in ['Date_of_Journey','Dep_Time','Arrival_Time']:
    change_into_datetime(col)


# In[23]:


data.dtypes


# In[24]:


data['Date_of_Journey'].dt.day


# In[25]:


data['Journey_day'] = data['Date_of_Journey'].dt.day


# In[26]:


data['Journey_month'] = data['Date_of_Journey'].dt.month


# In[27]:


data['Journey_year'] = data['Date_of_Journey'].dt.year


# In[28]:


data.head(3)


# #### P1:  Let's try to clean Dep_Time & Arrival_Time,then extract derrived attributes 

# data['Dep_Time_hour'] = data['Dep_Time'].dt.hour

# data['Dep_Time_min'] = data['Dep_Time'].dt.minute

# data.head(2)

# data['Arrival_Time_hour'] = data['Arrival_Time'].dt.hour

# data['Arrival_Time_min'] = data['Arrival_Time'].dt.minute

# data.head(2)

# In[29]:


#or we can make a fuction for this task 

def extract_hour_min(df, col):
    df[col+"_hour"] = df[col].dt.hour         #col_name = Dep_Time_hour / Arrival_Time_hour
    df[col+"_min"] = df[col].dt.minute     #col_name = Dep_Time_min / Arrival_Time_min
    return df.head(3)


# In[30]:


extract_hour_min(data,"Dep_Time")


# In[31]:


extract_hour_min(data,"Arrival_Time")


# In[32]:


#since we have extracted the time from column "Dep_Time" & "Arrival_Time", now we can remove those 2 columns
columns_to_drop = ['Dep_Time', 'Arrival_Time']
data.drop(columns_to_drop, axis=1, inplace=True)


# In[33]:


data.head(3)


# In[34]:


data.shape  #number of columns and rows in a table


# ## Data Analysis 

# #### P2: Let's analyse when will most of the flight's take-off

# In[35]:


data.columns


# In[36]:


def flight_dep_time(x):
    
    if (x>4) and (x<=8):
        return "Earlt Morning"
    
    if (x>8) and (x<=12):
        return "Morning"
    
    if (x>12) and (x<=16):
        return "Noon"
    
    if (x>16) and (x<=20):
        return "Evening"
    
    if (x>20) and (x<=24):
        return "Night"
    
    else:
        return "Late Night"


# In[37]:


data["Dep_Time_hour"].apply(flight_dep_time)


# In[38]:


data["Dep_Time_hour"].apply(flight_dep_time).value_counts()


# In[39]:


data["Dep_Time_hour"].apply(flight_dep_time).value_counts().plot()  #default_plot


# In[40]:


data["Dep_Time_hour"].apply(flight_dep_time).value_counts().plot(kind="bar", color='blue')  #bar_plot


# In[41]:


#to make the charts interactive we need to use Plotly 


# !pip install plotly

# !pip install chart_studio

# !pip install cufflinks

# In[42]:


import plotly
import cufflinks as cf
from cufflinks.offline import go_offline
from plotly.offline import plot, iplot, init_notebook_mode, download_plotlyjs
init_notebook_mode(connected = True)
cf.go_offline()


# In[43]:


data["Dep_Time_hour"].apply(flight_dep_time).value_counts().iplot(kind="bar")


# In[44]:


data.columns


# In[45]:


data.head(4)


# In[46]:


#as few flight as duration just in hour/min so we need to pre process the data.
def preprocess_duration(col):
    if 'h' not in col:
        col = '0h'+' '+col
    elif 'm' not in col:
        col = col+' '+'0m'
        
    return col


# In[47]:


data['Duration'] = data['Duration'].apply(preprocess_duration)


# In[48]:


data['Duration']


# In[49]:


data.head(4)


# In[50]:


data['Duration'][1]


# In[51]:


'7h 25m'.split(' ')


# In[52]:


'7h 25m'.split(' ')[0]


# In[53]:


'7h 25m'.split(' ')[0][0:-1]   #getting the number


# In[54]:


type('7h 25m'.split(' ')[0][0:-1])


# In[55]:


int('7h 25m'.split(' ')[0][0:-1])  #chinging to integer


# In[56]:


type('7h 25m'.split(' ')[0][0:-1])


# In[57]:


'7h 25m'.split(' ')[1]


# In[58]:


'7h 25m'.split(' ')[1][0:-1]


# In[59]:


int('7h 25m'.split(' ')[1][0:-1])


# In[60]:


#now let's make a lamda function to do this operation

data['Duration_hour'] = data['Duration'].apply(lambda x: int(x.split(' ')[0][0:-1]))


# In[61]:


data['Duration_min'] = data['Duration'].apply(lambda x: int(x.split(' ')[1][0:-1]))


# In[62]:


data.head(3)


# #### P3: Let's analyse weather duration impacts the price or not?

# h to be replaced by *60
# m to be replaced by *1

# In[64]:


data['Duration'].str.replace('h', '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)


# In[65]:


data['Duration_total_min'] = data['Duration'].str.replace('h', '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)


# In[66]:


data['Duration_total_min']


# In[67]:


data.columns


# In[68]:


sns.scatterplot(x='Duration_total_min', y='Price', data=data)


# In[69]:


sns.scatterplot(x='Duration_total_min', y='Price',hue='Total_Stops', data=data)


# In[70]:


sns.lmplot(x='Duration_total_min', y='Price',data=data)  #lmplot shows a regression line


# as the above plot shows as the duration increases the price also increases.

# #### P4: On which way Jet Airways is extremely used?

# #### P5: Airline vs Price Analysis   

# In[71]:


data['Airline']=='Jet Airways'


# In[72]:


data[data['Airline']=='Jet Airways']


# In[74]:


data[data['Airline']=='Jet Airways'].groupby('Route').count()


# In[83]:


data[data['Airline']=='Jet Airways'].groupby('Route').size().sort_values(ascending=False)


# In[84]:


#top 10 Routes of Jet Airways
data[data['Airline']=='Jet Airways'].groupby('Route').size().sort_values(ascending=False).head(10)


# For the 2nd question we will plot boxplot as it shows 25th percentile, 50th Percentile/Median, 75th percentile also it shows the outliers

# In[85]:


data.columns


# In[94]:


sns.boxplot(x='Airline', y='Price', data=data.sort_values('Price', ascending=False))
plt.xticks(rotation='vertical')
plt.show()


# ## Feature Engineering 

# #### P6: Apply One-Hot Encoding on data 

# In[99]:


category_col = [col for col in data.columns if data[col].dtype=='object']  #having data type as object


# In[100]:


numerical_col = [col for col in data.columns if data[col].dtype!='object']  #data type is not object


# In[101]:


category_col


# In[102]:


data['Source'].unique()


# In[107]:


data[data['Source']=='Banglore']


# In[108]:


data['Source'].apply(lambda x:1 if x=='Banglore' else 0)   #one-hot


# Let's do One-Hot

# In[109]:


for sub_category in data['Source'].unique():
    data['Source_'+sub_category] = data['Source'].apply(lambda x : 1 if x==sub_category else 0)


# In[110]:


data.head(3)


# In[112]:


category_col  #One-hot can be done only in categorical columns


# In[113]:


data['Airline'].nunique()


# As we have 12 sub-category in Airline column and if we do One-Hot in this column to make our ML Algo understand the column.
# Since, this many new columns will cause a problem known as 'Curse of Dimension'

# #### P7: Let's Perfrom target guided encoding on Data

# #### P8: Perfrom Manual Encoding on Data  

# In[120]:


data.groupby(['Airline'])['Price'].mean().sort_values()


# In[124]:


airlines = data.groupby(['Airline'])['Price'].mean().sort_values().index


# In[125]:


airlines


# now let's make a dictonary for airlines
# i.e. {key : value} == {airlines : index}

# In[127]:


dict_airlines = {key:index for index, key in enumerate (airlines, 0)}


# In[128]:


dict_airlines


# In[129]:


data['Airline'] = data['Airline'].map(dict_airlines)


# In[131]:


data['Airline']


# In[132]:


data.head(3)  #we can see that the ariline column got updated with the index values of dict_airline


# In[133]:


data['Destination'].unique()


# We see that Delhi and New Delhi are same so let's Replace 'New Delhi' with 'Delhi'

# In[134]:


data['Destination'].replace('New Delhi', 'Delhi', inplace=True)


# In[135]:


data['Destination'].unique()


# In[136]:


destination = data.groupby(['Destination'])['Price'].mean().sort_values().index


# In[137]:


dict_destination = {key:index for index, key in enumerate (destination, 0)}


# In[138]:


dict_destination


# In[139]:


data['Destination'] = data['Destination'].map(dict_destination)


# In[140]:


data.head(3)


# #### P9: Perfrom Manual Encoding on data 

# #### P10: Remove Un-necessary features 

# In[141]:


data.columns


# In[147]:


data['Total_Stops'].unique()


# In[148]:


dict_stops = {'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}


# In[149]:


dict_stops


# In[152]:


data['Total_Stops'] = data['Total_Stops'].map(dict_stops)


# In[154]:


data.head(1)


# In[156]:


data['Additional_Info'].unique()


# In[158]:


data['Additional_Info'].value_counts()    #most of the data has no info


# In[159]:


data['Additional_Info'].value_counts()/len(data)*100    #will show the % of data with no info


# Since, 78% of the data as no info so we can drop the column and also other columns for wich we have done one-hot and lable encoding 

# In[161]:


data.head(1)


# In[163]:


data.drop(columns = ['Date_of_Journey','Source','Route','Duration','Additional_Info','Journey_year'], axis=1, inplace = True)


# In[165]:


data.head(3)


# #### P11: Let's perform Outlier Detection on Price column 

# #### P12: How to deal with Outliers 

# In[167]:


sns.distplot(data['Price'])


# In[168]:


sns.boxplot(data['Price'])


# In[172]:


plt.hist(data['Price'])


# In[173]:


sns.distplot(data['Price'], kde=False)


# In[180]:


def plot(df, col):
    fig , (ax1, ax2, ax3) = plt.subplots(3,1)
    
    sns.distplot(df[col], ax=ax1)
    sns.boxplot(df[col], ax=ax2)
    sns.distplot(df[col], ax=ax3, kde=False)


# In[181]:


plot(data, 'Price')


# In[185]:


q1 = data['Price'].quantile(0.25)
q3 = data['Price'].quantile(0.75)

iqr = q3 - q1

maximum = q3 + 1.5*iqr
minimum = q1 - 1.5*iqr


# In[186]:


print(maximum)


# In[187]:


print(minimum)


# therefore, anything above maximum value and anything less then minimum value is a outlier

# In[188]:


print([price for price in data['Price'] if price>maximum or price<minimum])   #printing the outliers


# In[189]:


len([price for price in data['Price'] if price>maximum or price<minimum])   #total number of outliers


# So to handle this outliers any value say above 35000 will be replaced by the median 

# In[190]:


data['Price'] = np.where(data['Price']>35000, data['Price'].median(), data['Price'])


# In[191]:


plot(data, 'Price')


# #### P14: Perform feature selection 

# In[192]:


x = data.drop(['Price'], axis=1)


# In[193]:


y = data['Price']


# In[194]:


from sklearn.feature_selection import mutual_info_regression


# In[196]:


imp_features = mutual_info_regression(x,y)


# In[197]:


imp_features


# In[204]:


importance = pd.DataFrame(imp_features, index=x.columns)


# In[208]:


importance.sort_values(by=0, ascending=False)


# In[203]:





# #### P15: Let's Build ML model 

# In[209]:


from sklearn.model_selection import train_test_split


# In[211]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# In[212]:


from sklearn.ensemble import RandomForestRegressor


# In[213]:


ml_model = RandomForestRegressor()


# In[214]:


ml_model.fit(x_train, y_train)


# In[215]:


y_predict = ml_model.predict(x_test)


# In[216]:


y_predict


# Now let's check the accuracy of the model

# In[217]:


from sklearn import metrics


# In[219]:


metrics.r2_score(y_test, y_predict)*100


# #### P16: Let's Save Model 

# !pip install pickle  
# #if not working go for pickle-mixin

# !pip install pickle-mixin

# In[224]:


import pickle


# In[225]:


file = open(r'C:\Users\BHS7BAN\Desktop/rf_random.pk1', 'wb')  #wb - write mode


# In[227]:


pickle.dump(ml_model, file)


# In[228]:


model = open(r'C:\Users\BHS7BAN\Desktop/rf_random.pk1', 'rb')  #rb - read mode


# In[230]:


forest = pickle.load(model)


# In[231]:


y_predict2 = forest.predict(x_test)


# In[233]:


metrics.r2_score(y_test, y_predict)   # same results


# In[ ]:




