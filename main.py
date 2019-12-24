import numpy as np
import pandas as pd
#To use the Apriori Algorithm
from mlxtend.frequent_patterns import apriori, association_rules

data=pd.read_csv('Online Retail.csv')
print(data.head())
print(data.columns)
print("Size of data is ",len(data))

#remove all empty/trailing spaces from the description
data['Description']=data['Description'].str.strip()

#remove all empty invoice numbers
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
data['InvoiceNo']=data['InvoiceNo'].astype('str')
print("Size of data after drop is ",len(data))

#The size of data country wise#
print("UK country data =",data[data.Country=='United Kingdom'].shape)
print("France country data =",data[data.Country=='France'].shape)
print("Portugal data count is =",data[data.Country=='Portugal'].shape)

#Creating baskets based on the countries
basket_France = (data[data['Country'] =="France"] 
          .groupby(['InvoiceNo', 'Description'])['Quantity'] 
          .sum().unstack().reset_index().fillna(0) 
          .set_index('InvoiceNo')) 
print(basket_France.head(5))
basket_France.to_csv('OutputFrance.csv')
############################################

#Creating baskets based on the countries
basket_UK = (data[data['Country'] =="United Kingdom"] 
          .groupby(['InvoiceNo', 'Description'])['Quantity'] 
          .sum().unstack().reset_index().fillna(0) 
          .set_index('InvoiceNo')) 
print(basket_UK.head(5))
###########################################

#Creating baskets based on the countries
basket_portugal = (data[data['Country'] =="Portugal"] 
          .groupby(['InvoiceNo', 'Description'])['Quantity'] 
          .sum().unstack().reset_index().fillna(0) 
          .set_index('InvoiceNo')) 
print(basket_portugal.head(5))
############################################

##################---HOT ENCODING---########################
#This is done because many machine learning algorithms cannot work with categorical data directly. The categories must be converted into numbers. This is required for both input and output variables that are categorical
def hot_encode(x): 
    if(x<= 0): 
        return 0
    if(x>= 1): 
        return 1

######################################################
#Apply Encoding to baskets
basket_France=basket_France.applymap(hot_encode)
basket_UK=basket_UK.applymap(hot_encode)
basket_Portugal=basket_portugal.applymap(hot_encode)

#Create the model
freq_item=apriori(basket_France,min_support=0.5,use_colnames=True)

#Creating the dataframe from the generated model#
rules=association_rules(freq_item,metric="lift",min_threshold=0.3)
print(rules.head)