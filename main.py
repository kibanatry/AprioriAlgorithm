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