# explore the data and the types and name from data dictionary

import pandas as pd             ##the pandas module to transorm csv into tables

customer_df = pd.read_csv('../data/raw/customer.csv')
dictionary_df = pd.read_csv('../data/raw/data_dictionary.csv')
transaction_df = pd.read_csv('../data/raw/transaction.csv')

print("==== Customer DataFrame Info====")
print(customer_df.shape)                    ##Give the shape of the dataframe
print(customer_df.dtypes)                   ##The type o each column in the dataframe
print(customer_df.head())                   ##the first 5 rows
print(customer_df.isnull().sum())           ##the number of missing values in each column

print("\n=== TRANSACTIONS ===")
print(transaction_df.shape)
print(transaction_df.dtypes)
print(transaction_df.head())
print(transaction_df.isnull().sum())

print("\n=== DICTIONNAIRE ===")
print(dictionary_df)