# Explore the data and field definitions from the data dictionary.

import pandas as pd             ##Use pandas to load CSV files as DataFrames.

customer_df = pd.read_csv('../data/raw/customer.csv')
dictionary_df = pd.read_csv('../data/raw/data_dictionary.csv')
transaction_df = pd.read_csv('../data/raw/transaction.csv')

print("==== Customer DataFrame Info====")
print(customer_df.shape)                    ##Print the DataFrame dimensions.
print(customer_df.dtypes)                   ##Print each column's data type.
print(customer_df.head())                   ##Print the first five rows.
print(customer_df.isnull().sum())           ##Count missing values in each column.

print("\n=== TRANSACTIONS ===")
print(transaction_df.shape)
print(transaction_df.dtypes)
print(transaction_df.head())
print(transaction_df.isnull().sum())

print("\n=== DICTIONNAIRE ===")
print(dictionary_df)