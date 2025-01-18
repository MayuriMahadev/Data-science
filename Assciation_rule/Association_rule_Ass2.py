# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 20:09:46 2025

@author: Admin
"""


'''
Problem Statement: - 
The Departmental Store, has gathered the data of the products it sells on a Daily basis.
Using Association Rules concepts, provide the insights on the rules and the plots.

Business Objective:-
The objective is to identify relationships between products sold at the departmental store using association rule mining. These insights will help:
    
1. Optimize product placement.
2. Create bundled offers to increase sales.
3. Enhance customer satisfaction by offering relevant product recommendations.
 
Constraints:-
1. Data Quality: Incomplete or inconsistent transactional data may require extensive preprocessing.
2. Scalability: Algorithms must handle large datasets efficiently.
3. Thresholds: Defining appropriate thresholds for support and confidence to balance meaningfulness and granularity of the rules.
4. Interpretability: Insights must be actionable and comprehensible to stakeholders. 
'''

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load Dataset
df = pd.read_csv(r"D:/Assciation_rule/groceries.csv", on_bad_lines='skip', header=None)

# Display the first few rows of the dataset
print("Dataset Sample:")
print(df.head())

# Check dataset structure
print("\nDataset Info:")
print(df.info())

# Step 2: Data Preprocessing
# Convert the dataset into a list of lists
transactions = df.stack().groupby(level=0).apply(list).tolist()

# Step 3: Convert dataset into a suitable format using TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df1 = pd.DataFrame(te_ary, columns=te.columns_)

# Step 4: Apply the Apriori Algorithm
frequent_itemsets = apriori(df1, min_support=0.03, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Step 5: Generate Association Rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Optional: Filtering for lift >= 2
rules_filtered = rules[rules['lift'] >= 2]
print("\nAssociation Rules with Lift >= 2:")
print(rules_filtered[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Optional: Save Results to CSV
frequent_itemsets.to_csv("frequent_itemsets.csv", index=False)
rules.to_csv("association_rules.csv", index=False)

# Step 6: Visualize Results (Optional)
import matplotlib.pyplot as plt

# Bar plot for top 10 frequent itemsets
top_itemsets = frequent_itemsets.nlargest(10, 'support')
plt.barh(top_itemsets['itemsets'].astype(str), top_itemsets['support'], color='skyblue')
plt.xlabel('Support')
plt.ylabel('Itemsets')
plt.title('Top 10 Frequent Itemsets')
plt.show()
