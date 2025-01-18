# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:15:21 2025

@author: Admin
"""

import pandas as pd
df=pd.read_csv("D:/Assciation_rule/myphonedata.csv")
df.head()

#already encoded in format suitable for apriori
#apply apriori algorithm
from mlxtend.frequent_patterns import apriori,association_rules
frequent_itemsets=apriori(df,min_support=0.2,use_colnames=True)
frequent_itemsets

#association rules
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1)
rules

#Sort rules by lift in descending order
rules_sorted_by_lift = rules.sort_values('lift', ascending=False)
#plot lift for the 5 association rules
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(range(len(rules_sorted_by_lift)), rules_sorted_by_lift['lift'])
plt.xticks(range(len(rules_sorted_by_lift)), rules_sorted_by_lift.apply(lambda row: f"{list(row['antecedents'])} -> {list(row['consequents'])}", axis=1), rotation=45, ha='right')
plt.xlabel('Rules')
plt.ylabel('Lift')
plt.title('Rules by Lift')
plt.show()

#Sort rules by confidence in descending order
rules_sorted_by_confidence = rules.sort_values('confidence', ascending=False)
#bar chart for confidence
plt.figure(figsize=(10, 6))
plt.bar(range(len(rules_sorted_by_confidence)), rules_sorted_by_confidence['confidence'])
plt.xticks(range(len(rules_sorted_by_confidence)), rules_sorted_by_confidence.apply(lambda row: f"{list(row['antecedents'])} -> {list(row['consequents'])}", axis=1), rotation=45, ha='right')
plt.xlabel('Rules')
plt.ylabel('Confidence')
plt.title('Top 5 Rules by Confidence')
plt.show()

