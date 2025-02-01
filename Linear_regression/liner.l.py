# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 23:33:05 2025

@author: Admin
"""


import pandas as pd
import numpy as np
import scipy
from scipy import stats
#provide statistical function
#stats contains a variety of statistical tests
from statsmodels.stats import descriptivestats as sd
#provide descriptve statistics tools , inclusind the sign_test.
from statsmodels.stats.weightstats import ztest
#used  for conducting z-tests on datatest

#1 sample sign test
#whenever there is a single and data is not normal
marks=pd.read_csv(r"D:/Linear_regression/Signtest.csv")

#normal QQ plot
import pylab
stats.probplot(marks.Scores, dist='norm',plot=pylab)
#create a QQ plot to visualy check if the data follows a normal distributippn
#test for normally
shapiro_test=stats.shapiro(marks.Scores)
#performs the shapiro-Wilk test for normality
#H0 (null hypothesis): the data is normally distributed
#H1 (alternate hypothesis): the data is not normaly distributed
#outputs a test statisctics and p -value
print("Shapiro Test:",shapiro_test)
#p_value is 0.024 <0.05 , data is not normal

#Descriptive statistics
print(marks.Scores.describe())
#mean=84.20 and median=89.00
#1-sample  sign test
sign_test_result=sd.sign_test(marks.Scores,mu0=marks.Scores.mean())
print("sign Test Result:",sign_test_result)
#Result : p-value=0.82
#interpretation:
#H0 : The median of scores is equal to the mean of scores.
#H1 the median of scores is not equal to the mean scores.
#since the p-value  (0.82) is greater than 0.05, we fail to reject the null hypothesis
#concluusion: the median and mean of scores are statistically

#1 sample z-test
fabric = pd.read_csv(r"C:\13_Linear-regression\Fabric_data.csv")

#normality test
fabric_normality= stats.shapiro(fabric)
print("Fabric normality test:",fabric_normality)
#p value = 0.1460>0.05

fabric_mean=np.mean(fabric)
print("Mean Fabric Length:" ,fabric_mean)

# z-test
#z-test_result,p_val= ztest(fabric['Fabric_length'], value=150)
z_test_result, p_val = ztest(fabric['Fabric_length'], value=150)
print("Z-Test Result:" ,z_test_result,"P-value:",p_val)
#result : p-value =7.15 * 10^-6
#interpretation
#H0 : the mean fabric  length is 150
#H1 : the mean fabric length is not 150
#since the p-value is ectremely small(less than 0.05),we re
#conclusion:the mean fabric length significantly diffres from

#Mann-Whitney Test
fuel=pd.read_csv(r"C:\13_Linear-regression\mann_whitney_additive.csv")
fuel.columns=["Without_additive","With_additive"]

#normality test
print("Without  Additive Normality:" ,stats.shapiro(fuel.Without_additive))
#p=0.50>0.05: accept H0
print("With additive normality:",stats.shapiro(fuel.With_additive))
#0.04<0.05 : reject H0 data is not normal
#Mann-Whitney U Test
mannwhitney_result=stats.mannwhitneyu(fuel.Without_additive,fuel.With_additive)
print("Mann-Whitney test Result:", mannwhitney_result)
#result : p-value= 0.445
#interpretation:
#H0 : No diffrence in performance between without_additive and with_additive.
#H1 : A significant diffrence exists.
#since the  p-value (o.445) is greater than 0.05 , we fail to reject the null hypothesis
#conclusion adding fuel additive does not significanly impact performance.
#applies the mann-Whitney U Test to check if theres's a significant diffrence between
#H0 : No diffrence performance between two groups.
#H1 : Significant diffrence in performance.

#######################3
#Paired T-Test
sup=pd.read_csv(r"C:\13_Linear-regression\paired2.csv")

#normality test 
print("Supplier A Normality Test:",stats.shapiro(sup.SupplierA))
#p-value = 0.896199285 >0.05: fails to reject H0, data is normal
print("Supplier B Normality Test:",stats.shapiro(sup.SupplierB))
##p-value = 0.896199285 >0.05: fails to reject H0, data is normal
#paired T-test
t_test_result,p_val=stats.ttest.rel(sup['SupplierA'],sup['SupplierB'])
print("Paired T-test Result:",t_test_result,"P-value:",p_val)
#result: p-value = 0.00
#Interpretation:
#H0 : no significant diffrence in ttransaction time between supplier A and Supplier B
#H1 : A significant diffrence exists.
#since the p-value (0.00) is less than 0.05 , we reject the null hypothesis.
#conclusion:There is a significant diffrence in transation times between the two supplier