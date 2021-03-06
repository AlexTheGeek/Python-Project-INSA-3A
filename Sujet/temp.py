#Code de base
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statistics
import random
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
import scipy
from scipy import stats
import math
import csv
import re
import io
import sys
import json
import glob
from sklearn.naive_bayes import GaussianNB

dataset1 = pd.read_csv('weather.csv')
dataset1.sample()
dataset=dataset1.sample(1000)
print(dataset.shape)
print(dataset.describe())
print(dataset.info())
dataset.plot(x='Temp_Min', y='Temp_Max', style='o')

plt.title('Temp_Min vs Temp_Max')  
plt.xlabel('Temp_Min')  
plt.ylabel('Temp_Max')  
plt.show()
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['Temp_Max'])
plt.show()
X = dataset['Temp_Min'].values.reshape(-1,1)
y = dataset['Temp_Max'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actuelle(Mesurées)': y_test.flatten(), 'Prédiction(modèle)': y_pred.flatten()})
print(df)

df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

plt.title('Modèle ax+y')  
plt.xlabel('Temp_Min')  
plt.ylabel('Temp_Max')  
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
Z= dataset['Temp_Min']

Means = []
def  testSamples(numTrials, sampleSize):
    for t in range(numTrials ):
        Y=Z.sample(sampleSize)
        Means.append(sum(Y)/len(Y))
    return Means

#Means=pd.DataFrame(Means)
Means=testSamples(200, 100)
print(Means)

#testSamples(50, 30)

print(np.mean(Means))
print(np.mean(Z))
print(np.std(Means))
print(np.std(Z))
plt.figure(1)
plt.hist(Means, bins=int(10), histtype='step')
#plt.figure(2)
#plt.hist(vals,  bins=10)
plt.show()

from scipy.stats import shapiro
stat, p = shapiro(Means)
print('Statistics={}, p={}'.format(stat, p))
alpha = 0.05
if p > alpha:
    print('Sample looks Normal (do not reject H0)')
else:
    print('Sample does not look Normal (reject H0)')


plt.boxplot(Means)
plt.show()

import statsmodels.api as sm
import pylab
import pylab 
import scipy.stats as stats

#sm.qqplot(Means)
stats.probplot(Means, dist="norm", plot=pylab)
pylab.show()
#sm.qqplot(Means, line ='45') 
#pylab.show()
##########################################################
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy import mean
from numpy import std
from mlxtend.evaluate import paired_ttest_5x2cv
###################################################################
X, y = make_classification(n_samples=100, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# summarize the dataset
#print(X.shape, y.shape)
# evaluate model 1
model1 = LogisticRegression()
cv1 = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)
scores1 = cross_val_score(model1, X, y, scoring='accuracy', cv=cv1, n_jobs=-1)
print('LogisticRegression Mean Accuracy: %.3f (%.3f)' % (mean(scores1), std(scores1)))
# evaluate model 2
model2 = LinearDiscriminantAnalysis()
cv2 = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)
scores2 = cross_val_score(model2, X, y, scoring='accuracy', cv=cv2, n_jobs=-1)
print('LinearDiscriminantAnalysis Mean Accuracy: %.3f (%.3f)' % (mean(scores2), std(scores2)))
# plot the results
plt.boxplot([scores1, scores2], labels=['LR', 'LDA'], showmeans=True)
plt.show()

print('LinearDiscriminantAnalysis Mean Accuracy: %.3f (%.3f)' % (mean(scores2), std(scores2)))
# check if difference between algorithms is real
t, p = paired_ttest_5x2cv(estimator1=model1, estimator2=model2, X=X, y=y, scoring='accuracy', random_seed=1)
# summarize
print('P-value: %.3f, t-Statistic: %.3f' % (p, t))
# interpret the result
if p <= 0.05:
	print('Difference between mean performance is probably real')
else:
	print('Algorithms probably have the same performance')



from mlxtend.preprocessing import standardize
from mlxtend.feature_extraction import LinearDiscriminantAnalysis

X = standardize(X)

lda = LinearDiscriminantAnalysis(n_discriminants=2)
lda.fit(X, y)
X_lda = lda.transform(X)
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip((0, 1),
                        ('blue', 'red')):
        plt.scatter(X_lda[y == lab, 0],
                    X_lda[y == lab, 1],
                    label=lab,
                    c=col)
    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Linear Discriminant 2')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
