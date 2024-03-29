#!/usr/bin/python3
#Simplify Code
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
import csv
import scipy.stats as stats
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlxtend.evaluate import paired_ttest_5x2cv
from mlxtend.preprocessing import standardize
from mlxtend.feature_extraction import LinearDiscriminantAnalysis as ldaf
from function import fonction

print("\n  _____           _      _     __  __       _   _           ____          \n |  __ \         (_)    | |   |  \/  |     | | | |         |___ \   /\    \n | |__) | __ ___  _  ___| |_  | \  / | __ _| |_| |__  ___    __) | /  \   \n |  ___/ '__/ _ \| |/ _ \ __| | |\/| |/ _` | __| '_ \/ __|  |__ < / /\ \  \n | |   | | | (_) | |  __/ |_  | |  | | (_| | |_| | | \__ \  ___) / ____ \ \n |_|   |_|  \___/| |\___|\__| |_|  |_|\__,_|\__|_| |_|___/ |____/_/    \_\ \n                _/ |                                                      \n               |__/                                                       \n\n")

dataset1 = pd.read_csv('weather.csv') 
dataset1.sample() 
dataset=dataset1.sample(1000) 
print(dataset.shape) 
print(dataset.describe()) 
dataset.info()

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
reg = LinearRegression().fit(X_train, y_train)
print(reg.intercept_)
print(reg.coef_)
y_pred = reg.predict(X_test)
df = pd.DataFrame({'Actuelle(Mesurées)': y_test.flatten(), 'Prédiction(modèle)': y_pred.flatten()}) #use of pandas
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


Means = [] 
Means=fonction.testSamples(200, 100, dataset['Temp_Min'])
print(Means)
print(np.mean(Means))
print(np.mean(dataset['Temp_Min']))
print(np.std(Means))
print(np.std(dataset['Temp_Min']))

plt.figure(1)
plt.hist(Means, bins=10, histtype='step')
plt.show()


stat, p = stats.shapiro(Means)
print('Statistics={}, p={}'.format(stat, p))
alpha = 0.05
if p > alpha:
    print('Sample looks Normal (do not reject H0)')
else:
    print('Sample does not look Normal (reject H0)')


plt.boxplot(Means)
plt.show()



stats.probplot(Means, plot=plt)
plt.show()



X, y = make_classification(n_samples=100, n_features=10, n_informative=10, n_redundant=0, random_state=1)
model1 = LogisticRegression()
cv1 = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)
scores1 = cross_val_score(model1, X, y, scoring='accuracy', cv=cv1, n_jobs=-1)
print('LogisticRegression Mean Accuracy: %.3f (%.3f)' % (np.mean(scores1), np.std(scores1)))
model2 = LinearDiscriminantAnalysis()
cv2 = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)
scores2 = cross_val_score(model2, X, y, scoring='accuracy', cv=cv2, n_jobs=-1)
print('LinearDiscriminantAnalysis Mean Accuracy: %.3f (%.3f)' % (np.mean(scores2), np.std(scores2)))
plt.boxplot([scores1, scores2], labels=['LR', 'LDA'], showmeans=True)
plt.show()

t, p = paired_ttest_5x2cv(estimator1=model1, estimator2=model2, X=X, y=y, scoring='accuracy', random_seed=1)
print('P-value: %.3f, t-Statistic: %.3f' % (p, t))
if p <= 0.05:
	print('Difference between mean performance is probably real')
else:
	print('Algorithms probably have the same performance')





X = standardize(X)  

lda = ldaf(n_discriminants=2)
lda.fit(X, y)
X_lda = lda.transform(X)
plt.figure(figsize=(6, 4))
for lab, col in zip((0, 1),('blue', 'red')):
    plt.scatter(X_lda[y == lab, 0],X_lda[y == lab, 1],label=lab,c=col)
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


print("\n  _____   ____                ____   \n |  __ \ |  _ \       /\     |  _ \  \n | |  | || |_) |     /  \    | |_) | \n | |  | ||  _ <     / /\ \   |  _ <  \n | |__| || |_) |   / ____ \ _| |_) | \n |_____(_)____(_) /_/    \_(_)____(_)\n                                     \n")
