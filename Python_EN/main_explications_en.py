#!/usr/bin/python3
#Simplified code
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

###
#Comma-separated data retrieval from a csv file to generate a sample in a random way, and thus display information about this DataFrame (dimension, statistics, memory usage, non-zero values, columns, dtype index, ...).
###
dataset1 = pd.read_csv('weather.csv')   #Reading a file (csv) of comma-separated values in a DataFrame
dataset1.sample()                       #Generating a random sample of each group of an object in the DataFrame
dataset=dataset1.sample(1000)           #A new DataFrame (dataset) containing 1000 randomly sampled elements from dataset1
print(dataset.shape)                    #Display of a tuple representing the dimension of the DataFrame dataset
print(dataset.describe())               #Display of statistics for each type of DataFrame values
dataset.info()                          #Displays DataFrame information, including dtype index, colonnnes and non-zero values and memory usage.

###
#Display of the temperature distribution as a function of minimum and maximum temperatures in a scatter plot
###
dataset.plot(x='Temp_Min', y='Temp_Max', style='o') #Creating a DataFrame plot, with the x-axis representing Temp_min and the y-axis representing Temp_Max, in a scatter plot style.
plt.title('Temp_Min vs Temp_Max')                   #Setting the graph title
plt.xlabel('Temp_Min')                              #Setting the x-axis title
plt.ylabel('Temp_Max')                              #Setting the y-axis title
plt.show()                                          #Graphic/Figure display

###
#Creation of a histogram graph to visualize the distribution of the Temp_Max values of the DataFrame
###
plt.figure(figsize=(15,10))                     #Creation of a figure of a size defined by figsize in inch, 15 inch wide and 10 inch high
plt.tight_layout()                              #Adjusting the borders between and around the plot
seabornInstance.distplot(dataset['Temp_Max'])   #Allows you to draw a distribution plot on a FacetGrid, allowing you to view the Temp_Max DataFrame data in a histogram format (default).
plt.show()                                      #Graphic/Figure display

###
#Predicting Temp_Max from the previous file
###
X = dataset['Temp_Min'].values.reshape(-1,1)                                                        #The X value includes the Temp_Min attribute
y = dataset['Temp_Max'].values.reshape(-1,1)                                                        #The value includes the Temp_Max attribute
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)            #Allocation of 80% of the data to the training set and the rest to the test set
reg = LinearRegression().fit(X_train, y_train)                                                      #Training of the model using the training set
print(reg.intercept_)                                                                               #Intersection display 
print(reg.coef_)                                                                                    #Display of the directing coefficient of the regression line
y_pred = reg.predict(X_test)                                                                        #Using test data to make predictions about Temp_Max
df = pd.DataFrame({'Actuelle(Mesurées)': y_test.flatten(), 'Prédiction(modèle)': y_pred.flatten()}) #Building a DataFrame from predicted data and tests
print(df)                                                                                           #Displaying DataFrame


###
#Creation of the vertical bar graph showing the distribution of measured and predicted temperatures.
###
df1 = df.head(25)                                                       #Retrieving the first 25 lines of df to put them in df1
df1.plot(kind='bar',figsize=(16,10))                                    #Creation of DataFrame, size 16inch wide and 10inch high, and the style of the graph will be vertical bars
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')  #Creation of the internal grid (major) of the graphic with a solid line style, 0.5 points thick, one green colour
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')  #Creation of the frame (minor) of the graphic with a dotted line style, 0.5 points thick, one black colour
plt.show()                                                              #Graphic/Figure display

###
#Creation of a scatter plot in grey with the previously calculated regression line in red
###
plt.title('Modèle ax+y')                            #Setting the chart title
plt.xlabel('Temp_Min')                              #Setting the x-axis title
plt.ylabel('Temp_Max')                              #Setting the y-axis title
plt.scatter(X_test, y_test,  color='gray')          #Creation of a scatter diagram y_test vs. X_test of grey colour
plt.plot(X_test, y_pred, color='red', linewidth=2)  #CCreation of a DataFrame plot (of the ax+y function), with the x-axis representing the X_Test and the y-axis representing the y_pred, in red colour and 2 points wide.
plt.show()                                          #Graphic/Figure display


###
#Displays values based on predictions
###
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))              #Display of the calculation of the mean absolute values of the errors   
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))                #Display of mean squared error
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  #Display of the root mean squared error


###
#Addition of mean values in a table, display of this table and mean values and standard deviation
###
Means = []                                          #Creation of an empty table of average values
Means=testSamples(200, 100,dataset['Temp_Min'])     #Adding average values to the table with the previously created testSamples function
print(Means)                                        #Displaying the mean value table
print(np.mean(Means))                               #Displaying the average of the Means table
print(np.mean(dataset['Temp_Min']))                 #Displaying the average of the dataset['Temp_Min'] table
print(np.std(Means))                                #Displays the standard deviation of the values in the Means table.
print(np.std(dataset['Temp_Min']))                  #Displays the standard deviation of the values in the dataset['Temp_Min'] table

###
#Creation of a stepped histogram
###
plt.figure(1)                                   #Creation of a figure with a unique identifier equal to 1
plt.hist(Means, bins=10, histtype='step')       #Creation of a stepped histogram with a single line and without filling, with 10 steps of the same width.
plt.show()                                      #Graphic/Figure display


###
#Shapiro-Wilk test on a normally distributed population
###
stat, p = stats.shapiro(Means)                          #We do the Shapiro-Wilk test which verifies the null hypothesis according to Means' data, and returns the value of the test statistic and the p-value for the test hypothesis.
print('Statistics={}, p={}'.format(stat, p))            #Display of statistics and p-value
alpha = 0.05                                            #Initilization of alpha
if p > alpha:                                           #Test between p-value and alpha
    print('Sample looks Normal (do not reject H0)')     #Display if p-value > alpha
else:   
    print('Sample does not look Normal (reject H0)')    #Display if p-value < alpha


###
#A box and whisker plot. The box extends from the values in the lower quartile to the upper quartile of the data, with a line at the median. The whiskers extend from the box to show the extent of the data. Flier points are those past the end of the whiskers.
###
plt.boxplot(Means)  #Creation of a box and whisker plot
plt.show()          #Graphic/Figure display

###
#Probability curve
###
stats.probplot(Means, plot=plt) #Calculates the quantiles of the Means normal probability curve and the plot with matplotlib
plt.show()                      #Graphic/Figure display


###
#Writing the calculation of the linear regression coefficient in two different ways
###
X, y = make_classification(n_samples=100, n_features=10, n_informative=10, n_redundant=0, random_state=1) #Setting up the cloud plot
model1 = LogisticRegression()                                                                             #Initialization of the evaluation of model 1 by the LogisticRegression() function
cv1 = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)                                    #Setting the variable cv1   
scores1 = cross_val_score(model1, X, y, scoring='accuracy', cv=cv1, n_jobs=-1)                            #Setting the variable scores1
print('LogisticRegression Mean Accuracy: %.3f (%.3f)' % (np.mean(scores1), np.std(scores1)))              #Display of the average linear regression coefficient for model 1   
model2 = LinearDiscriminantAnalysis()                                                                     #Initiation of the evaluation of model 2 by the function 
cv2 = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)                                    #Setting the cv2 variable
scores2 = cross_val_score(model2, X, y, scoring='accuracy', cv=cv2, n_jobs=-1)                            #Setting the scores2 variable
print('LinearDiscriminantAnalysis Mean Accuracy: %.3f (%.3f)' % (np.mean(scores2), np.std(scores2)))      #Display of the average linear regression coefficient for model 2 

###
#Creation of a box and whisker plot
###
plt.boxplot([scores1, scores2], labels=['LR', 'LDA'], showmeans=True)           #Creation of a box and whisker plot from scores1 and scores2
plt.show()                                                                      #Graphic/Figure display


###
#Determination of P-value and T-stastitic for testing between the two models
###
t, p = paired_ttest_5x2cv(estimator1=model1, estimator2=model2, X=X, y=y, scoring='accuracy', random_seed=1) #Initialization of the value pair t and p
print('P-value: %.3f, t-Statistic: %.3f' % (p, t))                                                           #Display of the p-Value and t-Statistic initialised above
if p <= 0.05:                                                                                                #Test of the value of the variable p
	print('Difference between mean performance is probably real')                                            #Display if p<=0.05
else:
	print('Algorithms probably have the same performance')                                                   #Display if p>0.05


###
#Discriminant analysis
###
X = standardize(X)                                                      #Smoothing of variable X
lda = ldaf(n_discriminants=2)                                           #Initialization of the discriminator
lda.fit(X, y)                                                           #Training the model using the training package
X_lda = lda.transform(X)                                                #Transforms the values so that they can be used by the following functions
plt.figure(figsize=(6, 4))                                              #Creation of a figure with a certain size specified as an argument
for lab, col in zip((0, 1),('blue', 'red')):                            #Loop for plotting the cloud plot according to line and column
    plt.scatter(X_lda[y == lab, 0],X_lda[y == lab, 1],label=lab,c=col)  #Plotting the cloud plot
plt.xlabel('Linear Discriminant 1')                                     #Setting x-axis title 
plt.ylabel('Linear Discriminant 2')                                     #Setting y-axis title 
plt.legend(loc='lower right')                                           #Setting the legend on the bottom right-hand side
plt.tight_layout()                                                      #Adjusting the borders between and around the undercuts
plt.show()                                                              #Graphic/Figure display


print("\n  _____   ____                ____   \n |  __ \ |  _ \       /\     |  _ \  \n | |  | || |_) |     /  \    | |_) | \n | |  | ||  _ <     / /\ \   |  _ <  \n | |__| || |_) |   / ____ \ _| |_) | \n |_____(_)____(_) /_/    \_(_)____(_)\n                                     \n")
