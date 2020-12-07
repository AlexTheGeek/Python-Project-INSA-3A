#!/usr/bin/python3
#Simplify Code
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
import csv
import pylab
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
from mlxtend.feature_extraction import LinearDiscriminantAnalysis as lda
from function import fonction

print("\n  _____           _      _     __  __       _   _           ____          \n |  __ \         (_)    | |   |  \/  |     | | | |         |___ \   /\    \n | |__) | __ ___  _  ___| |_  | \  / | __ _| |_| |__  ___    __) | /  \   \n |  ___/ '__/ _ \| |/ _ \ __| | |\/| |/ _` | __| '_ \/ __|  |__ < / /\ \  \n | |   | | | (_) | |  __/ |_  | |  | | (_| | |_| | | \__ \  ___) / ____ \ \n |_|   |_|  \___/| |\___|\__| |_|  |_|\__,_|\__|_| |_|___/ |____/_/    \_\ \n                _/ |                                                      \n               |__/                                                       \n\n")

###
#Recuperation des donnees separees par des virgules à partir d'un fichier csv pour generer un echantillon de manière aléatoire, et ainsi afficher des informations sur cet DatamFrame (dimension, statistique, utilisatioon mémoire, valeurs non nulle, colonnes, index dtype, ...)
###
dataset1 = pd.read_csv('weather.csv')   #Lecture d'un fichier (csv) de valeurs separees par des virgules dans un DataFrame
dataset1.sample()                       #Generation d'un echantillon aleatoires de chaque groupe d'un objet du DataFrame
dataset=dataset1.sample(1000)           #Un nouvel DataFrame (dataset) contenenant 1000 elements echantillonnes de façon aleatoire a partir de dataset1
print(dataset.shape)                    #Affichage d'un tuple representant la dimension du DataFrame dataset
print(dataset.describe())               #Affichage des statistiques pour chaque type de valeurs du DataFrame
print(dataset.info())                   #Affichage des informations sur le DataFrame, notamment l'index dtype et les colonnnes et les valeurs non nulles et l'utilisation de la mémoire

###
#Affichage de la répartition des temperatures en fonction des températures min et températures max, dans un graphique en nuage de points
###
dataset.plot(x='Temp_Min', y='Temp_Max', style='o') #Creation d'un trace de DataFrame, avec l'axe x representant les Temp_min et l'axe y representant les Temp_Max, dans un style de nuge de points
plt.title('Temp_Min vs Temp_Max')                   #Parametrage du titre du graphique
plt.xlabel('Temp_Min')                              #Parametrage du titre de l'axe x
plt.ylabel('Temp_Max')                              #Parametrage du titre de l'axe y
plt.show()                                          #Affichage du graphique/figure

###
#Creation d'un graphique histogramme pour visualiser la répartion des valeur de Temp_Max du DataFrame
###
plt.figure(figsize=(15,10))                     #Creation d'un figure de taille definie par figsize en inch, 15 inch de largeur et 10 de hauteur
plt.tight_layout()                              #Ajustement des bordures entre et autour les sous trace
seabornInstance.distplot(dataset['Temp_Max'])   #Permet de dessiner un trace de distribution sur une FacetGrid, permettant de visualiser les donnees de DateFrame des Temp_Max dans un format d'histogramme (par defaut)
plt.show()                                      #Affichage du graphique/figure

###
#Prediction des Temp_Max a partir du fichier precedent
###
X = dataset['Temp_Min'].values.reshape(-1,1)                                                        #La valeur X inclut l'attribut Temp_Min
y = dataset['Temp_Max'].values.reshape(-1,1)                                                        #La valeur y includ l'attribut Temp_Max
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)            #Attribution de 80% des donnees a l'ensemble de formation et le reste a l'ensemble de test
regressor = LinearRegression()                                                                      #Creation un objet de regression lineaire  
regressor.fit(X_train, y_train)                                                                     #Entrainement du model en utilisant l'ensemble de formation
print(regressor.intercept_)                                                                         #Affichage de l'intersection 
print(regressor.coef_)                                                                              #Affichage du coefficient directeur de la droite de regression
y_pred = regressor.predict(X_test)                                                                  #Utilisation des donnees de test pour faire des predictions sur le Temp_Max
df = pd.DataFrame({'Actuelle(Mesurées)': y_test.flatten(), 'Prédiction(modèle)': y_pred.flatten()}) #Construction d'un DataFrame a partir des donnes predictes et de test
print(df)                                                                                           #Affichage de DataFrame


###
#Création du graphique à barre verticale montrant la répartitions des températures mesurées et prédites.
###
df1 = df.head(25)                                                       #Recuperation des 25 dernières lignes de df pour les mettre dans df1
df1.plot(kind='bar',figsize=(16,10))                                    #Création de DataFrame, de taille 16inch de largeur et 10inch de hauteur, et le style du graphique sera des barres verticales
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')  #Creation de la grille interne (makor) du graphique avec un style de trait plein, epaisseur de 0.5points, une couleur verte
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')  #Creation du cadre(minor) du graphique avec un style de trait à point, epaisseur de 0.5points, une couleur noire
plt.show()                                                              #Affichage du graphique/figure

###
#Creation d'un graphique de dispersion en gris avec la droite de regression en rouge calculee precedemment
###
plt.title('Modèle ax+y')                            #Parametrage du titre du graphique
plt.xlabel('Temp_Min')                              #Parametrage du titre de l'axe x
plt.ylabel('Temp_Max')                              #Parametrage du titre de l'axe y
plt.scatter(X_test, y_test,  color='gray')          #Cration d'un diagramme de dispersion y_test contre X_test de couleur grise
plt.plot(X_test, y_pred, color='red', linewidth=2)  #Creation d'un trace (de la fonction ax+y) de DataFrame, avec l'axe x representant les X_Test et l'axe y representant les y_pred, dans la couleur rouge et de largeur 2points
plt.show()                                          #Affichage du graphique/figure


###
#Affiche des valeurs en fonction des predictions
###
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))              #Affichage du calcul des valeurs absolues moyennes des erreurs   
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))                #Affichage du calcul de la moyenne des erreurs au carré
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  #Affichage du calcul de la racine carrée de la moyenne des erreurs quadratiques


###
#Ajout des valeurs moyenne dans un tableau et affichage de ce tableau et des valeurs moyennes et ecart-type
###
Means = []                                          #Creation d'un tableau vide des valeurs moyennes
Means=testSamples(200, 100,dataset['Temp_Min'])     #Ajout des valeurs moyenne dans le tableau avec la fonction testSamples cree precedemment
print(Means)                                        #Affichage du tableau des valeurs moyennes
print(np.mean(Means))                               #Affichage de la moyenne du tableau Means
print(np.mean(dataset['Temp_Min']))                 #Affichage de la moyenne du tableau dataset['Temp_Min']
print(np.std(Means))                                #Affiche l'ecart-type des valeurs du tableau Means
print(np.std(dataset['Temp_Min']))                  #Affiche l'ecart-tupe des valeurs du tableau dataset['Temp_Min']

###
#Creation d'un histogramme en escalier
###
plt.figure(1)                                   #Creation d'un figure avec un unique identifiant egale à 1
plt.hist(Means, bins=int(10), histtype='step')  #Creation d'un histogramme en escalier avec un seul trait et sans remplissage, avec 10 marches ayant la meme largeur
plt.show()                                      #Affichage du graphique/figure


###
#Test de Shapiro-Wilk sur une population distribuee normalement
###
stat, p = stats.shapiro(Means)                          #On fait le test de Shapiro-Wilk qui verifie l'hypothese nulle selon les donnees de Means, et retourne la valeur de la statistique du test et la p-value pour l'hypothese du test
print('Statistics={}, p={}'.format(stat, p))            #Affichage de la statistique et de la p-value
alpha = 0.05                                            #Iniatialisation de alpha
if p > alpha:                                           #Test entre p-value et alpha
    print('Sample looks Normal (do not reject H0)')     #Affichage si p-value > alpha
else:   
    print('Sample does not look Normal (reject H0)')    #Affichage si p-value < alpha


###
#Diagramme en boites et en moutaches. La boîte s'étend des valeurs du quartile inférieur au quartile supérieur des données, avec une ligne à la médiane. Les moustaches s'étendent à partir de la boîte pour montrer l'étendue des données. Les points de vol sont ceux qui se trouvent après l'extrémité des moustaches.
###
plt.boxplot(Means)  #Creation d'un diagramme en boites et en moutaches de Means
plt.show()          #Affichage du graphique/figure

###
#Courbe de probabilite
###
stats.probplot(Means, dist="norm", plot=pylab)  #Calcule les quantiles de la courbe de probabilite normale de Means et la trace avec pylab
pylab.show()                                    #Affichage du graphique/figure


###
#
###
X, y = make_classification(n_samples=100, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# summarize the dataset
#print(X.shape, y.shape)
# evaluate model 1
model1 = LogisticRegression()
cv1 = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)
scores1 = cross_val_score(model1, X, y, scoring='accuracy', cv=cv1, n_jobs=-1)
print('LogisticRegression Mean Accuracy: %.3f (%.3f)' % (np.mean(scores1), np.std(scores1)))
# evaluate model 2
model2 = LinearDiscriminantAnalysis()
cv2 = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)
scores2 = cross_val_score(model2, X, y, scoring='accuracy', cv=cv2, n_jobs=-1)
print('LinearDiscriminantAnalysis Mean Accuracy: %.3f (%.3f)' % (np.mean(scores2), np.std(scores2)))
# plot the results
plt.boxplot([scores1, scores2], labels=['LR', 'LDA'], showmeans=True)
plt.show() #Affichage du graphique/figure
#print('LinearDiscriminantAnalysis Mean Accuracy: %.3f (%.3f)' % (mean(scores2), std(scores2)))
# check if difference between algorithms is real
t, p = paired_ttest_5x2cv(estimator1=model1, estimator2=model2, X=X, y=y, scoring='accuracy', random_seed=1)
# summarize
print('P-value: %.3f, t-Statistic: %.3f' % (p, t))
# interpret the result
if p <= 0.05:
	print('Difference between mean performance is probably real')
else:
	print('Algorithms probably have the same performance')


###
#
###
X = standardize(X)
lda = lda(n_discriminants=2)
lda.fit(X, y)
X_lda = lda.transform(X)
plt.figure(figsize=(6, 4))
for lab, col in zip((0, 1),('blue', 'red')):
    plt.scatter(X_lda[y == lab, 0],X_lda[y == lab, 1],label=lab,c=col)
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show() #Affichage du graphique/figure


print("\n  _____   ____                ____   \n |  __ \ |  _ \       /\     |  _ \  \n | |  | || |_) |     /  \    | |_) | \n | |  | ||  _ <     / /\ \   |  _ <  \n | |__| || |_) |   / ____ \ _| |_) | \n |_____(_)____(_) /_/    \_(_)____(_)\n                                     \n")
