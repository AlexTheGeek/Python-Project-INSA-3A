#!/usr/bin/python3
#Code simplifié
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
#Récupération des données séparées par des virgules à partir d'un fichier csv pour générer un échantillon de manière aléatoire, et ainsi afficher des informations sur cet DatamFrame (dimension, statistique, utilisatioon mémoire, valeurs non nulle, colonnes, index dtype, ...)
###
dataset1 = pd.read_csv('weather.csv')   #Lecture d'un fichier (csv) de valeurs separées par des virgules dans un DataFrame
dataset1.sample()                       #Génération d'un échantillon aléatoires de chaque groupe d'un objet du DataFrame
dataset=dataset1.sample(1000)           #Un nouvel DataFrame (dataset) contenenant 1000 élèments echantillonnés de façon aléatoire à partir de dataset1
print(dataset.shape)                    #Affichage d'un tuple représentant la dimension du DataFrame dataset
print(dataset.describe())               #Affichage des statistiques pour chaque type de valeurs du DataFrame
print(dataset.info())                   #Affichage des informations sur le DataFrame, notamment l'index dtype et les colonnnes et les valeurs non nulles et l'utilisation de la mémoire

###
#Affichage de la répartition des temperatures en fonction des températures min et températures max, dans un graphique en nuage de points
###
dataset.plot(x='Temp_Min', y='Temp_Max', style='o') #Création d'un trace de DataFrame, avec l'axe x représentant les Temp_min et l'axe y représentant les Temp_Max, dans un style de nuge de points
plt.title('Temp_Min vs Temp_Max')                   #Paramétrage du titre du graphique
plt.xlabel('Temp_Min')                              #Paramétrage du titre de l'axe x
plt.ylabel('Temp_Max')                              #Paramétrage du titre de l'axe y
plt.show()                                          #Affichage du graphique/figure

###
#Création d'un graphique histogramme pour visualiser la répartion des valeur de Temp_Max du DataFrame
###
plt.figure(figsize=(15,10))                     #Création d'un figure de taille définie par figsize en inch, 15 inch de largeur et 10 de hauteur
plt.tight_layout()                              #Ajustement des bordures entre et autour les sous trace
seabornInstance.distplot(dataset['Temp_Max'])   #Permet de dessiner un trace de distribution sur une FacetGrid, permettant de visualiser les donnees de DataFrame des Temp_Max dans un format d'histogramme (par défaut)
plt.show()                                      #Affichage du graphique/figure

###
#Prédiction des Temp_Max à partir du fichier précédent
###
X = dataset['Temp_Min'].values.reshape(-1,1)                                                        #La valeur X inclut l'attribut Temp_Min
y = dataset['Temp_Max'].values.reshape(-1,1)                                                        #La valeur y inclut l'attribut Temp_Max
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)            #Attribution de 80% des données à l'ensemble de formation et le reste à l'ensemble de test
reg = LinearRegression().fit(X_train, y_train)                                                      #Entrainement du model en utilisant l'ensemble de formation
print(reg.intercept_)                                                                               #Affichage de l'intersection 
print(reg.coef_)                                                                                    #Affichage du coefficient directeur de la droite de régression
y_pred = reg.predict(X_test)                                                                        #Utilisation des données de test pour faire des prédictions sur le Temp_Max
df = pd.DataFrame({'Actuelle(Mesurées)': y_test.flatten(), 'Prédiction(modèle)': y_pred.flatten()}) #Construction d'un DataFrame à partir des données prédictes et de test
print(df)                                                                                           #Affichage de DataFrame


###
#Création du graphique à barre verticale montrant la répartitions des températures mesurées et prédites.
###
df1 = df.head(25)                                                       #Récuperation des 25 premières lignes de df pour les mettre dans df1
df1.plot(kind='bar',figsize=(16,10))                                    #Création de DataFrame, de taille 16inch de largeur et 10inch de hauteur, et le style du graphique sera des barres verticales
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')  #Création de la grille interne (major) du graphique avec un style de trait plein, épaisseur de 0.5points, une couleur verte
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')  #Création du cadre (minor) du graphique avec un style de trait à point, épaisseur de 0.5points, une couleur noire
plt.show()                                                              #Affichage du graphique/figure

###
#Création d'un graphique de dispersion en gris avec la droite de régression en rouge calculée précédemment
###
plt.title('Modèle ax+y')                            #Paramétrage du titre du graphique
plt.xlabel('Temp_Min')                              #Paramétrage du titre de l'axe x
plt.ylabel('Temp_Max')                              #Paramétrage du titre de l'axe y
plt.scatter(X_test, y_test,  color='gray')          #Création d'un diagramme de dispersion y_test contre X_test de couleur grise
plt.plot(X_test, y_pred, color='red', linewidth=2)  #Création d'un trace (de la fonction ax+y) de DataFrame, avec l'axe x représentant les X_Test et l'axe y représentant les y_pred, dans la couleur rouge et de largeur 2points
plt.show()                                          #Affichage du graphique/figure


###
#Affiche des valeurs en fonction des predictions
###
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))              #Affichage du calcul des valeurs absolues moyennes des erreurs   
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))                #Affichage du calcul de la moyenne des erreurs au carré
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  #Affichage du calcul de la racine carrée de la moyenne des erreurs quadratiques


###
#Ajout des valeurs moyenne dans un tableau et affichage de ce tableau et des valeurs moyennes et écart-type
###
Means = []                                          #Création d'un tableau vide des valeurs moyennes
Means=testSamples(200, 100,dataset['Temp_Min'])     #Ajout des valeurs moyenne dans le tableau avec la fonction testSamples créée précédemment
print(Means)                                        #Affichage du tableau des valeurs moyennes
print(np.mean(Means))                               #Affichage de la moyenne du tableau Means
print(np.mean(dataset['Temp_Min']))                 #Affichage de la moyenne du tableau dataset['Temp_Min']
print(np.std(Means))                                #Affiche l'écart-type des valeurs du tableau Means
print(np.std(dataset['Temp_Min']))                  #Affiche l'écart-type des valeurs du tableau dataset['Temp_Min']

###
#Création d'un histogramme en escalier
###
plt.figure(1)                                   #Création d'un figure avec un unique identifiant egale à 1
plt.hist(Means, bins=10, histtype='step')       #Création d'un histogramme en escalier avec un seul trait et sans remplissage, avec 10 marches ayant la même largeur
plt.show()                                      #Affichage du graphique/figure


###
#Test de Shapiro-Wilk sur une population distribuée normalement
###
stat, p = stats.shapiro(Means)                          #On fait le test de Shapiro-Wilk qui vérifie l’hypothèse se nulle selon les données de Means, et retourne la valeur de la statistique du test et la p-value pour l'hypothèse du test
print('Statistics={}, p={}'.format(stat, p))            #Affichage de la statistique et de la p-value
alpha = 0.05                                            #Iniatialisation de alpha
if p > alpha:                                           #Test entre p-value et alpha
    print('Sample looks Normal (do not reject H0)')     #Affichage si p-value > alpha
else:   
    print('Sample does not look Normal (reject H0)')    #Affichage si p-value < alpha


###
#Diagramme en boites et en moutaches. La boîte s'étend des valeurs du quartile inférieur au quartile supérieur des données, avec une ligne à la médiane. Les moustaches s'étendent à partir de la boîte pour montrer l'étendue des données. Les points de vol sont ceux qui se trouvent après l'extrémité des moustaches.
###
plt.boxplot(Means)  #Création d'un diagramme en boites et à moustaches de Means
plt.show()          #Affichage du graphique/figure

###
#Courbe de probabilite
###
stats.probplot(Means, plot=plt) #Calcule les quantiles de la courbe de probabilité normale de Means et la trace avec matplotlib
plt.show()                      #Affichage du graphique/figure


###
#Ecriture par deux manières différentes du calcul du coefficient de régression linéaire
###
X, y = make_classification(n_samples=100, n_features=10, n_informative=10, n_redundant=0, random_state=1) #Paramètrage du nuage de points
model1 = LogisticRegression()                                                                             #Initialisation de l'évaluation du modèle 1 par la fonction LogisticRegression()
cv1 = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)                                    #Paramétrage de la variable cv1   
scores1 = cross_val_score(model1, X, y, scoring='accuracy', cv=cv1, n_jobs=-1)                            #Paramétrage de la variable scores1
print('LogisticRegression Mean Accuracy: %.3f (%.3f)' % (np.mean(scores1), np.std(scores1)))              #Affichage du coefficient de regression linéaire moyen pour le modèle 1   
model2 = LinearDiscriminantAnalysis()                                                                     #Initialisation de l'évaluation du modèle 2 par la fonction 
cv2 = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)                                    #Paramétrage de la variable cv2
scores2 = cross_val_score(model2, X, y, scoring='accuracy', cv=cv2, n_jobs=-1)                            #Paramétrage de la variable scores2
print('LinearDiscriminantAnalysis Mean Accuracy: %.3f (%.3f)' % (np.mean(scores2), np.std(scores2)))      #Affichage du coefficient de regression linéaire moyen pour le modèle 2 

###
#Création du diagramme boite à moustaches
###
# plot the results
plt.boxplot([scores1, scores2], labels=['LR', 'LDA'], showmeans=True)           #Création d'un diagramme boite à moustaches à partir de scores1 et scores2
plt.show()                                                                      #Affichage du graphique/figure


###
#Détermination de la P-value et de T-stastitic pour faire un test entre les deux modèles
###
t, p = paired_ttest_5x2cv(estimator1=model1, estimator2=model2, X=X, y=y, scoring='accuracy', random_seed=1) #Initialisation du couple de valeur t et p
print('P-value: %.3f, t-Statistic: %.3f' % (p, t))                                                           #Affichage de la p-Value et de t-Statistic initialisé ci-dessus
if p <= 0.05:                                                                                                #Test de la valeur de la variable p
	print('Difference between mean performance is probably real')                                        #Affichage si p<=0.05
else:
	print('Algorithms probably have the same performance')                                               #Affichage si p>0.05


###
#Analyse du discriminant
###
X = standardize(X)                                                      #Lissage de la variable X
lda = ldaf(n_discriminants=2)                                           #Initialisation du discriminant
lda.fit(X, y)                                                           #Entrainement du modèle en utilisant l'ensemble de formation
X_lda = lda.transform(X)                                                #Transforme les valeurs pour qu'elles soient utilisable par les fonctions d'après
plt.figure(figsize=(6, 4))                                              #Création d'un figure avec une certaine taille précisé en argument
for lab, col in zip((0, 1),('blue', 'red')):                            #Boucle pour tracer le nuage de point en fonction de la ligne et de la colonne
    plt.scatter(X_lda[y == lab, 0],X_lda[y == lab, 1],label=lab,c=col)  #Tracage du nuage de point
plt.xlabel('Linear Discriminant 1')                                     #Paramètrage du titre de l'axe X
plt.ylabel('Linear Discriminant 2')                                     #Paramètrage du titre de l'axe Y
plt.legend(loc='lower right')                                           #Paramètrage de la légende située en bas à droite
plt.tight_layout()                                                      #Ajustement des bordures entre et autour les sous traces
plt.show()                                                              #Affichage du graphique/figure


print("\n  _____   ____                ____   \n |  __ \ |  _ \       /\     |  _ \  \n | |  | || |_) |     /  \    | |_) | \n | |  | ||  _ <     / /\ \   |  _ <  \n | |__| || |_) |   / ____ \ _| |_) | \n |_____(_)____(_) /_/    \_(_)____(_)\n                                     \n")
