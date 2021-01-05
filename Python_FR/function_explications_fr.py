#!/usr/bin/python3
class fonction:                                     #Création de la classe fonction pour toutes les fonctions utilisables pour temp.py
    def  testSamples(numTrials, sampleSize, data):  #Définition de la fonction testsamples prenant en paramètre numTrials (int), sampleSize (int), data (tableau de float)
        Means = []                                  #Création d'un tableau vide
        for t in range(numTrials):                  #Boucle for commençant de 0 allant a numTrials-1 par pas de 1
            Y=data.sample(sampleSize)               #Y récupère un échantillon aléatoire d'éléments de taille SampleSize
            Means.append(sum(Y)/len(Y))             #Ajout dans le tabelau de la division de la somme de Y divisé par la longuer de Y
        return Means                                #On retourne le tableau Means
