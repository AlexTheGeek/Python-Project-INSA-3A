#!/usr/bin/python3
class fonction:                                     #Creation de la classe fonction pour toutes les fonctions utilisables pour temp.py
    def  testSamples(numTrials, sampleSize, data):  #Definition de la fonction testsamples prenant en parametre numTrials (int), sampleSize (int), data (tableau de float)
        Means = []                                  #Creation d'un tableau vide
        for t in range(numTrials):                  #Boucle for commençant de 0 allant a numTrials-1 par pas de 1
            Y=data.sample(sampleSize)               #Y recupere un echantillon aleatoire d'elements a partir de SampleSize
            Means.append(sum(Y)/len(Y))             #Ajout dans le tabelau de la division de la somme de Y divise par la longuer de Y
        return Means                                #On retourne le tableau Means