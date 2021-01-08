#!/usr/bin/python3
class fonction:                                     #Creation of the function class for all functions that can be used for temp.py
    def testSamples(numTrials, sampleSize, data):  #Définition de la fonction testSamples prenant en paramètre numTrials (int), sampleSize (int), data (tableau de float)
        Means = []                                  #Definition of the testSamples function taking in parameters numTrials (int), sampleSize (int), data (float array)
        for t in range(numTrials):                  #Loop for starting from 0 going to numTrials-1 in steps of 
            Y=data.sample(sampleSize)               #Y retrieves a random sample of elements of size sampleSize
            Means.append(sum(Y)/len(Y))             #Addition in the table of the division of the sum of Y divided by the length of Y
        return Means                                #We return the Means table
