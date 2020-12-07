class fonction:
    def  testSamples(numTrials, sampleSize, data):
        Means = [] 
        for t in range(numTrials):
            Y=data.sample(sampleSize)
            Means.append(sum(Y)/len(Y))
        return Means