from os import listdir
from os.path import join, isdir
from scipy import ndimage as im
import numpy as np


def getImages(directory, area, imagesPerPerson, numberOfPeople, trainingNo, testNo):
    subjects = [f for f in listdir(directory) if isdir(join(directory, f))]

    training = np.zeros([trainingNo * numberOfPeople, area])
    test = np.zeros([testNo * numberOfPeople, area])

    namesTest = []
    namesTraining = []

    trainingImage = 0
    testImage = 0
    personImage = 0
    subjectNumber = 0
    for subject in subjects:
        for k in range(1, imagesPerPerson + 1):
            a = im.imread(directory + '/' + subject + '/{}'.format(k) + '.pgm')
            if personImage < trainingNo:
                training[trainingImage, :] = (np.reshape(a, [1, area])-127.5) / 127.5
                namesTraining.append(str(subject))
                trainingImage += 1
            else:
                test[testImage, :] = (np.reshape(a, [1, area])-127.5) / 127.5
                namesTest.append(str(subject))
                testImage += 1
            personImage += 1
        subjectNumber += 1
        if subjectNumber > numberOfPeople - 1:
            break
        personImage = 0

    return training, test, namesTraining, namesTest
