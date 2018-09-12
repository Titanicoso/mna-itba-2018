# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from numericalMethods import *
from sklearn import svm
import configparser
from utils import *
from PIL import Image

config = configparser.ConfigParser()
config.read("./config.ini")

V_SIZE = config.getint("DEFAULT", "V_SIZE")
H_SIZE = config.getint("DEFAULT", "H_SIZE")
IMAGES_PER_PERSON = config.getint("DEFAULT", "IMAGES_PER_PERSON")
NUMBER_OF_PEOPLE = config.getint("DEFAULT", "NUMBER_OF_PEOPLE")
IMAGE_DIR = config.get("DEFAULT", "IMAGE_DIR")

TEST_NUMBER = config.getint("DEFAULT", "TEST_NUMBER")
TRAINING_NUMBER = config.getint("DEFAULT", "TRAINING_NUMBER")
METHOD = config.get("DEFAULT", "METHOD")

QUERY = config.get("DEFAULT", "QUERY")
NUMBER_OF_EIGENFACES = config.getint("DEFAULT", "NUMBER_OF_EIGENFACES")

training = TRAINING_NUMBER * NUMBER_OF_PEOPLE
test = TEST_NUMBER * NUMBER_OF_PEOPLE
trainingImages, testImages, trainingNames, testNames = getImages(IMAGE_DIR, V_SIZE*H_SIZE, IMAGES_PER_PERSON, NUMBER_OF_PEOPLE, TRAINING_NUMBER, TEST_NUMBER)


if METHOD == 'KPCA':
    degree = 2
    K = (np.dot(trainingImages, trainingImages.T) / training + 1) ** degree

    # esta transformación es equivalente a centrar las imágenes originales...
    unoM = np.ones([training, training]) / training
    K = K - np.dot(unoM, K) - np.dot(K, unoM) + np.dot(unoM, np.dot(K, unoM))

    # Autovalores y autovectores
    w, alpha = eigen(K)
    lambdas = w


    for col in range(alpha.shape[1]):
        alpha[:, col] = alpha[:, col] / np.sqrt(lambdas[col])

    # pre-proyección
    improypre = np.dot(K.T, alpha)
    unoML = np.ones([test, training]) / training
    Ktest = (np.dot(testImages, trainingImages.T) / training + 1) ** degree
    Ktest = Ktest - np.dot(unoML, K) - np.dot(Ktest, unoM) + np.dot(unoML, np.dot(K, unoM))
    imtstproypre = np.dot(Ktest, alpha)

    nmax = alpha.shape[1]
    accs = np.zeros([nmax, 1])

else:
    # CARA MEDIA
    meanimage = np.mean(trainingImages, 0)
    fig, axes = plt.subplots(1, 1)
    axes.imshow(np.reshape(meanimage + 127.5, [V_SIZE, H_SIZE]) * 127.5, cmap='gray')
    fig.suptitle('Average Image')
    plt.show()

    # resto la media
    trainingImages = [trainingImages[k, :] - meanimage for k in range(trainingImages.shape[0])]
    testImages = [testImages[k, :] - meanimage for k in range(testImages.shape[0])]

    S, V = rsvAndEigenValues(np.asmatrix(trainingImages))
    nmax = V.shape[0]
    accs = np.zeros([nmax, 1])

if QUERY == 'TEST':
    for neigen in range(1, nmax):
        if METHOD == 'KPCA':
            improy = improypre[:, 0:neigen]
            imtstproy = imtstproypre[:, 0:neigen]
        else:
            B = V[0:neigen, :]
            improy = np.dot(trainingImages, np.transpose(B))
            imtstproy = np.dot(testImages, np.transpose(B))
        # SVM
        # entreno
        clf = svm.LinearSVC()
        clf.fit(improy, trainingNames)
        accs[neigen] = clf.score(imtstproy, testNames)
        print('Precision with {0} eigenfaces: {1} %\n'.format(neigen, accs[neigen] * 100))

    fig, axes = plt.subplots(1, 1)
    axes.semilogy(range(nmax), (1 - accs) * 100)
    axes.set_xlabel('No. eigenfaces')
    axes.grid(which='Both')
    fig.suptitle('Error')
    plt.show()

else:
    path = QUERY
    picture = im.imread(path)
    fig, axes = plt.subplots(1, 1)
    axes.imshow(picture, cmap='gray')
    fig.suptitle('Image to predict')
    plt.show()

    picture = np.reshape((picture - 127.5) / 127.5, [1, H_SIZE * V_SIZE])

    if METHOD == 'KPCA':
        improy = improypre[:, 0:NUMBER_OF_EIGENFACES]
        imtstproy = imtstproypre[:, 0:NUMBER_OF_EIGENFACES]
    else:
        B = V[0:NUMBER_OF_EIGENFACES, :]
        improy = np.dot(trainingImages, np.transpose(B))

    clf = svm.LinearSVC()
    clf.fit(improy, trainingNames)

    if METHOD == 'KPCA':
        unoML = np.ones([1, training]) / training
        Ktest = (np.dot(picture, trainingImages.T) / training + 1) ** degree
        Ktest = Ktest - np.dot(unoML, K) - np.dot(Ktest, unoM) + np.dot(unoML, np.dot(K, unoM))
        imtstproypre = np.dot(Ktest, alpha)
        pictureProy = imtstproypre[:, 0:NUMBER_OF_EIGENFACES]
    else:
        B = V[0:NUMBER_OF_EIGENFACES, :]
        picture -= meanimage
        pictureProy = np.dot(picture, B.T)
    print("Subject is: {} \n".format(clf.predict(pictureProy)[0]))



