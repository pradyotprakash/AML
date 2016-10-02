import tensorflow as tf
from scipy import misc
import os
import numpy as np

def ffnn(inp, numLayers, weights, biases, activationFunction):
    layer = inp

    for i in xrange(1, numLayers):
        wi = 'w' + str(i)
        bi = 'b' + str(i)

        p = tf.matmul(layer, weights[wi])
        layer = activationFunction(tf.add(p, biases[bi]))

    return layer


def createParams(numLayers, widths):
    weights = {}
    biases = {}

    for i in xrange(numLayers - 1):
        wi = 'w' + str(i+1)
        bi = 'b' + str(i+1)

        weights[wi] = tf.Variable(tf.random_normal([widths[i], widths[i+1]]))
        biases[bi] = tf.Variable(tf.random_normal([widths[i+1]]))

    return weights, biases


def readData(folder, inputD=None, outputD=None):
    count = 0
    images = []

    os.chdir(folder)
    print 'Starting reading up images from %s'%folder
    l = []
    for imfile in os.listdir('.'):
        if imfile[-4:] == '.png':
            l.append(imfile)
    l.sort(key=lambda x: int(x[:-4]))

    for imfile in l:
        count += 1

        if count % 100 == 0: print count
        image = misc.imread(imfile)
        images.append(image.flatten())

    images = np.asarray(images, dtype=np.float32)

    labelsTemp = []
    classes = set()
    with open('labels.txt') as f:
        for row in f:
            cl = int(row)
            classes.add(cl)
            labelsTemp.append(cl)

    if inputD is None:
        inputD = images.shape[1]
        outputD = len(classes)

    labels = np.zeros(shape=(images.shape[0], outputD), dtype=np.float32)
    for i in xrange(images.shape[0]):
        labels[i, labelsTemp[i]] = 1.0

    os.chdir('..')
    print 'Images read from %s'%folder

    del labelsTemp, classes

    return images, labels, inputD, outputD


def main(inputD, outputD, numLayers, widths, activationFunction, lossFunction, learningRate):
    weights, biases = createParams(numLayers, widths)

    x = tf.placeholder(tf.float32, [None, inputD])
    y = ffnn(x, numLayers, weights, biases, activationFunction)
    yOriginal = tf.placeholder(tf.float32, [None, outputD])

    loss = tf.reduce_mean(-tf.reduce_sum(lossFunction(yOriginal, y), reduction_indices=[1]))
    trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    batchSize = 100
    batches = trainImages.shape[0] // batchSize
    for epoch in xrange(1):
        print 'Epoch: %s'%epoch
        randIndices = np.random.choice(trainImages.shape[0], batchSize)
        xs = trainImages[randIndices]
        ys = trainLabels[randIndices]
        sess.run(trainStep, feed_dict={x: xs, yOriginal: ys})

    correctPrediction = tf.equal(tf.argmax(y, 1), tf.argmax(yOriginal, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

    ret =  100.0*sess.run(accuracy, feed_dict={x: trainImages, yOriginal: trainLabels})
    sess.close()

    return ret

if __name__ == '__main__':
    trainImages, trainLabels, inputD, outputD = readData('sampleTrain')
    testImages, testLabels, _, _ = readData('sampleValid', inputD, outputD)

    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # trainImages, trainLabels, inputD, outputD = mnist.train.images, mnist.train.labels, mnist.train.images.shape[1], mnist.train.labels.shape[1]
    # testImages, testLabels = mnist.test.images, mnist.test.labels

    crossEntropy = lambda yOrig, y: yOrig * tf.log(y + 1e-20)
    l2Loss = lambda yOrig, y: tf.nn.l2_loss(yOrig - y)    

    widths = [inputD, outputD]
    numLayers = len(widths)
    activationFunction = tf.nn.softmax # tf.tanh
    lossFunction = crossEntropy
    learningRate = 0.01

    print main(inputD, outputD, numLayers, widths, activationFunction, lossFunction, learningRate)