import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from skimage import io, filters, transform
import os

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
		image = io.imread(imfile)
		image = np.invert(image)
		image = filters.gaussian(image, 5)
		image = transform.resize(image, [80, 80])
		image = image.astype(float)
		image = image/np.max(image)
		image = image.flatten()
		images.append(image)

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

def getWeights(widths):
	l = []
	u = []
	for i in xrange(len(widths) - 1):
		w = tf.Variable(tf.random_normal([widths[i], widths[i+1]], stddev=0.01))
		b = tf.Variable(tf.random_normal([1, widths[i+1]], stddev=0.01))
		l.append(w)
		u.append(b)

	return l, u

def getModel(x, widths, activationFunction, probInput, probHidden):
	weights, biases = getWeights(widths)

	output = tf.nn.dropout(x, probInput)
	for i in xrange(len(weights) - 1):
		h = activationFunction(tf.matmul(output, weights[i]) + biases[i])
		output = tf.nn.dropout(h, probHidden)

	return tf.matmul(output, weights[-1])

def main():
	trainImages, trainLabels, inputD, outputD = readData('train')
	testImages, testLabels, _, _ = readData('valid', inputD, outputD)
	widths = [inputD, 1600, 400, outputD]

	x = tf.placeholder(tf.float32, [None, inputD])
	labels = tf.placeholder(tf.float32, [None, outputD])
	probInput = tf.placeholder(tf.float32)
	probHidden = tf.placeholder(tf.float32)

	y = getModel(x, widths, tf.nn.relu, probInput, probHidden)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, labels))
	trainStep = tf.train.RMSPropOptimizer(0.001).minimize(loss)
	predict = tf.argmax(y, 1)

	# saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		batchSize = 100
		numBatches = trainImages.shape[0] // batchSize
		for i in xrange(100):
			for j in xrange(numBatches):
				xs = trainImages[j*batchSize:(j+1)*batchSize]
				ys = trainLabels[j*batchSize:(j+1)*batchSize]
				sess.run(trainStep, feed_dict={x: xs, labels: ys, probInput: 0.8, probHidden: 0.5})

			predictions = sess.run(predict, feed_dict={x: testImages, labels: testLabels, probInput: 1.0, probHidden: 1.0})
			accuracy = np.mean(np.argmax(testLabels, axis=1) == predictions)
			print 'Epoch:', i+1, 'Accuracy(%):', 100*accuracy

		# save_path = saver.save(sess, "model_file")
		# print 'Model saved at:', save_path

if __name__ == '__main__':
	main()
