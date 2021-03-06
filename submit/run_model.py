import tensorflow as tf
import numpy as np
from skimage import io, filters, transform
from scipy import misc

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

	return tf.nn.softmax(tf.matmul(output, weights[-1]))

def get_probability(files):
	inputD = 6400
	outputD = 104
	classification_probs = np.zeros(shape=(len(files), outputD), dtype=np.float32)

	widths = [inputD, 1600, 400, outputD]
	x = tf.placeholder(tf.float32, [None, inputD])
	probInput = tf.placeholder(tf.float32)
	probHidden = tf.placeholder(tf.float32)
	y = getModel(x, widths, tf.nn.relu, probInput, probHidden)

	saver = tf.train.Saver()

	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state("./model")
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			print 'Model file not found'

		i = 0
		for imfile in files:
			image = io.imread(imfile)
			image = np.invert(image)
			image = filters.gaussian(image, 5)
			image = transform.resize(image, [80, 80])
			image = image.astype(float)
			image = image/np.max(image)
			image = image.flatten()

			pred = sess.run(y, feed_dict={x: [image], probInput: 1.0, probHidden: 1.0})
			classification_probs[i] = pred
			# print np.argmax(pred)
			i += 1

	return classification_probs


if __name__ == '__main__':
	l = []
	for i in xrange(100):
		l.append('../valid/' + str(i) + '.png')
	get_probability(l)