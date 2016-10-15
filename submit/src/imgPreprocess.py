import os
from skimage import io, filters, transform
import numpy as np
from scipy import misc

def processData(folder):
	count = 0

	os.chdir(folder)
	print 'Starting reading up images from %s'%folder

	for imfile in os.listdir('.'):
		if not imfile[-4:] == '.png':
			continue

		count += 1
		if count % 100 == 0: print count
		image = io.imread(imfile)
		image = np.invert(image)
		image = filters.gaussian(image, 5)
		image = transform.resize(image, [80, 80])
		image = image.astype(float)
		misc.imsave('../resized' + folder + '/' + imfile, image)

	os.chdir('..')
	print 'Images read from %s'%folder

if __name__ == '__main__':
	processData('train')
	processData('valid')