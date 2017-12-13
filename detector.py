########################################################################################

# Tomas Sykora, 2017                                                                  #
# Optic disc in a retina image detection in TensorFlow                                #

########################################################################################

import os
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave
import time
from scipy import misc, ndimage
import random
from scipy.signal import medfilt
import cv2
import math

class optic_disc_detector:

	def __init__(self, sess=None):
		self.imgs = tf.placeholder(tf.float32, [None, 45, 42, 1])
		self.train_mode = tf.placeholder(tf.bool)
		self.convlayers()
		self.output = self.conv1_6
		self.sess = sess

	def convlayers(self):

		print('\nconvlayers(): Initializing layers')

		self.parameters = []

		# normalize input
		with tf.name_scope('preprocess') as scope:
			images = self.imgs/255 - 0.5

		with tf.name_scope('conv1_1') as scope:
			kernel = tf.get_variable(initializer=tf.keras.initializers.he_normal(), shape=[3, 3, 1, 8], name='weights1_1')
			conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[8], dtype=tf.float32),
								 trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv1_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		self.pool1 = tf.nn.max_pool(self.conv1_1,
							   ksize=[1, 3, 3, 1],
							   strides=[1, 3, 3, 1],
							   padding='SAME',
							   name='pool1')

		self.dropout1 = tf.layers.dropout(self.pool1,
			                         rate=0.2,
			                         training=self.train_mode,
			                         name='dropout1')

		with tf.name_scope('conv1_2') as scope:
			kernel = tf.get_variable(initializer=tf.keras.initializers.he_normal(), shape=[3, 3, 8, 16], name='weights1_2')
			conv = tf.nn.conv2d(self.dropout1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32),
								 trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv1_2 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		self.pool2 = tf.nn.max_pool(self.conv1_2,
							   ksize=[1, 3, 3, 1],
							   strides=[1, 3, 3, 1],
							   padding='SAME',
							   name='pool2')

		self.dropout2 = tf.layers.dropout(self.pool2,
			                         rate=0.2,
			                         training=self.train_mode,
			                         name='dropout2')

		with tf.name_scope('conv1_3') as scope:
			kernel = tf.get_variable(initializer=tf.keras.initializers.he_normal(), shape=[3, 3, 16, 32], name='weights1_3')
			conv = tf.nn.conv2d(self.dropout2, kernel, [1, 1, 1, 1], padding='VALID')
			biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
								 trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv1_3 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		self.pool3 = tf.nn.max_pool(self.conv1_3,
							   ksize=[1, 3, 3, 1],
							   strides=[1, 3, 3, 1],
							   padding='SAME',
							   name='pool3')

		self.dropout3 = tf.layers.dropout(self.pool3,
			                         rate=0.2,
			                         training=self.train_mode,
			                         name='dropout3')

		with tf.name_scope('conv1_4') as scope:
			kernel = tf.get_variable(initializer=tf.keras.initializers.he_normal(), shape=[1, 1, 32, 16], name='weights1_4')
			conv = tf.nn.conv2d(self.dropout3, kernel, [1, 1, 1, 1], padding='VALID')
			biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32),
								 trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv1_4 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		with tf.name_scope('conv1_6') as scope:
			kernel = tf.get_variable(initializer=tf.keras.initializers.he_normal(), shape=[1, 1, 16, 2], name='weights1_6')
			conv = tf.nn.conv2d(self.conv1_4, kernel, [1, 1, 1, 1], padding='VALID')
			biases = tf.Variable(tf.constant(0.0, shape=[2], dtype=tf.float32),
								 trainable=True, name='biases')
			self.conv1_6 = tf.nn.bias_add(conv, biases)
			self.parameters += [kernel, biases]

		self.saver = tf.train.Saver({'W1': self.parameters[0], 'b1': self.parameters[1], 
			                         'W2': self.parameters[2], 'b2': self.parameters[3], 
			                         'W3': self.parameters[4], 'b3': self.parameters[5], 
			                         'W4': self.parameters[6], 'b4': self.parameters[7], 
			                         'W5': self.parameters[8], 'b5': self.parameters[9]})

		print('output: ', np.shape(self.conv1_6))

	def extract_patches(self, image, patchshape, overlap_allowed=0.5, cropvalue=None,
					crop_fraction_allowed=0.1):
		"""
		Given an image, extract patches of a given shape with a certain
		amount of allowed overlap between patches, using a heuristic to
		ensure maximum coverage.
		If cropvalue is specified, it is treated as a flag denoting a pixel
		that has been cropped. Patch will be rejected if it has more than
		crop_fraction_allowed * prod(patchshape) pixels equal to cropvalue.
		Likewise, patches will be rejected for having more overlap_allowed
		fraction of their pixels contained in a patch already selected.
		"""
		jump_cols = int(patchshape[1] * overlap_allowed)
		jump_rows = int(patchshape[0] * overlap_allowed)
		
		# Restrict ourselves to the rectangle containing non-cropped pixels
		if cropvalue is not None:
			rows, cols = np.where(image != cropvalue)
			rows.sort(); cols.sort()
			active =  image[rows[0]:rows[-1], cols[0]:cols[-1]]
		else:
			active = image

		rowstart = 0; colstart = 0

		# Array tracking where we've already taken patches.
		covered = np.zeros(active.shape, dtype=bool)
		patches = []

		while rowstart < active.shape[0] - patchshape[0]:
			# Record whether or not e've found a patch in this row, 
			# so we know whether to skip ahead.
			got_a_patch_this_row = False
			colstart = 0
			while colstart < active.shape[1] - patchshape[1]:
				# Slice tuple indexing the region of our proposed patch
				region = (slice(rowstart, rowstart + patchshape[0]),
						  slice(colstart, colstart + patchshape[1]))
				
				# The actual pixels in that region.
				patch = active[region]

				# The current mask value for that region.
				cover_p = covered[region]
				if cropvalue is None or \
				   frac_eq_to(patch, cropvalue) <= crop_fraction_allowed and \
				   frac_eq_to(cover_p, True) <= overlap_allowed:
					# Accept the patch.
					patches.append(patch)
					
					# Mask the area.
					covered[region] = True
					
					# Jump ahead in the x direction.
					colstart += jump_cols
					got_a_patch_this_row = True
					#print "Got a patch at %d, %d" % (rowstart, colstart)
				else:
					# Otherwise, shift window across by one pixel.
					colstart += 1

			if got_a_patch_this_row:
				# Jump ahead in the y direction.
				rowstart += jump_rows
			else:
				# Otherwise, shift the window down by one pixel.
				rowstart += 1

			# Return a 3D array of the patches with the patch index as the first
			# dimension (so that patch pixels stay contiguous in memory, in a 
			# C-ordered array).

		return np.concatenate([pat[np.newaxis, ...] for pat in patches], axis=0)

	def preprocess(self, images):

		new_images = []

		for i, img in enumerate(images):

			clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(40,40))
			img = clahe.apply(img)

			img = cv2.bilateralFilter(img, -1, 20, 20)

			intensities = medfilt(img, (21, 21))
			intensities = intensities.astype(np.float32)
			intensities_smoothed = cv2.bilateralFilter(intensities, -1, 70, 13)
			width, height = img.shape
			img[0:width, 0:height] = img[0:width, 0:height] + (90) - intensities_smoothed[0:width, 0:height]
			idx = img[:] > 210
			img[idx] = 18

			clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1,1))
			img = clahe.apply(img)

			new_images.append(img)

		return np.array(new_images)

	def eval_test_img(self, sess, idx, img, gt):

		test = img[np.newaxis, ... ]

		pred = sess.run(self.output, feed_dict={
							self.imgs: test, 
							self.train_mode: False
						})
		print('TEST_set[', idx, ']:')
		print('gr_t: ', gt)
		print('pred: ', pred)

	def exctract_disc_patches(self, images, ground_truth):
		""" Exctract patches with zero distance from otic disc. """
		patches = []
		img_h, img_w = images[0].shape
		for i, img in enumerate(images):

			center_x = int(ground_truth[i,1])
			center_y = int(ground_truth[i,0])

			half_win_h = 22
			half_win_w = 21

			x1 = center_x - half_win_w
			y1 = center_y - half_win_h

			if x1 < 0:
				x1 = 0
			if y1 < 0:
				y1 = 0
			if (x1 + 42) >= img_w:
				x1 = img_w - 42
			if (y1 + 45) >= img_h:
				y1 = img_h - 45

			x2 = x1 + 42
			y2 = y1 + 45

			patch = img[y1:y2, x1:x2]

			patches.append(patch)
			#patches.append(ndimage.rotate(patch, 180))

		return np.array(patches)

	def load_data(self, images_folder, gt_file):
		images = []
		for filename in os.listdir(images_folder):
			img = imread(os.path.join(images_folder, filename), mode='L')
			if img is not None:
				images.append(imresize(img, (201, 233)))
		train_images = np.array(images)

		train_output = np.loadtxt(gt_file)

		# delete images without annotations (values (-1, -1))
		indices_to_delete = ~(train_output==-1).any(1)
		train_images = train_images[indices_to_delete]
		train_output = train_output[indices_to_delete]

		# resize output
		train_output = np.rint(train_output / 3)

		return train_images, train_output

	def make_patches(self, images, gt):
		train_patches = []
		patch_dists = []	# ground truth distance of a patch from the optic disc
		print("Creating patches...")
		for img_id, img in enumerate(images):

			# the less overlap allowed, the more patches created
			patches = self.extract_patches(img, (45, 42), overlap_allowed=0.2, cropvalue=None, crop_fraction_allowed=0.1)
			train_patches.extend(patches)

			for patch_id, patch in enumerate(patches):
				patch_x2d = patch_id % 24 # patches in a row
				patch_y2d = patch_id / 24 # patches in a column              
				mid_x = patch_x2d * 8 + 21 # column step + width/2
				mid_y = patch_y2d * 9 + 23 # row step + height/2

				x_offset = gt[img_id, 1] - mid_x
				y_offset = gt[img_id, 0] - mid_y

				offset = np.array([y_offset, x_offset])
				patch_dists.append(offset)

		return np.array(train_patches), np.array(patch_dists)

	def shuffle(self, images, gt):
		shuffle = list(zip(images, gt))
		random.shuffle(shuffle)
		images, gt = zip(*shuffle)

		return np.array(images), np.array(gt)

	def prepare_test_data(self):
		images_folder = './images/test/'
		ground_truth = './images/gt_test.txt'

		train_images, train_output = self.load_data(images_folder, ground_truth)
		train_images = self.preprocess(train_images)

		disc_patches = self.exctract_disc_patches(train_images, train_output)
		zero_dists = np.zeros((len(disc_patches), 2))
		print('Disc patches: ', np.shape(disc_patches))

		train_images, train_output = self.make_patches(train_images, train_output)
		print('Orig patches: ', np.shape(train_images))

		train_output = np.concatenate((train_output, zero_dists), axis=0)
		train_images = np.concatenate((train_images, disc_patches), axis=0)

		# randomly shuffle train array
		train_images, train_output = self.shuffle(train_images, train_output)

		print("Train samples:", np.shape(train_images))
		print("Output samples:", np.shape(train_output))

		# create tensors
		train_output = train_output[:, np.newaxis, np.newaxis, : ]
		train_images = train_images[..., np.newaxis]

		return train_images, train_output

	def train(self, images_folder, ground_truth):

		print('train(): Images processing')

		train_images, train_output = self.load_data(images_folder, ground_truth)
		train_images = self.preprocess(train_images)

		disc_patches = self.exctract_disc_patches(train_images, train_output)
		zero_dists = np.zeros((len(disc_patches), 2))
		print('Disc patches: ', np.shape(disc_patches))

		train_images, train_output = self.make_patches(train_images, train_output)
		print('Orig patches: ', np.shape(train_images))

		train_output = np.concatenate((train_output, zero_dists), axis=0)
		train_images = np.concatenate((train_images, disc_patches), axis=0)

		# randomly shuffle train array
		train_images, train_output = self.shuffle(train_images, train_output)

		print("Train samples:", np.shape(train_images))
		print("Output samples:", np.shape(train_output))

		# create tensors
		train_output = train_output[:, np.newaxis, np.newaxis, : ]
		train_images = train_images[..., np.newaxis]

		train_set = train_images
		train_y = train_output
		print("train_y: ", np.shape(train_y))
		print("train_set: ", np.shape(train_set))

		test_images, test_output = self.prepare_test_data()
		test_set = test_images[:130, ...]
		test_y = test_output[:130, ...]
		print("test_y: ", np.shape(test_y))
		print("test_set: ", np.shape(test_set))

		y = tf.placeholder("float32", [None, 1, 1, 2])

		cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=self.output))
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

		training_epochs = 150
		display_step = 4
		batch_size = 130
		with self.sess as sess:           
			print('train(): Training started')
			sess.run(tf.global_variables_initializer())

			for epoch in range(training_epochs):

				avg_cost = 0.0
				total_batch = int(len(train_set) / batch_size) 
				x_batches = np.array_split(train_set, total_batch)
				y_batches = np.array_split(train_y, total_batch)

				for i in range(total_batch):

					batch_x, batch_y = x_batches[i], y_batches[i]

					_, c = sess.run([optimizer, cost], 
									feed_dict={
										self.imgs: batch_x, 
										y: batch_y, 
										self.train_mode: True
									})
					avg_cost += c / total_batch

				if epoch % display_step == 0:
					print("\nEpoch:", '%04d' % (epoch+1), "\nmse(train_set)=", \
		"{:.9f}".format(avg_cost))

					pred_y = sess.run(self.output, 
									  feed_dict={
										   self.imgs: test_set,
										   self.train_mode: False
									   })
					mse = tf.reduce_mean(tf.square(pred_y - test_y))
					print("MSE(test_set): %.4f" % sess.run(mse)) 

					self.eval_test_img(sess, 90, test_set[90], test_y[90])
					self.eval_test_img(sess, 100, test_set[100], test_y[100])
					self.eval_test_img(sess, 20, test_set[20], test_y[20])

			self.saver.save(sess, './model')

	def fine_tune(self, images_folder, ground_truth):
		""" Fine tuning with patches containing disc patches (400),
		    and with normal patches (1000). This helps the model to
		    converge and stay on the position of otpic disc, if found.
		"""

		train_images, train_output = self.load_data(images_folder, ground_truth)
		train_images = self.preprocess(train_images)

		disc_patches = self.exctract_disc_patches(train_images, train_output)
		zero_dists = np.zeros((len(disc_patches), 2))
		print('Disc patches: ', np.shape(disc_patches))

		train_images, train_output = self.make_patches(train_images, train_output)
		print('Orig patches: ', np.shape(train_images))	

		train_images, train_output = self.shuffle(train_images, train_output)

		print("Train samples:", np.shape(train_images))
		print("Output samples:", np.shape(train_output))

		train_images = train_images[:1000, ...]
		train_output = train_output[:1000, ...]

		train_output = np.concatenate((train_output, zero_dists), axis=0)
		train_images = np.concatenate((train_images, disc_patches), axis=0)

		train_images, train_output = self.shuffle(train_images, train_output)

		# create tensors
		train_y = train_output[:, np.newaxis, np.newaxis, : ]
		train_set = train_images[..., np.newaxis]

		print("train_y: ", np.shape(train_y))
		print("train_set: ", np.shape(train_set))

		y = tf.placeholder("float32", [None, 1, 1, 2])

		cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=self.output))
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

		training_epochs = 50
		display_step = 4
		batch_size = 130

		print('Starting session')
		with self.sess as sess:           
			sess.run(tf.global_variables_initializer())
			print('train(): Training started')
			self.saver.restore(sess, './model')
			print('Model restored')

			for epoch in range(training_epochs):

				avg_cost = 0.0
				total_batch = int(len(train_set) / batch_size) 

				x_batches = np.array_split(train_set, total_batch)
				y_batches = np.array_split(train_y, total_batch)

				for i in range(total_batch):

					batch_x, batch_y = x_batches[i], y_batches[i]

					_, c = sess.run([optimizer, cost], 
									feed_dict={
										self.imgs: batch_x, 
										y: batch_y, 
										self.train_mode: True
									})
					avg_cost += c / total_batch

				if epoch % display_step == 0:
					print("\nEpoch:", '%04d' % (epoch+1), "\nmse(train_set)=", \
		"{:.9f}".format(avg_cost))

			self.saver.save(sess, './model_tuned')

	def detect(self, image_file):

		image_orig = cv2.imread(image_file, 0)
		image_orig = imresize(image_orig, (201, 250))

		""" Preprocessing """
		clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(40,40))
		img = clahe.apply(image_orig)

		img = cv2.bilateralFilter(img, -1, 20, 20)

		intensities = medfilt(img, (21, 21))
		intensities = intensities.astype(np.float32)
		intensities_smoothed = cv2.bilateralFilter(intensities, -1, 70, 13)
		width, height = img.shape
		img[0:width, 0:height] = img[0:width, 0:height] + (90) - intensities_smoothed[0:width, 0:height]
		idx = img[:] > 210
		img[idx] = 18

		clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1,1))
		img = clahe.apply(img)

		img_h, img_w = img.shape
		patch_width = 42
		patch_height = 45
		start_idx = 60
		start_idy = 40

		input_img = img[start_idy:start_idy+patch_height, start_idx:start_idx+patch_width].copy()
		input_img = input_img[np.newaxis, ..., np.newaxis]

		iters = 20
		with self.sess as sess:
			self.saver.restore(sess, './model_tuned')
			print('After displaying the image, press any key to continue. (', iters, ' steps)')
			for i in range(iters):				
				print('Iter: ', i, '/', iters-1)

				pred = sess.run(self.output, 
					            feed_dict={
									      self.imgs: input_img, 
									      self.train_mode: False
								          })

				start_idx = start_idx + int(pred[0,0,0,1])
				start_idy = start_idy + int(pred[0,0,0,0])
				if start_idx < 0:
					start_idx = 0
				if start_idy < 0:
					start_idy = 0
				if (start_idx + patch_width) >= img_w:
					start_idx = img_w - patch_width
				if (start_idy + patch_height) >= img_h:
					start_idy = img_h - patch_height
				end_idx = start_idx + patch_width
				end_idy = start_idy + patch_height

				input_img = img[start_idy:end_idy, start_idx:end_idx].copy()
				input_img = input_img[np.newaxis, ..., np.newaxis]
				
				img_show = image_orig.copy()
				cv2.imshow('detection', cv2.rectangle(img_show,(start_idx,start_idy),(end_idx,end_idy),(0,255,0),1))
				cv2.waitKey(0)

	def eval_model(self, images_folder, gt_file):
		images = []
		for filename in os.listdir(images_folder):
			img = cv2.imread(os.path.join(images_folder, filename), 0)
			if img is not None:
				images.append(imresize(img, (201, 250)))	
		train_images = np.array(images)

		train_images = self.preprocess(train_images)
						
		img_h, img_w = images[0].shape
		print(img_h, img_w)

		with self.sess as sess:
			self.saver.restore(sess, './model_tuned')
			output = []
			patch_width = 42
			patch_height = 45

			images_cnt = len(train_images)
			for img in train_images:
				
				start_idx = 60
				start_idy = 40
				input_img = img[start_idy:start_idy+patch_height, start_idx:start_idx+patch_width]
				input_img = input_img[np.newaxis, ..., np.newaxis]

				iters = 15
				for i in range(iters):				
					pred = sess.run(self.output, 
						            feed_dict={
										      self.imgs: input_img, 
										      self.train_mode: False
									          })

					start_idx = start_idx + int(pred[0,0,0,1])
					start_idy = start_idy + int(pred[0,0,0,0])
					if start_idx < 0:
						start_idx = 0
					if start_idy < 0:
						start_idy = 0
					if (start_idx + patch_width) >= img_w:
						start_idx = img_w - patch_width
					if (start_idy + patch_height) >= img_h:
						start_idy = img_h - patch_height
					end_idx = start_idx + patch_width
					end_idy = start_idy + patch_height

					input_img = img[start_idy:end_idy, start_idx:end_idx]
					input_img = input_img[np.newaxis, ..., np.newaxis]

				x_final = start_idx + 21
				y_final = start_idy + 23

				output.append(np.array([y_final, x_final]))

			output = np.array(output)

			ground_truth = np.loadtxt(gt_file)

			total = 0
			radius_10 = 0
			radius_20 = 0
			for i, x in enumerate(ground_truth):
				d = math.sqrt((x[0]-output[i,0])**2 + ((x[1]-output[i,1])**2))
				total += d
				if d < 10:
					radius_10 += 1 
				if d < 20:
					radius_20 += 1

			total = total / images_cnt
			print('Avg. distance from gt: ', total)
			print('Detections with distance under 10: ', radius_10)
			print('Detections with distance under 20: ', radius_20)


def parseArguments():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='Train mode.', action='store_true')
	parser.add_argument('--detect', help='Single image detection.', action='store_true')
	parser.add_argument('--finetune', help='Fine tuning the trainedmodel.', action='store_true')
	parser.add_argument('--eval', help='Final evaluation on imageret database.', action='store_true')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	sess = tf.Session()

	detector = optic_disc_detector(sess)

	args = parseArguments()

	if args.train is True:
		detector.train('.images/train/', '.images/gt_train.txt')
	elif args.finetune is True:
		detector.fine_tune('./images/train/', './images/gt_train.txt')
	elif args.detect is True:
		detector.detect('./test_image.png')
	elif args.eval is True:
		detector.eval_model('./images', './images/imageret_gt.txt')
	else:
		print('No argument set.')
