#this is the code to read the files
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import glob

# filenames = ['/home/anton/workspace/kaggle/CatsVsDog/data/train/dog.9.jpg']
# Make a queue of file names including all the JPEG images files in the relative
# image directory.

trainpath = "../data/train/*.jpg"
length = len(glob.glob(trainpath))
length=3
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(trainpath))

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()

image = []
label = []

def get_label(filename):
    if 'cat' in filename:
        return 0

    if 'dog' in filename:
        return 1
    return 2

for i in range(length):
    # Read a whole file from the queue, the first returned value in the tuple is the
    # filename which we are ignoring.
    file_name, image_file = image_reader.read(filename_queue)

    # Decode the image as a JPEG file, this will turn it into a Tensor which we can
    # then use in training.
    image.append(tf.image.decode_jpeg(image_file))
    label.append(get_label(file_name.eval()))
# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    image_tensor = sess.run([image[0]])
    image_name = sess.run([file_name])
    print(len(image_tensor))
    print(image_name)

    # Finish off the filename queue coordinator.
    coord.request_stop()
coord.join(threads)
