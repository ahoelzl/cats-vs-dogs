import tensorflow as tf
import glob as glob
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
 
def get_label(filename):
    print(filename)
    if 'cat' in filename:
        return 0

    if 'dog' in filename:
        return 1
    return 2


def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    trainpath = "../data/train/*.jpg" 
    filenames = glob.glob(trainpath)
    #print("filenames: ",filenames.__class__.__name__)
    labels = list(map( get_label, filenames))
    return filenames, labels


def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    print(label)
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    print(example)
    return example, label

image_list, label_list = read_labeled_image_list("../data/train/*.jpg")


images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)

num_epochs = 1

# Makes an input queue
input_queue = tf.train.slice_input_producer([images, labels],
                                            num_epochs=num_epochs,
                                            shuffle=True)



image, label = read_images_from_disk(input_queue)

print(image)
print(label)

sess = tf.Session()

# Optional Image and Label Batching
image_batch, label_batch = tf.train.batch([image, label],
                                          batch_size=10)



init_op = tf.global_variables_initializer()
sess.run(init_op)
sess.run(image_batch)
sess.close()
