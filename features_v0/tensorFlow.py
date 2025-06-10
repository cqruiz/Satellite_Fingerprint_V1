import tensorflow as tf
import numpy as np

def serialize_example(text, length):
    """
    Creates a tf.train.Example message with text and length.
    """
    # Define feature dictionary for the Example message
    feature = {
        'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text.encode('utf-8')])),
        'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[length]))
    }
    # Create an Example protocol buffer message from the feature dictionary
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize the Example message to a string (binary format)
    return example_proto.SerializeToString()

# Example dataset (text strings and lengths)
dataset = [
    ("Hello, TensorFlow!", 16),
    ("This is a sample text.", 21),
    ("TFRecord format example", 23)
]

# Specify the output TFRecord file name
tfrecord_file = 'text_data.tfrecord'

# Open a TFRecord writer to write serialized examples to the TFRecord file
with tf.io.TFRecordWriter(tfrecord_file) as writer:
    # Iterate over each item in the dataset
    for text, length in dataset:
        # Serialize the current (text, length) pair into a TFRecord Example
        serialized_example = serialize_example(text, length)
        # Write the serialized Example to the TFRecord file
        writer.write(serialized_example)

# Print a success message once the TFRecord file has been created
print(f"TFRecord file '{tfrecord_file}' created successfully.")


def parse_example(serialized_example):
    """
    Parses a single serialized tf.train.Example and extracts features.
    """
    # Define feature description (expected feature keys and types)
    feature_description = {
        'text': tf.io.FixedLenFeature([], tf.string),
        'length': tf.io.FixedLenFeature([], tf.int64)
    }
    # Parse the serialized example using the feature description
    example = tf.io.parse_single_example(serialized_example, feature_description)
    # Decode the text feature from bytes to string
    text = example['text'].numpy().decode('utf-8')
    # Extract the length feature
    length = example['length']
    return text, length

# TFRecord file containing serialized examples
#tfrecord_file = 'text_data.tfrecord'

# Create a TFRecord dataset from the file(s)
dataset = tf.data.TFRecordDataset(tfrecord_file)

# Parse the serialized examples using the parse_example function
parsed_dataset = dataset.map(parse_example)

# Iterate over the parsed dataset to access the extracted features
for text, length in parsed_dataset:
    print(f"Text: '{text}', Length: {length}")
