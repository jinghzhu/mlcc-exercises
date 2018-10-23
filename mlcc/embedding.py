# 1. Convert movie-review string data to a sparse feature vector.
# 2. Implement a sentiment-analysis linear model using a sparse feature vector.
# 3. Implement a sentiment-analysis DNN model using an embedding that projects data into two dimensions.
# 4. Visualize the embedding to see what the model has learned about the relationships between words.

from __future__ import print_function
import io
import tensorflow as tf


"""
Step 1: Setup
"""
tf.logging.set_verbosity(tf.logging.ERROR)
train_url = 'https://download.mlcc.google.com/mledu-datasets/sparse-data-embedding/train.tfrecord'
train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
test_url = 'https://download.mlcc.google.com/mledu-datasets/sparse-data-embedding/test.tfrecord'
test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)


"""
Step 2: Building Input Pipeline
"""


def _parse_function(record):
    """Extracts features and labels.

    Args:
      record: File path to a TFRecord file
    Returns:
      A `tuple` `(labels, features)`:
        features: A dict of tensors representing the features
        labels: A tensor with the corresponding labels.
    """
    features = {
        "terms": tf.VarLenFeature(dtype=tf.string),  # terms are strings of varying lengths
        "labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32)  # labels are 0 or 1
    }

    parsed_features = tf.parse_single_example(record, features)

    terms = parsed_features['terms'].values
    labels = parsed_features['labels']

    return {'terms': terms}, labels


# Construct a TFRecordDataset for training data, and map data to features and labels using the function above.
ds = tf.data.TFRecordDataset(train_path)  # Create the Dataset object.
ds = ds.map(_parse_function)  # Map features and labels with the parse function.
ds

# Retrieve the first example from the training data set.
n = ds.make_one_shot_iterator().get_next()
sess = tf.Session()
sess.run(n)


# Create an input_fn that parses the tf.Examples from the given files, and split them into features and targets.
def _input_fn(input_filenames, num_epochs=None, shuffle=True):
    # Same code as above; create a dataset and map features and labels.
    ds = tf.data.TFRecordDataset(input_filenames)
    ds = ds.map(_parse_function)

    if shuffle:
        ds = ds.shuffle(10000)

    # Feature data is variable-length, so pad and batch each field of dataset structure to whatever size is necessary.
    ds = ds.padded_batch(25, ds.output_shapes)

    ds = ds.repeat(num_epochs)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels


"""
Step 3: Use a Linear Model with Sparse Inputs and an Explicit Vocabulary
"""
# 50 informative terms that compose model vocabulary.
informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
                     "excellent", "poor", "boring", "awful", "terrible",
                     "definitely", "perfect", "liked", "worse", "waste",
                     "entertaining", "loved", "unfortunately", "amazing",
                     "enjoyed", "favorite", "horrible", "brilliant", "highly",
                     "simple", "annoying", "today", "hilarious", "enjoyable",
                     "dull", "fantastic", "poorly", "fails", "disappointing",
                     "disappointment", "not", "him", "her", "good", "time",
                     "?", ".", "!", "movie", "film", "action", "comedy",
                     "drama", "family")
terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
    key="terms",
    vocabulary_list=informative_terms
)

# Construct LinearClassifier, train it on training set, and evaluate it on evaluation set.
my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
feature_columns = [ terms_feature_column ]

classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns, optimizer=my_optimizer,)
classifier.train(input_fn=lambda: _input_fn([train_path]), steps=1000)

evaluation_metrics = classifier.evaluate(input_fn=lambda: _input_fn([train_path]), steps=1000)
print("Training set metrics:")
for m in evaluation_metrics:
    print(m, evaluation_metrics[m])
print("---")

evaluation_metrics = classifier.evaluate(input_fn=lambda: _input_fn([test_path]), steps=1000)

print("Test set metrics:")
for m in evaluation_metrics:
    print(m, evaluation_metrics[m])
print("---")


"""
Step 4: Use DNN Model
"""
##################### Here's what changed #####################################
classifier = tf.estimator.DNNClassifier(                                      #
  feature_columns=[tf.feature_column.indicator_column(terms_feature_column)], #
  hidden_units=[20,20],                                                       #
  optimizer=my_optimizer,                                                     #
)                                                                             #
###############################################################################

try:
    classifier.train(input_fn=lambda: _input_fn([train_path]), steps=1000)
    evaluation_metrics = classifier.evaluate(input_fn=lambda: _input_fn([train_path]), steps=1)
    print("Training set metrics:")
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
    print("---")

    evaluation_metrics = classifier.evaluate(input_fn=lambda: _input_fn([test_path]), steps=1)

    print("Test set metrics:")
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
    print("---")
except ValueError as err:
    print(err)


"""
Step 5: Use Embedding with DNN Model
"""
######################## EMBEDDING CODE HERE ####################################
terms_embedding_column = tf.feature_column.embedding_column(terms_feature_column, dimension=2)
feature_columns = [ terms_embedding_column ]

my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

classifier = tf.estimator.DNNClassifier(
  feature_columns=feature_columns,
  hidden_units=[20,20],
  optimizer=my_optimizer
)
#################################################################################

classifier.train(input_fn=lambda: _input_fn([train_path]), steps=1000)
evaluation_metrics = classifier.evaluate(input_fn=lambda: _input_fn([train_path]), steps=1000)
print("Training set metrics:")
for m in evaluation_metrics:
    print(m, evaluation_metrics[m])
print("---")

evaluation_metrics = classifier.evaluate(input_fn=lambda: _input_fn([test_path]), steps=1000)
print("Test set metrics:")
for m in evaluation_metrics:
    print(m, evaluation_metrics[m])
print("---")


"""
Step 6: Validate Embedding
"""
# The above model used an embedding_column. But this doesn't tell us much about what's going on internally.
# How can we check that the model is actually using an embedding inside?
# To start, let's look at the tensors in the model:
classifier.get_variable_names()

# Is the embedding layer the correct shape?
# From the output below, we can find the embedding is a matrix to project a 50-dimensional vector down to 2 dimensions.
classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights').shape


"""
Step 7: Examine Embedding
"""
# Do the following:
# 1. Run code to see the embedding trained.
# 2. Re-train the model by rerunning the code before, and then run the embedding visualization below again.
# 3. Finally, re-train the model again using only 10 steps (which will yield a terrible model). Run the embedding
# visualization below again.
import numpy as np
import matplotlib.pyplot as plt

embedding_matrix = classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights')

for term_index in range(len(informative_terms)):
  # Create a one-hot encoding for our term. It has 0s everywhere, except for
  # a single 1 in the coordinate that corresponds to that term.
  term_vector = np.zeros(len(informative_terms))
  term_vector[term_index] = 1
  # Now project that one-hot vector into the embedding space.
  embedding_xy = np.matmul(term_vector, embedding_matrix)
  plt.text(embedding_xy[0], embedding_xy[1], informative_terms[term_index])

# Do a little setup to make sure the plot displays nicely.
plt.rcParams["figure.figsize"] = (15, 15)
plt.xlim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
plt.ylim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
plt.show()


"""
Step 8: Improve Model Performance
"""
# 1. Changing hyperparameters, or using a different optimizer like Adam.
# 2. Adding additional terms to informative_terms.
# Download the vocabulary file.
terms_url = 'https://download.mlcc.google.com/mledu-datasets/sparse-data-embedding/terms.txt'
terms_path = tf.keras.utils.get_file(terms_url.split('/')[-1], terms_url)

# Create a feature column from "terms", using a full vocabulary file.
informative_terms = None
with io.open(terms_path, 'r', encoding='utf8') as f:
    informative_terms = list(set(f.read().split()))  # Convert it to a set first to remove duplicates.

terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="terms",
                                                                                 vocabulary_list=informative_terms)
terms_embedding_column = tf.feature_column.embedding_column(terms_feature_column, dimension=2)
feature_columns = [terms_embedding_column]

my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 10],
    optimizer=my_optimizer
)
classifier.train(input_fn=lambda: _input_fn([train_path]), steps=1000)

evaluation_metrics = classifier.evaluate(input_fn=lambda: _input_fn([train_path]), steps=1000)
print("Training set metrics:")
for m in evaluation_metrics:
    print(m, evaluation_metrics[m])
print("---")

evaluation_metrics = classifier.evaluate(input_fn=lambda: _input_fn([test_path]), steps=1000)

print("Test set metrics:")
for m in evaluation_metrics:
    print(m, evaluation_metrics[m])
print("---")