
import tensorflow as tf
import pandas as pd
import os
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(__file__)

# Comment this to run for 2nd file
rel_file_path = "biomechanical/column_2C_weka.csv"

# Uncomment this to run for 2nd file
# rel_file_path = "biomechanical/column_3C_weka.csv"

# Random forest parameters
num_steps = 500
batch_size = 1
num_classes = 2  # 3 for 2nd dataset
num_features = 6
num_trees = 500
max_nodes = 5000

dataframe = pd.read_csv(os.path.join(script_dir, rel_file_path))

# Change classes to 0 and 1
# Comment this to run for the 2nd file
dataframe['class'] = dataframe['class'].replace({'Normal': 0, 'Abnormal': 1})

# Uncomment this to run for 2nd file
# dataframe['class'] = dataframe['class'].replace({
#     'Normal': 0,
#     'Hernia': 1,
#     'Spondylolisthesis': 2})

dataframe.info()

# Selecting only the columns that contain numbers without their classes
x_d = dataframe.loc[:, dataframe.columns != 'class'].values.astype('float')
# Selectiong only the classes column
y_d = dataframe.loc[:, 'class'].values.astype('int')

# Initializing tensroflow placeholders for the weights
X = tf.placeholder(tf.float32, shape=[None, num_features])
Y = tf.placeholder(tf.int32, shape=[None])

# Splitting the dataset for training and testing
x_train, x_test, y_train, y_test = train_test_split(x_d, y_d,
                                                    test_size=0.3,
                                                    random_state=1)

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

forest_graph = tensor_forest.RandomForestGraphs(hparams)

train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_vars = tf.group(tf.global_variables_initializer(),
                     resources.initialize_resources(resources.shared_resources()))

# Training and testing
with tf.Session() as sess:
    sess.run(init_vars)

    for i in range(1, num_steps + 1):
        _, l = sess.run([train_op, loss_op], feed_dict={X: x_train, Y: y_train})
        if i % 50 == 0 or i == 1:
            acc = sess.run(accuracy_op, feed_dict={X: x_train, Y: y_train})
            print('Step {}, Loss: {}, Acc: {}'.format(i, l, acc))

    print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: x_test, Y: y_test}))
