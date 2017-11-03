
# coding: utf-8

import math
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from IPython.display import clear_output, Image, display, HTML

###### Do not modify here ######
def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = graph_def
    #strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
###### Do not modify  here ######

###### Implement Data Preprocess here ######
housing = fetch_california_housing()
print("Shape of dataset:", housing.data.shape)
print("Shape of label:", housing.target.shape)

housing_data = housing.data
housing_target = housing.target

# normalize data

#housing_data_tf = tf.constant(housing_data)
#housing_data_mean = tf.reduce_mean(housing_data_tf, 0)
#housing_data_mean_diff = tf.subtract(housing_data_tf, housing_data_mean)
#housing_data_s = tf.reduce_sum(tf.square(housing_data_mean), 0)
#housing_data_s = tf.sqrt(
#                        tf.div(housing_data_s, (housing_data.shape[0] - 1)))
#housing_data_t_statistic = tf.div(housing_data_mean_diff, housing_data_s)
#housing_data = tf.Session().run(housing_data_t_statistic)
#
#np.set_printoptions(linewidth=200, edgeitems=10)
##print(tf.Session().run(housing_data_mean_diff))
##quit()

# set training set count
training_set_count = math.floor(housing.data.shape[0]*0.9)

# add bias
training_set_data = np.concatenate(
        (housing_data[:training_set_count], np.ones((training_set_count, 1), dtype=np.int)),
        axis=1
        )

testing_set_data = np.concatenate(
        (housing_data[training_set_count:], np.ones((housing.data.shape[0] - training_set_count, 1), dtype=np.int)),
        axis=1
        )
training_set_target = housing_target[:training_set_count]
testing_set_target = housing_target[training_set_count:]

print("Training set")
print("Shape of dataset:", training_set_data.shape)
print("Shape of label:", training_set_target.shape)
print("Testing set")
print("Shape of dataset:", testing_set_data.shape)
print("Shape of label:", testing_set_target.shape)
x_train = tf.constant(training_set_data)
y_train = tf.constant(training_set_target, shape=[training_set_count,1])
x_test = tf.constant(testing_set_data)
y_test = tf.constant(testing_set_target, shape=[housing.data.shape[0] - training_set_count, 1])


###### Implement Data Preprocess here ######

###### Start TF session ######
with tf.Session() as sess:
    w = tf.matrix_inverse(tf.matmul(x_train, x_train, transpose_a=True))
    w = tf.matmul(w, x_train, transpose_b=True)
    w = tf.matmul(w, y_train)
#    print("Weight:", sess.run(w))
###### Calculate error rate ######
    y_hat = tf.matmul(x_test, w)
    y_error_rate = tf.reduce_mean(tf.abs(tf.div(tf.subtract(y_hat, y_test), y_test)))
    print("Error rate:", sess.run(y_error_rate))
###### Calculate error rate ######
    show_graph(tf.get_default_graph().as_graph_def())
###### Start TF session ######

###### Graph ######

# accroding to linear regression formula, we can compute weight using feature and label. (MatMul -> MatrixInverse -> MatMul[1-2])
# after weight is comupted, we can apply weight to linear regression formula to get predicted result. (MatMul[3])
# also, we using tensorflow API to compute error rate. (Sub -> div -> Abs -> Mean)

