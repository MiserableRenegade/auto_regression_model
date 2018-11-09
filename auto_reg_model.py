from __future__ import print_function
import os
import sys
import numpy
import tensorflow as tf
from auto_reg_input import  *
import matplotlib.pyplot as plt

tf.app.flags.DEFINE_integer('training_iteration', 100,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '~/models/', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

def main(_):
    if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
    	print('Usage: mnist_export.py [--training_iteration=x] '
    	  '[--model_version=y] export_dir')
    	sys.exit(-1)
    if FLAGS.training_iteration <= 0:
    	print('Please specify a positive value for training iteration.')
    	sys.exit(-1)
    if FLAGS.model_version <= 0:
    	print('Please specify a positive value for version number.')
    	sys.exit(-1)
    # Hyperparameters
    learning_rate = 1e-3
    display_step = 1
    batch_size = 30
    hidden_layer_1 = 1
    # Construct Model
    sess = tf.InteractiveSession()
    x, train_x,test_x = input_data() # take data from linear_input_data, linear_input_data.input_data()
    train_X = numpy.asarray(train_x)
    test_X = numpy.asarray(test_x)
    train_Y = numpy.asarray(train_X)
    n_samples = train_X.shape[1]
    # tf Graph Input
    X = tf.placeholder(dtype='float32',shape=(None,n_samples),name="first_placeholder")
    Y = tf.placeholder(dtype='float32',shape=(None,1),name="second_placeholder")
    # Set model weights
    W = tf.Variable(tf.random_normal((n_samples,hidden_layer_1),stddev=0.01,dtype='float32'),name="weights")
    b = tf.Variable(tf.zeros((1,hidden_layer_1),dtype='float32'))
    # Construct a linear model
    prediction = tf.add(tf.matmul(X,W),tf.reduce_sum(b*W))
    # Gradient descent
    # Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
    loss = tf.reduce_sum(tf.square(prediction-Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    iteration = int(len(train_X)/batch_size)
    init = tf.global_variables_initializer()
    sess.run(tf.global_variables_initializer())
    for step in range(FLAGS.training_iteration-1):
        for epoch in range(iteration):
            batch_x1 = train_X[epoch:batch_size+epoch:]
            batch_y1 = train_X[epoch+1:batch_size+epoch+1:]
            sess.run(optimizer, feed_dict={X: batch_x1, Y: batch_y1})
            if (epoch+1) % display_step == 0:
                    cosst = sess.run(loss, feed_dict={X: batch_x1, Y:batch_y1}) # show information about cost, weights, bias every 50 steps
        training_cost = sess.run(loss, feed_dict={X: batch_x1, Y: batch_y1})
        print("Epoch=", step,"Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
        predict = sess.run(prediction, feed_dict={X: batch_x1})
    # Test our model
    test_batch_x1 = test_X[:batch_size+1,:]
    test_batch_x = test_batch_x1.reshape(-1,1)
    test_batch_y1 = test_X[:batch_size+1,:]
    test_batch_y = test_batch_y1.reshape(-1,1)
    predict_test = sess.run(prediction, feed_dict={X: test_batch_x})
    print("prediction:",predict_test,"Real data:",test_batch_x)
    plt.plot(predict_test,color='green') # Predicted line
    plt.ylabel('Sin')
    plt.plot(test_batch_y,color='blue') # Test line
    plt.show()
    # Path to save model
    export_path_base = sys.argv[-1]
    export_path = os.path.join(
    tf.compat.as_bytes(export_path_base),
    tf.compat.as_bytes(str(1)))
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    # Build the signature_def_map.
    regression_inputs = tf.saved_model.utils.build_tensor_info(X) # Save first_placeholder to take prediction
    regression_outputs_prediction = tf.saved_model.utils.build_tensor_info(prediction) # Save predcition function
    regression_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
    inputs={
        tf.saved_model.signature_constants.REGRESS_INPUTS:regression_inputs
    },
    outputs={
        tf.saved_model.signature_constants.REGRESS_OUTPUTS:regression_outputs_prediction,
    },
    method_name=tf.saved_model.signature_constants.REGRESS_METHOD_NAME
    ))

    tensor_info_x = tf.saved_model.utils.build_tensor_info(X) # Save first_placeholder to take prediction
    tensor_info_y = tf.saved_model.utils.build_tensor_info(prediction) # Save cost function

    prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
    inputs={'input_value':tensor_info_x},
    outputs={'output_value':tensor_info_y},
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        'predict_value':
            prediction_signature,
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            regression_signature,
    },
    legacy_init_op = legacy_init_op)

    builder.save()
    print("Done exporting!")

if __name__ == '__main__':
    tf.app.run()
