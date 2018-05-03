from __future__ import print_function
import numpy as np
import tensorflow as tf

lower_bound = 24
upper_bound = 102
span = upper_bound - lower_bound


class LSTM_model(object):
  def __init__(self, layer_size, batch_size, num_unrolling):
    self.layer_size = layer_size
    self.batch_size = batch_size
    self.num_unrolling = num_unrolling
    self._range = 2 * span
    self._build_lstm_cell()
    self._train_input_label()
  
  def _build_lstm_cell(self):
    # Parameters
    self.ifcox = tf.Variable(tf.truncated_normal([self._range, 4 * num_nodes], -0.1, 0.1))
    self.ifcom = tf.Variable(tf.truncated_normal([num_nodes, 4 * num_nodes], -0.1, 0.1))
    self.ifcob = tf.Variable(tf.zeros([1, 4 * num_nodes]))

    self.saved_output = tf.Variable(tf.zero([self.batch_size, num_nodes]), trainable = False)
    self.saved_state= tf.Variable(tf.zero([self.batch_size, num_nodes]), trainable = False)

    self.softmax_w = tf.Variable(tf.truncated_normal([num_nodes, self._range], -0.1, 0.1))
    self.softmax_b = tf.Variable(tf.zeros([self._range]))

  def _lstm_cell(self, i, o, state):
    all_gates_state = tf.matmul(i, self.ifcox) + tf.matmul(o, self.ifcom) + self.ifcob
    input_gate = tf.sigmoid(all_gates_state[:, 0:num_nodes])
    forget_gate = tf.sigmoid(all_gates_state[:, num_nodes: 2 * num_nodes])
    update = all_gates_state[:, 2 * num_nodes: 3 * num_nodes]
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(all_gates_state[:, 3 * num_nodes:])
    return output_gate * tf.tanh(state), state

  def _train_input_label(self):
    self.train_data = [tf.placeholder(tf.float32, shape=[batch_size, self._range])
      for _ in range(self.num_unrolling+1)]
    self.train_inputs = train_data[:-1]
    self.train_labels = train_data[1:]

  def _unrolled_output(self):
    self.outputs = list()
    output = self.saved_output
    state = self.saved_state
    for input_ in self.train_inputs:
      output, state = self._lstm_cell(input_, output, state)
      outputs.append(output)

    with tf.control_dependencies([self.saved_output.assign(output),
                                  self.saved_state.assign(state)]):
      logits = tf.nn.xw_plus_b(
          tf.concat(0, outputs), self.softmax_w, self.softmax_b)
      loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_woth_logits(
            logits, tf.concat(0, self.train_labels)

    # Optimizer
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
      10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(
      zip(gradients, v), global_step=global_step)

    # Predictions
    self._train_prediction = tf.nn.softmax(logits)

    # Sampling and validation eval: batch 1, no unrolling.
    sample_input = tf.placeholder(tf.float32, shape=[1, self._range])
    saved_sample_output = tf.Variable(tf.zeros([1, self.num_nodes]))
    saved_sample_state = tf.Variable(tf.zeros([1, self.num_nodes]))
    self.reset_sample_state = tf.group(
      saved_sample_output.assign(tf.zeros([1, self.num_nodes])),
      saved_sample_state.assign(tf.zeros([1, self.num_nodes])))
    sample_output, sample_state = self._lstm_cel(
      sample_input, saved_sample_output, saved_sample_state)
    with tf.control_dependencies([saved_output.assign(sample_output),
                                  saved_state.assign (sample_state )]):
      self._sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))


    







  


    
