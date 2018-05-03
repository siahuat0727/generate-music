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
  
  def _build_lstm_cell(self):
    # Parameters
    ifcox = tf.Variable(tf.truncated_normal([self._range, 4 * num_nodes], -0.1, 0.1))
    ifcom = tf.Variable(tf.truncated_normal([num_nodes, 4 * num_nodes], -0.1, 0.1))
    ifcob = tf.Variable(tf.zeros([1, 4 * num_nodes]))

    saved_output = tf.Variable(tf.zero([self.batch_size, num_nodes]), trainable = False)
    saved_state  = tf.Variable(tf.zero([self.batch_size, num_nodes]), trainable = False)

    w = tf.Variable(tf.truncated_normal([num_nodes, self._range], -0.1, 0.1))
    w = tf.Variable(tf.zeros([self._range]))

  def 

  def lstm_cell(self, i, o, state):
    all_gates_state = tf.matmul(i, ifcox) + tf.matmul(o, ifcom) + ifcob
    input_gate = tf.sigmoid(all_gates_state[:, 0:num_nodes])
    forget_gate = tf.sigmoid(all_gates_state[:, num_nodes: 2 * num_nodes])
    update = all_gates_state[:, 2 * num_nodes: 3 * num_nodes]
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(all_gates_state[:, 3 * num_nodes:])
    return output_gate * tf.tanh(state), state

  


    
