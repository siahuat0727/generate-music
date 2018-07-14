from __future__ import print_function
import numpy as np
import tensorflow as tf
from midi_statematrix import *
from lib import *

class LSTM_model(object):
  def __init__(self, layer_size, batch_size, num_unrolling):
    self.layer_size = layer_size
    self.batch_size = batch_size
    self.num_unrolling = num_unrolling
    self._range = 2 * span
    self._build_lstm_cell()
    self._train_input_label()
    self._unrolled_output()
  
  def _build_lstm_cell(self):
    # Parameters
    self.ifcox = tf.Variable(tf.truncated_normal([self._range, 4 * self.layer_size], -0.1, 0.1))
    self.ifcom = tf.Variable(tf.truncated_normal([self.layer_size, 4 * self.layer_size], -0.1, 0.1))
    self.ifcob = tf.Variable(tf.zeros([1, 4 * self.layer_size]))

    self.saved_output = tf.Variable(tf.zeros([self.batch_size, self.layer_size]), trainable = False)
    self.saved_state= tf.Variable(tf.zeros([self.batch_size, self.layer_size]), trainable = False)

    self.softmax_w = tf.Variable(tf.truncated_normal([self.layer_size, self._range], -0.1, 0.1))
    self.softmax_b = tf.Variable(tf.zeros([self._range]))

  def _lstm_cell(self, i, o, state):
    all_gates_state = tf.matmul(i, self.ifcox) + tf.matmul(o, self.ifcom) + self.ifcob
    input_gate = tf.sigmoid(all_gates_state[:, 0:self.layer_size])
    forget_gate = tf.sigmoid(all_gates_state[:, self.layer_size: 2 * self.layer_size])
    update = all_gates_state[:, 2 * self.layer_size: 3 * self.layer_size]
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(all_gates_state[:, 3 * self.layer_size:])
    return output_gate * tf.tanh(state), state

  def _train_input_label(self):
    self.train_data = [tf.placeholder(tf.float32, 
      shape=[self.batch_size, self._range]) for _ in range(self.num_unrolling+1)]
    self.train_inputs = self.train_data[:-1]
    self.train_labels = self.train_data[1:]

  def _unrolled_output(self):
    outputs = list()
    output = self.saved_output
    state = self.saved_state
    for input_ in self.train_inputs:
      output, state = self._lstm_cell(input_, output, state)
      outputs.append(output)

    with tf.control_dependencies([self.saved_output.assign(output),
                                  self.saved_state.assign(state)]):
      logits = tf.nn.xw_plus_b(
        tf.concat(outputs, 0), self.softmax_w, self.softmax_b)
      self.loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
          logits=logits, labels=tf.concat(self.train_labels, 0)))

    # Optimizer
    global_step = tf.Variable(0)
    self.learning_rate = tf.train.exponential_decay(
      10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(self.loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    self.optimizer = optimizer.apply_gradients(
      zip(gradients, v), global_step=global_step)

    # Predictions
    self._train_prediction = tf.nn.softmax(logits)

    # Sampling and validation eval: batch 1, no unrolling.
    self.sample_input = tf.placeholder(tf.float32, shape=[1, self._range])
    saved_sample_output = tf.Variable(tf.zeros([1, self.layer_size]))
    saved_sample_state = tf.Variable(tf.zeros([1, self.layer_size]))
    self.reset_sample_state = tf.group(
      saved_sample_output.assign(tf.zeros([1, self.layer_size])),
      saved_sample_state.assign(tf.zeros([1, self.layer_size])))
    sample_output, sample_state = self._lstm_cell(
      self.sample_input, saved_sample_output, saved_sample_state)
    with tf.control_dependencies([saved_sample_output.assign(sample_output), 
                                  saved_sample_state.assign(sample_state )]):
      self._sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(
        sample_output, self.softmax_w, self.softmax_b))

  def train(self, train_batches, valid_batches, train_step, summary_frequency=100):
    with tf.Session() as session:
      tf.global_variables_initializer().run()
      mean_loss = 0
      for step in range(train_step + 1):
        batches = train_batches.next() # TODO check  type and correctness
        feed_dict = dict()
        for train_data, batch in zip(self.train_data, batches):
          feed_dict[train_data] = batch
        _, l, predictions, lr = session.run(
            [self.optimizer, self.loss, self._train_prediction, self.learning_rate],
            feed_dict=feed_dict)
        mean_loss += l

        if step % summary_frequency == 0:
          # print mean loss
          if step > 0:
            mean_loss = mean_loss / summary_frequency
          print('Avg loss at step %d: %e learning rate: %f' % (
            step, mean_loss, lr), end='\t')
          mean_loss = 0

          # print perplpexity of minibatch
          labels = np.concatenate(list(batches)[1:]) # TODO check the diff with batches[1:]
          print('Minibatch perplpexity: %e' % float(
            np.exp(logprob(predictions, labels))), end='\t')

          # generate some sample
          if step > 0 and step % (summary_frequency * 10) == 0:
            print('=' * 80)
            for i in range(5):
              feed = np.zeros(shape=(1, self._range), dtype=np.float) # TODO check correctness, try random 
              statematrix = list()
              for _ in range(100): # TODO length of music
                predictions = self._sample_prediction.eval({self.sample_input:feed})
                # feed = self._binarized(predictions) # TODO think: class func. or import
                feed = sample(predictions)
                statematrix.append(input2state(feed)) # TODO think: class func. or import

              noteStateMatrixToMidi(statematrix, name='../sample/sample_%d_%d' % (step, i))
            print('generated 5 sample midi')
            print('=' * 80)

          # Measure validation set perplexity
          self.reset_sample_state.run()
          valid_logprob = 0
          valid_size = 10
          for _ in range(valid_size):
            b = valid_batches.next() 
            predictions = self._sample_prediction.eval({self.sample_input: b[0]})
            valid_logprob = valid_logprob + logprob(predictions, b[1])
          print('Validation set perplexity: %.2f' % float(np.exp(
            valid_logprob / valid_size)))

  def _binarized(self, predictions): # TODO class func. ?
    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1
    return predictions
    # return [0 if p < 0.5 else 1 for p in predictions]










