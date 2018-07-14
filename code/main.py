from multi_training import loadPieces
from mixed_generator import MixedGenarator
from batch_generator import BatchGenerator # TODO del
from model import LSTM_model
from lib import * # TODO del

layer_size = 100
batch_size = 10
num_unrolling = 10
num_valid = 1
valid_batch_size = 1
train_batch_size = 10

def main():
  # load midi
  dirpath = '../'
  pieces = loadPieces(dirpath)

  # divide train valid
  valid_pieces = pieces[:num_valid]
  train_pieces = pieces[num_valid:]
  valid_gen = BatchGenerator(pieces[0], valid_batch_size, 1)
  train_gen = MixedGenarator(pieces, batch_size, num_unrolling)
  
  # create model ans start training
  model = LSTM_model(layer_size, batch_size, num_unrolling)
  model.train(train_gen, valid_gen, train_step=10000, summary_frequency=100)

if __name__ == '__main__':
  main()
