#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import sys
import os

from matplotlib import pyplot as plt
from scipy.stats import  pearsonr as pearson

from rnnlm import RNNLM
from utilities import *
import argparse

commandLineParser = argparse.ArgumentParser (description = 'Compute features from labels.')
commandLineParser.add_argument ('--valid_size', type=int, default = 1000,
                                help = 'Specify the validation set size')
commandLineParser.add_argument ('--batch_size', type=int, default = 64,
                                help = 'Specify the training batch size')
commandLineParser.add_argument ('--learning_rate', type=float, default = 1e-0,
                                help = 'Specify the intial learning rate')
commandLineParser.add_argument ('--lr_decay', type=float, default = 0.94,
                                help = 'Specify the learning rate decay rate')
commandLineParser.add_argument ('--dropout', type=float, default = 1.0,
                                help = 'Specify the dropout keep probability')
commandLineParser.add_argument ('--n_epochs', type=int, default = 100,
                                help = 'Specify the number of epoch to run training for')
commandLineParser.add_argument ('--seed', type=int, default = 100,
                                help = 'Specify the global random seed')
commandLineParser.add_argument ('--name', type=str, default = 'model',
                                help = 'Specify the name of the model')
commandLineParser.add_argument ('--debug', type=int, choices= [0, 1, 2], default = 0,
                                help = 'Specify the name of the model')
commandLineParser.add_argument ('--load_path', type=str, default = None,
                                help = 'Specify path to model which should be loaded')

def main(argv=None):
  args = commandLineParser.parse_args()
  
  train_data = process_data_lm('train.dat', 'data', spId=False, input_index='input.wlist.index', output_index='input.wlist.index', bptt=20) 
  #train_data = process_data_lm('train.txt', 'data', spId=True, input_index='input.wlist.index') 
  valid_data = process_data_lm("valid.dat", path="data", spId=False, input_index='input.wlist.index', output_index='input.wlist.index', bptt=None)
  
  
  network_architecture = parse_params('./config')
  
  rnnlm = RNNLM(network_architecture=network_architecture, 
                    seed=args.seed, 
                    name=args.name, 
                    dir='./',
                    load_path=args.load_path,
                    debug_mode=args.debug)
  
  rnnlm.fit(valid_data, 
            train_data, 
            learning_rate=1e-2, 
            lr_decay=0.94, 
            batch_size=64, 
            dropout=args.dropout, 
            optimizer=tf.train.AdamOptimizer, 
            n_epochs=10)
  sys.exit()
  rnnlm.save()
  
if __name__ == '__main__':
  main()