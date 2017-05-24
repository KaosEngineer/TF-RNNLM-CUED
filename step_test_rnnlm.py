#! /usr/bin/env python

import numpy as np
import sys
import os

from rnnlm import RNNLM
from utilities import *
import argparse

from matplotlib import pyplot as plt
from scipy.stats import  pearsonr as pearson




#from matplotlib import pyplot as plt
from scipy.stats import  pearsonr as pearson
from sklearn.metrics import mean_squared_error as MSE

commandLineParser = argparse.ArgumentParser (description = 'Compute features from labels.')
commandLineParser.add_argument ('--dropout', type=float, default = None,
                                help = 'Specify the MCD keep probability')
commandLineParser.add_argument ('--samples', type=int, default = None,
                                help = 'Specify the numpy of MCD samples')
commandLineParser.add_argument ('--seed', type=int, default = 100,
                                help = 'Specify path to model which should be loaded')
commandLineParser.add_argument ('--name', type=str, default = 'attention_grader',
                                help = 'Specify path to model which should be loaded')
commandLineParser.add_argument ('--load_path', type=str, default = './',
                                help = 'Specify path to model which should be loaded')
commandLineParser.add_argument ('--debug', type=int, default = 0,
                                help = 'Specify path to model which should be loaded')
commandLineParser.add_argument ('--valid', type=bool, default = False,
                                help = 'Specify path to model which should be loaded')

def main(argv=None):
  args = commandLineParser.parse_args()
  if os.path.isdir('CMDs'):
    with open('CMDs/step_test_rnnlm.txt', 'a') as f:
      f.write(' '.join(sys.argv)+'\n')
  else:
    os.mkdir('CMDs')
    with open('CMDs/step_test_rnnlm.txt', 'a') as f:
      f.write(' '.join(sys.argv)+'\n')


  valid_data = process_data_lm("valid.dat", path="data", spId=False, input_index='input.wlist.index', output_index='input.wlist.index', bptt=None)
  
  network_architecture = parse_params('./config')
  
  rnnlm = RNNLM(network_architecture=network_architecture, 
                    seed=args.seed, 
                    name=args.name, 
                    dir='./',
                    load_path=args.load_path,
                    debug_mode=args.debug)
  
  
  rnnlm.predict(valid_data)
    
if __name__ == '__main__':
  main()
