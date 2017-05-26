import numpy as np
import os
from scipy.stats import  pearsonr as pearson
from matplotlib import pyplot as plt
font = {'family' : 'normal',
        'size'   : 16}

plt.rc('font', **font)

#plt.rcParams.update({'font.size': 42})
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import auc
from sklearn import gaussian_process
from math import ceil
from sklearn.manifold.t_sne import TSNE
try:
  import cPickle as pickle
except:
  import pickle
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import itertools
import sys

def load_data_local(path, bulats=False, shuffle=False, integer=False, seed=100):
  """
  Helper function used to ALTA dataset into numpy arrays along with targets and metadata (L1, etc...)
  name: Name of dataset along with Acoustic Model and feature generation. Ex: BLXXXgrd01/HC3/F2
  directory: directory where it is located.
  shuffle: If true, pre-shuffle data
  seed: seed for shuffling.
  """

  features_raw = np.loadtxt(path+'/features_L1.dat', dtype=np.float32)
  if bulats:
    targets = np.loadtxt(path+'/targets_bulats.dat', dtype=np.float32)
  else:
    targets = np.loadtxt(path+'/targets.dat', dtype=np.float32)

  """
  with open("Data/sim.csv") as f:
    ncols = len(f.readline().split(','))

data = np.loadtxt("Data/sim.csv", delimiter=',', skiprows=1, usecols=range(1,ncols+1))
"""


  if shuffle:
    data = np.c_[features_raw.reshape(len(features_raw), -1), targets.reshape(len(targets), -1)]
    np.random.seed(seed)
    np.random.shuffle(data)
    features_raw = data[:, :features_raw.size//len(features_raw)].reshape(features_raw.shape)
    targets = data[:, features_raw.size//len(features_raw):].reshape(targets.shape)

  features = features_raw[:,:-1]
  L1_indicator = np.asarray(features_raw[:,-1], dtype=np.int32)

  if integer:
    return np.around(targets[:,-1:]), features, L1_indicator
  else:
    return targets[:,-1:], features, L1_indicator

def load_data(path, use_aux=False, expert=False, shuffle=False, integer=False, seed=100):
  """
  Helper function used to ALTA dataset into numpy arrays along with targets and metadata (L1, etc...)
  name: Name of dataset along with Acoustic Model and feature generation. Ex: BLXXXgrd01/HC3/F2
  directory: directory where it is located.
  shuffle: If true, pre-shuffle data
  seed: seed for shuffling.
  """

  if expert:
    gfile='grades-expert.txt'
  else:
    gfile='grades.txt'
  with open(path+'/features.txt') as f:
    ncols = len(f.readline().split())
    ncols = len(f.readline().split())
    features = np.loadtxt(path+'/features.txt', dtype=np.float32, delimiter=' ', skiprows=1, usecols=range(1,ncols))
    print features.shape
  if use_aux:
    with open(path+'/aux.txt') as f:
      ncols = len(f.readline().split())
      aux = np.loadtxt(path+'/aux.txt', dtype=np.float32,  skiprows=1, usecols=range(1,ncols))
  with open(path+'/'+gfile) as f:
    ncols = len(f.readline().split())
    targets = np.loadtxt(path+'/'+gfile, dtype=np.float32,  skiprows=1, usecols=range(1,ncols))

  if len(targets.shape) < 2:
    targets = targets[:, np.newaxis]
  if shuffle:
    if use_aux:
      data = np.c_[features.reshape(len(features), -1), aux.reshape(len(aux), -1), targets.reshape(len(targets), -1)]
      np.random.seed(seed)
      np.random.shuffle(data)
      print features.size//len(features)
      features = data[:, :features.size//len(features)].reshape(features.shape)
      aux = data[:, features.size//len(features):aux.size//len(aux)+features.size//len(features)].reshape(aux.shape)
      targets = data[:, aux.size//len(aux)+features.size//len(features):].reshape(targets.shape)
    else:
      data = np.c_[features.reshape(len(features), -1), targets.reshape(len(targets), -1)]
      np.random.seed(seed)
      np.random.shuffle(data)
      features = data[:, :features.size//len(features)].reshape(features.shape)
      targets = data[:, features.size//len(features):].reshape(targets.shape)



  if integer:
    targets =  np.around(targets)


  if use_aux:
    return targets[:,-1:], features, np.asarray(aux, dtype=np.int32)
  else:
    return targets[:,-1:], features

def _create_dict(path, index):
        dict = {}
        path = os.path.join(path, index)
        with open(path, 'r') as f:
          for line in f.readlines():
                  line = line.replace('\n', '').split()
                  dict[line[1]]= int(line[0])+1
        return dict

def _word_to_id(data, path, index):
        vocab = _create_dict(path, index)
        return [[vocab[word] if vocab.has_key(word) else 0 for word in line] for line in data]


def process_data_lm(data, path, input_index, output_index, bptt, spId=False):

  data_path = os.path.join(path, data)

  with open(data_path, 'r') as f:
    data = []
    slens=[]
    for line in f.readlines():
      line = line.replace('\n', '').split()
      if spId:
        line = line[1:]
      data.append(line)
      slens.append( len(line) -1 )
  in_data = _word_to_id(data, path, input_index)
  out_data =_word_to_id(data, path, output_index)


  if bptt==None:
    slens = np.asarray(slens, dtype=np.int32)
    input_processed_data = np.zeros((len(slens), np.max(slens)), dtype=np.int32)
    target_processed_data = np.zeros((len(slens), np.max(slens)), dtype=np.int32)

    for i in xrange(len(in_data)):
      input = in_data[i][:-1]
      output = out_data[i][1:]
      input_processed_data[i][0:slens[i]] = input
      target_processed_data[i][0:slens[i]] = output
    return target_processed_data, input_processed_data, slens
  else:
    sequence_lengths = []
    for s in slens:
      if s <= bptt:
        sequence_lengths.append(s)
      else:
        lines = int(np.floor(s/float(bptt)))
        lens = [bptt]*lines
        if len(lens) > 0: sequence_lengths.extend(lens)
        s = s % bptt
        if s > 0:
          sequence_lengths.append(s)
    sequence_lengths = np.asarray(sequence_lengths, dtype=np.int32)
    #print np.mean(sequence_lengths), np.std(sequence_lengths),

    #print sequence_lengths.shape[0], len(id_data)
    input_processed_data = np.zeros((len(sequence_lengths), bptt), dtype=np.int32)
    target_processed_data = np.zeros((len(sequence_lengths), bptt), dtype=np.int32)
    row = 0
    for i, length in zip(xrange(len(in_data)), slens):
      #if length>0:
          #input = in_data[i][:-1]
      #else:
      input = in_data[i]
      output = out_data[i]#[1:]
      lines = int(np.ceil(length/float(bptt)))
      for j in xrange(lines):
        input_processed_data[row+j][0:len(input[j*bptt:(j+1)*bptt])] =input[j*bptt:(j+1)*bptt]
        target_processed_data[row+j][0:len(input[j*bptt:(j+1)*bptt])] = output[j*bptt:(j+1)*bptt]
      row+=lines

    return target_processed_data, input_processed_data, sequence_lengths

def process_data(data, path, spId, input_index):
  data_path = os.path.join(path, data)
  with open(data_path, 'r') as f:
    data = []
    slens=[]
    for line in f.readlines():
            line = line.replace('\n', '').split()
            if spId:
              line = line[1:]
            line = line[1:-1] #strip off sentence start and sentence end
            if len(line) == 0:
              pass
            else:
              data.append(line)
              slens.append( len(line) )
  data = _word_to_id(data, path, input_index)

  slens = np.asarray(slens, dtype=np.int32)
  print np.mean(slens), np.std(slens), np.max(slens)

  processed_data = np.zeros((len(slens), np.max(slens)), dtype=np.int32)

  for i, length in zip(xrange(len(data)), slens):
    processed_data[i][0:slens[i]] = data[i]

  return processed_data, slens, np.int32(np.max(slens))



def process_data_bucket(data, path, spId, input_index):
  data_path = os.path.join(path, data)
  with open(data_path, 'r') as f:
    data = []
    slens=[]
    for line in f.readlines():
            line = line.replace('\n', '').split()
            if spId:
              line = line[1:]
            data.append(line)
            slens.append( len(line) )
  data = _word_to_id(data, path, input_index)

  slens = np.array(slens, dtype=np.int32)
  print np.mean(slens), np.std(slens), np.max(slens)

  processed_data = []

  for dat in data:
    processed_data.append(np.asarray(dat, dtype=np.int32))

  return processed_data, slens, np.int32(np.max(slens)-1)

def scatter(targets, preds, name=None, dir=None):
  """ Function to create a scatter plot of ALTA grades. """
  # Scatter Plots
  plt.plot([y for y in targets], [y for y in preds], 'r^')
  plt.xlim(0, 6)
  plt.ylim(0, 6)
  plt.ylabel('Automatic Score')
  plt.xlabel('Expert Score')
  #plt.show()
  plt.savefig(os.path.join(dir, 'scatter_'+name+'.png'), bbox_inches='tight')
  plt.close()


  plt.hexbin([y for y in targets], [y for y in preds], gridsize=6, cmap=plt.cm.Blues)
  #cb = plt.colorbar()
  plt.xlim(0, 6)
  plt.ylim(0, 6)
  plt.ylabel('Automatic Score')
  plt.xlabel('Expert Score')
  plt.savefig(os.path.join(dir, 'scatter_density_'+name+'.png'), bbox_inches='tight')
  plt.close()

def interpolate(targets, model_1, model_2, dir, name, mse_plot=False):
  """ Function to create an interpolation plot of two models.
  targets: Targets which both models are compared with
  model_1: Predictions from model 1
  model_2: Predictions from model 2
  dir: Directory where to save to.
  name: name of chart
  """

  #Interpolation
  correlations = []
  MSEs = []
  for i in xrange(100):
    interp = (100.0-float(i))/100.0*model_1 + model_2*(float(i))/100.0
    p = pearson(interp, targets)
    mse = MSE(interp, targets)
    correlations.append([float(i)/100.0, p[0]])
    MSEs.append([float(i)/100.0, mse])

  print np.max(np.asarray(correlations)[:, 1])
  print np.min(np.asarray(MSEs)[:, 1])

  plt.plot([i[0] for i in correlations], [i[1] for i in correlations])
  plt.xlabel('DNN Fraction')
  plt.ylabel('Pearson Correlation')
  plt.savefig(os.path.join(dir,'interpolation_pearson_'+name+'.png'))
  plt.close()

  if mse_plot:
    plt.plot([i[0] for i in MSEs], [i[1] for i in MSEs])
    plt.xlabel('DNN Fraction')
    plt.ylabel('MSE')
    plt.savefig(os.path.join(dir,'interpolation_mse_'+name+'.png'))
    plt.close()

def reject(y, y_exp, var, plot=False, L1=None, name='Rejection.png'):

  error = (y-y_exp)**2
  P_0 = pearson(y, y_exp)[0][0]
  if L1 is None:
    array = np.concatenate((y, y_exp, error, var), axis=1)
  else:
    L1[196:]=40
    array = np.concatenate((y, y_exp, error, var, L1[:, np.newaxis]), axis=1)
  sorted_array = array[array[:,2].argsort()]
  results=[[0.0, P_0]]
  results_var=[[0.0, P_0]]
  results_min = [[0.0, P_0]]
  for i in xrange(1, array.shape[0]):
    x = np.concatenate((sorted_array[:-i,0], sorted_array[-i:,1]), axis=0)
    p = pearson(x, sorted_array[:, 1])[0]
    results.append([float(i)/float(array.shape[0]), p])
    results_min.append([float(i)/float(array.shape[0]), P_0 + (1.0-P_0)*float(i)/float(array.shape[0])])
  if L1 is not None:
    L1_best = sorted_array[:, 4]
  tpr = []
  sorted_array = array[array[:,3].argsort()]
  for i in xrange(1, array.shape[0]):
    x = np.concatenate((sorted_array[:-i,0], sorted_array[-i:,1]), axis=0)
    p = pearson(x, sorted_array[:, 1])[0]
    if ( float(i)/float(array.shape[0]) <= 0.100001 ) and (float(i)/float(array.shape[0]) >= 0.090009):
      tpr.append(p)
    results_var.append([float(i)/float(array.shape[0]), p])
  if L1 is not None:
    L1_var = sorted_array[:, 4]

  max_auc = auc([x[0] for x in results], [x[1] - P_0 for x in results], reorder=True)
  var_auc = auc([x[0] for x in results_var], [x[1] - P_0 for x in results_var], reorder=True)
  min_auc = auc([x[0] for x in results_min], [x[1] - P_0 for x in results_min], reorder=True)


  if plot:
    plt.scatter([x[0] for x in results], [x for x in np.asarray(sorted(var, reverse=True))])
    plt.xlim(0.0, 1.0)
    plt.savefig('Variance.png', bbox_inches='tight')
    plt.close()
    if L1 is not None:
      plt.scatter([x[0] for x in results], [x[1] for x in results],  c=L1_best, cmap=plt.cm.winter)
      plt.scatter([x[0] for x in results_var], [x[1] for x in results_var], c=L1_var, cmap=plt.cm.winter)
      plt.scatter([x[0] for x in results_var], [x[1] for x in results_min], c=L1_var, cmap=plt.cm.winter)
    else:
      plt.plot([x[0] for x in results], [x[1] for x in results],  'b^',
               [x[0] for x in results_var], [x[1] for x in results_var], 'ro',
               [x[0] for x in results_var], [x[1] for x in results_min], 'go')
    plt.legend(['Optimal-Rejection', 'Model-Rejection', 'Expected Random-Rejection'],loc=4, prop={'size':18.5})
    plt.xlim(0.0, 1.0)
    plt.ylim(0.86, 1.0)
    plt.xlabel('Rejection Fraction')
    plt.ylabel('Pearson Correlation')
    #plt.show()
    plt.savefig(name, bbox_inches='tight')
    plt.close()

    print 'AUC', auc([x[0] for x in results_var], [x[1] for x in results_var], reorder=True)
  return var_auc/(1.0-P_0), max_auc/(1.0-P_0), min_auc/(1.0-P_0), (var_auc-min_auc)/(max_auc-min_auc), np.mean(tpr)


def reject_fill(y, y_exp, var, plot=False, L1=None, name='Rejection.png'):

  error = (y-y_exp)**2
  P_0 = pearson(y, y_exp)[0][0]
  if L1 is None:
    array = np.concatenate((y, y_exp, error, var), axis=1)
  else:
    L1[196:]=40
    array = np.concatenate((y, y_exp, error, var, L1[:, np.newaxis]), axis=1)
  sorted_array = array[array[:,2].argsort()]
  results=[[0.0, P_0]]
  results_var=[[0.0, P_0]]
  results_min = [[0.0, P_0]]
  for i in xrange(1, array.shape[0]):
    x = np.concatenate((sorted_array[:-i,0], sorted_array[-i:,1]), axis=0)
    p = pearson(x, sorted_array[:, 1])[0]
    results.append([float(i)/float(array.shape[0]), p])
    results_min.append([float(i)/float(array.shape[0]), P_0 + (1.0-P_0)*float(i)/float(array.shape[0])])
  if L1 is not None:
    L1_best = sorted_array[:, 4]

  sorted_array = array[array[:,3].argsort()]
  for i in xrange(1, array.shape[0]):
    x = np.concatenate((sorted_array[:-i,0], sorted_array[-i:,1]), axis=0)
    p = pearson(x, sorted_array[:, 1])[0]
    results_var.append([float(i)/float(array.shape[0]), p])
  if L1 is not None:
    L1_var = sorted_array[:, 4]

  max_auc = auc([x[0] for x in results], [x[1] - P_0 for x in results], reorder=True)
  var_auc = auc([x[0] for x in results_var], [x[1] - P_0 for x in results_var], reorder=True)
  min_auc = auc([x[0] for x in results_min], [x[1] - P_0 for x in results_min], reorder=True)

  if plot:
    fig, ax = plt.subplots()
    plt.fill_between([x[0] for x in results], np.zeros(224), P_0*np.ones(224), alpha=0.01, color='g')
    plt.fill_between([x[0] for x in results], P_0*np.ones(224), [x[1] for x in results_min], alpha=0.07, color='g')
    plt.fill_between([x[0] for x in results_var], [x[1] for x in results_min], [x[1] for x in results_var], alpha=0.5, color='r')
    plt.fill_between([x[0] for x in results_var], [x[1] for x in results_var], [x[1] for x in results], alpha=0.5, color='b')
    #plt.legend([r'AUC $\rho$', 'AUC Random', 'AUC Variance', 'AUC Maximum'],loc=4 )
    plt.xlim(0.0, 1.0)
    plt.ylim(0.86, 1.0)
    ypoints = [0.86, P_0, 0.897, 1.0]
    xpoints = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.yticks(ypoints, ['0.0' , 'PCC', '10% Rej.\nPCC', '1.0'], fontsize=17)
    plt.yticks(ypoints, fontsize=17)
    plt.xticks(xpoints, ['0.0', '0.1', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=17)
    plt.xlabel('Rejection Fraction', fontsize=17)
    plt.ylabel('Pearson Correlation', fontsize=17)
    #plt.show()
    plt.savefig('auc_diagramm.png', bbox_inches='tight')
    plt.close()


def tsne_embedding(trn_data, trn_labels=None, test_data=None, seed=100):

    mean= np.mean(trn_data, axis=0)
    std = np.sqrt(1e-8 + np.var(trn_data, axis=0))

    trn_data = (trn_data - mean)/std
    model = TSNE(n_components=2, init='random', random_state=seed)
    projection= model.fit_transform(trn_data)
    #projection = scale(projection, axis=0, with_mean=True, with_std=True, copy=True )

    trn_labels=None
    if trn_labels:
      labels = np.arange(0.0, 113.0)
      #print trn_labels
      #plt.scatter([x[0] for x in projection], [x[1] for x in projection], c=labels)
      fig, ax = plt.subplots(nrows=1, ncols=1)
      for x_,y_,label in zip([x[0] for x in projection], [x[1] for x in projection],labels):
        ax.scatter([x_], [y_], c=label)
      ax.legend([lab for lab in trn_labels], loc=4)
      for point in ax.collections:
        point.set_cmap(plt.cm.winter)
        point.set_color(0.5)
      fig.show()
    else:
      plt.scatter([x[0] for x in projection], [x[1] for x in projection])
    plt.show()
    plt.close()

def gp_grade(trn_data, test_data, L1_FEATURES=False, seed=100):
  gp = gaussian_process.GaussianProcess ( theta0=1e-2,
                                          thetaL=1e-4,
                                          thetaU=1e-1,
                                          nugget = 0.2)

  def map(i):
    if i ==107:
      j = 0
    elif i == 24:
      j = 1
    elif i == 32:
      j = 2
    elif i == 4:
      j = 3
    elif i == 73:
      j = 4
    elif i == 88:
      j = 5
    elif i == 98:
      j = 7
    return j

  #L1=trn_data[2]
  #test_L1 = test_data[2]
  #L1_one_hot = np.zeros((L1.shape[0], 113), dtype=np.float32)
  #test_L1_one_hot  = np.zeros((test_L1.shape[0], 113), dtype=np.float32)

  #for i in L1:
  #  L1_one_hot[int(i)]=1.0


  #for i in test_L1:
  #  test_L1_one_hot[int(i)] = 1.0

  if L1_FEATURES:
    features = np.concatenate((trn_data[1], L1_one_hot), axis=1)
    print features.shape
    test_features = np.concatenate((test_data[1], test_L1_one_hot), axis=1)
    gp.fit(features, trn_data[0])
    predictions, variances = gp.predict (test_features, eval_MSE = True)
  else:
    gp.fit(trn_data[1], trn_data[0])
    predictions, variances = gp.predict (test_data[1], eval_MSE = True)


  variances = variances[:, np.newaxis]
  return predictions, variances

def parse_params(path):
  param_list = []
  with open(path, 'rb') as f:
    for line in f.readlines():
      line = line.replace('\n', '').replace(':', '').split()
      if line[0] == 'n_in' or line[0] == 'n_hid' or line[0] == 'n_hidr' or line[0] == 'n_out' or line[0] == 'n_z' or line[0] == 'n_layers' or line[0] == 'n_L1' or line[0] == 'n_not_tied':
        param_list.append((line[0], int(line[1])))
      elif line[0] == 'L2':
        param_list.append((line[0], float(line[1])))
      elif line[0] == 'activation_fn':
        if line[1] == 'tanh':
          fn = tf.nn.tanh
        elif line[1] == 'sigmoid':
          fn = tf.nn.sigmoid
        elif line[1] == 'relu':
          fn = tf.nn.relu
        elif line[1] == 'elu':
          fn = tf.nn.elu
        else:
          print 'Incorrect option. Exiting'
          sys.exit()
        param_list.append((line[0], fn))
      elif line[0] == 'L1' or line[0] == 'BN' or line[0] == 'RNN_A' or line[0] == 'ACU' or line[0] == 'RNN_Q' or line[0] == 'BIDI':
        if line[1] == 'True':
          param_list.append((line[0], True))
        elif line[1] == 'False':
          param_list.append((line[0], False))
        else:
          print 'Incorrect option. Exiting'
          sys.exit()
      elif line[0] == 'intializer':
        if line[1] == 'xavier':
          init = tf.contrib.layers.xavier_initializer
        else:
          print 'Incorrect option. Exiting'
          sys.exit()
      elif line[0] == 'model_type'  or line[0] == 'attention' or line[0] == 'focus':
        param_list.append((line[0], line[1]))
      else:
        print 'Incorrect option. Exiting'
        sys.exit()
  return param_list

def plot_confusion_matrix(targets, preds, classes, name=None, dir=None):
    labels = []
    for i in xrange(classes):
      labels.append(str(i))
    cm = confusion_matrix(targets, preds)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #cm = np.flipud(cm)
    np.set_printoptions(precision=2)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(classes)
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


    cm = np.around(cm,2)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.savefig(os.path.join(dir, 'confusion_matrix_'+name+'.png'), bbox_inches='tight')
    plt.close()
