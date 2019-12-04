import os
import sys

from hyperopt import fmin, tpe, hp,Trials, STATUS_OK
from helper_functions import Dataset_Helper
from aliaser import keras
import numpy as np
from models.LSTM import LSTMModel
from models.Dense import DenseModel
from models.Bidirectional import BidirectionalModel
from models.GRU import GRUModel
from models.EmbeddingLSTM import EmbeddingLSTMModel
from results_saver import LogWriter


def resolve_network_type(network_type):
    if network_type == 'lstm':
        return LSTMModel()
    elif network_type == 'gru':
        return GRUModel()
    elif network_type== 'bidi':
        return BidirectionalModel()
    elif network_type == 'dense':
        return DenseModel()
    elif network_type == 'embedding':
        return EmbeddingLSTMModel()


def optimize_model(args):
    model = resolve_network_type(args['network_type'])
    model.set_params(args)
    model.fit_generator(datasets_helper=args['dataset_helper'], tokenizer=args['tokenizer'], validation_count=500)
    results = model.evaluate_generator(datasets_helper=args['dataset_helper'], tokenizer=args['tokenizer'])
    return -np.amax(results.history['val_acc'])

def optimize_lstm(args):
    model = LSTMModel()
    model.set_params(args)
    model.fit_generator(datasets_helper=args['dataset_helper'],tokenizer=args['tokenizer'],validation_count=500)
    results = model.evaluate_generator(datasets_helper=args['dataset_helper'],tokenizer=args['tokenizer'])
    return -np.amax(results.history['val_acc'])

def find_optimal_dense_params():
    pass

def find_optimal_GRU_params():
    pass

def find_optimal_Bidirectional_params():
    pass

def find_optimal_embedding_params():
    pass

def create_base_params(network_type,dataset_helper:Dataset_Helper, tokenizer):
    space = {
        'network_type': network_type,
        'topic_nums': dataset_helper.get_num_of_topics(),
        'tokenizer': tokenizer,
        'dataset_helper': dataset_helper,
        'num_of_words': hp.choice('num_of_words',[5000,10000,15000]),
        'preprocess': False,
        'max_len': 100,
        #'max_len': hp.choice('max_len',[50,100,200]),
        'num_of_layers': hp.randint('num_of_layers',3),#TODO check how much it can handle
        'num_of_neurons': hp.choice('num_of_neurons',[16,32,64,128,256,512]),
        'activation_function': hp.choice('activation_function',['relu','tanh']),
        'dropouts': hp.randint('dropouts',3),
        'dropout_values': hp.uniform('dropout_values',0.01,0.5),
        'epochs': hp.randint('epochs',20),
        'batch_size': hp.choice('batch_size',[64,128,256,512]),
        'optimizer': hp.choice('optimizer',[keras.optimizers.Adam(learning_rate=hp.choice('learning_rate',[0.001,0.0001,0.01,0.0005])), keras.optimizers.RMSprop(hp.choice('learning_rate',[0.001,0.0001,0.01,0.0005])),keras.optimizers.SGD(hp.choice('learning_rate',[0.01,0.001,0.1,0.005]))])
    }
    return space

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


datasets_helper = Dataset_Helper(False)
results_saver = LogWriter(log_file_desc="hyperopt-best-param-search")
results = []
num_of_words = 10000
datasets_helper.set_wanted_datasets([3])
models_to_test = ['lstm','dense','embedding','bidi','gru']
for model in models_to_test:
    while datasets_helper.next_dataset():
        best = fmin(optimize_model,
            space=hp.uniform('x', -10, 10),
            algo=tpe.suggest,
            max_evals=20)
        results_saver.add_log('Best params for network type {} and dataset {} are: {}'.format(model,datasets_helper.get_dataset_name(),best))
        results_saver.write_any('best_params',[model,datasets_helper.get_dataset_name(),best],'a')
        #results_saver.write_2D_list([[model,datasets_helper.get_dataset_name(),best]],'best_params','a')
    datasets_helper.reset_dataset_counter()


"""best_run, best_model = optim.minimize(model=test,
                                          data=[],
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())"""
