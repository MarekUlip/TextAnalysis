import os
import sys

from hyperopt import fmin, tpe, hp,Trials, STATUS_OK,space_eval
from hyperopt.pyll.stochastic import sample
from helper_functions import Dataset_Helper
from aliaser import keras, Tokenizer
import numpy as np
from models.LSTM import LSTMModel
from models.Dense import DenseModel
from models.Bidirectional import BidirectionalModel
from models.GRU import GRUModel
from models.EmbeddingLSTM import EmbeddingLSTMModel
from results_saver import LogWriter
import tensorflow as tf


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
    tokenizer = Tokenizer(num_words=args['num_of_words'])
    # filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n',
    # lower=False, split=' ')
    generator = args['dataset_helper'].text_generator()
    tokenizer.fit_on_texts(generator)
    args['optimizer'] = create_optimizer(args['optimizer'],args['learning_rate'])
    model = resolve_network_type(args['network_type'])
    model.set_params(args)
    if args['network_type'] == 'embedding':
        model.tokenizer = tokenizer
    model.compile_model()
    model.fit_generator(datasets_helper=args['dataset_helper'], tokenizer=tokenizer, validation_count=500)
    results = model.evaluate_generator(datasets_helper=args['dataset_helper'], tokenizer=tokenizer)
    print(results)
    del model
    del tokenizer
    del generator
    tf.compat.v1.keras.backend.clear_session()
    return -np.amax(results[1])

def optimize_lstm(args):
    model = LSTMModel()
    model.set_params(args)
    model.fit_generator(datasets_helper=args['dataset_helper'],tokenizer=args['tokenizer'],validation_count=500)
    results = model.evaluate_generator(datasets_helper=args['dataset_helper'],tokenizer=args['tokenizer'])
    return -np.amax(results.history['val_acc'])

def create_optimizer(name, learn_rate):
    if name == 'adam':
        return keras.optimizers.Adam(learning_rate=learn_rate)
    elif name == 'rmsprop':
        return keras.optimizers.RMSprop(learning_rate=learn_rate)

def find_optimal_dense_params():
    pass

def find_optimal_GRU_params():
    pass

def find_optimal_Bidirectional_params():
    pass

def find_optimal_embedding_params():
    pass

def create_base_params(network_type,dataset_helper:Dataset_Helper):
    if network_type == 'embedding':
        batch_size = hp.choice('batch_size',[128])
        num_of_layers = hp.choice('num_of_layers',[1,2])
    else:
        batch_size = hp.choice('batch_size',[64,128,256])
        num_of_layers = hp.choice('num_of_layers',[1,2,3,4])
    space = {
        'network_type': hp.choice('network_type',[network_type]),
        'topic_nums': hp.choice('topic_nums',[dataset_helper.get_num_of_topics()]),
        #'tokenizer': tokenizer,
        'dataset_helper': hp.choice('dataset_helper', [dataset_helper]),
        'num_of_words': hp.choice('num_of_words',[5000,10000,12500]),
        #'preprocess': False,
        'max_len': hp.choice('max_len',[100,200,300]),
        'num_of_layers': num_of_layers,#TODO check how much it can handle
        'num_of_neurons': hp.choice('num_of_neurons',[32,64,128,256]),
        'activation_function': hp.choice('activation_function',['relu','tanh']),
        'dropouts': hp.randint('dropouts',3),
        'dropout_values': hp.uniform('dropout_values',0.01,0.5),
        'epochs': hp.choice('epochs',[20]),#hp.randint('epochs',20),
        'batch_size': batch_size,
        'learning_rate': hp.choice('learning_rate',[0.001,0.0001,0.01,0.0005]),
        'optimizer': hp.choice('optimizer',['adam', 'rmsprop'])
    }
    return space

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


datasets_helper = Dataset_Helper(False)
results_saver = LogWriter(log_file_desc="hyperopt-best-param-search")
results = []
datasets_helper.set_wanted_datasets([3])
models_to_test = ['lstm','dense','embedding','bidi','gru']
"""datasets_helper.next_dataset()
space = create_base_params('lstm',datasets_helper)
smpl = sample(space)
print(sample(space))"""
for model in models_to_test:
    while datasets_helper.next_dataset():
        space = create_base_params(model,datasets_helper)
        best = fmin(optimize_model,
            space=space,
            algo=tpe.suggest,
            max_evals=30,
            max_queue_len=1,
            verbose=False)
        results_saver.add_log('Best params for network type {} and dataset {} are: {}\n{}'.format(model,datasets_helper.get_dataset_name(),best,space_eval(space,best)))
        results_saver.write_any('best_params',[model,datasets_helper.get_dataset_name(),space_eval(space,best)],'a')
        #results_saver.write_2D_list([[model,datasets_helper.get_dataset_name(),best]],'best_params','a')
    datasets_helper.reset_dataset_counter()


"""best_run, best_model = optim.minimize(model=test,
                                          data=[],
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())"""
