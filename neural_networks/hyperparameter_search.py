import os
import sys

from hyperopt import fmin, tpe, hp, space_eval
from dataset_loader.dataset_helper import Dataset_Helper
from neural_networks.aliaser import keras, Tokenizer
import numpy as np
from neural_networks.models.LSTM import LSTMModel
from neural_networks.models.Dense import DenseModel
from neural_networks.models.Bidirectional import BidirectionalModel
from neural_networks.models.GRU import GRUModel
from neural_networks.models.EmbeddingLSTM import EmbeddingLSTMModel
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

def get_important_params_from_args(accuracy,args):
    params = {
        'accuracy': accuracy,
        'dataset_num': args['dataset_num'],
        'network_type': args['network_type'],
        'num_of_words': 10000,
        #'preprocess': False,
        'max_len':args['max_len'],
        'num_of_layers': args['num_of_layers'],
        'num_of_neurons': args['num_of_neurons'],
        'activation_function': args['activation_function'],
        'dropouts': args['dropouts'],
        'dropout_values': args['dropout_values'],
        'batch_size': args['batch_size'],
        'learning_rate': args['learning_rate'],
        'optimizer': args['optimizer']
    }
    return params

def optimize_model(args):
    print(args)
    datasets_helper = Dataset_Helper(False)
    datasets_helper.set_wanted_datasets([args['dataset_num']])
    datasets_helper.next_dataset()
    tokenizer = Tokenizer(num_words=args['num_of_words'])
    generator = datasets_helper.text_generator()
    tokenizer.fit_on_texts(generator)
    optimizer = create_optimizer(args['optimizer'],args['learning_rate'])
    model = resolve_network_type(args['network_type'])
    model.set_params(args)
    model.optimizer = optimizer
    if args['network_type'] == 'embedding':
        model.tokenizer = tokenizer
    model.compile_model()
    model.fit(datasets_helper=datasets_helper, tokenizer=tokenizer, validation_count=500)
    results = model.evaluate(datasets_helper=datasets_helper, tokenizer=tokenizer)
    print(results)
    args['results_saver'].write_any('logs',[get_important_params_from_args(results[1],args)],'a')
    del model
    del tokenizer
    del generator
    del datasets_helper
    tf.compat.v2.keras.backend.clear_session()
    return -np.amax(results[1])

def optimize_lstm(args):
    model = LSTMModel()
    model.set_params(args)
    model.fit(datasets_helper=args['dataset_helper'], tokenizer=args['tokenizer'], validation_count=500)
    results = model.evaluate(datasets_helper=args['dataset_helper'], tokenizer=args['tokenizer'])
    return -np.amax(results.history['val_acc'])

def create_optimizer(name, learn_rate):
    if name == 'adam':
        return keras.optimizers.Adam(learning_rate=learn_rate)
    elif name == 'rmsprop':
        return keras.optimizers.RMSprop(learning_rate=learn_rate)

def create_base_params(network_type,dataset_helper:Dataset_Helper,results_saver):
    if network_type == 'embedding':
        batch_size = hp.choice('batch_size',[64,128])
        num_of_layers = hp.choice('num_of_layers',[1,2])
        num_of_neurons = hp.choice('num_of_neurons',[32,64,128])
    else:
        batch_size = hp.choice('batch_size',[64,128,256])
        num_of_layers = hp.choice('num_of_layers',[1,2,3,4])
        num_of_neurons = hp.choice('num_of_neurons',[32,64,128,256])
    space = {
        'dataset_num': datasets_helper.dataset_position,
        'network_type': network_type,
        'topic_nums': dataset_helper.get_num_of_topics(),
        'num_of_words': hp.choice('num_of_words',[10000]),
        #'preprocess': False,
        'max_len': hp.choice('max_len',[100,200,300]),
        'num_of_layers': num_of_layers,
        'num_of_neurons': num_of_neurons,
        'activation_function': hp.choice('activation_function',['relu','tanh']),
        'dropouts': hp.randint('dropouts',3),
        'dropout_values': hp.uniform('dropout_values',0.01,0.2),
        'epochs': 20,#hp.randint('epochs',20),
        'batch_size': batch_size,
        'learning_rate': hp.choice('learning_rate',[0.001,0.01,0.0005]),
        'optimizer': hp.choice('optimizer',['adam', 'rmsprop']),
        'results_saver': results_saver
    }
    return space

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


datasets_helper = Dataset_Helper(False)
results_saver = LogWriter(log_file_desc="hyperopt-best-param-search")
results = []
datasets_helper.set_wanted_datasets([1])
models_to_test = ['lstm','dense','embedding','bidi']
"""datasets_helper.next_dataset()
space = create_base_params('lstm',datasets_helper)
smpl = sample(space)
print(sample(space))"""
for model in models_to_test:
    while datasets_helper.next_dataset():
        space = create_base_params(model,datasets_helper,results_saver)
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
