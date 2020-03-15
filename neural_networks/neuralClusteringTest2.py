from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from neural_networks.aliaser import Tokenizer
nmi_func = normalized_mutual_info_score
ari_func = adjusted_rand_score
from neural_networks.neuralClusteringTestGeneral import *
from text_generators.training_text_generator import TrainingTextGenerator
import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt
from dataset_loader.dataset_helper import Dataset_Helper

def autoencoder_bad(dims, act='relu', init='glorot_uniform'):
    n_stacks = len(dims) - 1
    # input
    input_img = Input(shape=(dims[0],), name='input')
    x = input_img
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

    # output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    input_img = Input(shape=(dims[0],), name='input')
    x = input_img
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

    # output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')
num_of_words = 10000
batch_size = 256
dataset_helper = Dataset_Helper(False)
dataset_helper.set_wanted_datasets([2])
while dataset_helper.next_dataset():
    tokenizer = Tokenizer(num_of_words)
    text_generator = TrainingTextGenerator(dataset_helper.get_train_file_path(),batch_size,dataset_helper.get_num_of_train_texts(),num_of_words,tokenizer,';',dataset_helper.get_num_of_topics(), preload_dataset=True, is_predicting=False)
    tokenizer.fit_on_texts(dataset_helper.text_generator())
    dataset = text_generator.get_dataset()
    x = dataset[0]
    y = dataset[1]
    n_clusters = dataset_helper.get_num_of_topics()
    dims = [x.shape[-1], 500, 500, 2000, 10]
    init = keras.initializers.VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')
    pretrain_optimizer = keras.optimizers.SGD(lr=1, momentum=0.9)
    pretrain_epochs = 100
    save_dir = '../results'

    autoencoder, encoder = autoencoder(dims, init=init)
    autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
    autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)  # , callbacks=cb)
    autoencoder.save_weights(save_dir + '/ae_weights.h5')
    autoencoder.load_weights(save_dir+'/ae_weights.h5')
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input,
                outputs=[clustering_layer, autoencoder.output])

    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(encoder.predict(x))
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
    y_pred_last = np.copy(y_pred)

    model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=pretrain_optimizer)
    loss = 0
    index = 0
    maxiter = 8000
    update_interval = 140
    index_array = np.arange(x.shape[0])
    tol = 0.001
    y_true_nums = y.argmax(1)
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q, _  = model.predict(x, verbose=0)
            p = target_distribution(q)  # update the auxiliary target distribution p

            # evaluate the clustering performance
            y_pred = q.argmax(1)
            if y is not None:
                acc = np.round(count_accuracy(y_true_nums, y_pred), 5)
                nmi = np.round(nmi_func(y_true_nums, y_pred), 5)
                ari = np.round(ari_func(y_true_nums, y_pred), 5)
                loss = np.round(loss, 5)
                print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

            # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
        loss = model.train_on_batch(x=x[idx], y=[p[idx], x[idx]])
        index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

    model.save_weights(save_dir + '/b_DEC_model_final.h5')

    model.load_weights(save_dir + '/b_DEC_model_final.h5')

    # Eval.
    q, _ = model.predict(x, verbose=0)
    p = target_distribution(q)  # update the auxiliary target distribution p

    # evaluate the clustering performance
    y_pred = q.argmax(1)
    if y is not None:
        acc = np.round(count_accuracy(y_true_nums, y_pred), 5)
        nmi = np.round(nmi_func(y_true_nums, y_pred), 5)
        ari = np.round(ari_func(y_true_nums, y_pred), 5)
        loss = np.round(loss, 5)
        print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)



    sns.set(font_scale=3)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true_nums, y_pred)

    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    plt.show()

