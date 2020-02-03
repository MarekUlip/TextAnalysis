import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import scipy.optimize
from aliaser import Input, Dense, Model
from aliaser import keras, tf
nmi_func = normalized_mutual_info_score
ari_func = adjusted_rand_score
from sklearn.cluster import KMeans
from neuralClusteringTestGeneral import *



def autoencoder(dims, act='relu', init='glorot_uniform'):
    n_stacks = len(dims) - 1
    input_text = Input(shape=(dims[0],),name='input')
    x = input_text
    for i in range(n_stacks-1):
        x = Dense(dims[i+1], activation=act, kernel_initializer=init,name='encoder_%d' % i)(x)
    encoded = Dense(dims[-1], kernel_initializer=init, name='endored_%d' % (n_stacks-1))(x)
    x = encoded
    for i in range(n_stacks-1,0,-1):
        x = Dense(dims[i], activation=act,kernel_initializer=init,name='decoder_%d' % i)(x)
    x = Dense(dims[0],kernel_initializer=init,name='decoder_0')(x)
    decoded = x
    return Model(inputs=input_text,outputs = decoded, name='AE'), Model(inputs=input_text, outputs=encoded, name='encoder')

x = np.array()
y = np.array()
dims = [x.shape[-1],500,500,2000,10]
init = keras.initializers.VarianceScaling(scale=1./3., mode='fan_in', distribution='uniform')
pretrain_optimizer = keras.optimizers.SGD(lr=1,momentum=0.9)
pretrain_epochs = 300
batch_size = 256
save_dir = './nn_cluster'
autoencoder, encoder = autoencoder(dims, init=init)
#TODO plot model
n_clusters = 4
autoencoder.compile(optimizer=pretrain_optimizer,loss='mse')
autoencoder.fit(x,x, batch_size=batch_size,epochs=pretrain_epochs)
autoencoder.save_weights(save_dir+'/conv_ae_weights.h5')
clustering_layer = ClusteringLayer(n_clusters,name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)
model.compile(optimizer=keras.optimizers.SGD(0.01,0.9), loss='kld')
kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict((encoder.predict(x)))
y_pred_last = np.copy(y_pred)
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])



loss = 0
index = 0
maxiter = 8000
update_interval = 140
index_array = np.arange(x.shape[0])
tol = 0.001

for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(x, verbose=0)
        p = target_distribution(q)

        y_pred = q.argmax(1)
        if y is not None:
            acc = np.round(count_accuracy(y, y_pred),5)
            nmi = np.round(nmi_func(y, y_pred),5)
            ari = np.round(ari_func(y, y_pred),5)
            loss = np.round(loss, 5)
            print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
    loss = model.train_on_batch(x=x[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
model.save_weights(save_dir + '/conv_DEC_model_final.h5')

q = model.predict(x, verbose=0)
p = target_distribution(q)

y_pred = q.argmax(1)
if y is not None:
    acc = np.round(count_accuracy(y, y_pred), 5)
    nmi = np.round(nmi_func(y, y_pred), 5)
    ari = np.round(ari_func(y, y_pred), 5)
    loss = np.round(loss, 5)
    print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)








