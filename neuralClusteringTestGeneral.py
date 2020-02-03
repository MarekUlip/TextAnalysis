import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import scipy.optimize
from aliaser import Input, Dense, Model
from aliaser import keras, tf
nmi_func = normalized_mutual_info_score
ari_func = adjusted_rand_score
from sklearn.cluster import KMeans

class ClusteringLayer(keras.layers.Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha= alpha
        self.initial_weights = weights
        self.input_spec = keras.layers.InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = keras.layers.InputSpec(dtype=keras.backend.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (keras.backend.sum(keras.backend.square(keras.backend.expand_dims(inputs, axis=1) - self.clusters), axis=2)/self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = keras.backend.transpose(keras.backend.transpose(q) / keras.backend.sum(q,axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters':self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def count_accuracy(y_true, y_pred):
    """
       Calculate clustering accuracy. Require scikit-learn installed
       # Arguments
           y: true labels, numpy.array with shape `(n_samples,)`
           y_pred: predicted labels, numpy.array with shape `(n_samples,)`
       # Return
           accuracy, in [0,1]
       """
    y_true = y_true.astype(np.int64)
    #y_true = y_true.argmax(1)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    result = 0
    for i in range(len(ind[0])):
        result+=w[ind[0][i],ind[1][i]]
    return result
    #return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T