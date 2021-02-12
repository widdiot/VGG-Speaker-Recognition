from __future__ import print_function
from __future__ import absolute_import
import keras
import tensorflow as tf
import keras.backend as K

import backbone
weight_decay = 1e-4


class ModelMGPU(keras.Model):
    def __init__(self, ser_model, gpus):
        pmodel = keras.utils.multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


class VladPooling(keras.engine.Layer):
    '''
    This layer follows the NetVlad, GhostVlad
    '''
    def __init__(self, mode, k_centers, g_centers=0, **kwargs):
        self.k_centers = k_centers
        self.g_centers = g_centers
        self.mode = mode
        super(VladPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cluster = self.add_weight(shape=[self.k_centers+self.g_centers, input_shape[0][-1]],
                                       name='centers',
                                       initializer='orthogonal')
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape
        return (input_shape[0][0], self.k_centers*input_shape[0][-1])

    def call(self, x):
        # feat : bz x W x H x D, cluster_score: bz X W x H x clusters.
        feat, cluster_score = x
        num_features = feat.shape[-1]

        # softmax normalization to get soft-assignment.
        # A : bz x W x H x clusters
        max_cluster_score = K.max(cluster_score, -1, keepdims=True)
        exp_cluster_score = K.exp(cluster_score - max_cluster_score)
        A = exp_cluster_score / K.sum(exp_cluster_score, axis=-1, keepdims = True)

        # Now, need to compute the residual, self.cluster: clusters x D
        A = K.expand_dims(A, -1)    # A : bz x W x H x clusters x 1
        feat_broadcast = K.expand_dims(feat, -2)    # feat_broadcast : bz x W x H x 1 x D
        feat_res = feat_broadcast - self.cluster    # feat_res : bz x W x H x clusters x D
        weighted_res = tf.multiply(A, feat_res)     # weighted_res : bz x W x H x clusters x D
        cluster_res = K.sum(weighted_res, [1, 2])

        if self.mode == 'gvlad':
            cluster_res = cluster_res[:, :self.k_centers, :]

        cluster_l2 = K.l2_normalize(cluster_res, -1)
        outputs = K.reshape(cluster_l2, [-1, int(self.k_centers) * int(num_features)])
        return outputs


def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)

def tuplemax_loss(y_true, y_pred):
    """
    pairwise tuplemax loss.
    https://arxiv.org/pdf/1811.12290.pdf
    Eq. (2)
    """
    print(y_true.shape)
    print(y_pred.shape)
    false_class_logits = tf.gather(params=y_pred,indices=tf.where(tf.math.equal(y_true,0)))
    true_class_logits = tf.ones_like(false_class_logits) * tf.gather(params=y_pred,indices=tf.where(tf.math.equal(y_true,1)))
    loss_array = tf.math.log(tf.math.exp(true_class_logits)/(tf.math.exp(true_class_logits)+tf.math.exp(false_class_logits)))
    return -tf.math.reduce_mean(loss_array)

def tuplemax_loss_batched(y_true, y_pred,batch_size=32,num_labels=10):
    """
    pairwise tuplemax loss.
    https://arxiv.org/pdf/1811.12290.pdf
    Eq. (2)
    """
    false_class_logits = tf.reshape(tf.gather_nd(params=y_pred,indices=tf.where(tf.math.equal(y_true,0))),[batch_size,num_labels-1])
    true_class_logits = tf.broadcast_to(tf.reshape(tf.gather_nd(params=y_pred,indices=tf.where(tf.math.equal(y_true,1))),[batch_size,1]),[batch_size,num_labels-1])
    loss_array = tf.math.log(tf.math.exp(true_class_logits)/(tf.math.exp(true_class_logits)+tf.math.exp(false_class_logits)))
    return -tf.math.reduce_mean(loss_array)

def vggvox_resnet2d_icassp(input_dim=(257, 250, 1), num_class=8631, mode='train', args=None):
    net=args.net
    loss=args.loss
    vlad_clusters=args.vlad_cluster
    ghost_clusters=args.ghost_cluster
    bottleneck_dim=args.bottleneck_dim
    aggregation = args.aggregation_mode
    mgpu = len(keras.backend.tensorflow_backend._get_available_gpus())

    if net == 'resnet34s':
        inputs, x = backbone.resnet_2D_v1(input_dim=input_dim, mode=mode)
    else:
        inputs, x = backbone.resnet_2D_v2(input_dim=input_dim, mode=mode)
    # ===============================================
    #            Fully Connected Block 1
    # ===============================================
#    print("inputs:",inputs.shape)
#    print("X:",x.shape)
    x_fc = keras.layers.Conv2D(bottleneck_dim, (7, 1),
                               strides=(1, 1),
                               activation='relu',
                               kernel_initializer='orthogonal',
                               use_bias=True, trainable=True,
                               kernel_regularizer=keras.regularizers.l2(weight_decay),
                               bias_regularizer=keras.regularizers.l2(weight_decay),
                               name='x_fc')(x)
#    print(x_fc.shape)
    # ===============================================
    #            Feature Aggregation
    # ===============================================
    if aggregation == 'avg':
        if mode == 'train':
            x = keras.layers.AveragePooling2D((1, 5), strides=(1, 1), name='avg_pool')(x)
            x = keras.layers.Reshape((-1, bottleneck_dim))(x)
        else:
            x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = keras.layers.Reshape((1, bottleneck_dim))(x)

    elif aggregation == 'vlad':
        x_k_center = keras.layers.Conv2D(vlad_clusters, (7, 1),
                                         strides=(1, 1),
                                         kernel_initializer='orthogonal',
                                         use_bias=True, trainable=True,
                                         kernel_regularizer=keras.regularizers.l2(weight_decay),
                                         bias_regularizer=keras.regularizers.l2(weight_decay),
                                         name='vlad_center_assignment')(x)
        x = VladPooling(k_centers=vlad_clusters, mode='vlad', name='vlad_pool')([x_fc, x_k_center])

    elif aggregation == 'gvlad':
        x_k_center = keras.layers.Conv2D(vlad_clusters+ghost_clusters, (7, 1),
                                         strides=(1, 1),
                                         kernel_initializer='orthogonal',
                                         use_bias=True, trainable=True,
                                         kernel_regularizer=keras.regularizers.l2(weight_decay),
                                         bias_regularizer=keras.regularizers.l2(weight_decay),
                                         name='gvlad_center_assignment')(x)
        x = VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='gvlad', name='gvlad_pool')([x_fc, x_k_center])

    else:
        raise IOError('==> unknown aggregation mode')

    # ===============================================
    #            Fully Connected Block 2
    # ===============================================
    x = keras.layers.Dense(bottleneck_dim, activation='relu',
                           kernel_initializer='orthogonal',
                           use_bias=True, trainable=True,
                           kernel_regularizer=keras.regularizers.l2(weight_decay),
                           bias_regularizer=keras.regularizers.l2(weight_decay),
                           name='fc6')(x)


    model = keras.models.Model(inputs, x, name='vggvox_resnet2D_{}_{}'.format(loss, aggregation))
    return model
