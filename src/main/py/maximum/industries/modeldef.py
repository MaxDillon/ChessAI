import os, time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Activation, Input, Dropout, Dense, Conv2D
from tensorflow.keras.layers import Flatten, Add, Softmax, BatchNormalization, Subtract, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l2
from tensorflow.compat.v1.graph_util import convert_variables_to_constants
from google.protobuf.internal.encoder import _VarintBytes
from google.protobuf.internal.decoder import _DecodeVarint32
from maximum.industries.loader import NUM_INPUT_CHANNELS, DTYPE
from maximum.industries.normalization import FixedNormalization

# initialize TF default session to use GPU growth so all memory is not immediately reserved
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))
K.clear_session() # this is what clears the default graph, not setting a session.
K.set_floatx(DTYPE)
if DTYPE == 'float16':
    K.set_epsilon(1e-4) # use a larger epsilon for float16

def get_conv(filters, kernel_size=3, activation=tf.nn.relu):
    return Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                  activation=activation, data_format='channels_first',
                  kernel_initializer=initializers.glorot_normal(),
                  bias_initializer=initializers.zeros(),
                  kernel_regularizer=l2(0.001),
                  bias_regularizer=l2(0.001))

def get_dense(units, regu=0.001):
    return Dense(units,
                 activation=tf.nn.relu,
                 kernel_initializer=initializers.glorot_normal(),
                 bias_initializer=initializers.glorot_normal(),
                 kernel_regularizer=l2(regu),
                 bias_regularizer=l2(regu),
                 name='dense_%d_relu' % units)
    
def get_norm(freeze_batch_norm, name, scale=True):
    return (FixedNormalization(axis=1, scale=scale, name=name)
            if freeze_batch_norm else
            BatchNormalization(axis=1, scale=scale, name=name))

def get_residual_block(x1, freeze_batch_norm, i):
    filters = K.int_shape(x1)[1]
    x2 = get_conv(filters=filters, activation=None)(x1)
    x2 = get_norm(freeze_batch_norm, 'batchnorm-%d-a' % i, scale=False)(x2)
    x2 = Activation(tf.nn.relu)(x2)
    x2 = get_conv(filters=filters, activation=None)(x2)
    x2 = get_norm(freeze_batch_norm, 'batchnorm-%d-b' % i, scale=True)(x2)
    x2 = Add()([x1, x2])
    return Activation(tf.nn.relu)(x2)

def make_model(filters=160, blocks=8, kernels=(5,1), rate=0.001, freeze_batch_norm=False):
    input = Input(shape=(NUM_INPUT_CHANNELS, 8, 8), name='input')

    # initial convolution
    x = get_conv(filters=filters, kernel_size=kernels[0])(input)
    
    # residual blocks
    for i in range(blocks): x = get_residual_block(x, freeze_batch_norm, i)

    # value tower
    vt = Flatten()(x)
    vt = get_dense(40, regu=0.01)(vt)
    vt = Dropout(rate=0.5)(vt)
    vt = get_norm(freeze_batch_norm, 'batchnorm-vt')(vt)
    vt = get_dense(30, regu=0.02)(vt)
    vt = Dropout(rate=0.5)(vt)
    value = Dense(1, activation=tf.nn.tanh, name='value',
                  kernel_initializer=initializers.glorot_normal(),
                  bias_initializer=initializers.zeros(),
                  bias_regularizer=l2(1.0),
                  activity_regularizer=l2(0.01))(vt)

    px = get_conv(filters=8*8, activation=None, kernel_size=kernels[1])(x)
    pf = Flatten()(px)
    policy = Softmax(name='policy')(pf)

    model = Model(inputs=input, outputs=[value, policy])
    losses = { 'value': 'mean_squared_error', 'policy': 'categorical_crossentropy' }
    weights = { 'value': 2.0, 'policy': 1.0 }
    optimizer = Adam(rate)
    model.compile(optimizer=optimizer, loss=losses, loss_weights=weights, metrics=[])

    print('Model parameters: %d' % model.count_params())
    return model

def save_model(model, output_dir):
    timestamp = int(time.time())
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    model.save('%s/%d.h5' % (output_dir, timestamp)) # save reloadable .h5 so we can restart training
    tmp_name = '/tmp/weights-%s.h5' % timestamp
    model.save_weights(tmp_name)

    # determine the number of filters and blocks in the model
    last_add = [l for l in model.layers if 'add' in l.name][-1]
    blocks = int(last_add.name.split('_')[-1]) + 1
    filters = int(last_add.input[0].shape[1])
    kernels = [int([l for l in model.layers if 'conv2d' in l.name][i].weights[0].shape[0])
               for i in [0,-1]]
    
    with tf.Graph().as_default():
        with tf.Session().as_default() as freeze_sess:
            # in new graph and session create an identical model, except with custom batch normalization
            # layers that can be frozen without any training ops.
            model2 = make_model(filters=filters, blocks=blocks, kernels=kernels, freeze_batch_norm=True)
            # load weights into the new network and get a frozen graph def.
            model2.load_weights(tmp_name)
            freeze_var_names = [v.name for v in model2.variables]
            output_node_names = ['input', 'value/Tanh', 'policy/Softmax']
            output_node_names += [v.op.name for v in model2.variables]
            input_graph_def = freeze_sess.graph.as_graph_def()
            frozen_graph_def = convert_variables_to_constants(freeze_sess, 
                                                              input_graph_def,
                                                              output_node_names)
            # create a new graph and sesion containing the frozen graph and save
            with tf.Graph().as_default():
                with tf.Session().as_default() as save_sess:
                    tf.graph_util.import_graph_def(frozen_graph_def, name='')
                    builder = tf.saved_model.builder.SavedModelBuilder('%s/%d' % (output_dir, timestamp))
                    builder.add_meta_graph_and_variables(save_sess, [tf.saved_model.tag_constants.SERVING])
                    builder.save(False)
