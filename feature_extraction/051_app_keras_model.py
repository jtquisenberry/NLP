import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta, Adam, RMSprop
from keras.utils import np_utils
import pandas as pd
import tensorflow as tf
from sklearn.manifold import TSNE
import pickle

from keras.models import load_model
model = load_model(r'E:\Corpora\PII_Directory_20190507\keras_model_001.mdl')

filehandler = open(r"E:\Corpora\PII_Directory_20190507\keras_vectors_001.pkl", 'rb')
tfidf_trained_vectorizer = pickle.load(filehandler)
filehandler.close()


print(model.weights)

weights_layer_0 = model.get_weights()[0][:]
weights_df = pd.DataFrame(weights_layer_0).head()
# print(weights_df)

weights_layer_0 = model.weights[0]
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

with sess.as_default():
    weights_array = weights_layer_0.eval() #numpy.ndarray
    weights_df = pd.DataFrame(weights_array)

    X_embedded = TSNE(n_components=1).fit_transform(weights_array)
    X_embedded_df = pd.DataFrame(X_embedded)
    X_embedded_series = X_embedded_df[0]
    indices = X_embedded_series.argsort()
    indices_descending = list(reversed(list(indices)))

    print('TOP FEATURES')
    print()

    features_array = np.array(tfidf_trained_vectorizer.get_feature_names())
    print(features_array[indices_descending][:40])

    # ['read book' 'siemen' 'ingr' 'needle say' 'articl 1993apr16' 'despit'
    #  'electron' 'employ opinion' 'hydra gatech' 'fluke' 'alchemi' 'theodor'
    #  'wesleyan' 'bike' 'documentari' 'specif' 'help appreci' 'certainli'
    #  '1993apr18' 'rid' 'geoffrey' 'psi' 'uxh' 'latch' 'perfect' 'rewrit'
    #  'close' 'appropri' 'guild org' 'thing like' 'bush' 'nsa' 'chicago'
    #  'socket' 'incent' '278' 'convex' 'exist singl' 'charli' '50mhz']



    a = 1

a = 1