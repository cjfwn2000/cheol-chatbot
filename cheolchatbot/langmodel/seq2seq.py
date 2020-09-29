import tensorflow as tf

class Seq2seqEncoder(tf.keras.Model):
    def __init__(self, vocabSize, embeddingDim, encUnits, batchSz):
        super(Seq2seqEncoder, self).__init__()
        raise Exception('Not implemented yet.')

    def call(self, x, hidden):
        raise Exception('Not implemented yet.')

    def initializeHiddenState(self):
        raise Exception('Not implemented yet.')


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        raise Exception('Not implemented yet.')

    def call(self, query, values):
        raise Exception('Not implemented yet.')

    
class Seq2seqDecoder(tf.keras.Model):
    def __init__(self, vocabSize, embeddingDim, decUnits, batchSz):
        raise Exception('Not implemented yet.')
    
    def call(self, x, hidden, encOutput):
        raise Exception('Not implemented yet.')

    