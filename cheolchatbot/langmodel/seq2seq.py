import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocabSize, embeddingDim, encUnits, batchSz):
        super(Encoder, self).__init__()
        
        # The batch Size (한 묶음 당 row, 즉 문-답 문장쌍 갯수; 보통 Data fetch과정에서 정해놨을 것)
        self.batchSz = batchSz
        # The size; the output of one cell of RNN(GRU here)
        self.encUnits = encUnits
        
        # Embedding layer; The first layer
        self.embedding = tf.keras.layers.Embedding(vocabSize, embeddingDim)
        # GRU; RNN Layer; The second layer
        self.gru = tf.keras.layers.GRU(
            self.encUnits,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )


    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state


    def initializeHiddenState(self):
        return tf.zeros((self.batchSz, self.encUnits))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        raise Exception('Not implemented yet.')

    def call(self, query, values):
        raise Exception('Not implemented yet.')

    
class Decoder(tf.keras.Model):
    def __init__(self, vocabSize, embeddingDim, decUnits, batchSz):
        raise Exception('Not implemented yet.')
    
    def call(self, x, hidden, encOutput):
        raise Exception('Not implemented yet.')

    