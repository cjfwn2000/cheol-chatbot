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
    # https://arxiv.org/pdf/1409.0473.pdf

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # queryWithTimeAxis.shape == (batchSz, 1, hidden size)
        queryWithTimeAxis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(queryWithTimeAxis) + self.W2(values)
        ))
        # attentionWeights.shape == (batchSz, seq-max-len, 1)
        attentionWeights = tf.nn.softmax(score, axis=1)
        # contextVector.shape (after sum) == (batchSz, hidden size)
        contextVector = attentionWeights * values
        contextVector = tf.reduce_sum(contextVector, axis=1)

        return contextVector, attentionWeights

    
class Decoder(tf.keras.Model):
    def __init__(self, vocabSize, embeddingDim, decUnits, batchSz):
        super(Decoder, self).__init__()
        self.batchSz = batchSz
        self.decUnits = decUnits
        # Embedding layer는 왜 필요하지?
        self.embedding = tf.keras.layers.Embedding(vocabSize, embeddingDim)
        self.gru = tf.keras.layers.GRU(
            self.decUnits,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        # Embedding의 역
        self.fc = tf.keras.layers.Dense(vocabSize)
        # 위의 Attention layer
        self.attention = BahdanauAttention(self.decUnits)
    
    def call(self, x, hidden, encOutput):
        # encOutput shape == (batchSz, seq-max-len, hidden size)
        contextVector, attentionWeights = self.attention(hidden, encOutput)
        # x shape after embedding == (batchSz, 1, embbeddingDim)
        x = self.embedding(x)
        # x shape after concatenation == (batchSz, 1, embeddingDim + hidden size)
        x = tf.concat([tf.expand_dims(contextVector, 1), x], axis=-1)
        # GRU
        output, state = self.gru(x)
        # output shape == (batchSz * 1, hidden size)
        output = tf.reshape(output, (-1, output.shape[2]))
        # output shape == (batchSz, vocab)
        x = self.fc(output)

        return x, state, attentionWeights

    