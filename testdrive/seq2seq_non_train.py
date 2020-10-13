import tensorflow as tf
import cheolchatbot.langmodel.seq2seq as cseq2

# Encoder와 Decoder의 입력으로 들어갈 Tensor
# inputdata.data.DataFetcher에서 나올 수 있는 Tensor
BATCH = 64
SEQ_LEN_DEM = 17
SEQ_LEN_RES = 14
VOCABSIZE = 6700
sampleInputDem = tf.random.uniform(
    (BATCH, SEQ_LEN_DEM), minval=0,  maxval=VOCABSIZE, dtype=tf.int32)
sampleInputRes = tf.random.uniform(
    (BATCH, SEQ_LEN_RES), minval=0,  maxval=VOCABSIZE, dtype=tf.int32)

print("sampleInputDem.shape")
print(sampleInputDem.shape)
print("sampleInputRes.shape")
print(sampleInputRes.shape)
print()

# Models and layers
EMB_DIM = 512
CODER_UNITS = 1024
ATT_UNITS = 10

myEncoder = cseq2.Encoder(
    vocabSize=VOCABSIZE,
    embeddingDim=EMB_DIM,
    encUnits=CODER_UNITS,
    batchSz=BATCH
)
myAttentionLayer = cseq2.BahdanauAttention(ATT_UNITS)
myDecoder = cseq2.Decoder(
    vocabSize=VOCABSIZE,
    embeddingDim=EMB_DIM,
    decUnits=CODER_UNITS,
    batchSz=BATCH
)


# Seq2seq의 Encoder는 Hidden state를 다뤄야 한다.
# Hidden state의 초기값은 이번 개발 단계에서 주어지지 않았다.
# 소프트웨어 완성품에서야 우리가 직접 만들 필요가 사라진다.
hidden = myEncoder.initializeHiddenState()
print("hidden.shape")
print(hidden.shape) #(batchSize, encUnits)
print()

# Test: Encoder
outputFromEnc, hiddenAfter = myEncoder(sampleInputDem, hidden)
print("outputFromEnc.shape")
print(outputFromEnc.shape) #(batchSize, seqLen, encUnits)
print("hiddenAfter.shape")
print(hiddenAfter.shape) #(batchSize, encUnits)
print()

# Test: BahdanauAttention
attResult, attWeights = myAttentionLayer(hiddenAfter, outputFromEnc)
print("Shape Att result a.k.a. context vector")
print( attResult.shape ) #(batchSize, hiddenUnits=encUnits)
print("Shape Att weights")
print( attWeights.shape ) #(batchSize, seqMaxLen, 1)
print()

# Test: Decoder
outputFromDec, _, _ = myDecoder(
    tf.random.uniform((BATCH, 1)),
    hiddenAfter, outputFromEnc )
print("Shape output from dec")
print(outputFromDec.shape) #(batchSize, vocab)
print()
# 주의해야 할 점은
# Decoder는 Encoder와 달리, 그 입력이 '단타'성이라는 것이다.
# 쉽게 말해 Decoder에게는 입력을 (BATCH, SEQ_LEN)으로 주어서는 안 되고
# (BATCH, 1) 로 주어야 한다는 것이다.
# 그리하여 BATCH개의 각 단어 하나씩 나온 게 결과다. => (BATCH, VOCAB_SIZE)
# 실제 운용: 희망 문장은 하나 (즉 BATCH=1), <END>가 나올 때까지 Loop.

