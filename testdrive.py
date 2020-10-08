import tensorflow as tf
import cheolchatbot.langmodel.seq2seq as cseq2

# Encoder와 Decoder의 입력으로 들어갈 Tensor
# inputdata.data.DataFetcher에서 나올 수 있는 Tensor
BATCH = 64
SEQ_LEN_DEM = 17
SEQ_LEN_RES = 14
VOCABSIZE = 6700
sampleInputDem = tf.random.uniform(
    (BATCH, SEQ_LEN_DEM), minval=0,  maxval=VOCABSIZE+1, dtype=tf.int32)
sampleInputRes = tf.random.uniform(
    (BATCH, SEQ_LEN_RES), minval=0,  maxval=VOCABSIZE+1, dtype=tf.int32)

print("sampleInputDem.shape")
print(sampleInputDem.shape)
print("sampleInputRes.shape")
print(sampleInputRes.shape)
print()

# Encoder가 잘 만들어졌는지에 대한 테스트를 한다.
EMB_DIM = 512
CODER_UNITS = 1024

myEncoder = cseq2.Encoder(
    vocabSize=VOCABSIZE,
    embeddingDim=EMB_DIM,
    encUnits=CODER_UNITS,
    batchSz=BATCH
)

# Seq2seq의 Encoder는 Hidden state를 다뤄야 한다.
# Hidden state의 초기값은 이번 개발 단계에서 주어지지 않았다.
# 소프트웨어 완성품에서야 우리가 직접 만들 필요가 사라진다.
hidden = myEncoder.initializeHiddenState()
print("hidden.shape")
print(hidden.shape) #(batchSize, encUnits)
print()

# Showtime
output, hiddenAfter = myEncoder(sampleInputDem, hidden)
print("output.shape")
print(output.shape) #(batchSize, seqLen, encUnits)
print("hiddenAfter.shape")
print(hiddenAfter.shape) #(batchSize, encUnits)
