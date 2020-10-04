import cheolchatbot.inputdata.data as cda
import cheolchatbot.langmodel.seq2seq as cseq2

dataFetcher = cda.MysqlChatDataFetcher('localhost', 'plise', 'plise4434', 'acxchatbot')
# Now the data is loaded.

train, vali = dataFetcher.toTfDataset()

# I got tf.dataset of train/val
# 이제 문-답 사례 텐서 하나(oneSequence, oneSequence)만 추출... 그 중에서 '문'(dem; oneSequence)만 따오자.
# 이것으로 모델 유닛테스트를 할 것이다.
for dselem in vali.take(1):
    dem, res = dselem

# dem = 64개 문장이 있음 (batch로 묶었기 때문)
print("dem.shape")
print(dem.shape) #(batchSize, seqLen)

# Encoder가 잘 만들어졌는지에 대한 테스트를 한다.
myEncoder = cseq2.Encoder(
    vocabSize=dataFetcher.vocabSize(),
    embeddingDim=512,
    encUnits=1024,
    batchSz=64
)

# Seq2seq의 Encoder는 Hidden state를 다뤄야 한다.
# Hidden state의 초기는 이번 개발 단계에서 주어지지 않았다.
# 그 초기값은 나중에 모델 통합 단계에서 주어진다.
# 그래서 우리가 직접 만들어준다.
hidden = myEncoder.initializeHiddenState()
print("hidden.shape")
print(hidden.shape) #(batchSize, encUnits)

# Showtime
output, hiddenAfter = myEncoder(dem, hidden)
print("output.shape")
print(output.shape) #(batchSize, seqLen, encUnits)
print("hiddenAfter.shape")
print(hiddenAfter.shape) #(batchSize, encUnits)
