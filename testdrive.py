import cheolchatbot.inputdata.data as cda

joso = cda.MysqlChatDataFetcher(host='localhost', user='plise', password='plise4434', dbName='acxchatbot')
dsTrain, _ = joso.toTfDataset()
print(dsTrain.element_spec)
print("------------------------------")
toker = joso.tokenCoder()

keys = list(toker.dvNumToWord.keys())
for i in range(5):
    gus = toker.dvNumToWord[keys[i]]
    print(gus)
    