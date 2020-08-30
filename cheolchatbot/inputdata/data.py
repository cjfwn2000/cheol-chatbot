'''
훈련용 데이터를 가져오는 일을 한다.
'''

import tensorflow as tf
from . import texts
import pymysql.cursors
import csv #writeTuplelistTempCsv
import tempfile #.gettempdir()

def writeTuplelistTempCsv(tuplegen, header):
    '''tuplegen이 yield하는 tuple을 임시파일에 csv 써내어 그 파일 주소를 return한다.
    또한 yield된 횟수를 return한다. tuplegen에 내재된 항목 갯수를 알기 위한 사용자를 위함이다.
    header는 문자열 tuple이며 csv의 맨 위에 쓰여진다. 쓰지 않을 것이면 None이면 된다.'''
    # MysqlChatDataset에서 MySQL->csv 전략을 수행하기 위한 요량으로 만들었다.
    fn = tempfile.gettempdir() + '/acxchat_export.csv'
    i = 0 #how many rows...
    with open(fn, 'w', newline='') as tempcsv:
        writer = csv.writer(tempcsv, quoting=csv.QUOTE_NONNUMERIC)
        if header: writer.writerow(header)
        for tu in tuplegen:
            writer.writerow(tu)
            i += 1
    return fn, i


class MysqlChatDataFetcher:
    '''
    이 객체가 제대로 데이터를 가져올 수 있게
    사용자는 mysql계정, 비밀번호, 테이블 이름 등등을 설정해 줄 필요가 있다.
    '''

    def __init__(self, host, user, password, dbName, trainRate=0.7, batchSize=64):
        self._mysqlSetting = {'host':host, 'user':user, 'password':password, 'db':dbName}
        self._trainRate = trainRate
        self._batchSize = batchSize
        
        dsTrain, dsVal, toker = self._fetchChatdata()
        self._datasetTrain = dsTrain
        self._datasetVal = dsVal
        self._tokenCoder = toker
        return None
    
    def _fetchChatdata(self):
        '''Returns dsTrain, dsVal, toker.'''
        # Strategy:
        # MySQL -> csv -> tf의 CsvDataset
        
        # MySQL... -> csv
        filenameCsv = None
        howManyRows = 0
        ms = self._mysqlSetting
        connection = pymysql.connect(host=ms['host'],
                                     user=ms['user'],
                                     password=ms['password'],
                                     db=ms['db'],
                                     charset='utf8mb4',
                                     cursorclass=pymysql.cursors.SSCursor)
        try:
            sql = 'SELECT question, answer FROM Chats'
            with connection.cursor() as cursor:
                cursor.execute(sql)
                logs = cursor.fetchall_unbuffered()
                filenameCsv, howManyRows = writeTuplelistTempCsv(logs, header=('question', 'answer'))
            
        finally:
            connection.close()

        # Loading the csv file saved above.
        # -> Dataset [(tf.string, tf.string), ...]
        assert not filenameCsv == None
        dsCrude = tf.data.experimental.CsvDataset(filenameCsv, [tf.string, tf.string], header=True)
        
        # Splitting into train/val.
        lenTrain = int(howManyRows * self._trainRate)
        dsTrain = dsCrude.take(lenTrain)
        dsVal = dsCrude.skip(lenTrain)
        
        # Preparing a TokenCoder to encode the strings of ds into numbers.
        def genSentences(ds):
            for dem, res in ds:
                yield dem.numpy() #type(dem) == tf.string
                yield res.numpy()
        toker = texts.TokenCoder.fromTexts(genSentences(genSentences(dsTrain)))
        
        # ds를 Tensorflow model의 입력으로 삼을 수 있게 만든다.
        # 즉 Token-coding한다. 또한 Shuffling, padding 등등도 행한다.
        bufferSizeToShuffle = 20000
        batchSize = self._batchSize
        staTok, endTok = toker.endNum()+1, toker.endNum()+2
        def tokencode(dem, res):
            demNew = [staTok] + toker.encode(dem.numpy()) + [endTok]
            resNew = [staTok] + toker.encode(res.numpy()) + [endTok]
            return demNew, resNew
        def tfTokencode(dem, res):
            # Dataset.map runs in graph mode.
            # Graph tensors do not have a value.
            # In graph mode you can only use TensorFlow Ops and functions.
            demTf, resTf = tf.py_function(
                tokencode, [dem, res], [tf.int64, tf.int64] )
            demTf.set_shape([None])
            resTf.set_shape([None])
            return demTf, resTf
        
        dsTrainTK = dsTrain.map(tfTokencode)
        dsTrainTK = dsTrainTK.cache()
        # cache를 하면, dataset을 메모리로 올리게 되고 읽음이 빨라진다.
        dsTrainTK = dsTrainTK.shuffle(bufferSizeToShuffle).padded_batch(batchSize)
        dsTrainTK = dsTrainTK.prefetch(tf.data.experimental.AUTOTUNE)
        dsValTK = dsVal.map(tfTokencode)
        dsValTK = dsValTK.padded_batch(batchSize)
                     
        return dsTrainTK, dsValTK, toker                                 
    

    def toTfDataset(self):
        '''Tensorflow 모델 입력으로 쓰이기 위한 Train dataset과 Validation dataset을 return한다.'''
        return self._datasetTrain, self._datasetVal
    

    def tokenCoder(self):
        '''이 데이터를 전처리함에 있어 만들어진 TokenCoder를 return한다.'''
        return self._tokenCoder

    
    def vocabSize(self):
        '''토큰 번호의 총 가짓수.'''
        # self.toker.vocabSize() + len([(Padding을 나타내는 0번 토큰), (시작토큰), (끝토큰)])
        #= len([0, 1, 2, ..., self.toker.vocabSize(), staTok, endTok])
        #= self.toker.vocabSize() + 3
        # 의문 : https://www.tensorflow.org/tutorials/text/transformer#set_hyperparameters 를 따르면, 위의 값이 self.toker.vocabSize()+2여야 한다더라.
        #그렇다면 TokenCoder.vocabSize()는 원래 [0]까지 포함한 계산 결과여야 했나?
        
        return self._tokenCoder.vocabSize() + 3

