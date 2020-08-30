from konlpy.tag import Okt
import json

# vocabDict에 없을 문자, Unknown Character
UNK = ('UNK', 'UNK')

# 아래 클래스에서 자주 쓰이게 될 '문장 분해'를 하려고 도구를 준비해 준다.
okt = Okt()
def elemSent(s):
    # '여기 있는 감자튀김 ...' =>
    # [('여기', 'Noun'), ('있는', 'Adjective'), ('감자', 'Noun'), ('튀김', 'Noun'), ...]
    return okt.pos(s, norm=True)


class TokenCoder:
    '''
    한국어 텍스트를 일련번호배열로 만들어 준다.
    tfds.features.text.TextEncoder와 같은 기능이
    우리의 프로젝트에서도 쓰여질 수 있도록 해보았다.
    e.g.
    encode('여기 있는 감자튀김 다 내꺼야.') == [34, 23, 402, 130, 76, 549]
    decode([34, 23, 402, 130, 76, 549]) == [여기 있는 감자 튀김 다 내꺼야]
    encode의 입력과 decode의 출력이 다를 수 있음에 주의하라.
    번호 0, 1은 특수번호다. 각각 Padding과 UNK를 가리킨다.
    '''
    
    def __init__(self):
        '''생성자이지만 단독호출하지 말라. fromTexts 메소드를 참고하라.'''
        
        # Dictionaries of Vocab. 변환표. Number로도 Word로도 접근할 수 있게 만든다.
        '''K:V = Word:Number  ;for Encoding'''
        self.dvWordToNum = {}
        '''K:V = Number:Word  ;for Decoding'''
        self.dvNumToWord = {}
        
        
    @classmethod
    def fromTexts(cls, textGenerator):
        '''텍스트 문장을 일일히 흡수해 TokenCoder를 내어준다.
        textGenerator는 텍스트 문장 하나를 yield해야 한다.'''
        
        # 여기에 Dictionaries를 채워넣어라.
        t = cls()
        # 문장들에서 모아온 형태소가 중복되지 않기 위해.
        wordSet = set()
        
        for s in textGenerator:
            pos = elemSent(s)
            wordSet.update(pos)
            
        # t에게 줄 Dictionaries를 만든다
        numbersforWords = range(2, len(wordSet)+2)
        dvWordToNum = dict( zip(wordSet, numbersforWords) )
        dvNumToWord = dict( zip(numbersforWords, wordSet) )
        # 특수번호 0, 1; 0은 skip.
        dvWordToNum[UNK] = 1
        dvNumToWord[1] = UNK
        
        t.dvWordToNum = dvWordToNum
        t.dvNumToWord = dvNumToWord
        return t
        
    
    def vocabSize(self):
        return len(self.dvNumToWord)
    
    
    def endNum(self):
        # Dictionaries에 배정된 번호는 2,3,4,...,endNum,1
        # 1은 마지막에 추가된 UNK이며, 0은 배정되있지 않다.
        # 따라서 vocabSize = len[1..endNum] = endNum
        # 의문: 배정번호가 1..endNum으로 연속이지 않을 수 있는가? 아직은 없다고 본다.
        return self.vocabSize()
    
    
    # encode가 self.dvWordToNum를 직접 건드릴 필요를 없도록 하기 위해.
    # decode도 마찬가지.
    def mapToNum(self, word):
        wn = self.dvWordToNum
        if word in wn: return wn[word]
        else: return wn[UNK]
    
    def mapToWord(self, num):
        nw = self.dvNumToWord
        if num in nw: return nw[num]
        else: return UNK  #번호 1도 아니고 쌩판 미확인 번호인 경우에도.
    
    
    def encode(self, s):
        ''' s to [num...] '''
        ss = elemSent(s)
        return [self.mapToNum(w) for w in ss]
    
    
    def decode(self, nn):
        ''' nn=[num...] to s '''
        ss = [self.mapToWord(n) for n in nn]
        #return " ".join(ss)
        return self.reconstructOktSentence(ss)
        
    def reconstructOktSentence(self, poslist):
        return " | ".join([str(e) for e in poslist])
    
    
    def saveDictionaryFile(self, filename):
        '''filename 파일로 이 객체의 설정을 내보낸다.
        프로그램 종료 후에도 재사용이 가능하도록 도와준다.'''
        
        # Number to word dictionary만 저장해도 다른 하나를 복구할 수 있을 것.
        nw = self.dvNumToWord
        with open(filename, 'w') as f:
            # json.dump(mySavedSetup, f)
            for n,w in nw.items():
                # 549(tab)내꺼야(tab)Verb
                f.write('{}\t{}\t{}\n'.format(n, w[0], w[1]))
                
    
    @classmethod
    def fromDictionaryFile(cls, filename):
        '''filename 파일에서 이 객체의 설정을 불러온다.
        프로그램 종료 후에도 재사용이 가능하도록 도와준다.'''
        
        # 파일엔 Number to word dictionary만 적혀 있으므로
        # 이를 가지고 Word to number dictionary를 구한다.
        def genEachline(file):
            line = file.readline()
            while line:
                yield line
                line = file.readline()
        
        nw = {}
        with open(filename, 'r') as f:
            #myLoadedSetup = json.load(f)
            for line in genEachline(f):
                line = line.rstrip() #마지막 \n을 제거한다
                # 549(tab)내꺼야(tab)Verb
                args = line.split('\t')
                n, w = int(args[0]), (args[1], args[2])
                nw[n] = w
        nw[1] = UNK
        
        # 여기에 Dictionaries를 채워넣어라.
        t = cls()
        dvNumToWord = nw
        dvWordToNum = dict([(word,num) for num,word in dvNumToWord.items()]) #reversing
        t.dvNumToWord = dvNumToWord
        t.dvWordToNum = dvWordToNum
        return t
