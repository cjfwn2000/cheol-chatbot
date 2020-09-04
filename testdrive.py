import cheolchatbot.inputdata.texts as cte

sampleTexts = [
    "오늘 점심 뭐 먹었음?",
    "얘, 봄감자가 맛있단다",
    "언제까지 과제 해야?",
    "이거 뭐죠?",
    "오늘 점심 맛있냐"
    ]
# 위 문장들로 TokenCoder를 만든다
textGen = (t for t in sampleTexts)
tokenCoder = cte.TokenCoder.fromTexts(textGen)

# 만들어진 TokenCoder의 Dictionary 현황
dict_tok2num = tokenCoder.dvWordToNum
for tok in dict_tok2num:
    print( tok, '=>', dict_tok2num[tok] )
print("--------------------")

# 테스트: encode
# 충분히 encode할 수 있는 문장
print("오늘 과제 뭐죠")
print( tokenCoder.encode("오늘 과제 뭐죠") )
# 몇몇 형태소는 UNK(1번)에 해당할 것
print("오늘 저녁 감자가 맛있대")
print( tokenCoder.encode("오늘 저녁 감자가 맛있대") )
