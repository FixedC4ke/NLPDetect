from tnize import GetTokens

rusdictfile = open('dict-utf8.txt', 'r')
rusdict = rusdictfile.read().split('\n')
rusdictfile.close()

corpus=[] #текста на вход
corpus.append('Съешь еще этих мягких булок, да выпей чаю')

for sentence in corpus: #дополнение словаря отсутсвующими словами
    senttokens = GetTokens(sentence)
    for word in senttokens:
        if word not in rusdict:
            rusdict.append(word)

rusdictfile = open('dict-utf8.txt', 'w') #перезапись словаря с добавленными словами
for word in rusdict:
    rusdictfile.write(word+'\n')
rusdictfile.close()

print(rusdict[-1])