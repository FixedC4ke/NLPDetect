def GetDict():
    rusdictfile = open('dict-utf8.txt', 'r', encoding='utf-8')
    rusdict = rusdictfile.read().split('\n')
    rusdictfile.close()
    return rusdict