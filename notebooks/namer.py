_DEFAULT_WORDS = open('../local_data/naming/wordlist.100000', 'r').read().splitlines()

def get_name ( objid, word_list=None ):
    if word_list is None:
        word_list = _DEFAULT_WORDS
    wordcount = len(word_list)
    hostnumber = int(str(objid)[:8])
    hostword1 = word_list[hostnumber % wordcount]
    hostword2 =  word_list[sum([int(w) for w in str(hostnumber)])]
    galword = word_list[int(str(objid)[8:]) % wordcount]
    return '_'.join([hostword1,hostword2,galword])

