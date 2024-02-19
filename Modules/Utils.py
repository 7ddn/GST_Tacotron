import cmudict

cmu = cmudict.dict()

def to_ARPA(text, cmu = cmu, num = False):
    text= text.strip()
    text = [cmu[i.lower()] for i in text.split()]
    text = [' '.join(i[0]) for i in text if len(i)>0]
    text =  ' '.join(text).strip()

    if not num:
        for i in range(10):
            text = text.replace(str(i), '')
    
    return text
