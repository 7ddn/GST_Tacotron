import cmudict
import re

arpa_dict = cmudict.dict()

def str_to_arpa(text, no_stress = True):
    def get_first_arpa(arpa_list):
        if arpa_list == []:
            return ''
        return arpa_list[0]

    if isinstance(text, list):
        text = ' '.join(text)
    text = re.findall(r"[\w']+", text)    
    arpa = []
    for t in text:
        arpa.append(' '.join(get_first_arpa(arpa_dict[t.lower()])))
    arpa = ' '.join(arpa)
    if no_stress:
        return re.sub(r'[0-9]', '', arpa)
    return arpa
