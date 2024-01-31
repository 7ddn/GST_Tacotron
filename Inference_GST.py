from Model import GST_Tacotron
import os
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--max_num', default = math.inf)
parser.add_argument('--min_num', default = 0)
args = parser.parse_args()

max_num = float(args.max_num)
min_num = float(args.min_num)

def remove_letter(text):
    return tf.strings.regex_replace(text, f'[0-9]', '')

gst_taco = GST_Tacotron(is_Training=False)

gst_taco.Restore()

with open(os.path.expanduser('~/accented_speech_archive/language.py')) as f:
    exec(f.read())
    # language_List = ['japanese', 'english', 'german', 'portuguese', 'polish', 'french', 'turkish', 'mandarin']
    # language_Number = [27, 579, 36, 48, 34, 63, 37, 65]

# ban_list = ['turkish', 'english']
ban_list = None
# allow_list = ['japanese', 'mandarin']
allow_list = None

wav_List = []
tag_List = []
for l in language_List:
    # wav_List = []
    # tag_List = []
    if language_Number[l] < min_num:
        print('skipped because reach the min')
        continue
    
    if (language_Number[l] > max_num):
        print('skipped because reach the max')
        continue
    
    if ban_list is not None and l in ban_list:
        continue
    elif allow_list is not None and l not in allow_list:
        continue

    idx = 1
    for _ in range(language_Number[l]):
        while os.path.isfile(os.path.expanduser('~/accented_speech_archive/wavs/') + l + str(idx) + '.mp3.wav') is False:
            idx += 1
        wav_List.append(os.path.expanduser('~/accented_speech_archive/wavs/') + l + str(idx) + '.mp3.wav')
        idx += 1

    tag_List += [
        l for _ in range(1, language_Number[l])]

    print(f'Add utterances of language {l} into the list, number of utterances is {language_Number[l]}')

print(f'Starting GST Inference, used {len(wav_List)} utterances')

gst_taco.Inference_GST(wav_List, tag_List)
