from Model import GST_Tacotron
import os
import argparse
import math

inf_parser = argparse.ArgumentParser()
inf_parser.add_argument('--max_num', default = math.inf)
inf_parser.add_argument('--min_num', default = 0)
args = inf_parser.parse_args()

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

ban_list = ['english']
# ban_list = None
# allow_list = ['japanese', 'french']
allow_list = None

# wav_List = []
for l in language_List:
    if language_Number[l] < min_num:
        continue

    if ban_list is not None and l in ban_list:
        continue
    elif allow_list is not None and l not in allow_list:
        continue

    wav_List = []

    idx = 1
    for _ in range(language_Number[l]):
        while os.path.isfile(os.path.expanduser('~/accented_speech_archive/wavs/') + l + str(idx) + '.mp3.wav') is False:
            idx += 1
        wav_List.append(os.path.expanduser('~/accented_speech_archive/wavs/') + l + str(idx) + '.mp3.wav')
        idx += 1
    
    # print(wav_List)
    print(f'Starting inference for language {l}, number of reference speeches is {len(wav_List)}')

    sentence_List = [
        'Please call Stella. Ask her to bring these things with her from the store.'
        ] * len(wav_List)

    gst_taco.Inference(
        sentence_List = sentence_List,
        wav_List_for_GST = wav_List,
        label = l,
        )


