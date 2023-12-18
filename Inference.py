from Model import GST_Tacotron
import os
import argparse

inf_parser = argparse.ArgumentParser()
inf_parser.add_argument('--max_num')
args = inf_parser.parse_args()
max_num = int(args.max_num) if args.max_num is not None else None


def remove_letter(text):
    return tf.strings.regex_replace(text, f'[0-9]', '')

gst_taco = GST_Tacotron(is_Training=False)

gst_taco.Restore()

language_List = ['japanese', 'english', 'german', 'portuguese', 'polish', 'french', 'turkish', 'mandarin']
language_Number = [27, 579, 36, 48, 34, 63, 37, 65]

ban_list = ['mandarin']
# ban_list = None
# allow_list = ['japanese', 'french']
allow_list = None

# wav_List = []
for idx, l in enumerate(language_List):

    if max_num is not None and idx > max_num:
        break
    
    if ban_list is not None and l in ban_list:
        continue
    elif allow_list is not None and l not in allow_list:
        continue


    wav_List = [
        os.path.expanduser('~/accented_speech_archive/wavs/') + l + str(i) + '.mp3.wav' for i in range(1, language_Number[idx])]
    
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


