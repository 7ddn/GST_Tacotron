from Model import GST_Tacotron
import os

def remove_letter(text):
    return tf.strings.regex_replace(text, f'[0-9]', '')

gst_taco = GST_Tacotron(is_Training=False)

gst_taco.Restore()

language_List = ['japanese', 'english', 'german', 'portuguese', 'polish', 'french', 'turkish', 'mandarin']
language_Number = [27, 579, 36, 48, 34, 63, 37, 65]

# ban_list = ['turkish', 'english']
ban_list = None
# allow_list = ['japanese', 'mandarin']
allow_list = None

# wav_List = []
# tag_List = []
for idx, l in enumerate(language_List):
    
    if ban_list is not None and l in ban_list:
        continue
    elif allow_list is not None and l not in allow_list:
        continue

    wav_List = [
        os.path.expanduser('~/accented_speech_archive/wavs/') + l + str(i) + '.mp3.wav' for i in range(1, language_Number[idx])]

    tag_List = [
        l for _ in range(1, language_Number[idx])]

    print(f'Starting inference GST for language {l}, number of utterances is {len(wav_List)}')

    gst_taco.Inference_GST(wav_List, tag_List)
