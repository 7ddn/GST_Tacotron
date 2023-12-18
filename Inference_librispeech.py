from Model import GST_Tacotron
import os


def remove_letter(text):
    return tf.strings.regex_replace(text, f'[0-9]', '')

gst_taco = GST_Tacotron(is_Training=False)

gst_taco.Restore()

wav_List = [os.path.expanduser(f'~/phoneme_librispeech/train-clean-100/103/wav/103-1240-00{i:02d}.wav') for i in range(43)]

# print(wav_List)
print(f'Starting inference, number of reference speeches is {len(wav_List)}')

sentence_List = [
    'The grass is always greener in the other side of the fence.'
    ] * len(wav_List)

gst_taco.Inference(
    sentence_List = sentence_List,
    wav_List_for_GST = wav_List,
    label = 'librispeech',
    )
