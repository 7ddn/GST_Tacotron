from Model import GST_Tacotron
gst_taco = GST_Tacotron(is_Training=False)

gst_taco.Restore()

language_List = ['japanese', 'english', 'german', 'portuguese', 'polish', 'french', 'turkish', 'mandarin']
language_Number = [27, 579, 36, 48, 34, 63, 37, 65]

ban_list = ['turkish', 'english']
# allow_list = ['german', 'polish']

wav_List = []
for idx, l in enumerate(language_List):
    if l in ban_list:
        continue
    wav_List += [
        '../accented_speech_archive/wavs/' + l + str(i) + '.mp3.wav' for i in range(1, language_Number[idx])]

print(wav_List)

sentence_List = [
    'Please call Stella. Ask her to bring these things with her from the store.'
    ] * len(wav_List)

gst_taco.Inference(
    sentence_List = sentence_List,
    wav_List_for_GST = wav_List,
    label = 'VCTK',
    )
