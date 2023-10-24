from Model import GST_Tacotron
gst_taco = GST_Tacotron(is_Training=False)

gst_taco.Restore()

language_List = ['japanese', 'english', 'german', 'portuguese', 'polish', 'french', 'turkish', 'mandarin']
language_Number = [27, 579, 36, 48, 34, 63, 37, 65]

# ban_list = ['turkish', 'english']
allow_list = ['english', 'mandarin']

wav_List = []
tag_List = []
for idx, l in enumerate(language_List):
    if not l in allow_list:
        continue
    wav_List += [
        '../accented_speech_archive/wavs/' + l + str(i) + '.mp3.wav' for i in range(1, language_Number[idx])]

    tag_List += [
        l for i in range(1, language_Number[idx])]

print(wav_List)
print(tag_List)

gst_taco.Inference_GST(wav_List, tag_List)
