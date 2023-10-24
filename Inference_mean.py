from Model_mean import GST_Tacotron
gst_taco = GST_Tacotron(is_Training=False)

gst_taco.Restore()

wav_List_for_GST = [
    '../LJSpeech-1.1/wavs/LJ009-0138.wav',
    #'../accented_speech_archive/wavs/portuguese1.mp3.wav', 
    #'../accented_speech_archive/wavs/portuguese2.mp3.wav', 
    #'../accented_speech_archive/wavs/portuguese3.mp3.wav', 
    #'../accented_speech_archive/wavs/portuguese4.mp3.wav', 
    #'../accented_speech_archive/wavs/portuguese5.mp3.wav'
    ]

sentence_List = [
    'The grass is always greener in the other side.'
    ] * len(wav_List_for_GST)

print(wav_List_for_GST)

gst_taco.Inference(
    sentence_List = sentence_List, 
    wav_List_for_GST = wav_List_for_GST,
    label = 'Result',
    )

