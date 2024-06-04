import math
import pickle
from textgrids import TextGrid
from tqdm import tqdm
import os

db_dir = os.path.expanduser('~/phoneme_librispeech/train-clean-360')

phone_count = {}    

frame_length = 10
sr = 16000
frame_time = frame_length / sr


for root, _, files in os.walk(db_dir):
    for file in tqdm(files):
        if not file.endswith("TextGrid"):
                continue
        textgrid_path = os.path.join(root, file)
        tg = TextGrid()
        tg.read(textgrid_path)
        phs = tg['phones']

        for ph in phs:
            if ph.text == '':
                continue
            elif ph.text == 'sp' or ph.text == 'sil':
                continue
            phone_number = math.floor((ph.xmax - ph.xmin) / frame_time)

            if phone_number != 0:
                if not ph.text in phone_count.keys():
                    phone_count[ph.text] = phone_number
                else:
                    phone_count[ph.text] += phone_number

print(f'{len(phone_count.keys())} type of phonemes has been recodered, with a total number of {len(phone_count)}')

with open('./Pretrained/phone_counts.pkl') as f:
    pickle.dump(phone_count, f)
    print(f'Phone counts saved at {f.name}')

