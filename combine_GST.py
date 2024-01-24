import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default = '.')
    args = parser.parse_args()  
    
    title = ''
    texts = []

    dir_name = args.dir.strip('/').split('/')
    if dir_name[-1] == 'GST':
        dir_name = dir_name[-2]
    else:
        dir_name = dir_name[-1]

    for root, _, files in os.walk(args.dir):
        for file in files:
            if file == 'combined_gst.TXT' or file == f'{dir_name}.TXT':
                continue
            with open(os.path.join(root, file)) as f:
                text = f.readlines()
                if 'VCTK' in text or 'LJ' in text:
                    continue
                title = text.pop(0)
                texts += text
    texts = [title] + texts


    filename = os.path.join(args.dir, f'{dir_name}.TXT')
    with open(filename, 'w') as f:
        f.write(''.join(texts))
    print(f'Combined GST saved at {filename}')

    
