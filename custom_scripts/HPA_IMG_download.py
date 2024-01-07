import os
import gzip
import time
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from io import BytesIO
from multiprocessing.pool import Pool

from PIL import Image
import imageio

tqdm.pandas()

def get_ID(url):
    fn = '_'.join(url.split('/')[-2:])
    ID = '_'.join(fn.split('_')[:4])
    return ID

def download_tif_from_df(url, save_dir, overwrite):
    fn = get_ID(url)+'.tiff'
    if os.path.exists(os.path.join(save_dir, fn)) and not overwrite:
        print(f"Image {fn} already exists. Skipping...")
    else: 
        print(f'downloading {fn}...')
        try:
            image = []
            for channel in ['blue','green','red','yellow']:
                img_file = url.replace('_blue_red_green.jpg', f'_{channel}.tif.gz')
                print(img_file)
                r = requests.get(img_file)
                tf = gzip.open(BytesIO(r.content)).read()
                im = imageio.imread(tf, 'tiff', is_ome=False)
                image.append(im)
            image = np.stack(image).T
            imageio.imwrite(os.path.join(save_dir, fn), image)
            return image
        except Exception as e:
            print(f'{e}')
            print(f'{url} broke...')

def download_png_from_df(url, save_dir, overwrite):
    fn = get_ID(url)+'.png'
    if os.path.exists(os.path.join(save_dir, fn)) and not overwrite:
        print(f"Image {fn} already exists. Skipping...")
    else: 
        print(f'downloading {fn}...')
        try:
            image = []
            for channel in ['blue','green','red','yellow']:
                img_file = url.replace('_blue_red_green', f'_{channel}')
                r = requests.get(img_file)
                im = Image.open(BytesIO(r.content))
                im = im.convert('L')
                image.append(im)
            image = Image.merge('RGBA', image)
            image.save(os.path.join(save_dir, fn), 'png')
        except Exception as e:
            print(f'{e}')
            print(f'{url} broke...')

def download_from_df(pid, img_dict, save_dir, overwrite, ftype='jpg'):
    if ftype == 'jpg': img_dict.img_file.progress_apply(download_png_from_df, args=[save_dir, overwrite])
    elif ftype == 'tiff': img_dict.img_file.progress_apply(download_tif_from_df, args=[save_dir, overwrite])
    else: 
        print(f"file-type {ftype} not implemented.")
        sys.exit() 

if __name__ == '__main__':
    t00 = time.time()
    # parameters:
    parser = argparse.ArgumentParser()

    parser.add_argument('input_csv', type=str, help='Specify path to image dictionary pickle file, e.g. img_dict.pkl')
    parser.add_argument('save_dir', type=str, help='Specify path output directory')
    parser.add_argument('-f', '--ftype', type=str, default='jpg', choices=['jpg', 'tiff'], help='Specify the image-file type you wish to download.')
    parser.add_argument('-w', '--workers', type=int, default=1, help='Specify number of workers (default: 1)')
    parser.add_argument('-o', '--overwrite', type=bool, default=False, help='Overwrite output dir and files if dir exists?')
    args = parser.parse_args()

    # create output directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    print(f'Downloading images to {args.save_dir}...')

    print('Parent process %s.' % os.getpid())
    d = pd.read_csv(args.input_csv)
    d = d.iloc[:10,:] # activate for testing
    list_len = len(d)

    p = Pool(args.workers)
    jump = int(list_len / args.workers)
    jumps = [[0, jump]]
    for w in range(1, args.workers):
        jumps.append((jumps[-1][-1], jumps[-1][1] + jump))
    for i in range(args.workers):
        start = jumps[i][0]
        end = jumps[i][1]
        subset_to_download = d.iloc[start:end]

        p.apply_async(
            download_from_df, args=(str(i), subset_to_download, args.save_dir, args.overwrite, args.ftype)
        )
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    print(f"Total execution time: {time.time() - t00}")
    print(f"Time-per-file: {(time.time() - t00)/list_len}")