from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from train import MotionTextDataset

split = "val"
CHUNK_SIZE = 1000
CHUNK_LOCATION = Path(f'data/chunks/{split}')
CHUNK_LOCATION.mkdir(parents=True, exist_ok=True)



def chunk_and_save(data, filename_key, data_key):
        save_motion_filename = CHUNK_LOCATION / Path("/".join(data[filename_key].parts[-3:]))
        save_motion_filename.parent.mkdir(parents=True, exist_ok=True)
        suffixes = save_motion_filename.suffixes

        while save_motion_filename.suffix:
            save_motion_filename = save_motion_filename.with_suffix("") 

        motion_data = data[data_key].split(CHUNK_SIZE, dim=0)
        
        for i, data_m in enumerate(motion_data):
            pd.DataFrame(data_m.numpy()).to_pickle(save_motion_filename.with_suffix(f".{i}{''.join(suffixes)}"))
            


def main():
    train_dataset = MotionTextDataset(f'data/genea2023/{split}', agents=['main-agent'])
    for i, data in tqdm(enumerate(train_dataset), total=len(train_dataset)):
        chunk_and_save(data, 'motion_filename', 'motion')
        chunk_and_save(data, 'text_filename', 'text')
        chunk_and_save(data, 'audio_filename', 'audio') 

        


    


if __name__ == '__main__':
    main()