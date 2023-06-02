from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from clip.lightning_module import MotionTextDataset


def chunk_and_save(data, filename_key, data_key, chunk_location, chunk_size, sliding_window_size=None):
        save_motion_filename = chunk_location / Path("/".join(data[filename_key].parts[-3:]))
        save_motion_filename.parent.mkdir(parents=True, exist_ok=True)
        suffixes = save_motion_filename.suffixes
        if sliding_window_size is None:
            sliding_window_size = chunk_size

        while save_motion_filename.suffix:
            save_motion_filename = save_motion_filename.with_suffix("") 

        motion_data = data[data_key]
        # pad with zeros to make window slide over all the data
        motion_data = F.pad(motion_data, (0, 0, 0, sliding_window_size - motion_data.shape[0] % sliding_window_size))
        motion_data = motion_data.unfold(0, chunk_size, sliding_window_size).transpose(1, 2)
       
        
        for i, data_m in enumerate(motion_data):
            pd.DataFrame(data_m.numpy()).to_pickle(save_motion_filename.with_suffix(f".{i}{''.join(suffixes)}"))
            


def main():
    split = "trn"
    CHUNK_SIZE = 500
    WINDOW_SIZE = 250
    CHUNK_LOCATION = Path(f'data/chunks/{split}')
    CHUNK_LOCATION.mkdir(parents=True, exist_ok=True)

    train_dataset = MotionTextDataset(f'data/genea2023/{split}')
    for i, data in tqdm(enumerate(train_dataset), total=len(train_dataset)):
        chunk_and_save(data, 'motion_filename', 'motion', CHUNK_LOCATION, CHUNK_SIZE)
        chunk_and_save(data, 'text_filename', 'text', CHUNK_LOCATION, CHUNK_SIZE)
        chunk_and_save(data, 'audio_filename', 'audio', CHUNK_LOCATION, CHUNK_SIZE) 
        
    split = "val"    
    CHUNK_LOCATION = Path(f'data/chunks/{split}')
    CHUNK_LOCATION.mkdir(parents=True, exist_ok=True)
    val_dataset = MotionTextDataset(f'data/genea2023/{split}')
    for i, data in tqdm(enumerate(val_dataset), total=len(val_dataset)):
        chunk_and_save(data, 'motion_filename', 'motion', CHUNK_LOCATION, CHUNK_SIZE)
        chunk_and_save(data, 'text_filename', 'text', CHUNK_LOCATION, CHUNK_SIZE)
        chunk_and_save(data, 'audio_filename', 'audio', CHUNK_LOCATION, CHUNK_SIZE) 

        


    


if __name__ == '__main__':
    main()