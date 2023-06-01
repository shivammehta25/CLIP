from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from clip.model import CLIP

meta_data_structure = {
    'agent': None,
    'motion_filename': None,
    'motion': None,
    'original_text': None,
    'text_filename': None,
    'text': None,
    'audio': None,
    'audio_filename': None,
}

def remove_none(d):
    return {k:v for k,v in d.items() if v is not None}

class MotionTextDataset(Dataset):
    AGENTS_AVAILABLE = ['main-agent', 'interloctr']
    FILEEXT = {
        'motion': 'bvh_expmap_30fps.pkl',
        'text': 'pkl',
        'audio': 'muted.pkl'
    }
    FOLDERNAME = {
        'motion': 'diffusion_genea23_sm0_0_30fps',
        'text': 'data2vec-text',
        'audio': 'data2vec-audio'
    }

    error_files = (
        'trn_2023_v0_139_main-agent.bvh_expmap_30fps.pkl'
    )
    
    def __init__(self, data_folder, agents=None, modules_to_load=['motion', 'text', 'audio']) -> None:
        super().__init__()
        if isinstance(data_folder, str):
            data_folder = Path(data_folder)
            assert data_folder.exists(), f'Folder {data_folder} does not exist'

        if agents is None:
            agents = self.AGENTS_AVAILABLE

        self.data_folder = data_folder
        self.modules_to_load = modules_to_load

        self.data_files = []
        for agent in agents:
            for file in (self.data_folder / agent / self.FOLDERNAME['motion']).glob(f'*.{self.FILEEXT["motion"]}'):
                if file.name in self.error_files:
                    continue
                if 'mirrored' in file.name:
                    continue
                
                data_point = deepcopy(meta_data_structure)
                data_point['agent'] = agent
                file_name = file.with_suffix('').with_suffix('').name
                if 'motion' in modules_to_load:
                    data_point['motion_filename'] = file
                
                if 'text' in modules_to_load:
                    data_point['text_filename'] = self.data_folder / agent / self.FOLDERNAME['text'] / f"{file_name}.{self.FILEEXT['text']}"
                    
                if 'audio' in modules_to_load:
                    data_point['audio_filename'] = self.data_folder / agent / self.FOLDERNAME['audio'] / f"{file_name}.{self.FILEEXT['audio']}"

                self.data_files.append(remove_none(data_point))

    def __len__(self) -> int:
        return len(self.data_files)

    def load_motion(self, motion_filename):
        x = pd.read_pickle(motion_filename)
        return torch.from_numpy(x.values).float()

    def load_audio(self, audio_filename):
        x = pd.read_pickle(audio_filename)
        return torch.from_numpy(x.values).float()

    def load_text(self, text_filename):
        x = pd.read_pickle(text_filename)
        return torch.from_numpy(x.values).float()

    def ensure_same_length(self, data_point):
        def interpolate(x, n):
            """Interpolate on first dimension with F.interpolate"""
            return rearrange(F.interpolate(rearrange(x, 't c -> 1 c t'), size=n), '1 c t -> t c')

        ensure_length = None
        if 'motion' in self.modules_to_load:
            if ensure_length is None:
                ensure_length = data_point['motion'].shape[0]
            else:
                data_point['motion'] = interpolate(data_point['motion'], ensure_length)

        if 'text' in self.modules_to_load:
            if ensure_length is None:
                ensure_length = data_point['text'].shape[0]
            else:
                data_point['text'] = interpolate(data_point['text'], ensure_length)

        if 'audio' in self.modules_to_load:
            if ensure_length is None:
                ensure_length = data_point['audio'].shape[0]
            else:
                data_point['audio'] = interpolate(data_point['audio'], ensure_length)

    def __getitem__(self, index: int):
        data_point = self.data_files[index]
        if 'motion' in self.modules_to_load:
            data_point['motion'] = self.load_motion(data_point['motion_filename'])
        if 'text' in self.modules_to_load:
            data_point['text'] = self.load_text(data_point['text_filename'])
        if 'audio' in self.modules_to_load:
            data_point['audio'] = self.load_audio(data_point['audio_filename'])

        self.ensure_same_length(data_point)

        return data_point
    
def custom_collate(batch):
    """Outputs will be of shape: b t c"""
    collated_batch = defaultdict(list) 
    for data_point in batch:
        collated_batch['agent'].append(data_point['agent'])
        for k, v in data_point.items():
            if isinstance(v, str):
                collated_batch[k].append(v)
            elif isinstance(v, torch.Tensor):
                collated_batch[k].append(v)
                collated_batch[f"{k}_length"].append(v.shape[0])
        
    for k, v in collated_batch.items():
        if k.endswith('_length'):
            collated_batch[k] = torch.tensor(v)
        elif isinstance(v[0], torch.Tensor):
            collated_batch[k] = pad_sequence(v, batch_first=True)
        else:
            collated_batch[k] = v
         
    return collated_batch

class DataModule(L.LightningDataModule):
    
    def __init__(self, data_loc, batch_size: int = 32, num_worker: int = 1):
        super().__init__()
        self.data_loc = data_loc
        self.batch_size = batch_size
        self.num_workers = num_worker

    def setup(self, stage: str):
        self.train_dataset = MotionTextDataset(f'{self.data_loc}/trn', agents=['main-agent'])
        self.val_dataset = MotionTextDataset(f'{self.data_loc}/val', agents=['main-agent'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=custom_collate, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=custom_collate, num_workers=self.num_workers)


class ClipModel(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.model = nn.Linear(512, 2)
        
    def forward(self, batch):
        return self.model(batch['text'], batch['motion'])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.02)





def main(args):
    data_module = DataModule(args.data_dir, batch_size=args.batch_size, num_worker=args.num_workers)
    model = ClipModel(hparams={})
    
    trainer = L.Trainer(devices=args.gpus)
    trainer.fit(model, data_module)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument_group("data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="data/chunks")
    parser.add_argument("--gpus", nargs='+', type=int, default=[3])
    args = parser.parse_args()
    main(args) 