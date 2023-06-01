import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import grad_norm
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
        'audio': 'pkl'
    }
    FOLDERNAME = {
        'motion': 'diffusion_genea23_sm0_0_30fps',
        'text': 'data2vec-text',
        'audio': 'data2vec-audio'
    }

    error_files = (
        'trn_2023_v0_139_main-agent.bvh_expmap_30fps.pkl'
        'trn_2023_v0_181_interloctr.bvh_expmap_30fps.pkl'
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
    
    def __init__(self, data_loc, agents, modalities, batch_size: int = 32, num_worker: int = 1):
        super().__init__()
        self.save_hyperparameters()

        self.data_loc = data_loc
        self.agents = agents
        self.batch_size = batch_size
        self.num_workers = num_worker

    def setup(self, stage: str):
        self.train_dataset = MotionTextDataset(f'{self.data_loc}/trn', agents=self.agents)
        self.val_dataset = MotionTextDataset(f'{self.data_loc}/val', agents=self.agents)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=custom_collate, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=custom_collate, num_workers=self.num_workers)
    
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--agents", type=str, nargs='+', default=['main-agent', 'interloctr'], help="Agents' data to use")
        parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
        parser.add_argument("--num_workers", type=int, default=20, help="number of workers for training")
        parser.add_argument("--data_dir", type=str, default="data/chunks", help="path to data")
        parser.add_argument("--motion_dim", type=int, default=60, help="dimension of motion")
        parser.add_argument("--inputs_dim", type=int, default=768, help="dimension of motion")
        parser.add_argument("--modalities", type=str, nargs='+', default=['motion', 'text', 'audio'], help="modalities to train on, motion will be the final output") 
        return parser


class ClipModel(L.LightningModule):
    def __init__(
        self,
        args,
        inputs_dim,
        motion_dim,
        embed_dim,
        context_length,
        transformer_width,
        transformer_heads,
        transformer_layers,
        input_modalities=["audio", "text"],
    ):
        super().__init__()
        self.save_hyperparameters(args)
        self.input_modalities = input_modalities
        self.input_modalities.remove("motion")
        
        self.model = CLIP(
            inputs_dim=inputs_dim,
            motion_dim=motion_dim,
            embed_dim=embed_dim,
            context_length=context_length,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
        ) 
        self.loss = nn.CrossEntropyLoss()

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embed_dim", type=int, default=512, help="embedding dimension")
        parser.add_argument("--context_length", type=int, default=500, help="context length")
        parser.add_argument("--transformer_width", type=int, default=768, help="transformer width")
        parser.add_argument("--transformer_heads", type=int, default=8, help="transformer heads")
        parser.add_argument("--transformer_layers", type=int, default=6, help="transformer layers")
        parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
        return parser 

    def training_step(self, batch, batch_idx):
        loss = self._run_loss_computation(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._run_loss_computation(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss
    
    def _run_loss_computation_second(self, batch):
        batch_size = batch['motion'].shape[0]
        inputs = torch.stack([batch[modality] for modality in self.input_modalities]).sum(0)
        similarity_matrix = self.model(inputs, batch['motion'])
        ground_truth = torch.eye(similarity_matrix.shape[1], device=similarity_matrix.device).unsqueeze(0).expand(batch_size,-1,-1)
        total_loss = F.mse_loss(similarity_matrix,ground_truth)
        return total_loss
    
    def _run_loss_computation(self, batch):
        batch_size = batch['motion'].shape[0]
        inputs = torch.stack([batch[modality] for modality in self.input_modalities]).sum(0)
        logits_per_image, logits_per_text = self.model(inputs, batch['motion'])
        ground_truth = torch.arange(batch_size, device=logits_per_image.device)
        total_loss = (self.loss(logits_per_image,ground_truth) + self.loss(logits_per_text,ground_truth))/2
        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,weight_decay=1e-6)
    
    def on_before_optimizer_step(self, optimizer):
        self.log_dict(grad_norm(self, norm_type=2))


def main(args):
    data_module = DataModule(args.data_dir, args.agents, args.modalities, batch_size=args.batch_size, num_worker=args.num_workers)
    model = ClipModel(
        args,
        args.inputs_dim,
        args.motion_dim,
        args.embed_dim,
        args.context_length,
        args.transformer_width,
        args.transformer_heads,
        args.transformer_layers,
        input_modalities=args.modalities,
    )
    
    tb_logger = TensorBoardLogger(save_dir=args.logdir, name=args.run_name)
    print(args.gpus, type(args.gpus))
    trainer = L.Trainer(
        devices=args.gpus,
        log_every_n_steps=10,
        callbacks=[ModelCheckpoint(monitor="val_loss")],
        logger=tb_logger,
        precision='bf16-mixed',
        max_epochs=100,
        )
    trainer.fit(model, data_module, args.checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs='+', type=int, default=[3])
    parser.add_argument("--run_name", type=str, default="clip_run")
    parser.add_argument("--logdir", type=str, default="lightning_logs")
    parser.add_argument("--checkpoint_path", "-c", type=str, default=None, help="path to checkpoint to resume training")
    parser = DataModule.add_argparse_args(parser)
    parser = ClipModel.add_argparse_args(parser)
    args = parser.parse_args()
    print(args)
    main(args) 