import argparse

import lightning as L
from lightning.pytorch.callbacks import (DeviceStatsMonitor, EarlyStopping,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import TensorBoardLogger

from clip.lightning_module import ClipModel, DataModule


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
    
    tb_logger = TensorBoardLogger(save_dir=args.logdir, name=args.run_name,
                                  sub_dir=args.subdir if hasattr(args, 'subdir') else None)
    print(args.gpus, type(args.gpus))
    trainer = L.Trainer(
        devices=args.gpus,
        log_every_n_steps=10,
        callbacks=[
            ModelCheckpoint(monitor="val_loss"),
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            DeviceStatsMonitor(cpu_stats=False)
        ],
        logger=tb_logger,
        precision='bf16-mixed',
        max_epochs=100,
        )
    trainer.fit(model, data_module, args.checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs='+', type=int, default=[3])
    parser.add_argument("--run_name", type=str, default="HPTuning")
    parser.add_argument("--logdir", type=str, default="lightning_logs")
    parser.add_argument("--checkpoint_path", "-c", type=str, default=None, help="path to checkpoint to resume training")
    parser = DataModule.add_argparse_args(parser)
    parser = ClipModel.add_argparse_args(parser)
    args = parser.parse_args()
    
    
    
    for lr in [1e-6, 5e-6, 3e-6]: 
        for transformer_width in [384, 512, 768]:
            args.transformer_width = transformer_width
            args.lr = lr
            args.subdir = f"lr_{lr}_transformer_width_{transformer_width}"
            print(args)
            main(args) 