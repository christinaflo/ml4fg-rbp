import click
import json
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelSummary, EarlyStopping, ModelCheckpoint

from model import RNAModel
from rbp_dataset import CLIPDataset


class RNARunner(pl.LightningModule):
    def __init__(self, config):
        super(RNARunner, self).__init__()
        self.save_hyperparameters(config)

        self.model = RNAModel(self.hparams["dim"], self.hparams["include_ss"], self.hparams["ss_separate"],
                              self.hparams.get("include_graph", False))
        self.criterion = nn.BCEWithLogitsLoss()
        self.valid_auc = torchmetrics.AUROC(task='binary', pos_label=1)
        self.test_auc = torchmetrics.AUROC(task='binary', pos_label=1)

    def forward(self, batch):
        return self.model(*batch)

    def step(self, batch, stage):
        y, x = batch
        y_true = y.to(dtype=torch.float32)

        pred_y = self.forward(x)

        loss = self.criterion(pred_y, y_true)

        self.log(f'{stage}_{self.hparams["target"]}_loss_pl', loss,
                 on_step=False, on_epoch=True, logger=True)

        if stage == 'val':
            self.valid_auc.update(pred_y.cpu().detach(), y.cpu().detach())
            self.log(f'{stage}_{self.hparams["target"]}_aucroc_pl', self.valid_auc,
                         on_step=False, on_epoch=True, logger=True)
        elif stage == 'test':
            self.test_auc.update(pred_y.cpu().detach(), y.cpu().detach())
            self.log(f'{stage}_{self.hparams["target"]}_aucroc_pl', self.test_auc,
                         on_step=False, on_epoch=True, logger=True)

        return loss

    def training_step(self, train_batch, batch_idx):
        stage = 'train'
        loss = self.step(train_batch, stage)
        return loss

    def validation_step(self, val_batch, batch_idx):
        stage = 'val'
        self.step(val_batch, stage)

    def test_step(self, test_batch, batch_idx):
        stage = 'test'
        self.step(test_batch, stage)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.hparams['lr'])
        t_max = self.trainer.max_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'epoch'
        }
        return [optimizer], [lr_scheduler]


class RNABatchCollator:
    def __call__(self, batch):
        pad_seq = lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)

        seq = pad_seq([data.seq for data in batch])
        struct = pad_seq([data.struct for data in batch])
        label = pad_seq([data.label for data in batch])
        mask = pad_seq([data.seq.new_ones(data.seq.shape[0], dtype=torch.float32) for data in batch])

        edge_attr = None
        if 'edge_attr' in batch[0].edge_attrs():
            edge_attr = torch.stack([F.pad(data.edge_attr.unsqueeze(-1),
                               pad=(0, seq.size(1) - data.edge_attr.size(0), 0, seq.size(1) - data.edge_attr.size(0)))
                         for data in batch]).to(dtype=torch.float32)

        return label, (seq, struct, edge_attr, mask)


class RNADataModule(pl.LightningDataModule):
    def __init__(self, config, num_devices):
        super().__init__()
        self.config = config
        self.num_devices = max(num_devices, 1)

    @property
    def effective_batch_size(self) -> int:
        return self.config['batch_size'] // self.num_devices

    def setup(self, stage: Optional[str] = None):
        train_val_dataset = CLIPDataset(root='./data/CLIP', exp_id=self.config['exp_id'], is_training=True,
                                        include_graph=self.config.get('include_graph', False))
        test_dataset = CLIPDataset(root='./data/CLIP', exp_id=self.config['exp_id'], is_training=False,
                                   include_graph=self.config.get('include_graph', False))

        self.train_data, self.val_data = train_val_dataset.train_val_split(ideeps_split=self.config["ideeps_split"])
        self.test_data = test_dataset.shuffle()

        self.batch_collator = RNABatchCollator()

    def train_dataloader(self):
        return DataLoader(self.train_data, self.config['batch_size'], shuffle=True,
                          collate_fn=self.batch_collator, num_workers=self.config['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_data, self.config['batch_size'], shuffle=False,
                          collate_fn=self.batch_collator, num_workers=self.config['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test_data, self.config['batch_size'], shuffle=False,
                          collate_fn=self.batch_collator, num_workers=self.config['num_workers'])


@click.group()
def cli():
    pass


@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path: str) -> None:
    with open(config_path) as f:
        config = json.load(f)

    wandb_logger = WandbLogger(name=config['name'], project=config['wandb_project'])

    # Set random seeds
    pl.seed_everything(config['seed'], workers=True)

    # Create dataset and configuration sets
    data_module = RNADataModule(config=config, num_devices=torch.cuda.device_count())
    data_module.prepare_data()
    data_module.setup("fit")

    # train
    model = RNARunner(config)

    checkpoint_callback = ModelCheckpoint(
        monitor=f'val_{config["target"]}_loss_pl',
        filename=config["name"] + "-{epoch}",
        dirpath=f'checkpoints/{config["target"]}',
        save_top_k=1,
        mode="min",
    )

    model_summary = ModelSummary(max_depth=5)
    early_stopping = EarlyStopping(monitor=f'val_{config["target"]}_loss_pl', mode="min", patience=15)

    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        gpus=torch.cuda.device_count(),
        logger=[wandb_logger],
        callbacks=[ model_summary, early_stopping]
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


@cli.command()
@click.argument('load_path', type=click.Path(exists=True))
@click.option('--save-dir', default='outputs', type=click.Path())
def test(load_path: str, save_dir: str) -> None:
    model = RNARunner.load_from_checkpoint(load_path)
    config = model.hparams

    # Set random seeds
    pl.seed_everything(config['seed'], workers=True)

    csv_logger = CSVLogger(save_dir, name=config['name'])

    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         logger=csv_logger)

    # Create dataset and configuration sets
    data_module = RNADataModule(config=config, num_devices=torch.cuda.device_count())
    data_module.prepare_data()
    data_module.setup("test")

    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    cli()
