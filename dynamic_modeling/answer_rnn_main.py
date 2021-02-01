import os
import json
import torch
import getpass
import datetime
import argparse
from dotmap import DotMap
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from agents.answer_rnn import RNNAgent
from dataset import TrajTutorState, tutor_collate_fn

DATA_DIR = '/data/anie/offline_rl/data/'
OUT_DIR = '/data/anie/offline_rl_dynamics/'

def tags_from_args(args):
    args = vars(args)
    tags = []
    ks = sorted(list(args.keys()))
    for k in ks:
        if k == 'tags':
            continue
        v = args[k]
        if type(v) == bool:
            if v: tags.append(k)
        else:
            tags.append(v)

    tags.extend(args['tags'])

    return tags

def get_date():
    date = datetime.date.today().strftime('%b%d').lower()
    return date

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help='path to config file')
    parser.add_argument("--tags", nargs='+', type=str,  help='optional descriptive tags for this run', default=[])
    parser.add_argument("--gpu-device", type=int, default=0)
    args = parser.parse_args()
    return args

def setup(args):
    with open(args.config) as fp:
        config = json.load(fp)

    config = DotMap(config)

    train_ds = TrajTutorState(
        os.path.join(
            DATA_DIR,
            #config.dataset.folder,
            config.dataset.train_split,
        ),
        os.path.join(DATA_DIR, 'state_ids.json'),
    )
    val_ds = TrajTutorState(
        os.path.join(
            DATA_DIR,
            #config.dataset.folder,
            config.dataset.valid_split,
        ),
        os.path.join(DATA_DIR, 'state_ids.json')
    )

    config.dataset.feat_dim = train_ds.dset_args()['feat_dim']
    config.dataset.num_states = 481
    config.dataset.num_actions = 3
    return config, train_ds, val_ds


def create_dataloader(dataset, dl_config, shuffle=True):
    loader = DataLoader(
        dataset,
        batch_size=dl_config.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=False,
        collate_fn=tutor_collate_fn,
        num_workers=dl_config.num_workers,
    )
    return loader


def main(args):
    config, train_ds, val_ds = setup(args)
    train_dl = create_dataloader(train_ds, config.dataloader)
    val_dl = create_dataloader(val_ds, config.dataloader, shuffle=False)

    tags = tags_from_args(args)
    tags = [str(t) for t in tags]
    name = ':'.join(tags)
    config.pprint()

    AgentClass = globals()[config.agent]
    agent = AgentClass(config)

    save_dir = os.path.join(OUT_DIR, get_date())
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    wandb_logger = WandbLogger(
        project='tutorial-answer',
        entity='windweller',
        save_dir=save_dir,
        tags=tags,
        name=name,
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        weights_summary='full',
        max_epochs=config.epochs,
        gpus=[int(args.gpu_device)],
    )
    trainer.fit(
        agent,
        train_dataloader=train_dl,
        val_dataloaders=val_dl,
    )

if __name__ == '__main__':
    torch.manual_seed(42)
    args = parse_args()
    main(args)
