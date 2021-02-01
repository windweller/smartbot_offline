import os
import json
import torch
import getpass
import datetime
import argparse
from dotmap import DotMap
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger

from agents.joint_rnn import RNNAgent
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

def setup(args, student_id):
    with open(args.config) as fp:
        config = json.load(fp)

    config = DotMap(config)
    assert config.dataset.pred_target in {'nlg_score', 'post_test'}
    assert "{}" in config.dataset.train_split

    print(student_id)

    config.dataset.train_split = config.dataset.train_split.format(student_id)
    config.dataset.valid_split = config.dataset.valid_split.format(student_id)

    if not config.dataset.pred_target:
        raise Exception("can't be empty")

    train_ds = TrajTutorState(
        os.path.join(
            DATA_DIR,
            # config.dataset.folder,
            config.dataset.train_split,
        ),
        os.path.join(DATA_DIR, 'state_ids.json'),
        os.path.join(DATA_DIR, "nlg_score.csv"),
        pred_target=config.dataset.pred_target
    )
    val_ds = TrajTutorState(
        os.path.join(
            DATA_DIR,
            # config.dataset.folder,
            config.dataset.valid_split,
        ),
        os.path.join(DATA_DIR, 'state_ids.json'),
        os.path.join(DATA_DIR, "nlg_score.csv"),
        pred_target=config.dataset.pred_target
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
    train_students = json.load(open(os.path.join(DATA_DIR, "train_students_ids.json")))
    test_students = json.load(open(os.path.join(DATA_DIR, "test_students_ids.json")))

    students_ids = train_students + test_students

    for student_id in students_ids:

        config, train_ds, val_ds = setup(args, student_id)
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

        csv_logger = CSVLogger(save_dir=os.path.join(OUT_DIR, "leave_one_out_cv", student_id))
        experiment = csv_logger.experiment
        experiment.log = experiment.log_metrics  # create a function alias, so we don't have to change other things

        trainer = pl.Trainer(
            logger=csv_logger,
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
