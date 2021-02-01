"""
We loop around the dataset, fake a training_step
but use the same validation loop
"""

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

from agents.state_mlp import MLPAgent
from dataset import MarkovTutorStates

from collections import defaultdict
import numpy as np

import sklearn.metrics as skm
from tqdm import tqdm

DATA_DIR = '/data/anie/offline_rl/data/'
OUT_DIR = '/data/anie/offline_rl_dynamics/'

class MarkovAgent(object):
    def __init__(self, train_data):
        self.data = train_data
        self.tabular_dic = {}
        self.next_state_prob = {}

        self.state_num = 481
        self.state_vocab = train_data.state_vocab

        self.val_state_not_in_train = 0

    def add_state(self, state):
        if state not in self.tabular_dic:
            self.tabular_dic[state] = defaultdict(int)

    def train(self):
        for i, tup in tqdm(enumerate(self.data)):
            state, next_state = tup['state'], tup['targets']
            self.add_state(state)
            self.tabular_dic[state][next_state] += 1

        print("Computing probabilities...")

        for state in tqdm(self.tabular_dic.keys()):
            self.next_state_prob[state] = {}
            total = sum(self.tabular_dic[state].values())
            for n_s in self.tabular_dic[state]:
                self.next_state_prob[state][n_s] = self.tabular_dic[state][n_s] / total

        print("Training finished")

    def get_true_state_np(self, state):
        a = np.zeros(self.state_num)
        a[state] = 1.
        return a

    def get_pred_state_np(self, dist_dic):
        a = np.zeros(self.state_num)
        for state, prob in dist_dic.items():
            a[state] = prob
        return a

    def pred_next_state(self, state):
        if state not in self.next_state_prob:
            self.val_state_not_in_train += 1
            return np.zeros(self.state_num)

        dist_dic = self.next_state_prob[state]
        pred = self.get_pred_state_np(dist_dic)
        return pred

    def evaluate(self, valid_data):
        pred_probs, y_probs = [], []
        for i, tup in enumerate(valid_data):
            state, next_state = tup['state'], tup['targets']
            pred_prob = self.pred_next_state(state)
            y = self.get_true_state_np(next_state)
            pred_probs.append(pred_prob)
            y_probs.append(y)

        pred_probs = np.vstack(pred_probs)
        y_probs = np.vstack(y_probs)

        preds = np.argmax(pred_probs, axis=1)
        ys = np.argmax(y_probs, axis=1
                       )
        acc = skm.accuracy_score(ys, preds)
        xent = skm.log_loss(y_probs, pred_probs)

        print(f"Number of val states not found in train: {self.val_state_not_in_train}")
        print(f"Validation acc: {acc}, cross entropy: {xent}")

def setup(args):
    with open(args.config) as fp:
        config = json.load(fp)

    config = DotMap(config)

    train_ds = MarkovTutorStates(
        os.path.join(
            DATA_DIR,
            #config.dataset.folder,
            config.dataset.train_split,
        ),
        os.path.join(DATA_DIR, 'state_ids.json'),
        include_correctness=config.dataset.include_correctness
    )
    val_ds = MarkovTutorStates(
        os.path.join(
            DATA_DIR,
            #config.dataset.folder,
            config.dataset.valid_split,
        ),
        os.path.join(DATA_DIR, 'state_ids.json'),
        include_correctness=config.dataset.include_correctness
    )

    config.dataset.feat_dim = train_ds.dset_args()['feat_dim']
    config.dataset.num_states = 481
    config.dataset.num_actions = 3
    return config, train_ds, val_ds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help='path to config file')
    args = parser.parse_args()
    return args

def main(args):
    config, train_ds, val_ds = setup(args)
    agent = MarkovAgent(train_ds)
    agent.train()
    agent.evaluate(val_ds)

if __name__ == '__main__':
    torch.manual_seed(42)
    args = parse_args()
    main(args)

