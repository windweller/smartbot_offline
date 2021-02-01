import os
import copy
import pickle
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader


def load_data(data_path):
    print('Loading file...')
    orig = pd.read_csv(data_path)


# ==== State-model ====

# One Markov Data Loader
class MarkovTutorStates(Dataset):
    def __init__(self, data_path, state_ids_path, include_correctness=False):
        print('Loading file...')
        data = pd.read_csv(data_path)
        self.data = data

        self.include_correctness = include_correctness

        self.action_vocab = ['', 'tell', 'elicit']
        self.state_vocab = json.load(open(state_ids_path))
        self.KC_list = [1, 14, 20, 21, 22, 24, 27, 28]

        self.kc_used_headers = [f'KC_{i}_used' for i in self.KC_list]
        self.kc_pretest_headers = [f'Pre_test_KC_{i}_score' for i in self.KC_list]
        self.kc_cum_correct_headers = [f'KC{i}_correct_cum' for i in self.KC_list]
        self.kc_cum_incorrect_headers = [f'KC{i}_incorrect_cum' for i in self.KC_list]

        if include_correctness:
            self.feats = self.data[self.kc_used_headers + self.kc_pretest_headers + self.kc_cum_correct_headers + \
                                   self.kc_cum_incorrect_headers]
        else:
            self.feats = self.data[self.kc_used_headers + self.kc_pretest_headers]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # this really speaks to the advantage of having
        # a feat data matrix pre-processed
        row = self.data.iloc[index]
        # feats: {action, prev_state, KCs, pre-test KCs}
        # 8 + 8 = 16

        feats = self.feats.iloc[index]
        feats = np.array(feats)

        target = self.state_vocab.index(row['Next_state'])

        action = row['Action']
        if pd.isna(action):
            action = ''

        item = dict(action=self.action_vocab.index(action),
                    state=self.state_vocab.index(row['State_id']),
                    feats=torch.FloatTensor(feats),
                    targets=target)

        return item

    def dset_args(self):
        """
        Return a dict of args that models might need to know e.g. input dimension of features.
        This dictionary will be available in the model config.
        """
        if self.include_correctness:
            return dict(feat_dim=32)
        else:
            return dict(feat_dim=16)  # self.feat.shape[1]


# One Traj Data Loader
# TODO: add final score
# TODO: final score needs an additional loading thingy
# TODO: but adding it shouldn't be too difficult
# TODO: it's just a different prediction target
class TrajTutorState(Dataset):
    def __init__(self, data_path, state_ids_path, nlg_score_path="./data/nlg_score.csv",
                 include_correctness=False,
                 include_answer_pred=False, pred_target='post_test'):
        print('Loading file...')
        data = pd.read_csv(data_path)
        self.data = data

        # load scores
        nlg_score = pd.read_csv(nlg_score_path)
        nlg_score['Student_id'] = nlg_score['student_id'].map(lambda x: 'Exp' + x)
        self.nlg_score = nlg_score.drop(columns='student_id')
        self.pred_target = pred_target

        unique_uids = np.unique(data['Student_id'])
        self.unique_uids = unique_uids

        self.include_correctness = include_correctness

        self.action_vocab = ['', 'tell', 'elicit']
        self.state_vocab = json.load(open(state_ids_path))
        self.correctness_vocab = [0, 1, -1]  # -1 means we don't need to predict correctness
        self.KC_list = [1, 14, 20, 21, 22, 24, 27, 28]

        self.kc_used_headers = [f'KC_{i}_used' for i in self.KC_list]
        self.kc_pretest_headers = [f'Pre_test_KC_{i}_score' for i in self.KC_list]
        self.kc_cum_correct_headers = [f'KC{i}_correct_cum' for i in self.KC_list]
        self.kc_cum_incorrect_headers = [f'KC{i}_incorrect_cum' for i in self.KC_list]

        if include_correctness:
            self.feats = self.data[self.kc_used_headers + self.kc_pretest_headers + self.kc_cum_correct_headers + \
                                   self.kc_cum_incorrect_headers]
        else:
            self.feats = self.data[self.kc_used_headers]  #  + self.kc_pretest_headers

        self.feats = self.feats.to_numpy().astype(np.dtype('float32'))

    def __len__(self):
        return len(self.unique_uids)

    def __getitem__(self, index):
        user_id = self.unique_uids[index]
        df = self.data[self.data['Student_id'] == user_id]

        # we grab everything
        indices = np.where(self.data['Student_id'] == user_id)[0]
        feats = self.feats[indices]
        seq_len = feats.shape[0]

        # feats: {action, prev_state, KCs, pre-test KCs}
        # 8 + 8 = 16
        targets = df['Next_state'].apply(self.state_vocab.index)

        actions = df['Action'].fillna(value="").apply(self.action_vocab.index)
        states = df['State_id'].apply(self.state_vocab.index)

        # we add answer correctness prediction in here for the joint task
        # otherwise we can ignore this
        # add a mask, only predict under "elicit"
        # answer prediction masks
        answer_correctness = df['Correctness'].apply(self.correctness_vocab.index)

        # we take all prediction targets (whether it's NLG or raw post-test score)
        stu_id = user_id
        # kc_list - 1 is to line up with pandas row index
        perf = self.nlg_score[self.nlg_score['Student_id'] == stu_id].iloc[np.array(self.KC_list)-1]
        scores = perf[self.pred_target].to_numpy()
        pre_scores = perf['pre_test'].to_numpy()

        item = dict(user_id=user_id,
                    action=torch.LongTensor(np.asarray(actions)),
                    state=torch.LongTensor(np.asarray(states)),
                    feats=torch.FloatTensor(feats),
                    targets=torch.LongTensor(np.asarray(targets)),
                    answer_correctness=torch.LongTensor(np.asarray(answer_correctness)),
                    answer_masks=torch.LongTensor(np.asarray(actions) == self.action_vocab.index("elicit")),
                    seq_len=int(seq_len),
                    score_targets=torch.FloatTensor(scores),
                    pre_test_scores=torch.FloatTensor(pre_scores))

        return item

    def dset_args(self):
        """
        Return a dict of args that models might need to know e.g. input dimension of features.
        This dictionary will be available in the model config.
        """
        if self.include_correctness:
            return dict(feat_dim=32)
        else:
            return dict(feat_dim=self.feats.shape[1])


def pad_tensor(A, num_pad, fill=0):
    shape = A.shape
    if len(shape) > 1:
        p_shape = copy.deepcopy(list(shape))
        p_shape[0] = num_pad
        P = torch.zeros(*p_shape) + fill
    else:
        P = torch.zeros(num_pad) + fill
    A = torch.cat([A, P], dim=0)
    return A


def tutor_collate_fn(batch):
    u_ids = [b['user_id'] for b in batch]
    seq_lens = [b['seq_len'] for b in batch]
    max_seq_len = max(seq_lens)

    seq_lens = torch.LongTensor(seq_lens)

    actions = []
    states = []
    feats = []
    masks = []
    targets = []
    answers = []
    answer_masks = []
    score_targets = []
    pre_test_scores = []

    for row in batch:
        seq_len_i = row['seq_len']
        pad_len = max_seq_len - seq_len_i
        mask_i = [1 for _ in range(seq_len_i)] + [0 for _ in range(pad_len)]
        mask_i = torch.LongTensor(mask_i)

        action_i = pad_tensor(row['action'], pad_len)
        state_i = pad_tensor(row['state'], pad_len)
        feats_i = pad_tensor(row['feats'], pad_len)
        targets_i = pad_tensor(row['targets'], pad_len)
        answer_corr_i = pad_tensor(row['answer_correctness'], pad_len)
        answer_mask_i = pad_tensor(row['answer_masks'], pad_len)  # this only works if fill=0, otherwise error

        actions.append(action_i)
        states.append(state_i)
        feats.append(feats_i)
        targets.append(targets_i)
        masks.append(mask_i)
        answers.append(answer_corr_i)
        answer_masks.append(answer_mask_i)
        score_targets.append(row['score_targets'])
        pre_test_scores.append(row['pre_test_scores'])

    score_targets = torch.stack(score_targets)
    pre_test_scores = torch.stack(pre_test_scores)

    actions = torch.stack(actions)
    states = torch.stack(states)
    targets = torch.stack(targets)
    feats = torch.stack(feats)
    masks = torch.stack(masks)
    answers = torch.stack(answers)
    answer_masks = torch.stack(answer_masks)

    batch = dict(user_id=u_ids,
                 action=actions,
                 state=states,
                 feats=feats,
                 targets=targets,
                 seq_len=seq_lens,
                 masks=masks,
                 answers=answers,
                 action_masks=answer_masks,
                 score_targets=score_targets,
                 pre_test_scores=pre_test_scores)

    return batch

# ===== Correctness Tracking =====

# TODO: add an option to filter out cases like 'tell' or no action

class MarkovTutorAnswers(Dataset):
    def __init__(self, data_path, state_ids_path):
        print('Loading file...')
        data = pd.read_csv(data_path)
        self.data = data

        self.action_vocab = ['', 'tell', 'elicit']
        self.state_vocab = json.load(open(state_ids_path))
        self.KC_list = [1, 14, 20, 21, 22, 24, 27, 28]
        self.correctness_vocab = [0, 1, -1]  # -1 means we don't need to predict correctness

        self.kc_used_headers = [f'KC_{i}_used' for i in self.KC_list]
        self.kc_pretest_headers = [f'Pre_test_KC_{i}_score' for i in self.KC_list]
        self.kc_cum_correct_headers = [f'KC{i}_correct_cum' for i in self.KC_list]
        self.kc_cum_incorrect_headers = [f'KC{i}_incorrect_cum' for i in self.KC_list]

        self.feats = self.data[self.kc_used_headers + self.kc_pretest_headers + self.kc_cum_correct_headers + \
                               self.kc_cum_incorrect_headers]

        # TODO: we could use a mask here too...but too complicated and not necessary

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # this really speaks to the advantage of having
        # a feat data matrix pre-processed
        row = self.data.iloc[index]
        # feats: {action, prev_state, KCs, pre-test KCs}
        # 8 + 8 = 16

        feats = self.feats.iloc[index]
        feats = np.array(feats)

        target = self.correctness_vocab.index(int(row['Correctness']))

        action = row['Action']
        if pd.isna(action):
            action = ''

        item = dict(action=self.action_vocab.index(action),
                    state=self.state_vocab.index(row['State_id']),
                    feats=torch.FloatTensor(feats),
                    action_masks=action == 'elicit',  # elicit!
                    targets=target)

        return item

    def dset_args(self):
        """
        Return a dict of args that models might need to know e.g. input dimension of features.
        This dictionary will be available in the model config.
        """
        return dict(feat_dim=32)


if __name__ == '__main__':
    DATA_DIR = '/data/anie/offline_rl/data/'

    print("========== State Dataset =========")
    dset = MarkovTutorStates(os.path.join(DATA_DIR, 'dynamics_dataset_train.csv'),
                             os.path.join(DATA_DIR, 'state_ids.json'))
    dl = DataLoader(dset, batch_size=8)
    for dp in dl:
        print(dp)
        break

    print("========== Answer Dataset =========")
    dset = MarkovTutorAnswers(os.path.join(DATA_DIR, 'dynamics_dataset_train.csv'),
                              os.path.join(DATA_DIR, 'state_ids.json'))
    dl = DataLoader(dset, batch_size=8)
    for dp in dl:
        print(dp)
        break
