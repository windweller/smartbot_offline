import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from os.path import join as pjoin

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm, t

from smartbot_offline.offline_pg import set_random_seed, MLPPolicy, wis_ope, clipped_is_ope

class BCDataset(Dataset):
    def __init__(self, df, model_round_as_feature=False):
        # df: raw pandas df
        self.df = df
        self.model_round_as_feature = model_round_as_feature
        # preprocess a little
        # self.df['postive_feedback'] = df['postive_feedback'].apply(lambda x: 0 if x == 'None' else 1)
        # array(['None', 'True'], dtype=object)
        # True -> 1, None -> 0
        # self.df['stage'].fillna(5, inplace=True)
        # the nan stage is actually just the end of final stage

        feature_names = ['hr_state', 'sysbp_state', 'glucose_state',
                         'antibiotic_state', 'vaso_state', 'vent_state']
        # 'grade_norm' and 'pre' are relatively categorical/discrete
        # unused_features = ['input_message_kid', 'time_stored', 'grade']
        # categorical_features = ['stage']  # postive_feedback

        # feature_names = ['grade_norm', 'pre-score_norm', 'stage_norm', 'failed_attempts_norm',
        #                 'pos_norm', 'neg_norm', 'hel_norm', 'anxiety_norm']

        feature_names = feature_names  # + categorical_features
        self.feature_names = feature_names

        if model_round_as_feature:
            self.feature_names += ['model_round']

        self.target_names = ["beh_p_0", "beh_p_1", "beh_p_2", "beh_p_3", "beh_p_4", "beh_p_5", "beh_p_6", "beh_p_7"]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        features = row[self.feature_names]
        targets = row[self.target_names]

        features = features.to_numpy().astype(float)
        targets = targets.to_numpy().astype(float)

        return {'features': torch.from_numpy(features).float(),
                'targets': torch.from_numpy(targets).float()}

feature_names = ['hr_state', 'sysbp_state', 'glucose_state', 'antibiotic_state', 'vaso_state', 'vent_state']
target_names = ["beh_p_0", "beh_p_1", "beh_p_2", "beh_p_3", "beh_p_4", "beh_p_5", "beh_p_6", "beh_p_7"]

MAX_TIME = 20

def compute_is_weights_for_nn_policy(behavior_df, eval_policy, eps=0.05, temp=0.1,
                                     reward_column='Reward', no_grad=True,
                                     gr_safety_thresh=0.0, is_train=True, normalize_reward=False, use_knn=False,
                                     model_round_as_feature=False):
    # is_weights with batch processing
    df = behavior_df
    user_ids = df['Trajectory'].unique()
    n = len(user_ids)

    # now max_time is dynamic, finally!!
    MAX_TIME = max(behavior_df.groupby('Trajectory').size())

    assert reward_column in ['Reward']

    pies = torch.zeros((n, MAX_TIME))  # originally 20
    pibs = torch.zeros((n, MAX_TIME))
    rewards = torch.zeros((n, MAX_TIME))
    lengths = np.zeros((n))  # we are not doing anything to this

    # compute train split reward mean and std
    # (n,): and not average
    user_rewards = df.groupby("Trajectory")[reward_column].mean()
    train_reward_mu = user_rewards.mean()
    train_reward_std = user_rewards.std()

    for idx, user_id in enumerate(user_ids):
        data = df[df['Trajectory'] == user_id]
        # get features, targets
        if not model_round_as_feature:
            features = np.asarray(data[feature_names]).astype(float)
        else:
            raise Exception("No model round in this dataset")
            # features = np.asarray(data[feature_names + ['model_round']]).astype(float)
        targets = np.asarray(data[target_names]).astype(float)
        actions = np.asarray(data['Action_taken']).astype(int)

        length = features.shape[0]
        lengths[idx] = length

        T = targets.shape[0]
        # shape: (1, T)
        beh_probs = torch.from_numpy(np.array([targets[i, a] for i, a in enumerate(actions)])).float()
        pibs[idx, :T] = beh_probs

        gr_mask = None
        if gr_safety_thresh > 0:
            if not is_train:
                # we only use KNN during validation
                if not use_knn:
                    gr_mask = None
                else:
                    raise Exception("not implemented")
                    # knn_targets = np.asarray(data[knn_target_names]).astype(float)
                    # assert knn_targets.shape[0] == targets.shape[0]
                    # beh_action_probs = torch.from_numpy(knn_targets)
                    #
                    # gr_mask = beh_action_probs >= gr_safety_thresh
            else:
                beh_action_probs = torch.from_numpy(targets)

                # gr_mask = beh_probs >= gr_safety_thresh
                gr_mask = beh_action_probs >= gr_safety_thresh

            # need to renormalize behavior policy as well?

        # assign rewards (note adjusted_score we only assign to last step)
        # reward, we assign to all
        if reward_column == 'Reward':
            reward = np.asarray(data[reward_column])[-1]
            # only normalize reward during training
            if normalize_reward and is_train:
                # might not be the best -- we could just do a plain shift instead
                # like -1 shift
                reward = (reward - train_reward_mu) / train_reward_std
            rewards[idx, T - 1] = reward
        else:
            # normal reward
            # rewards[idx, :T] = torch.from_numpy(np.asarray(data[reward_column])).float()
            raise Exception("We currrently do not offer training in this mode")

        # last thing: model prediction
        eval_action_probs = eval_policy.get_action_probability(torch.from_numpy(features).float(), no_grad,
                                                               action_mask=gr_mask)

        pies[idx, :T] = torch.hstack([eval_action_probs[i, a] for i, a in enumerate(actions)])

    return pibs, pies, rewards, lengths.astype(int), MAX_TIME

def step(policy, batch, mse_loss=False, batch_mean=True):
    features = batch['features']
    targets = batch['targets']

    # logp: [batch_size, prob_per_action]
    logp = policy(features)
    if batch_mean:
        loss = -torch.mean(torch.sum(targets * logp, dim=-1))
    else:
        loss = torch.sum(targets * logp, dim=-1)

    return loss

def evaluate(policy, val_dataloader, batch_mean=True):
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            val_loss = step(policy, batch, batch_mean=batch_mean)
            if batch_mean:
                losses.append(val_loss.detach().item())
            else:
                losses.append(val_loss.detach().numpy())

    if batch_mean:
        return np.mean(losses)
    else:
        loss_np = np.concatenate(losses)
        return - np.mean(loss_np)

def bc_train_policy(policy, train_df, valid_df, lr=1e-4, epochs=10, verbose=True, early_stop=False,
                    train_ess_early_stop=25, val_ess_early_stop=25, return_weights=False, model_round_as_feature=False):

    if model_round_as_feature:
        import warnings
        warnings.warn('model_round_as_feature is on; not able to do PG training on this model')

    train_data = BCDataset(train_df, model_round_as_feature=model_round_as_feature)
    valid_data = BCDataset(valid_df, model_round_as_feature=model_round_as_feature)

    train_loader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )
    optimizer = Adam(policy.parameters(), lr=lr)
    mean_train_losses, train_esss, mean_val_losses, val_esss = [], [], [], []
    train_weights, valid_weights = [], []
    train_opes, val_opes = [], []

    for e in range(epochs):
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss = step(policy, batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().item())

        mean_train_loss = np.mean(train_losses) # note that this is the average of average
        mean_val_loss = evaluate(policy, valid_loader, batch_mean=False)  # note that this is just average

        pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(train_df, policy, no_grad=True, is_train=True,
                                                                                  model_round_as_feature=model_round_as_feature)
        train_wis, weights = wis_ope(pibs, pies, rewards, lengths, max_time=max_time)

        train_opes.append(train_wis.item())
        train_weights.append(weights)

        train_ESS = 1 / (torch.sum(weights ** 2, axis=0))

        pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(valid_df, policy, no_grad=True, is_train=False,
                                                                                  model_round_as_feature=model_round_as_feature)
        valid_wis, weights = wis_ope(pibs, pies, rewards, lengths, max_time=max_time)

        valid_weights.append(weights)
        val_opes.append(valid_wis.item())

        val_ESS = 1 / (torch.sum(weights ** 2, axis=0))

        if verbose:
            # print(
            #     f"Epoch {e} train loss: {mean_train_loss}, train OPE: {train_wis}, train ESS: {train_ESS}, val loss: {mean_val_loss}, val OPE: {valid_wis}, val ESS: {val_ESS}")
            print("Epoch {} train loss: {:.3f}, train OPE: {:.3f}, train ESS: {:.3f}, val loss: {:.3f}, val OPE: {:.3f}, val ESS: {:.3f}".format(
                e, mean_train_loss, train_wis, train_ESS, mean_val_loss, valid_wis, val_ESS
            ))

        mean_train_losses.append(mean_train_loss)
        mean_val_losses.append(mean_val_loss)
        train_esss.append(train_ESS.item())
        val_esss.append(val_ESS.item())

        if early_stop:
            if train_ESS >= train_ess_early_stop and val_ESS >= val_ess_early_stop:
                break

    if not return_weights:
        return mean_train_losses, train_esss, mean_val_losses, val_esss
    else:
        return mean_train_losses, train_esss, mean_val_losses, val_esss, train_weights, valid_weights, train_opes, val_opes


def ope_step(policy, df, ope_method, lambda_ess=4, gr_safety_thresh=0.0, is_train=True, return_weights=False,
            normalize_reward=False):
    # we use this to compute loss/gradients
    # lambda_ess=4 is POPCORN's default value

    pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(df, policy, is_train=is_train, no_grad=False,
                                                                              gr_safety_thresh=gr_safety_thresh,
                                                                              normalize_reward=normalize_reward)
    ope_score, weights = ope_method(pibs, pies, rewards, lengths, max_time=max_time)
    ess_theta = 1 / (torch.sum(weights ** 2, dim=0))

    # POPCORN style loss
    loss = -(ope_score - (lambda_ess / torch.sqrt(ess_theta)))

    if not return_weights:
        return loss, ope_score.detach().item(), ess_theta.detach().item()
    else:
        return loss, ope_score.detach().item(), ess_theta.detach().item(), weights


def ope_evaluate(policy, df, ope_method, gr_safety_thresh, is_train=True, reward_column='Reward', return_weights=False, use_knn=False):
    with torch.no_grad():
        pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(df, policy, is_train=is_train,
                                                                                  gr_safety_thresh=gr_safety_thresh,
                                                                                  reward_column=reward_column,
                                                                                  use_knn=use_knn)
        ope_score, weights = ope_method(pibs, pies, rewards, lengths, max_time=max_time)
        ess = 1 / (torch.sum(weights ** 2, dim=0))

    if return_weights:
        return ope_score, ess, weights
    else:
        return ope_score, ess

class PGDataset(Dataset):
    def __init__(self, df, model_round_as_feature=False):
        # we only return student user_ids

        self.df = df
        self.model_round_as_feature = model_round_as_feature
        # preprocess a little
        # self.df['postive_feedback'] = df['postive_feedback'].apply(lambda x: 0 if x == 'None' else 1)
        # array(['None', 'True'], dtype=object)
        # True -> 1, None -> 0
        # self.df['stage'].fillna(5, inplace=True)
        # the nan stage is actually just the end of final stage
        self.student_ids = self.df['Trajectory']

    def __len__(self):
        return self.student_ids.shape[0]

    def __getitem__(self, index):
        student_id = self.student_ids.iloc[index]

        return {'student_id': torch.from_numpy(np.array([student_id])).int()}

def minibatch_offpolicy_pg_training(policy, train_df, valid_df, train_ope_method, folder_path, lr=1e-4, epochs=10, lambda_ess=4,
                          eval_ope_method=None, verbose=True, early_stop_ess=0., gr_safety_thresh=0.0, return_weights=False,
                          use_knn=False, normalize_reward=False, clip_lower=1e-16, clip_upper=1e2):
    # gr_safety_thresh: action prob lower than this will be masked out
    # a mask on top of the policy output probabilities to remove any actions under some threshold t

    train_data = PGDataset(train_df)

    train_loader = DataLoader(
        train_data,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    eval_ope_method = train_ope_method if eval_ope_method is None else eval_ope_method

    optimizer = Adam(policy.parameters(), lr=lr)
    train_opes, train_esss, val_opes, val_esss = [], [], [], []

    train_weights, valid_weights = [], []

    ckpt_path = pjoin(folder_path, 'model.ckpt') # pjoin(args.folder, 'model.ckpt')
    best_epoch = 0
    for e in range(epochs):

        train_losses = []
        train_opes = []
        train_ess = []

        for batch in train_loader:
            student_ids = batch['student_id'].squeeze(1).numpy().tolist()
            batch_df = train_df[train_df['Trajectory'].isin(student_ids)]

            optimizer.zero_grad()
            # clipped IS, minibatch

            pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(batch_df, policy, is_train=True,
                                                                                      no_grad=False,
                                                                                      gr_safety_thresh=gr_safety_thresh,
                                                                                      normalize_reward=normalize_reward)

            ope_score, weights = clipped_is_ope(pibs, pies, rewards, lengths, max_time=max_time, clip_upper=clip_upper, clip_lower=clip_lower)
            normalized_weights = weights / weights.sum(dim=0)
            ess_theta = 1 / (torch.sum(normalized_weights ** 2, dim=0))

            # POPCORN style loss
            loss = -(ope_score - (lambda_ess / torch.sqrt(ess_theta)))

            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().item())
            train_opes.append(ope_score.detach().item())
            train_ess.append(ess_theta.detach().item())

        val_ope, val_ess, weights = ope_evaluate(policy, valid_df, eval_ope_method, gr_safety_thresh, is_train=False,
                                                 return_weights=True, use_knn=use_knn)

        valid_weights.append(weights)

        val_opes.append(val_ope)
        val_esss.append(val_ess)

        if verbose:
            # train loss: {:.2f},
            # loss.detach().item(),
            print("Epoch {} train OPE Score: {:.2f}, ".format(e, np.mean(train_opes)) + \
                  "train loss: {:.2f}, train ESS: {:.2f}, val OPE Score: {:.2f}, val ESS: {:.2f}".format(np.mean(train_losses), np.mean(train_ess), val_ope, val_ess))

        # break without saving
        # if early_stop_ess != 0:
        #     if np.mean(train_ess) <= early_stop_ess:
        #         break

        # save model
        torch.save(policy.state_dict(), ckpt_path)
        best_epoch = e

    # loading the previous saved model
    if e > 0:
        policy.load_state_dict(torch.load(ckpt_path))

    #best_train_ope = train_opes[best_epoch]
    best_val_ope = val_opes[best_epoch]
    # best_train_ess = train_esss[best_epoch]
    best_valid_ess = val_esss[best_epoch]

    best_train_ope, best_train_ess = 0, 0
    train_losses = []

    if return_weights:
        return train_losses, train_opes, train_esss, val_opes, val_esss, valid_weights
    else:
        return train_losses, train_opes, train_esss, val_opes, val_esss, (best_train_ope, best_val_ope, best_train_ess, best_valid_ess)