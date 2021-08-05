# write this one
# Procedure:
# We get 25 random seeds (fixed)
# We try it on 4 Lambda ESS [1,2,4,10], with action masking of [0.03, 0.05]
# 1. We early stop on training, using ESS = 20 as threshold (or something)
# 2. We evaluate each policy on validation for ESS and Score
# 3. We compute BCa bootstrap bound on training and validation for each policy
# 4. Log everything to a folder

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

def normalize(ob):
    return (ob - [3, 4, 3, 10, 0.5, 0.5, 0.5, 27])/ [1, 4, 3, 10, 0.5, 0.5, 0.5, 18]

def unnormalize(norm_ob):
    return np.array(norm_ob) * np.array([1, 4, 3, 10, 0.5, 0.5, 0.5, 18]) + np.array([3, 4, 3, 10, 0.5, 0.5, 0.5, 27])

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def masked_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        # This line causes error to be silent
        # while mask.dim() < vector.dim():
        #     mask = mask.unsqueeze(1)
        
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    # return torch.nn.functional.log_softmax(vector, dim=dim)
    return torch.nn.functional.softmax(vector, dim=dim)

norm_names = ['grade_norm', 'pre-score_norm', 'stage_norm', 'failed_attempts_norm',
         'pos_norm', 'neg_norm', 'hel_norm', 'anxiety_norm']

raw_names = ['grade', 'pre', 'stage', 'failed_attempts', 'pos', 'neg', 'help', 'anxiety']

def generate_features(norm_ob_row, unnormed_ob_row):
    assert len(unnormed_ob_row.shape) == 1

    norm_obs_dict = dict(zip(norm_names, norm_ob_row))
    raw_obs_dict = dict(zip(raw_names, unnormed_ob_row))

    features = [norm_obs_dict['stage_norm'],
                norm_obs_dict['failed_attempts_norm'],
                norm_obs_dict['pos_norm'],
                norm_obs_dict['neg_norm'],
                norm_obs_dict['hel_norm'],
                norm_obs_dict['anxiety_norm'],
                norm_obs_dict['grade_norm'],
                int(raw_obs_dict['pre']),
                int(raw_obs_dict['anxiety']),
                int(raw_obs_dict['stage'])]

    return features

# add the feature wrapper around this
# TODO: 0. DeployPolicy does the following:
# TODO   - implement all methods that are needed by Sherry
# TODO   - Take in
# TODO: 1. re-generate traj of two students
# TODO: 2. Re-train? BC+Eval (not too hard)
# TODO: 3. Train on full dataset; or only on split 5
class DeployPolicy(nn.Module):
    def __init__(self):
        super().__init__()

class MLPPolicy(nn.Module):
    def __init__(self, sizes, activation=nn.GELU,
                 output_activation=nn.Identity):
        super().__init__()
        self.nA = 4
        self.is_linear = False
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        self.net = nn.Sequential(*layers)

    def get_action_probability(self, obs, no_grad=True, action_mask=None):
        # obs: [batch_size, obs_dim]
        # we add a mask to forbid certain actions being taken
        if no_grad:
            with torch.no_grad():
                logits = self.net(obs)
                # mask here
                if action_mask is not None:
                    probs = masked_softmax(logits, action_mask)
                else:
                    probs = F.softmax(logits, dim=-1)
        else:
            logits = self.net(obs)
            # mask here
            if action_mask is not None:
                probs = masked_softmax(logits, action_mask)
            else:
                probs = F.softmax(logits, dim=-1)

        return probs

    def forward(self, obs):
        logits = self.net(obs)
        logp = F.log_softmax(logits, dim=-1)
        return logp

class LinearPolicy(nn.Module):
    def __init__(self, sizes,
                 output_activation=nn.Identity):
        super().__init__()
        self.nA = 4
        assert len(sizes) == 2, "Linear Policy can only have input/ouput in sizes"
        self.net = nn.Linear(sizes[0], sizes[1])
        self.is_linear = True

    def get_action_probability(self, obs, no_grad=True, action_mask=None):
        # obs: [batch_size, obs_dim]
        # we add a mask to forbid certain actions being taken
        if no_grad:
            with torch.no_grad():
                logits = self.net(obs)
                # mask here
                if action_mask is not None:
                    probs = masked_softmax(logits, action_mask)
                else:
                    probs = F.softmax(logits, dim=-1)
        else:
            logits = self.net(obs)
            # mask here
            if action_mask is not None:
                probs = masked_softmax(logits, action_mask)
            else:
                probs = F.softmax(logits, dim=-1)

        return probs

    def forward(self, obs):
        logits = self.net(obs)
        logp = F.log_softmax(logits, dim=-1)
        return logp

class BCDataset(Dataset):
    def __init__(self, df, model_round_as_feature=False):
        # df: raw pandas df
        self.df = df
        self.model_round_as_feature = model_round_as_feature
        # preprocess a little
        # self.df['postive_feedback'] = df['postive_feedback'].apply(lambda x: 0 if x == 'None' else 1)
        # array(['None', 'True'], dtype=object)
        # True -> 1, None -> 0
        self.df['stage'].fillna(5, inplace=True)
        # the nan stage is actually just the end of final stage

        feature_names = ['stage_norm', 'failed_attempts_norm', 'pos_norm', 'neg_norm',
                         'hel_norm', 'anxiety_norm', 'grade_norm', 'pre', 'anxiety']
        # 'grade_norm' and 'pre' are relatively categorical/discrete
        # unused_features = ['input_message_kid', 'time_stored', 'grade']
        categorical_features = ['stage']  # postive_feedback

        # feature_names = ['grade_norm', 'pre-score_norm', 'stage_norm', 'failed_attempts_norm',
        #                 'pos_norm', 'neg_norm', 'hel_norm', 'anxiety_norm']

        feature_names = feature_names + categorical_features
        self.feature_names = feature_names

        if model_round_as_feature:
            self.feature_names += ['model_round']

        self.target_names = ["p_hint", "p_nothing", "p_encourage", "p_question"]

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


feature_names = ['stage_norm', 'failed_attempts_norm', 'pos_norm', 'neg_norm',
                 'hel_norm', 'anxiety_norm', 'grade_norm', 'pre', 'anxiety']
# 'grade_norm' and 'pre' are relatively categorical/discrete
#unused_features = ['input_message_kid', 'time_stored', 'grade']
categorical_features = ['stage']  # postive_feedback

# feature_names = ['grade_norm', 'pre-score_norm', 'stage_norm', 'failed_attempts_norm',
#                 'pos_norm', 'neg_norm', 'hel_norm', 'anxiety_norm']

target_names = ["p_hint", "p_nothing", "p_encourage", "p_question"]

feature_names = feature_names + categorical_features

MAX_TIME = 28

def compute_is_weights_for_nn_policy(behavior_df, eval_policy, eps=0.05, temp=0.1,
                                     reward_column='adjusted_score', no_grad=True,
                                     gr_safety_thresh=0.0, is_train=True, normalize_reward=False, use_knn=False,
                                     model_round_as_feature=False):
    # is_weights with batch processing
    df = behavior_df
    user_ids = df['user_id'].unique()
    n = len(user_ids)

    # now max_time is dynamic, finally!!
    MAX_TIME = max(behavior_df.groupby('user_id').size())

    assert reward_column in ['adjusted_score', 'reward']

    pies = torch.zeros((n, MAX_TIME))  # originally 20
    pibs = torch.zeros((n, MAX_TIME))
    rewards = torch.zeros((n, MAX_TIME))
    lengths = np.zeros((n))  # we are not doing anything to this

    # compute train split reward mean and std
    # (n,): and not average
    user_rewards = df.groupby("user_id")[reward_column].mean()
    train_reward_mu = user_rewards.mean()
    train_reward_std = user_rewards.std()

    for idx, user_id in enumerate(user_ids):
        data = df[df['user_id'] == user_id]
        # get features, targets
        if not model_round_as_feature:
            features = np.asarray(data[feature_names]).astype(float)
        else:
            features = np.asarray(data[feature_names + ['model_round']]).astype(float)
        targets = np.asarray(data[target_names]).astype(float)
        actions = np.asarray(data['action']).astype(int)

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
                    knn_targets = np.asarray(data[knn_target_names]).astype(float)
                    assert knn_targets.shape[0] == targets.shape[0]
                    beh_action_probs = torch.from_numpy(knn_targets)

                    gr_mask = beh_action_probs >= gr_safety_thresh
            else:
                beh_action_probs = torch.from_numpy(targets)

                # gr_mask = beh_probs >= gr_safety_thresh
                gr_mask = beh_action_probs >= gr_safety_thresh

            # need to renormalize behavior policy as well?

        # assign rewards (note adjusted_score we only assign to last step)
        # reward, we assign to all
        if reward_column == 'adjusted_score':
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


def wis_ope(pibs, pies, rewards, length, no_weight_norm=False, max_time=MAX_TIME, per_sample=False, clip_lower=1e-16,
            clip_upper=1e3):
    # even for batch setting, this function should still work fine
    # but for WIS, batch setting won't give accurate normalization
    n = pibs.shape[0]
    weights = torch.ones((n, MAX_TIME))

    for i in range(n):
        last = 1
        for t in range(int(length[i])):
            assert pibs[i, t] != 0
            last = last * (pies[i, t] / pibs[i, t])
            weights[i, t] = last
        weights[i, length[i]:] = weights[i, length[i] - 1]
    # weights = torch.clip(weights, 1e-16, 1e3)
    weights = torch.clip(weights, clip_lower, clip_upper)
    if not no_weight_norm:
        weights_norm = weights.sum(dim=0)
        weights /= weights_norm  # per-step weights (it's cumulative)
    else:
        weights /= n

    # this accumulates
    if not per_sample:
        return (weights[:, -1] * rewards.sum(dim=-1)).sum(dim=0), weights[:, -1]
    else:
        # return w_i associated with each N
        return weights[:, -1] * rewards.sum(dim=-1), weights[:, -1]


def is_ope(pibs, pies, rewards, length, max_time=MAX_TIME):
    return wis_ope(pibs, pies, rewards, length, no_weight_norm=True, max_time=max_time)

def clipped_is_ope(pibs, pies, rewards, length, max_time=MAX_TIME, clip_lower=1e-16,
            clip_upper=1e3):
    return wis_ope(pibs, pies, rewards, length, no_weight_norm=True, max_time=max_time,
                   clip_lower=clip_lower, clip_upper=clip_upper)

def cwpdis_ope(pibs, pies, rewards, length, max_time=MAX_TIME):
    # this computes a consistent weighted per-decision IS
    # following POPCORN paper / Thomas' thesis
    n = pibs.shape[0]
    weights = torch.ones((n, max_time))
    wis_weights = torch.ones(n)

    for i in range(n):
        last = 1
        for t in range(int(length[i])):
            assert pibs[i, t] != 0
            # changed these two lines so gradient can flow...
            last = last * (pies[i, t] / pibs[i, t])
            weights[i, t] = last

        wis_weights[i] = last

    with torch.no_grad():
        masks = (weights != 0).detach().numpy()
        masks = torch.FloatTensor(masks)

    weights = torch.clip(weights, 1e-16, 1e3)

    weights_norm = weights.sum(dim=0)

    # step 1: \sum_n r_nt * w_nt
    weighted_r = (rewards * weights).sum(dim=0)

    # step 2: (\sum_n r_nt * w_nt) / \sum_n w_nt
    score = weighted_r / weights_norm

    # step 3: \sum_t ((\sum_n r_nt * w_nt) / \sum_n w_nt)
    score = score.sum()

    # sum through the trajectory, we get CWPDIS(θ), and ESS(θ)

    wis_weights = torch.clip(wis_weights, 1e-16, 1e3)
    weights_norm = wis_weights.sum(dim=0)
    wis_weights = wis_weights / weights_norm

    return score, wis_weights


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


def ope_evaluate(policy, df, ope_method, gr_safety_thresh, is_train=True, reward_column='adjusted_score', return_weights=False, use_knn=False):
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


def offpolicy_pg_training(policy, train_df, valid_df, train_ope_method, folder_path, lr=1e-4, epochs=10, lambda_ess=4,
                          eval_ope_method=None, verbose=True, early_stop_ess=0., gr_safety_thresh=0.0, return_weights=False,
                          use_knn=False, normalize_reward=False):
    # gr_safety_thresh: action prob lower than this will be masked out
    # a mask on top of the policy output probabilities to remove any actions under some threshold t

    eval_ope_method = train_ope_method if eval_ope_method is None else eval_ope_method

    optimizer = Adam(policy.parameters(), lr=lr)
    train_opes, train_esss, val_opes, val_esss = [], [], [], []
    train_losses = []

    train_weights, valid_weights = [], []

    ckpt_path = pjoin(folder_path, 'model.ckpt') # pjoin(args.folder, 'model.ckpt')
    best_epoch = 0
    for e in range(epochs):

        optimizer.zero_grad()

        # No KNN for training
        loss, train_ope, train_ess = ope_step(policy, train_df, train_ope_method, lambda_ess,
                                              gr_safety_thresh, is_train=True, normalize_reward=normalize_reward)

        # add Lasso loss
        # if policy.is_linear:
        #     W = next(policy.parameters())
        #     l1_loss = torch.norm(W, p=1, dim=1).sum()
        #     loss = loss + l1_loss * 2e-2

        loss.backward()
        optimizer.step()

        val_ope, val_ess, weights = ope_evaluate(policy, valid_df, eval_ope_method, gr_safety_thresh, is_train=False,
                                                 return_weights=True, use_knn=use_knn)

        valid_weights.append(weights)

        train_opes.append(train_ope)
        train_esss.append(train_ess)

        val_opes.append(val_ope)
        val_esss.append(val_ess)

        train_losses.append(loss.detach().item())

        if verbose:
            # train loss: {:.2f},
            # loss.detach().item(),
            print("Epoch {} train OPE Score: {:.2f}, ".format(e, train_ope) + \
                  "train loss: {:.2f}, train ESS: {:.2f}, val OPE Score: {:.2f}, val ESS: {:.2f}".format(loss.detach().item(), train_ess, val_ope, val_ess))

        # break without saving
        if early_stop_ess != 0:
            if train_ess <= early_stop_ess:
                break

        # save model
        torch.save(policy.state_dict(), ckpt_path)
        best_epoch = e

    # loading the previous saved model
    if e > 0:
        policy.load_state_dict(torch.load(ckpt_path))

    best_train_ope = train_opes[best_epoch]
    best_val_ope = val_opes[best_epoch]
    best_train_ess = train_esss[best_epoch]
    best_valid_ess = val_esss[best_epoch]

    if return_weights:
        return train_losses, train_opes, train_esss, val_opes, val_esss, valid_weights
    else:
        return train_losses, train_opes, train_esss, val_opes, val_esss, (best_train_ope, best_val_ope, best_train_ess, best_valid_ess)

class PGDataset(Dataset):
    def __init__(self, df, model_round_as_feature=False):
        # we only return student user_ids

        self.df = df
        self.model_round_as_feature = model_round_as_feature
        # preprocess a little
        # self.df['postive_feedback'] = df['postive_feedback'].apply(lambda x: 0 if x == 'None' else 1)
        # array(['None', 'True'], dtype=object)
        # True -> 1, None -> 0
        self.df['stage'].fillna(5, inplace=True)
        # the nan stage is actually just the end of final stage
        self.student_ids = self.df['user_id']

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
            batch_df = train_df[train_df['user_id'].isin(student_ids)]

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


seeds = [42, 331659, 317883, 763090, 388383, 361473, 5406, 586703, 621359, 350927,
         96707, 659342, 457113, 708574, 937808, 986639, 296980, 998083, 799922, 636508,
         202442, 295876, 77463, 472346, 11761, 771297, 405306, 13421, 287019, 29800, 826720,
         307768, 292204, 630011, 171933, 646715, 284200, 898505, 274865, 66661, 524648, 188708,
         345047, 944273, 135057, 632339, 827946, 855243, 610525, 516977,
         670487, 116739, 26225, 777572, 288389, 256787, 234053, 146316, 772246, 107473]


def set_random_seed(seed):
    import torch
    import random
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


df = pd.read_csv("./primer_data/offline_with_probabilities_cleaned.csv")

B = 100
# n_users / 2 because we do 2-fold validation
# n_users = int(len(df['user_id'].unique()) / 2)

def th_to_np(np_array):
    return torch.from_numpy(np_array)


# 0.67, 0.95, 0.99
def bca_bootstrap(pibs, pies, rewards, length, ope_method=wis_ope, alpha=0.05, max_time=MAX_TIME):
    pibs = pibs.numpy()
    pies = pies.numpy()
    rewards = rewards.numpy()

    n_users = pibs.shape[0] # 89
    n_subsample = n_users  # n_val_patients

    wis_list = []
    for b in range(B):
        ids = np.random.choice(n_users, n_subsample)
        sam_rewards = rewards[ids, :]
        sam_pibs = pibs[ids, :]
        sam_pies = pies[ids, :]
        sam_length = length[ids]
        wis_pie, _ = ope_method(th_to_np(sam_pibs), th_to_np(sam_pies), th_to_np(sam_rewards), sam_length, max_time=max_time)
        wis_list.append(wis_pie.numpy())
    y = []
    for i in range(n_users):
        sam_rewards = np.delete(rewards, i, axis=0)
        sam_pibs = np.delete(pibs, i, axis=0)
        sam_pies = np.delete(pies, i, axis=0)
        sam_length = np.delete(length, i, axis=0)
        wis_pie, _ = ope_method(th_to_np(sam_pibs), th_to_np(sam_pies), th_to_np(sam_rewards), sam_length, max_time=max_time)
        y.append(wis_pie.numpy())

    wis_list = np.array(wis_list)
    wis_list = np.sort(wis_list)
    y = np.array(y)
    avg, _ = ope_method(th_to_np(pibs), th_to_np(pies), th_to_np(rewards), length, max_time=max_time)
    avg = avg.numpy()

    ql, qu = norm.ppf(alpha), norm.ppf(1 - alpha)

    # Acceleration factor
    num = np.sum((y.mean() - y) ** 3)
    den = 6 * np.sum((y.mean() - y) ** 2) ** 1.5
    ahat = num / den

    # Bias correction factor
    zhat = norm.ppf(np.mean(wis_list < avg))
    a1 = norm.cdf(zhat + (zhat + ql) / (1 - ahat * (zhat + ql)))
    a2 = norm.cdf(zhat + (zhat + qu) / (1 - ahat * (zhat + qu)))

    # print('Accel: %0.3f, bz: %0.3f, a1: %0.3f, a2: %0.3f' % (ahat, zhat, a1, a2))
    return np.quantile(wis_list, [a1, a2]), wis_list


def evaluate_policy_for_ci(policy, df, ope_method, gr_safety_thresh, alpha=0.05, is_train=True,
                           reward_column='adjusted_score', use_knn=False):
    # df = valid_df
    # df = train_df

    with torch.no_grad():
        pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(df, policy, is_train=is_train, gr_safety_thresh=gr_safety_thresh,
                                                                                  reward_column=reward_column, use_knn=use_knn)
        lb_ub, wis_list = bca_bootstrap(pibs, pies, rewards, lengths, alpha=alpha, ope_method=ope_method,
                                        max_time=max_time)  # 0.01, 0.05, 0.33

        avg, _ = ope_method(pibs, pies, rewards, lengths, max_time=max_time)

    #     policy_name = str(policy).lstrip("<class '__main__.").rstrip("'>")
    #     ope_name = str(cwpdis_ope).split()[1]

    #     print("{}, {} score: {:.2f}, ESS: {}".format(policy_name, ope_name, ope_folds / 2, ess_folds / 2))
    return lb_ub, avg.item(), wis_list


def compute_ci_for_policies(bc_policy, df, ope_method, gr_safety_thresh, alpha=0.05, get_wis=False, is_train=True,
                            reward_column='adjusted_score', use_knn=False):
    lb_ub, wis, wis_list = evaluate_policy_for_ci(bc_policy, df, ope_method, gr_safety_thresh, alpha=alpha, is_train=is_train,
                                                  reward_column=reward_column, use_knn=use_knn)
    score, ess = ope_evaluate(bc_policy, df, ope_method, gr_safety_thresh, is_train=is_train)

    # print(extra_message + "{} with CI: {}, val ESS: {}".format(wis, lb_ub, ess))

    if get_wis:
        return wis_list
    else:
        return wis, np.mean(wis_list), lb_ub, ess

import os
import csv
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, default='1', help="split")
    parser.add_argument("--folder", type=str, default='results_feb23', help="split")
    parser.add_argument("--append_result", action='store_true', help="write mode a: append")
    args = parser.parse_args()

    # if args.data_split == '1':
    #     train_df = pd.read_csv("./primer_data/train_df_clean.csv")
    #     valid_df = pd.read_csv("./primer_data/valid_df_clean.csv")
    # else:
    #     train_df = pd.read_csv(f"./primer_data/train_df_clean_split{args.data_split}.csv")
    #     valid_df = pd.read_csv(f"./primer_data/valid_df_clean_split{args.data_split}.csv")

    # train_df = pd.read_csv(f"./primer_data/train_df_by_round_clean_split{args.data_split}_fixed.csv")
    # valid_df = pd.read_csv(f"./primer_data/valid_df_by_round_clean_split{args.data_split}_fixed.csv")

    train_df = pd.read_csv(f"./primer_data/train_df_new_new_split{args.data_split}.csv")
    valid_df = pd.read_csv(f"./primer_data/valid_df_new_new_split{args.data_split}.csv")

    # pg_lr_set = {0.5:3e-4, 1:3e-4, 2:3e-4, 4:3e-4, 8:1e-5, 10:1e-5}
    split_to_early_stop = {}

    # pbar = tqdm(total=4*25) # 4 * 25
    # pbar = tqdm(total=2*25) # 4 * 25
    pbar = tqdm(total=64) # 6 * 25 18 * 5

    os.makedirs(args.folder, exist_ok=True)

    # one danger: different lambda_ess requires different early stop criteria
    write_mode = 'a' if args.append_result else 'w'
    logfile = open(pjoin(args.folder, "log.csv"), write_mode)
    log_writer = csv.writer(logfile)

    if not args.append_result:
        log_writer.writerow(['exp_name', 'seed', 'bc_train_ess', 'bc_val_ess', 'stop', 'train_loss', 'train_ope',
                         'train_ess', 'train_lb', 'train_ub', 'val_ope', 'val_ess', 'val_lb', 'val_ub', 'bh_rew'])

    bh_rew = valid_df.groupby('user_id')['adjusted_score'].mean().mean()

    # loading KNN
    # if int(args.data_split) <= 5:
    import pickle
    with open(f"./primer_data/knn_prob_new_new_split{args.data_split}", 'rb') as f:
        dic = pickle.load(f)
        # knn_prob_train = dic['knn_prob_train']
        knn_prob_valid = dic['knn_prob_valid']

    knn_target_names = []
    for i, t_n in enumerate(target_names):
        # train_df['knn_' + t_n] = knn_prob_train[:, i]
        valid_df['knn_' + t_n] = knn_prob_valid[:, i]
        knn_target_names.append('knn_' + t_n)

    for early_stop_epoch in [5, 10, 15, 25]:  # [5, 10, 20, 30, 50]
        for lambda_ess in [0.01, 0.1, 0.5, 4]: # [0.5, 1, 2, 4, 8, 10] [1, 2, 4, 10]:
            for gr_thresh in [0.01]: # [0.05, 0.1, 0.15] [0.01, 0.05, 0.1]:  # [0, 0.01, 0.03, 0.05]
                # gr_safety_thresh = 0.03
                for seed in [72]: # seeds[:1]:
                    for use_knn in [False, True]:
                        for normalize_reward in [False, True]:
                            set_random_seed(seed)  # seed

                            # bc_policy = MLPPolicy([10, 16, 16, 4])  # 8 x 8 before
                            # bc_policy = MLPPolicy([10, 16, 4])
                            bc_policy = MLPPolicy([10, 4, 4])
                            # bc_policy = LinearPolicy([10, 4])  # this can tell us if higher-order interaction is necessary

                            bc_train_losses, bc_train_esss, bc_val_losses, bc_val_esss = bc_train_policy(bc_policy, train_df, valid_df,
                                                                                                         epochs=70, lr=1e-3, verbose=False,
                                                                                                         early_stop=True,
                                                                                                         train_ess_early_stop=25, val_ess_early_stop=15)

                            # pg_lr = pg_lr_set[lambda_ess]
                            pg_lr = 3e-4
                            train_losses, train_opes, train_esss, val_opes, val_esss, best_summary = offpolicy_pg_training(bc_policy, train_df, valid_df,
                                                                                                             wis_ope, args.folder, lr=pg_lr,
                                                                                                             epochs=early_stop_epoch, lambda_ess=lambda_ess,
                                                                                                             gr_safety_thresh=gr_thresh,
                                                                                                             use_knn=use_knn, normalize_reward=normalize_reward,
                                                                                                             verbose=False,
                                                                                                             # for better plot, take this out to train fully
                                                                                                             early_stop_ess=0)  # 20 # 17.5

                            best_train_ope, best_val_ope, best_train_ess, best_valid_ess = best_summary

                            # CI on train
                            set_random_seed(seed)
                            train_wis, train_avg_sampled_wis, train_lb_ub, train_ess = compute_ci_for_policies(bc_policy, train_df,
                                                                                                               wis_ope,
                                                                                                               gr_safety_thresh=gr_thresh, alpha=0.05,
                                                                                                               is_train=True)

                            # CI on validation
                            set_random_seed(seed)
                            wis, avg_sampled_wis, lb_ub, ess = compute_ci_for_policies(bc_policy, valid_df, wis_ope, gr_safety_thresh=gr_thresh, alpha=0.05,
                                                                                       is_train=False, use_knn=use_knn)

                            # file_name = "lambda_ess_{}_gr_thres_{}_seed_{}.pkl".format(lambda_ess, str(gr_thresh).replace(".", ""), seed)
                            # cloudpickle.dump([
                            #     bc_train_losses, bc_train_esss, bc_val_losses, bc_val_esss,
                            #     train_losses, train_opes, train_esss, val_opes, val_esss,
                            #     train_wis, train_avg_sampled_wis, train_lb_ub, train_ess,
                            #     wis, avg_sampled_wis, lb_ub, ess
                            # ], open(pjoin(args.folder, file_name), 'wb'))+

                            knn_str = 'T' if use_knn else 'F'
                            norm_str = 'T' if normalize_reward else 'F'

                            exp_name = 'l_ess_{}_gr_{}_knn_{}_norm_{}_stop_{}'.format(lambda_ess, str(gr_thresh).replace(".", ""), knn_str, norm_str, early_stop_epoch)
                            try:
                                log_writer.writerow([exp_name, str(seed), bc_train_esss[-1], bc_val_esss[-1],
                                                     early_stop_epoch, train_losses[-1],
                                                     best_train_ope, best_train_ess, # train_opes[-1], train_esss[-1],
                                                     train_lb_ub[0], train_lb_ub[1],  best_val_ope.item(), best_valid_ess.item(), lb_ub[0], lb_ub[1],
                                                     bh_rew])
                            except:
                                print("error!")
                                pass

                            pbar.update(1)
                            logfile.flush()

    logfile.close()