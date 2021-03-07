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
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    # return torch.nn.functional.log_softmax(vector, dim=dim)
    return torch.nn.functional.softmax(vector, dim=dim)


class MLPPolicy(nn.Module):
    def __init__(self, sizes, activation=nn.GELU,
                 output_activation=nn.Identity):
        super().__init__()
        self.nA = 4
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


class BCDataset(Dataset):
    def __init__(self, df):
        # df: raw pandas df
        self.df = df
        # preprocess a little
        self.df['postive_feedback'] = df['postive_feedback'].apply(lambda x: 0 if x == 'None' else 1)
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
                                     reward_column='adjusted_score', no_grad=True, gr_safety_thresh=0.0):
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

    for idx, user_id in enumerate(user_ids):
        data = df[df['user_id'] == user_id]
        # get features, targets
        features = np.asarray(data[feature_names]).astype(float)
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
            gr_mask = beh_probs >= gr_safety_thresh
            # need to renormalize behavior policy as well?

        # assign rewards (note adjusted_score we only assign to last step)
        # reward, we assign to all
        if reward_column == 'adjusted_score':
            reward = np.asarray(data[reward_column])[-1]
            rewards[idx, T - 1] = reward
        else:
            # normal reward
            rewards[idx, :T] = torch.from_numpy(np.asarray(data[reward_column])).float()

        # last thing: model prediction
        eval_action_probs = eval_policy.get_action_probability(torch.from_numpy(features).float(), no_grad,
                                                               action_mask=gr_mask)

        pies[idx, :T] = torch.hstack([eval_action_probs[i, a] for i, a in enumerate(actions)])

    return pibs, pies, rewards, lengths.astype(int), MAX_TIME


def wis_ope(pibs, pies, rewards, length, no_weight_norm=False, max_time=MAX_TIME, per_sample=False):
    # even for batch setting, this function should still work fine
    # but for WIS, batch setting won't give accurate normalization
    n = pibs.shape[0]
    weights = torch.ones((n, MAX_TIME))

    for i in range(n):
        last = 1
        for t in range(int(length[i])):
            assert pibs[i, t] != 0
            weights[i, t] = last * (pies[i, t] / pibs[i, t])
            last = weights[i, t]
        weights[i, length[i]:] = weights[i, length[i] - 1]
    weights = torch.clip(weights, 1e-16, 1e3)
    if not no_weight_norm:
        weights_norm = weights.sum(dim=0)
        weights /= weights_norm  # per-step weights (it's cumulative)

    # this accumulates
    if not per_sample:
        return (weights[:, -1] * rewards.sum(dim=-1)).sum(dim=0), weights[:, -1]
    else:
        # return w_i associated with each N
        return weights[:, -1] * rewards.sum(dim=-1), weights[:, -1]


def is_ope(pibs, pies, rewards, length, max_time=MAX_TIME):
    return wis_ope(pibs, pies, rewards, length, no_weight_norm=True, max_time=max_time)


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


def step(policy, batch, mse_loss=False):
    features = batch['features']
    targets = batch['targets']

    # logp: [batch_size, prob_per_action]
    logp = policy(features)
    loss = -torch.mean(torch.sum(targets * logp, dim=-1))

    return loss


def evaluate(policy, val_dataloader):
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            val_loss = step(policy, batch)
            losses.append(val_loss.detach().item())
    return np.mean(losses)


def bc_train_policy(policy, train_df, valid_df, lr=1e-4, epochs=10, verbose=True, early_stop=False,
                    train_ess_early_stop=25, val_ess_early_stop=25):
    train_data = BCDataset(train_df)
    valid_data = BCDataset(valid_df)

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

    for e in range(epochs):
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss = step(policy, batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().item())

        mean_train_loss = np.mean(train_losses)
        mean_val_loss = evaluate(policy, valid_loader)

        pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(train_df, policy, no_grad=True)
        wis_pie, weights = wis_ope(pibs, pies, rewards, lengths, max_time=max_time)

        train_ESS = 1 / (torch.sum(weights ** 2, axis=0))

        pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(valid_df, policy, no_grad=True)
        wis_pie, weights = wis_ope(pibs, pies, rewards, lengths, max_time=max_time)

        val_ESS = 1 / (torch.sum(weights ** 2, axis=0))

        if verbose:
            print(
                f"Epoch {e} train loss: {mean_train_loss}, train ESS: {train_ESS}, val loss: {mean_val_loss}, val ESS: {val_ESS}")

        mean_train_losses.append(mean_train_loss)
        mean_val_losses.append(mean_val_loss)
        train_esss.append(train_ESS.item())
        val_esss.append(val_ESS.item())

        if early_stop:
            if train_ESS >= train_ess_early_stop and val_ESS >= val_ess_early_stop:
                break

    return mean_train_losses, train_esss, mean_val_losses, val_esss


def ope_step(policy, df, ope_method, lambda_ess=4, gr_safety_thresh=0.0):
    # we use this to compute loss/gradients
    # lambda_ess=4 is POPCORN's default value

    pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(df, policy, no_grad=False,
                                                                              gr_safety_thresh=gr_safety_thresh)
    ope_score, weights = ope_method(pibs, pies, rewards, lengths, max_time=max_time)
    ess_theta = 1 / (torch.sum(weights ** 2, dim=0))

    # POPCORN style loss
    loss = -(ope_score - (lambda_ess / torch.sqrt(ess_theta)))

    return loss, ope_score.detach().item(), ess_theta.detach().item()


def ope_evaluate(policy, df, ope_method):
    with torch.no_grad():
        pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(df, policy)
        ope_score, weights = ope_method(pibs, pies, rewards, lengths, max_time=max_time)
        ess = 1 / (torch.sum(weights ** 2, dim=0))

    return ope_score, ess


def offpolicy_pg_training(policy, train_df, valid_df, train_ope_method, lr=1e-4, epochs=10, lambda_ess=4,
                          eval_ope_method=None, verbose=True, early_stop_ess=0., gr_safety_thresh=0.0):
    # gr_safety_thresh: action prob lower than this will be masked out
    # a mask on top of the policy output probabilities to remove any actions under some threshold t

    eval_ope_method = train_ope_method if eval_ope_method is None else eval_ope_method

    optimizer = Adam(policy.parameters(), lr=lr)
    train_opes, train_esss, val_opes, val_esss = [], [], [], []
    train_losses = []

    ckpt_path = pjoin(args.folder, 'model.ckpt')
    for e in range(epochs):

        optimizer.zero_grad()

        loss, train_ope, train_ess = ope_step(policy, train_df, train_ope_method, lambda_ess, gr_safety_thresh)

        loss.backward()
        optimizer.step()

        val_ope, val_ess = ope_evaluate(policy, valid_df, eval_ope_method)

        train_opes.append(train_ope)
        train_esss.append(train_ess)

        val_opes.append(val_ope)
        val_esss.append(val_ess)

        train_losses.append(loss.detach().item())

        if verbose:
            # train loss: {:.2f},
            # loss.detach().item(),
            print("Epoch {} train OPE Score: {:.2f}, ".format(e, train_ope) + \
                  "train ESS: {:.2f}, val OPE Score: {:.2f}, val ESS: {:.2f}".format(train_ess, val_ope, val_ess))

        # break without saving
        if early_stop_ess != 0:
            if train_ess <= early_stop_ess:
                break

        # save model
        torch.save(policy.state_dict(), ckpt_path)

    # loading the previous saved model
    if e != 0:
        policy.load_state_dict(torch.load(ckpt_path))

    return train_losses, train_opes, train_esss, val_opes, val_esss


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
n_users = 89
n_subsample = n_users  # n_val_patients


def th_to_np(np_array):
    return torch.from_numpy(np_array)


# 0.67, 0.95, 0.99
def bca_bootstrap(pibs, pies, rewards, length, ope_method=wis_ope, alpha=0.05, max_time=MAX_TIME):
    pibs = pibs.numpy()
    pies = pies.numpy()
    rewards = rewards.numpy()

    wis_list = []
    for b in range(B):
        ids = np.random.choice(n_users, n_subsample)
        sam_rewards = rewards[ids, :]
        sam_pibs = pibs[ids, :]
        sam_pies = pies[ids, :]
        sam_length = length[ids]
        wis_pie, _ = ope_method(th_to_np(sam_pibs), th_to_np(sam_pies), th_to_np(sam_rewards), sam_length, max_time)
        wis_list.append(wis_pie.numpy())
    y = []
    for i in range(n_users):
        sam_rewards = np.delete(rewards, i, axis=0)
        sam_pibs = np.delete(pibs, i, axis=0)
        sam_pies = np.delete(pies, i, axis=0)
        sam_length = np.delete(length, i, axis=0)
        wis_pie, _ = ope_method(th_to_np(sam_pibs), th_to_np(sam_pies), th_to_np(sam_rewards), sam_length, max_time)
        y.append(wis_pie.numpy())

    wis_list = np.array(wis_list)
    wis_list = np.sort(wis_list)
    y = np.array(y)
    avg, _ = ope_method(th_to_np(pibs), th_to_np(pies), th_to_np(rewards), length, max_time)
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

    print('Accel: %0.3f, bz: %0.3f, a1: %0.3f, a2: %0.3f' % (ahat, zhat, a1, a2))
    return np.quantile(wis_list, [a1, a2]), wis_list


def evaluate_policy_for_ci(policy, df, ope_method, alpha=0.05):
    # df = valid_df
    # df = train_df

    with torch.no_grad():
        pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(df, policy)
        lb_ub, wis_list = bca_bootstrap(pibs, pies, rewards, lengths, alpha=alpha, ope_method=ope_method,
                                        max_time=max_time)  # 0.01, 0.05, 0.33

        avg, _ = ope_method(pibs, pies, rewards, lengths, max_time=max_time)

    #     policy_name = str(policy).lstrip("<class '__main__.").rstrip("'>")
    #     ope_name = str(cwpdis_ope).split()[1]

    #     print("{}, {} score: {:.2f}, ESS: {}".format(policy_name, ope_name, ope_folds / 2, ess_folds / 2))
    return lb_ub, avg.item(), wis_list


def compute_ci_for_policies(bc_policy, df, alpha=0.05, get_wis=False):
    lb_ub, wis, wis_list = evaluate_policy_for_ci(bc_policy, df, cwpdis_ope, alpha=alpha)
    score, ess = ope_evaluate(bc_policy, valid_df, cwpdis_ope)

    # print(extra_message + "{} with CI: {}, val ESS: {}".format(wis, lb_ub, ess))

    if get_wis:
        return wis_list
    else:
        return wis, np.mean(wis_list), lb_ub, ess

import os
import cloudpickle
import csv
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, default='1', help="split")
    parser.add_argument("--folder", type=str, default='results_feb23', help="split")
    args = parser.parse_args()

    if args.data_split == '1':
        train_df = pd.read_csv("./primer_data/train_df_clean.csv")
        valid_df = pd.read_csv("./primer_data/valid_df_clean.csv")
    else:
        train_df = pd.read_csv(f"./primer_data/train_df_clean_split{args.data_split}.csv")
        valid_df = pd.read_csv(f"./primer_data/valid_df_clean_split{args.data_split}.csv")

    pg_lr_set = {1:3e-4, 2:3e-4, 4:3e-4, 10:1e-5}

    # pbar = tqdm(total=4*25) # 4 * 25
    pbar = tqdm(total=2*25) # 4 * 25

    os.makedirs(args.folder, exist_ok=True)

    # one danger: different lambda_ess requires different early stop criteria

    logfile = open(pjoin(args.folder, "log.csv"), 'w')
    log_writer = csv.writer(logfile)

    log_writer.writerow(['exp_name', 'seed', 'bc_train_ess', 'bc_val_ess', 'train_ope',
                         'train_ess', 'train_lb', 'train_ub', 'val_ope', 'val_ess', 'val_lb', 'val_ub'])

    for lambda_ess in [4]: # [1, 2, 4, 10]:
        for gr_safety_thresh in [0.05, 0.1]:
            # gr_safety_thresh = 0.03
            for seed in seeds[:25]:
                set_random_seed(seed)  # seed

                bc_policy = MLPPolicy([10, 64, 64, 4])
                bc_train_losses, bc_train_esss, bc_val_losses, bc_val_esss = bc_train_policy(bc_policy, train_df, valid_df,
                                                                                             epochs=80, lr=1e-3, verbose=False,
                                                                                             early_stop=True)

                pg_lr = pg_lr_set[lambda_ess]
                train_losses, train_opes, train_esss, val_opes, val_esss = offpolicy_pg_training(bc_policy, train_df, valid_df,
                                                                                                 cwpdis_ope, lr=pg_lr,
                                                                                                 epochs=40, lambda_ess=lambda_ess,
                                                                                                 gr_safety_thresh=gr_safety_thresh,
                                                                                                 verbose=False,
                                                                                                 # for better plot, take this out to train fully
                                                                                                 early_stop_ess=25)  # 20 # 17.5

                # CI on train
                set_random_seed(seed)
                train_wis, train_avg_sampled_wis, train_lb_ub, train_ess = compute_ci_for_policies(bc_policy, train_df, alpha=0.05)

                # CI on validation
                set_random_seed(seed)
                wis, avg_sampled_wis, lb_ub, ess = compute_ci_for_policies(bc_policy, valid_df, alpha=0.05)

                file_name = "lambda_ess_{}_gr_thres_{}_seed_{}.pkl".format(lambda_ess, str(gr_safety_thresh).replace(".", ""), seed)
                # cloudpickle.dump([
                #     bc_train_losses, bc_train_esss, bc_val_losses, bc_val_esss,
                #     train_losses, train_opes, train_esss, val_opes, val_esss,
                #     train_wis, train_avg_sampled_wis, train_lb_ub, train_ess,
                #     wis, avg_sampled_wis, lb_ub, ess
                # ], open("./results_feb23/" + file_name, 'wb'))

                exp_name = 'lambda_ess_{}_gr_thres_{}'.format(lambda_ess, str(gr_safety_thresh).replace(".", ""))
                try:
                    log_writer.writerow([exp_name, str(seed), bc_train_esss[-1], bc_val_esss[-1],
                                         train_opes[-2], train_esss[-2], # train_opes[-1], train_esss[-1],
                                         train_lb_ub[0], train_lb_ub[1],  val_opes[-2].item(), val_esss[-2].item(), lb_ub[0], lb_ub[1]])
                except:
                    print("error!")
                    pass

                pbar.update(1)
                logfile.flush()

    logfile.close()