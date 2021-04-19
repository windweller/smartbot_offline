import pickle
import pandas as pd
import torch
import numpy as np

class TrainKNNPolicy(object):
    def __init__(self, split_num=1, K=5):
        # K=5: 5 closest neighbors' action probabilities averaged
        self.nA = 4
        self.K = K
        knn_data = pickle.load(open(f'./primer_data/knn_states_new_new_split{split_num}', 'rb'))
        self.train_knn_state_index = knn_data['train_knn_state_index']
        self.valid_knn_state_index = knn_data['valid_knn_state_index']

        self.train_df = pd.read_csv(f"./primer_data/train_df_new_new_split{split_num}.csv")
        self.target_names = ["p_hint", "p_nothing", "p_encourage", "p_question"]

    def get_action_probability(self, x_indices, no_grad=True, action_mask=None):
        # we look up the most similar state in training
        batch_size = x_indices.size
        action_prob = torch.zeros(batch_size, self.nA)

        for i, ind in enumerate(x_indices):
            k_train_indices = self.valid_knn_state_index[ind][:self.K]
            action_probs = self.train_df.iloc[k_train_indices][self.target_names]
            action_prob[i, :] = torch.from_numpy(np.asarray(action_probs.mean(0), dtype=float))

        action_prob[:, 0] = 1.0
        return action_prob

target_names = ["p_hint", "p_nothing", "p_encourage", "p_question"]
feature_names= ['stage_norm', 'failed_attempts_norm', 'pos_norm', 'neg_norm',
                 'hel_norm', 'anxiety_norm', 'grade_norm', 'pre', 'anxiety', 'stage']

def compute_is_weights_for_knn_policy(behavior_df, eval_policy, eps=0.05, temp=0.1,
                                     reward_column='adjusted_score', no_grad=True,
                                     gr_safety_thresh=0.0, is_train=True, normalize_reward=False, use_knn=False):
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
        # instead of `torch.from_numpy(features).float()`
        eval_action_probs = eval_policy.get_action_probability(data.index, no_grad,
                                                               action_mask=gr_mask)

        pies[idx, :T] = torch.hstack([eval_action_probs[i, a] for i, a in enumerate(actions)])

    return pibs, pies, rewards, lengths.astype(int), MAX_TIME


def compute_is_weights_for_bc_policy(behavior_df, eval_policy, eps=0.05, temp=0.1,
                                     reward_column='adjusted_score', no_grad=True,
                                     gr_safety_thresh=0.0, is_train=True, normalize_reward=False, use_knn=False,
                                     use_model_round=False):
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
        if not use_model_round:
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