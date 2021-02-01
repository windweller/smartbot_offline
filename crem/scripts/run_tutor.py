import gym
import numpy as np
import torch
import argparse
import uuid
import json
import os

from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))

d2 = dirname(dirname(dirname(abspath(__file__))))

import sys
sys.path.append(d)
sys.path.append(d2)

import rem.utils as utils
import rem.DDPG as DDPG
import rem.BCQ as BCQ
import rem.TD3 as TD3
import rem.REM as REM
import rem.conv_REM as conv_REM
import rem.conv_BCQ as conv_BCQ
import rem.RSEM as RSEM
import rem.DDPG_REM as DDPG_REM

from tutor_env import make_tutor_env

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_type", default="Robust")             # Prepends name to filename.
    parser.add_argument("--eval_freq", default=4, type=float)         # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=200, type=float)     # Max time steps to run environment for, 1e6
    parser.add_argument("--agent_name", default="BCQ")
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--num_heads", default=10, type=int)  # 100
    parser.add_argument("--prefix", default="default")
    parser.add_argument("--output_dir", default="results")

    parser.add_argument("--kc", default=1, type=int)

    args = parser.parse_args()

    args.env_name = "tutor-v0"

    file_name = "%s_%s_%s_%s" % (args.agent_name, args.env_name, str(args.seed), str(args.lr))
    if args.agent_name == 'REM':
        file_name += '_%s' % (args.num_heads)
        if args.prefix != "default":
            file_name += '_%s' % (args.prefix)
    # buffer_name = "%s_%s_%s" % (args.buffer_type, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: " + file_name)
    print("---------------------------------------")

    #results_dir = os.path.join(args.output_dir, args.agent_name, str(uuid.uuid4()))
    #os.makedirs(results_dir, exist_ok=True)
    # with open(os.path.join(results_dir, 'params.json'), 'w') as params_file:
    #     json.dump({
    #         'env_name': args.env_name,
    #         'seed': args.seed,
    #     }, params_file)

    env = make_tutor_env()

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = 1
    max_action = 1 # float(env.action_space.high[0])

    # Initialize policy
    kwargs = {'lr': args.lr}
    if args.agent_name in ['conv_REM', 'REM', 'RSEM', 'DDPG_REM']:
        kwargs.update(num_heads=args.num_heads)

    if args.agent_name == 'BCQ':
        policy_agent = BCQ.BCQ
    elif args.agent_name == 'TD3':
        policy_agent = TD3.TD3
    elif args.agent_name == 'REM':
        policy_agent = REM.REM

    policy = policy_agent(state_dim, action_dim, max_action, **kwargs)

    # Load buffer
    replay_buffer = utils.ReplayBuffer()
    # replay_buffer.load(buffer_name)
    dataset = env.get_dataset(h5path='/data/anie/offline_rl/data/train_student_to_states_KC{}.h5'.format(args.kc))
    N = dataset['rewards'].shape[0]

    episode_step = 0
    for i in range(N - 1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i + 1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        # Don't apply terminals on the last step of an episode
        if episode_step == env._max_episode_steps - 1:
            episode_step = 0
            continue
        if done_bool:
            episode_step = 0

        # s, s2, a, r, d
        replay_buffer.add((obs, new_obs, action, reward, done_bool))
        episode_step += 1

    # training is here...
    evaluations = []

    episode_num = 0
    done = True

    training_iters = 0
    losses = []
    while training_iters < args.max_timesteps:
        per_freq_losses = policy.train(replay_buffer, iterations=int(args.eval_freq))

        # evaluations.append(evaluate_policy(policy))
        # np.save(results_dir + "/" + file_name, evaluations)

        training_iters += args.eval_freq
        # print("Training iterations: " + str(training_iters))

        losses.extend(per_freq_losses)
    print("KC_{}_{}_policy finished".format(args.kc, args.agent_name))

    os.makedirs('result_loss', exist_ok=True)
    json.dump(losses, open("./result_loss/kc_{}_{}_policy_losses.json".format(args.kc, args.agent_name), 'w'))