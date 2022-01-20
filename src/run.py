import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

############################################## from maddpg-pytorch ##############################################
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
import copy

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu, constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  
            # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def onehot_from_logits(logits, eps=0.0, dim=1):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(dim, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(th.eye(logits.shape[1])[[np.random.choice(range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return th.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in enumerate(th.rand(logits.shape[0]))])


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=th.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -th.log(-th.log(U + eps) + eps)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature, dim=1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    sampled = sample_gumbel(logits.shape, tens_type=type(logits.data))

    if logits.device != 'cpu':
        sampled = sampled.to(logits.device)

    y = logits + sampled
    
    return F.softmax(y / temperature, dim=dim)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False, dim=1):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, dim=dim)
    if hard:
        y_hard = onehot_from_logits(y, dim=dim)
        y = (y_hard - y).detach() + y
    return y


class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic=None, hidden_dim=64, lr=0.01):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(num_in_pol, num_out_pol, hidden_dim=hidden_dim, constrain_out=True)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol, hidden_dim=hidden_dim, constrain_out=True)
        hard_update(self.target_policy, self.policy)
        
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        
    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if explore:
            action = gumbel_softmax(action, hard=True)
        else:
            action = onehot_from_logits(action)
        return action

    def load_params_without_optims(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer = None
############################################## from maddpg-pytorch ##############################################

def run(_run, _config, _log):
    th.set_num_threads(1)
    
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def load_pretrained_opponents(opponents, scenario_name, filename, num_cooperating_agents, num_opponent_agents, device='gpu'):
    save_dict = th.load(filename, map_location=th.device('cpu')) if not th.cuda.is_available() else th.load(filename)

    if device == 'gpu' and th.cuda.is_available():
        fn = lambda x: x.cuda()
    else:
        fn = lambda x: x.cpu()

    if scenario_name in ['simple_tag', 'simple_world']:
        opponent_params = save_dict['agent_params'][num_cooperating_agents:]
    elif scenario_name in ['simple_adversary', 'simple_crypto']:
        opponent_params = save_dict['agent_params'][:num_opponent_agents]

    for i, params in zip(range(len(opponents)), opponent_params):
        opponents[i].load_params_without_optims(params)

        opponents[i].policy.eval()
        opponents[i].target_policy.eval()
        opponents[i].policy = fn(opponents[i].policy)
        opponents[i].target_policy = fn(opponents[i].target_policy)

    print ('finished loading pretrained opponents from: {}'.format(filename))


def is_opponent(scenario_name, agent_idx, num_cooperating_agents, num_opponent_agents):
    is_opponent_flag = False
    if scenario_name in ['simple_tag', 'simple_world']:
        if agent_idx >= num_cooperating_agents:
            is_opponent_flag = True
    elif scenario_name in ['simple_adversary', 'simple_crypto']:
        if agent_idx < num_opponent_agents:
            is_opponent_flag = True
    return is_opponent_flag


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    if args.res:
        scheme['future_discounted_return'] = {"vshape": (1,)}
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(
        scheme, 
        groups, 
        args.buffer_size, 
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    opponents = None
    if args.env == 'mpe_env':
        num_agents, num_cooperating_agents, num_opponent_agents = env_info['agent_num_infos']
        opponent_init_params = []
        for agent_idx in range(num_agents):
            if is_opponent(args.scenario_name, agent_idx, num_cooperating_agents, num_opponent_agents):
                opponent_init_params.append({
                    'num_in_pol': env_info['observation_space'][agent_idx].shape[0], 
                    'num_out_pol': env_info['action_space'][agent_idx].n
                })
        opponents = [DDPGAgent(**params) for params in opponent_init_params]

        pretrained_opponent_model_dir = './src/pretrained_models/{}_{}_episode_{}.pt'.format(
            args.scenario_name, 
            args.opponent_algo, 
            args.load_opponent_model_episode
        )
        load_pretrained_opponents(opponents, args.scenario_name, pretrained_opponent_model_dir, num_cooperating_agents, num_opponent_agents)

    # Give runner the scheme
    runner.setup(
        scheme=scheme, groups=groups, preprocess=preprocess, mac=mac,
        opponents=opponents
    )

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    episode_idx = 0

    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        episode_batch, episode_return = runner.run(test_mode=False)

        # compute episode returns
        if args.res:
            # episode_batch.data.transition_data['reward']: 1, episode_len + 1, 1
            episode_batch_rewards = episode_batch.data.transition_data['reward'].squeeze(-1).squeeze(0)

            episode_batch_future_discounted_returns = [0 for _ in range(episode_batch_rewards.shape[0])]
            future_discounted_return = 0.
            for step_idx in range(episode_batch_rewards.shape[0] - 1, -1, -1):
                future_discounted_return = episode_batch_rewards[step_idx] + args.gamma * future_discounted_return
                episode_batch_future_discounted_returns[step_idx] = future_discounted_return
            
            episode_batch_future_discounted_returns = th.tensor(episode_batch_future_discounted_returns).unsqueeze(0).unsqueeze(-1).cuda()
            episode_batch.data.transition_data['future_discounted_return'] = episode_batch_future_discounted_returns

            assert len(episode_batch.data.episode_data.items()) == 0

        buffer.insert_episode_batch(episode_batch)

        episode_idx += 1

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        if args.env != 'mpe_env':
            n_test_runs = max(1, args.test_nepisode // runner.batch_size)
            if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
                logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
                last_time = time.time()

                last_test_T = runner.t_env
                for _ in range(n_test_runs):
                    runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            stats = logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
