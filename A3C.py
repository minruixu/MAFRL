import os
import gym
import time
import tqdm
import torch
import pprint
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import BasicLogger,tqdm_config, MovAvg
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, Collector, VectorReplayBuffer, ReplayBuffer, to_numpy, Batch
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.policy import A2CPolicy, ImitationPolicy
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer
from net import Net, ServerActor, RelayActor, NFActor, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='FLIM-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--il-lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--step-per-epoch', type=int, default=32)
    parser.add_argument('--il-step-per-epoch', type=int, default=1000)
    parser.add_argument('--episode-per-collect', type=int, default=16)
    parser.add_argument('--step-per-collect', type=int, default=16)
    parser.add_argument('--update-per-step', type=float, default=1 / 16)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--layer-num', type=int, default=2)
    parser.add_argument('--hidden-sizes', type=int,
                        nargs='*', default=[64, 64])
    parser.add_argument('--imitation-hidden-sizes', type=int,
                        nargs='*', default=[128])
    parser.add_argument('--training-num', type=int, default=3)
    parser.add_argument('--test-num', type=int, default=3)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    # a2c special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=None)
    parser.add_argument('--gae-lambda', type=float, default=1.)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    # env special
    parser.add_argument('--data-quality', type=float,default=0)
    parser.add_argument('--data-quantity', type=float,default=0)
    parser.add_argument('--psi', type=float, default=0)
    parser.add_argument('--nu', type=float, default=0)
    args = parser.parse_known_args()[0]
    return args

def build_policy(no,args):
    if no == 0:
        # server policy
        net = Net(args.layer_num, args.state_shape, device=args.device)
        actor = ServerActor(net, (10,)).to(args.device)
        critic = Critic(net).to(args.device)
        # orthogonal initialization
        for m in list(actor.modules()) + list(critic.modules()):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        optim = torch.optim.Adam(list(
            actor.parameters()) + list(critic.parameters()), lr=args.lr)
        dist = torch.distributions.Categorical
        policy = A2CPolicy(
        actor, critic, optim, dist,
        discount_factor=args.gamma, 
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef, 
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm, 
        reward_normalization=args.rew_norm)
    elif no == 1:
        # ND policy
        net = Net(args.layer_num, (4,), device=args.device)
        actor = NFActor(net, (10,)).to(args.device)
        critic = Critic(net).to(args.device)
        # orthogonal initialization
        for m in list(actor.modules()) + list(critic.modules()):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        optim = torch.optim.Adam(list(
            actor.parameters()) + list(critic.parameters()), lr=args.lr)
        dist = torch.distributions.Categorical
        policy = A2CPolicy(
        actor, critic, optim, dist,
        discount_factor=args.gamma, 
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef, 
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm, 
        reward_normalization=args.rew_norm)
    elif no == 2:
        net = Net(args.layer_num, (4,), device=args.device)
        actor = RelayActor(net, (10,)).to(args.device)
        critic = Critic(net).to(args.device)
        # orthogonal initialization
        for m in list(actor.modules()) + list(critic.modules()):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        optim = torch.optim.Adam(list(
            actor.parameters()) + list(critic.parameters()), lr=args.lr)
        dist = torch.distributions.Categorical
        policy = A2CPolicy(
        actor, critic, optim, dist,
        discount_factor=args.gamma, 
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef, 
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm, 
        reward_normalization=args.rew_norm)
    else:
        net = Net(args.layer_num, (4,), device=args.device)
        actor = NFActor(net, (10,)).to(args.device)
        critic = Critic(net).to(args.device)
        # orthogonal initialization
        for m in list(actor.modules()) + list(critic.modules()):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        optim = torch.optim.Adam(list(
            actor.parameters()) + list(critic.parameters()), lr=args.lr)
        dist = torch.distributions.Categorical
        policy = A2CPolicy(
        actor, critic, optim, dist,
        discount_factor=args.gamma, 
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef, 
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm, 
        reward_normalization=args.rew_norm)
    return policy

def test_a2c_with_il(args=get_args()):
    torch.set_num_threads(1)  # for poor CPU
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = gym.make(args.task)
    # train_envs = DummyVectorEnv(
    #     [lambda: gym.make(args.task) for _ in range(args.training_num)])
    # # test_envs = gym.make(args.task)
    # test_envs = DummyVectorEnv(
    #     [lambda: gym.make(args.task) for _ in range(args.test_num)])
    if args.data_quantity != 0:
        env.set_data_quantity(args.data_quantity)
    if args.data_quality != 0:
        env.set_data_quality(args.data_quality)
    if args.psi != 0:
        env.set_psi(args.psi)
    if args.nu != 0:
        env.set_nu(args.nu)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)
    # model

    # server policy
    server_policy = build_policy(0,args)

    # client policy
    ND_policy = build_policy(1, args)
    RD_policy = build_policy(2, args)
    FD_policy = build_policy(3, args)
    # 不用collector，用replaybuffer
    server_buffer = ReplayBuffer(args.buffer_size)
    ND_buffer = ReplayBuffer(args.buffer_size)
    RD_buffer = ReplayBuffer(args.buffer_size)
    FD_buffer = ReplayBuffer(args.buffer_size)

    # log
    log_path = os.path.join(args.logdir, args.task, 'a2c')
    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))


    start_time = time.time()
    _server_obs, _ND_obs, _RD_obs, _FD_obs = env.reset()
    _server_act = _server_rew = _done = _info = None
    server_buffer.reset()
    _ND_act = _ND_rew = _RD_act = _RD_rew = _FD_act = _FD_rew = [None]
    ND_buffer.reset()
    RD_buffer.reset()
    FD_buffer.reset()
    all_server_costs = []
    all_ND_utility = []
    all_RD_utility = []
    all_FD_utility = []
    all_leak_probability = []
    for epoch in range(1, 1 + args.epoch):
        # 每个epoch收集N*T数据，然后用B训练M次
        server_costs = []
        ND_utility = []
        FD_utility = []
        RD_utility = []
        leak_probability = []
        payment = []
        expected_time = []
        training_time = []
        for e in range(5):
            with tqdm.tqdm(total=args.step_per_epoch, desc=f'Epoch #{epoch}',
                        **tqdm_config) as t:
                while t.n < t.total:
                    # 收集数据,不用梯度
                    # server
                    _server_obs, _ND_obs, _RD_obs, _FD_obs = env.reset()
                    server_batch = Batch(obs=_server_obs, act=_server_act, rew=_server_rew,
                        done=_done, obs_next=None, info=_info, policy=None)
                    with torch.no_grad():
                        server_result = server_policy(server_batch, None)
                    _server_policy = [{}]
                    _server_act = to_numpy(server_result.act)
                    # ND
                    ND_batch = Batch(obs=_ND_obs, act=_ND_act, rew=_ND_rew,
                                        done=_done, obs_next=None, info=_info, policy=None)
                    with torch.no_grad():
                        ND_result = ND_policy(ND_batch, None)
                    _ND_policy = [{}]
                    _ND_act = to_numpy(ND_result.act)
                    # RD
                    RD_batch = Batch(obs=_RD_obs, act=_RD_act, rew=_RD_rew,
                                    done=_done, obs_next=None, info=_info, policy=None)
                    with torch.no_grad():
                        RD_result = RD_policy(RD_batch, None)
                    _RD_policy = [{}]
                    _RD_act = to_numpy(RD_result.act)
                    # FD
                    FD_batch = Batch(obs=_FD_obs, act=_FD_act, rew=_FD_rew,
                                    done=_done, obs_next=None, info=_info, policy=None)
                    with torch.no_grad():
                        FD_result = FD_policy(FD_batch, None)
                    _FD_policy = [{}]
                    _FD_act = to_numpy(FD_result.act)
                    # print(_ND_act.shape)
                    server_obs_next, ND_obs_next, RD_obs_next, FD_obs_next, _server_rew, _client_rew, _done, _info = env.step(_server_act[0],_ND_act[0],_RD_act[0],_FD_act[0])

                    server_costs.append(_server_rew)
                    ND_utility.append(_client_rew[0])
                    RD_utility.append(_client_rew[1])
                    FD_utility.append(_client_rew[2])
                    leak_probability.append(_info[0]["leak"])
                    payment.append(env.payment)
                    expected_time.append(env.expected_time)
                    training_time.append(env.global_time*env.time_lambda)
                    # 加入replay buffer
                    server_buffer.add(
                        Batch(obs=_server_obs[0], act=_server_act[0], rew=_server_rew[0],
                        done=_done[0], obs_next=server_obs_next[0], info=_info[0],
                        policy=_server_policy[0]))
                    ND_buffer.add(
                        Batch(obs=_ND_obs[0], act=_ND_act[0], rew=_client_rew[0],
                        done=_done[0], obs_next=ND_obs_next[0], info=_info[0],
                        policy=_ND_policy[0]))
                    RD_buffer.add(
                        Batch(obs=_RD_obs[0], act=_RD_act[0], rew=_client_rew[1],
                        done=_done[0], obs_next=RD_obs_next[0], info=_info[0],
                        policy=_RD_policy[0]))
                    FD_buffer.add(
                        Batch(obs=_FD_obs[0], act=_FD_act[0], rew=_client_rew[2],
                        done=_done[0], obs_next=FD_obs_next[0], info=_info[0],
                        policy=_FD_policy[0]))
                    t.update(1)
                    _server_obs = server_obs_next
                    _ND_obs = ND_obs_next
                    _RD_obs = RD_obs_next
                    _FD_obs = FD_obs_next
        all_server_costs.append(np.array(server_costs).mean())
        all_ND_utility.append(np.array(ND_utility).mean())
        all_RD_utility.append(np.array(RD_utility).mean())
        all_FD_utility.append(np.array(FD_utility).mean())
        all_leak_probability.append(np.array(leak_probability).mean())
        print("current bandwidth:",env.bandwidth)
        print("leak signal:",env.leak_NU,env.leak_FU)
        print("current server cost:",np.array(server_costs).mean())
        print("current device utility:",all_ND_utility[-1],all_RD_utility[-1],all_FD_utility[-1])
        print("leak probability:", all_leak_probability[-1])
        print("server_act:",_server_act[0])
        print("device_acts:",_ND_act[0],_RD_act[0],_FD_act[0])
        print("payment cost:",np.array(payment).mean())
        print("Expected time cost:",np.array(expected_time).mean())
        print("Training time cost:",np.array(training_time).mean())
        # print("server_act:",_server_act)
        # print("client_act:",_client_act)
        print("info:", env.communication_time, env.computation_time, env.K_theta)
        server_batch_data, server_indice = server_buffer.sample(0)
        server_batch_data = server_policy.process_fn(server_batch_data, server_buffer, server_indice)
        server_policy.learn( server_batch_data,args.batch_size,args.repeat_per_collect)
        # server_buffer.reset()

        ND_batch_data, ND_indice = ND_buffer.sample(0)
        ND_batch_data = ND_policy.process_fn(ND_batch_data, ND_buffer, ND_indice)
        ND_policy.learn(ND_batch_data, args.batch_size, args.repeat_per_collect)
        # ND_buffer.reset()

        RD_batch_data, RD_indice = RD_buffer.sample(0)
        RD_batch_data = RD_policy.process_fn(RD_batch_data, RD_buffer, RD_indice)
        RD_policy.learn(RD_batch_data, args.batch_size, args.repeat_per_collect)
        # RD_buffer.reset()

        FD_batch_data, FD_indice = FD_buffer.sample(0)
        FD_batch_data = FD_policy.process_fn(FD_batch_data, FD_buffer, FD_indice)
        FD_policy.learn(FD_batch_data, args.batch_size, args.repeat_per_collect)
        # FD_buffer.reset()
    print("all_server_cost:",all_server_costs)
    print("all_ND_utility:",all_ND_utility)
    print("all_RD_utility:", all_RD_utility)
    print("all_FD_utility:", all_FD_utility)
    print("all_leak_probability:",all_leak_probability)
    plt.plot(all_server_costs)
    plt.show()


if __name__ == '__main__':
    test_a2c_with_il()
