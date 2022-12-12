import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import time
from collections import deque
from copy import deepcopy
from typing import Type

import numpy as np
import torch.optim as optim
import lbforaging
from lbforaging import ForagingEnv
from multicapture import MulticaptureEnv
from torch.nn.utils.rnn import pad_sequence

import config
from agent import RQNetwork
from utils.misc import *
from utils.memory import Buffer, EpisodeBuffer
from utils.mixers import VDNMixer, QMixer

def eval_episode(env, agents, qnet, max_steps, averages=5):
    toReturn = []
    for _ in range(averages):
        returns = 0
        ep_step = 0
        done = False

        observations = obs_list_array_to_list_tensor(env.reset(), device)

        action = {k: th.empty(1) for k in agents}
        h = {k: qnet[0].init_hidden() for k in agents}
        h_ = {k: qnet[0].init_hidden() for k in agents}

        while not done and ep_step < max_steps:
            ep_step += 1
            with th.no_grad():
                for k in agents: 
                    h[k] = h_[k]
                    (action[k], h_[k]) = qnet[k].get_action(observations[k], h[k], 0.0)

                observations_, rewards, done, infos = env.step(tuple((int(action[k]) for k in agents)))
                observations = obs_list_array_to_list_tensor(observations_, device)

            returns += sum(rewards)/len(agents)
        toReturn.append(returns)
    return np.mean(toReturn)

def obs_list_array_to_list_tensor(data: List[Array], device: th.device, astype: Type = th.float32) -> List[Tensor]:
    return [th.as_tensor(d[np.newaxis], dtype=astype).to(device) for d in data]
    
def list_array_to_list_tensor(data: List[Array], device: th.device) -> List[Tensor]:
    return [th.as_tensor(d).to(device) for d in data]

def _get_pad_and_mask_from_obs(observations):
    zeropad_observations = pad_sequence(observations, padding_value=th.tensor(float('0')), batch_first=True)
    nanpad_observations = pad_sequence(observations, padding_value=th.tensor(float('nan')), batch_first=True)
   
    return zeropad_observations, ~th.isnan(nanpad_observations).any(-1)

def make_env(args) -> gym.Env:
    
    if args.env == "lbforaging":
        n_agents = args.n_agents
        max_player_level = args.player_lvl
        grid_shape = args.grid_size
        max_food = args.n_food
        sight = args.partial_obs
        max_steps = args.max_steps
        force_coop = args.coop
        penalty = args.load_penalty
        step_penalty = args.step_penalty

        if sight:
            s = 2
        else:
            s = grid_shape
        
        return ForagingEnv(n_agents,max_player_level, (grid_shape, grid_shape), max_food, s, max_steps, force_coop, penalty=penalty, step_penalty=step_penalty)
    elif args.env == "multicapture":
        n_agents = args.n_agents
        n_targets = args.n_targets
        grid_shape = args.grid_size
        target_capacity = args.target_capacity
        sight = args.partial_obs
        max_steps = args.max_steps
        target_reward = args.target_reward
        penalty = args.load_penalty
        step_penalty = args.step_penalty

        if sight:
            s = 2
        else:
            s = grid_shape
        
        return MulticaptureEnv(n_agents, (grid_shape, grid_shape), n_targets, target_capacity, s, max_steps, target_reward=target_reward, penalty=penalty, step_penalty=step_penalty)
    else:
        raise(Exception("Other environments not implemented!")) 
    

if __name__ == "__main__":
    args = config.parse_args()

    run_name = f"{args.env}__{args.tag}__{args.seed}__{int(time.time())}__{np.random.randint(0, 100)}"
    summary_w, wandb_path = init_loggers(run_name, args)

    # Torch init and seeding 
    set_seeds(args.seed, args.th_deterministic)
    device = set_torch(args.n_cpus, args.cuda)

    env = make_env(args) 
    env.reset()  
    
    n_agents = env.n_agent
    agents = range(n_agents)

    # RQ-Networks setup
    qnet, tg_qnet, qnet_opt, buffer = [{} for _ in range(4)]
    for k in agents:
        qnet[k] = RQNetwork(env, args.h_size, args.n_hidden).to(device)    # th.jit.script
        tg_qnet[k] = deepcopy(qnet[k])
        for weights_tg_qnet in tg_qnet[k].parameters(): weights_tg_qnet.requires_grad = False
        qnet_opt[k] = optim.Adam(qnet[k].parameters(), lr=args.lr)    
        buffer[k] = Buffer(env, args.buffer_size, args.batch_size, args.h_size)

    if args.mixer == "VDN":
        net_params = []
        for k in agents:
            net_params += list(qnet[k].parameters())
        mixer, tg_mixer = VDNMixer(), VDNMixer()
        net_params += list(mixer.parameters())
        qnet_opt = optim.Adam(net_params, lr=args.lr) 
    elif args.mixer == "QMIX":
        net_params = []
        for k in agents:
            net_params += list(qnet[k].parameters())
        mixer= QMixer(args.n_agents, args.mix_embed_dim, args.hypernet_embed_dim, args.hypernet_layers, env.state_size)
        tg_mixer = deepcopy(mixer)
        net_params += list(mixer.parameters())
        qnet_opt = optim.Adam(net_params, lr=args.lr) 
    else:
        mixer = None
        tg_mixer = None

    # Epsilon decay
    epsilon = args.eps_start
    eps_lin_decay = (args.eps_end - args.eps_start) / (args.eps_frac * (args.tot_episodes - args.delay_start))

    # Hysteretic lr -- set with epsilon decay values --
    hyst, hyst_end = args.hyst_start, args.hyst_end
    hyst_lin_decay = (args.hyst_end - args.hyst_start) / (args.eps_frac * (args.tot_episodes - args.delay_start))

    # Training metrics
    global_step = 0
    start_time = time.time()
    reward_q = deque(maxlen=args.last_n)

    eval_freq = 10
   
    for ep in range(1, args.tot_episodes):

        if ep % eval_freq == 0:
            eval_reward = eval_episode(env, agents, qnet, args.max_steps, averages=5)
            record = {
                    'Episode': ep,
                    'Global_Step': global_step,
                    'Eval_Reward': eval_reward
                    }

            if args.verbose:
                    print(f"E: {ep},\n\t "
                        f"Eval_Reward: {eval_reward},\n\t "
                        f"Global_Step: {global_step},\n\t "
                    )
            
            if args.tb_log: summary_w.add_scalars('Test', record, global_step) 
            if args.wandb_log: wandb.log(record)


        # Environment reset
        observations = obs_list_array_to_list_tensor(env.reset(), device)
        ep_buffer = {k: EpisodeBuffer(env, args.max_steps, args.h_size, device) for k in agents}
        
        # Maybe we can just init these as an empty dictionary (expect for the two valid)
        obs = {k: th.empty(observations[0].shape) for k in agents}
        action = {k: th.empty(1) for k in agents}
        h = {k: qnet[0].init_hidden() for k in agents}
        h_ = {k: qnet[0].init_hidden() for k in agents}
        ep_step, ep_reward = 0, 0

        state = env.get_state()

        while True:
            global_step += 1
            ep_step += 1
            
            with th.no_grad():
                for k in agents:
                    obs[k] = observations[k]  
                    h[k] = h_[k] 
                    (action[k], h_[k]) = qnet[k].get_action(observations[k], h[k], epsilon)
            
            observations_, rewards, done, infos = env.step(tuple((int(action[k]) for k in agents)))
            
            next_state = env.get_state()

            rewards = list_array_to_list_tensor(rewards, device)
            observations = obs_list_array_to_list_tensor(observations_, device)
            
            for k in agents:            
                ep_buffer[k].store(
                    obs[k],
                    action[k],
                    rewards[k],
                    observations[k],
                    done,
                    h[k],
                    h_[k],
                    th.from_numpy(state).view(-1),
                    th.from_numpy(next_state).view(-1)
                )     

            state = next_state

            ep_reward += sum(rewards)/n_agents

            if done or ep_step == args.max_steps - 1:
                
                for k in agents:
                    buffer[k].store(ep_buffer[k])

                reward_q.append(ep_reward)

                record = {
                    'Episode': ep,
                    'Global_Step': global_step,
                    'Reward': ep_reward,
                    'Avg_Reward': np.mean(reward_q),
                    'Epsilon': epsilon,
                    'Hysteretic': hyst
                }
                

                if args.tb_log: summary_w.add_scalars('Training', record, global_step) 
                if args.wandb_log: wandb.log(record)

                if args.verbose and ep % 50 == 0:
                    print(f"E: {ep},\n\t "
                        f"Reward: {record['Reward']},\n\t "
                        f"Avg_Reward: {record['Avg_Reward']},\n\t "
                        f"Global_Step: {record['Global_Step']},\n\t "
                        f"Epsilon: {record['Epsilon']},\n\t "
                        f"Hysteretic: {record['Hysteretic']},\n\t "
                        f"Step: {ep_step},\n\t "
                    )
                    
                epsilon = max(eps_lin_decay * ep + args.eps_start, args.eps_end)
                hyst = min(hyst_lin_decay * ep + args.hyst_start, args.hyst_end)

                break

            # Training
            if ep >= args.delay_start and global_step % args.update_freq == 0:
                
                agent_qs, target_qs, j_rewards = [], [], []

                for i, k in enumerate(agents):
                    #for _ in range(args.update_freq):  # The rate between update_freq and n° updates is usually 1

                    # Sequential sample (i.e., the same episodes and same steps)
                    if i == 0: batch, ep_idxs, start_idxs = buffer[k].sample(device)
                    else: batch, _, _ = buffer[k].sample(device, ep_idxs, start_idxs)           

                    lengths = batch['lengths']

                    states = batch['states']
                    next_states = batch['next_states']

                    batch['observations'] = th.split_with_sizes(batch['observations'], lengths)
                    batch['observations_'] = th.split_with_sizes(batch['observations_'], lengths)
                    batch['hs'] = th.split_with_sizes(batch['hs'], lengths)
                    batch['hs_'] = th.split_with_sizes(batch['hs_'], lengths)

                    rewards = batch['rewards'].view(-1, 1)
                    dones = batch['dones'].view(-1, 1)
                    actions = batch['actions'].view(-1, 1).to(th.int64)
                    
                    pad_observations, mask_observations  = _get_pad_and_mask_from_obs(batch['observations'])
                    pad_observations_, mask_observations_  = _get_pad_and_mask_from_obs(batch['observations_'])
                    pad_histories, _ = _get_pad_and_mask_from_obs(batch['hs'])      # Could just take 1° element from each object in the tuple
                    pad_histories_, _ = _get_pad_and_mask_from_obs(batch['hs_'])
                    pad_histories = pad_histories[:,0][np.newaxis]
                    pad_histories_ = pad_histories_[:,0][np.newaxis]

                    with th.no_grad():
                        # Get only non padded entries
                        tg_qvalues_, _ = tg_qnet[k](pad_observations_)
                        qvalues_, _ = qnet[k](pad_observations_)
                    
                        tg_qvalues_ = tg_qvalues_[mask_observations_]
                        qvalues_ = qvalues_[mask_observations_]
                        
                        actions_ = th.argmax(qvalues_, dim=-1, keepdims=True).type(th.int64)
                        tg_qvalue_ = tg_qvalues_.gather(-1, actions_)

                        if mixer == None:
                            y = rewards + args.gamma * tg_qvalue_ * (1 - dones)
                        else:
                            target_qs.append(tg_qvalue_)
                    
                    qvalues, _ = qnet[k](pad_observations)
                    qvalues = qvalues[mask_observations]
                    
                    qvalues = qvalues.gather(-1, actions)
                    if mixer == None:
                        td_err = (y - qvalues)
                        if args.hysteretic: td_err = th.max(hyst * td_err, td_err)
                        qnet_loss = (td_err**2).mean()

                        qnet_loss.backward()
                        qnet_opt[k].step()

                        qnet_opt[k].zero_grad(True)
                    else:
                        agent_qs.append(qvalues)
                        j_rewards.append(rewards)

                    with th.no_grad():
                        for tg_weights, weights in zip(tg_qnet[k].parameters(), qnet[k].parameters()): 
                            tg_weights.data.copy_(args.tau * tg_weights.data + (1.0 - args.tau) * weights.data)

                        if mixer != None:
                            for tg_weights, weights in zip(mixer.parameters(), tg_mixer.parameters()):
                                tg_weights.data.copy_(args.tau * tg_weights.data + (1.0 - args.tau) * weights.data)

                if mixer != None:
                    qvalue = th.hstack(agent_qs)
                    tg_qvalue_ = th.hstack(target_qs).detach()

                    reward = th.sum(th.hstack(j_rewards), dim=-1, keepdim=True)

                    vdn_qvalue = mixer(qvalue, states)
                    
                    vdn_tg_qvalue_ = tg_mixer(tg_qvalue_, next_states).detach()
                    
                    y = reward + args.gamma * vdn_tg_qvalue_ * (1 - dones)

                    td_err = (y - vdn_qvalue)
                    
                    if args.hysteretic: td_err = th.max(hyst * td_err, td_err)

                    qnet_loss = (td_err**2).mean()
                    
                    qnet_loss.backward()

                    qnet_opt.step()
                    
                    qnet_opt.zero_grad(True)

                if args.wandb_log: wandb.log({'MSE': qnet_loss})

                sps = int(global_step / (time.time() - start_time))
                if args.tb_log: summary_w.add_scalar('Training/SPS', sps, global_step) 
                if args.wandb_log: wandb.log({'SPS': sps})
        
    try:      
        env.close()
        if args.tb_log: summary_w.close()
        if args.wandb_log: 
            wandb.finish()
            if args.wandb_mode == 'offline':
                import subprocess
                subprocess.run(['wandb', 'sync', wandb_path])  
    except:
        pass

