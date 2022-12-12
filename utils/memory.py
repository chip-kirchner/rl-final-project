"""Memory buffer script

This manages the memory buffer. 
"""
from collections import deque
from copy import deepcopy

from .misc import *

class Buffer:
    """
    Class for the Buffer creation
    """

    def __init__(self, env: gym.Env, size: int, batch_size: int, h_size: int):
        
        self.o_dim = int(np.prod(env.observation_space[0].shape))
        self.h_size = h_size

        self.memory = deque(maxlen=size)

        self.priority_mem = deque()
        self.priority_score = float('-inf')

        self.batch_size = batch_size

    def store(self, episode, score=None):
        self.memory.append(episode)

        if score != None:
            if score > self.priority_score:
                # reset score
                self.priority_score = score
                # reset priority mem
                self.priority_mem = deque()
                # add this to it
                self.priority_mem.append(episode)
            elif score == self.priority_score:
                self.priority_mem.append(episode)


    def sample(self, device, ep_idxs=None, sample_idxs=None):
        batch_size = min(len(self.memory), self.batch_size)
        
        if ep_idxs is None: ep_idxs = np.random.choice(range(len(self.memory)), batch_size, replace=False)
        
        batch = {} #batch = self._get_empty_batch(batch_size, device)

        start_idxs = np.zeros(batch_size, dtype=np.int32)
        for i, idx in enumerate(ep_idxs):   
            episode = self.memory[idx]
            
            # Additional +1 as randint has high exluded
            if sample_idxs is None: start_idx = np.random.randint(0, (episode.idx) + 1)    
            else: start_idx = sample_idxs[i]

            ep_batch = episode.sample(start_idx)
            
            if i == 0:
                batch['observations'] = ep_batch['observations']
                batch['actions'] = ep_batch['actions']
                batch['rewards'] = ep_batch['rewards']
                batch['observations_'] = ep_batch['observations_']
                batch['dones'] = ep_batch['dones']
                batch['hs'] = ep_batch['hs']
                batch['hs_'] = ep_batch['hs_']
                batch['lengths'] = [ep_batch['observations'].shape[0]]
                batch['states'] = ep_batch['states']
                batch['next_states'] = ep_batch['next_states']
            else:
                
                batch['observations'] = th.cat((batch['observations'],ep_batch['observations']), 0)
                batch['actions'] = th.cat((batch['actions'], ep_batch['actions']), 0)
                batch['rewards'] = th.cat((batch['rewards'], ep_batch['rewards']), 0)
                batch['observations_'] = th.cat((batch['observations_'], ep_batch['observations_']), 0)
                batch['dones'] = th.cat((batch['dones'], ep_batch['dones']), 0)
                batch['hs'] = th.cat((batch['hs'], ep_batch['hs']), 0)
                batch['hs_'] = th.cat((batch['hs_'], ep_batch['hs_']), 0)
                batch['lengths'].append(ep_batch['observations'].shape[0])
                batch['states'] = th.cat((batch['states'], ep_batch['states']), 0)
                batch['next_states'] = th.cat((batch['next_states'], ep_batch['next_states']), 0)

            start_idxs[i] = start_idx

        return batch, ep_idxs, start_idxs

class EpisodeBuffer:
    """
    Class for the Buffer creation
    """

    def __init__(self, env: gym.Env, max_steps: int, h_size: int, device: th.device):
        
        self.capacity = max_steps

        obs_size = int(np.prod(env.observation_space[0].shape))
        state_size = int(np.prod(env.state_size))

        self.b_observations = th.zeros((self.capacity, obs_size)).to(device) 
        self.b_actions = th.zeros((self.capacity, 1), dtype=th.int32).to(device)
        self.b_rewards = th.zeros((self.capacity, 1)).to(device)
        self.b_observations_ = deepcopy(self.b_observations)
        self.b_dones = deepcopy(self.b_rewards)
        self.b_hs = th.zeros((self.capacity, h_size)).to(device)
        self.b_hs_ = deepcopy(self.b_hs)
        self.b_state = th.zeros((self.capacity, state_size)).to(device)
        self.b_state_ = th.zeros((self.capacity, state_size)).to(device)

        self.idx = 0

    def store(self, observation, action, reward, observation_, done, h, h_, state, next_state):
        self.b_observations[self.idx] = observation
        self.b_actions[self.idx] = action
        self.b_rewards[self.idx] = reward
        self.b_observations_[self.idx] = observation_
        self.b_dones[self.idx] = done
        self.b_hs[self.idx] = h
        self.b_hs_[self.idx] = h_
        self.b_state[self.idx] = state
        self.b_state_[self.idx] = next_state

        self.idx = self.idx + 1

    def sample(self, start_idx):
        # Sample the entire stored episode
        idxs = slice(0, self.idx, 1)
       
        batch = {
            'observations': self.b_observations[idxs],
            'actions': self.b_actions[idxs],
            'rewards': self.b_rewards[idxs],
            'observations_': self.b_observations_[idxs],
            'dones': self.b_dones[idxs],
            'hs': self.b_hs[idxs],
            'hs_': self.b_hs_[idxs],
            'states': self.b_state[idxs],
            'next_states': self.b_state_[idxs]
        }  

        return batch
