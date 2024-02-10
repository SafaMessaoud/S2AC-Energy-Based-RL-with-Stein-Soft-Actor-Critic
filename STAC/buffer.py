import torch 
import numpy as np
from utils import AttrDict
import pickle

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, size, load_replay, replay_path, device, env_name, episode_max_steps=500): 
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.env_name = env_name
        
        self.load_replay = load_replay
        self.replay_path = replay_path


        self.goals = [0, 0]
 
            
        if self.load_replay:
            with open(self.replay_path + 'buffer.pkl', 'rb') as f:
                buffer = pickle.load(f)
            self.obs_buf = buffer['obs_buf']
            self.obs2_buf = buffer['obs2_buf']
            self.act_buf = buffer['act_buf']
            self.rew_buf = buffer['rew_buf']
            self.done_buf = buffer['done_buf']
            self.ptr, self.size = buffer['ptr'], buffer['size']
            self.goals = buffer['goals']
        else:
            self.obs_buf = np.zeros((size, self.obs_dim), dtype=np.float32)
            self.obs2_buf = np.zeros((size, self.obs_dim), dtype=np.float32)
            self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
            self.rew_buf = np.zeros((size,), dtype=np.float32)
            self.done_buf = np.zeros((size,), dtype=np.float32)
            self.ptr, self.size = 0, 0
        self.max_size = size
        self.device = device

    def store(self, obs, act, rew, next_obs, done, env_info=None, step_itr=None):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size) 
        batch = AttrDict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in batch.items()}

    def save(self):
        with open(self.replay_path + 'buffer.pkl', 'wb') as f:
            pickle.dump({
                'obs_buf': self.obs_buf,
                'obs2_buf': self.obs2_buf,
                'act_buf': self.act_buf,
                'rew_buf': self.rew_buf,
                'done_buf': self.done_buf,
                'ptr': self.ptr,
                'size': self.size,
                'goals': self.goals
            }, f)