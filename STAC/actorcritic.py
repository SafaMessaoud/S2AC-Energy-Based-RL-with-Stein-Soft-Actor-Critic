import torch.nn as nn
from networks import mlp, MLPFunction, CNNFunction
from actors.actor_sac import ActorSac
from actors.actor_svgd import ActorSvgd
from actors.actor_sql import ActorSql
from actors.actor_diffusion import ActorDiffusion
from utils import AttrDict
import torch

class ActorCritic(nn.Module):
    def __init__(self, actor, observation_space, action_space, save_path, 
                 test_time, model_path, 
                 critic_kwargs=AttrDict(), actor_kwargs=AttrDict()):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.save_path = save_path
        self.actor_name = actor 
        self.test_time = test_time
        self.model_path = model_path

        dict_actors = {
            'sac': ActorSac,
            'svgd_nonparam': ActorSvgd,
            'svgd_p0_pram': ActorSvgd,
            'svgd_p0_kernel_pram': ActorSvgd,
            'svgd_sql': ActorSql,
            'diffusion': ActorDiffusion}

        Net = MLPFunction if not critic_kwargs.pop('critic_cnn') else CNNFunction
        self.q1 = Net(obs_dim, act_dim, 1, **critic_kwargs)
        self.q2 = Net(obs_dim, act_dim, 1, **critic_kwargs)
        
        if 'svgd' in actor:
            actor_kwargs.q1 = self.q1.forward
            actor_kwargs.q2 = self.q2.forward
            
        self.pi = dict_actors[actor](actor, obs_dim, act_dim, act_limit, **actor_kwargs)

        if self.test_time:
            self.load()

    def forward(self, obs, action_selection=None, with_logprob=True, in_q_loss=False, itr=None):
        return self.pi.act(obs, action_selection, with_logprob, in_q_loss, itr)

    def save(self, itr=0):
        torch.save(self.state_dict(), self.save_path + '/' + self.actor_name + '_' + str(itr))
        
    def load(self):
        self.load_state_dict(torch.load(self.model_path))
