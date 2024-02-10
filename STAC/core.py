from copy import deepcopy
import itertools
import numpy as np
import torch 
from torch.optim import Adam
from actorcritic import ActorCritic
from utils import AttrDict
from buffer import ReplayBuffer
from debugging import Debugger
import pickle
from tqdm import tqdm
from render_browser import render_browser
from gc import collect

class MaxEntrRL():
    def __init__(self, train_env, test_env, env, actor, critic_kwargs=AttrDict(), actor_kwargs=AttrDict(), device="cuda",   
        RL_kwargs=AttrDict(), optim_kwargs=AttrDict(), tb_logger=None, need_q=False):
        self.env_name = env
        self.actor = actor 
        self.device = device
        self.critic_kwargs = critic_kwargs
        self.actor_kwargs = actor_kwargs
        self.RL_kwargs = RL_kwargs
        self.optim_kwargs = optim_kwargs
        self.with_amor_infer = actor_kwargs.pop('with_amor_infer', False)
        self.need_q = need_q
        
        # instantiating the environment
        self.env, self.test_env = train_env, test_env

        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = ActorCritic(self.actor, self.env.observation_space, self.env.action_space, self.RL_kwargs.evaluation_data_path, 
            self.RL_kwargs.test_time, self.RL_kwargs.model_path, self.critic_kwargs, self.actor_kwargs)
        # self.ac.state_dict
        self.ac_targ = deepcopy(self.ac)

        # move models to device
        self.ac = self.ac.to(self.device)
        self.ac_targ = self.ac_targ.to(self.device)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.RL_kwargs.replay_size, load_replay=self.RL_kwargs.load_replay, replay_path=self.RL_kwargs.replay_path, device=self.device, env_name=self.env_name)

        if next(self.ac.pi.parameters(), None) is not None:
            self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.optim_kwargs.lr_actor)
        
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.q_optimizer = Adam(self.q_params, lr=self.optim_kwargs.lr_critic)

        self.debugger = Debugger(tb_logger, self.ac, self.env_name, self.env, self.test_env, self.RL_kwargs.update_after, self.RL_kwargs.max_steps)

        self.evaluation_data = AttrDict()
        self.evaluation_data['train_episodes_return'] = []
        self.evaluation_data['train_episodes_length'] = []
        self.evaluation_data['max' + 'test_episodes_return'] = []
        self.evaluation_data['max' + 'test_episodes_length'] = []
        self.evaluation_data['softmax' + 'test_episodes_return'] = []
        self.evaluation_data['softmax' + 'test_episodes_length'] = []


    def compute_loss_q(self, data, itr):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        # Target actions come from *current* policy
        o2 = o2.view(-1,1,self.obs_dim).repeat(1,self.ac.pi.num_particles,1).view(-1,self.obs_dim)
        a2, logp_a2 = self.ac(o2, action_selection=None, with_logprob=True, in_q_loss=False) 
        
        with torch.no_grad(): 
            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2).view(-1, self.ac.pi.num_particles)
            q2_pi_targ = self.ac_targ.q2(o2, a2).view(-1, self.ac.pi.num_particles)

            
            if self.actor == 'svgd_sql':
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                V_soft_ = self.RL_kwargs.alpha_c * torch.logsumexp(q_pi_targ / self.RL_kwargs.alpha_c, dim=-1)
                V_soft_ += self.RL_kwargs.alpha_c * (self.act_dim * np.log(2) - np.log(self.ac.pi.num_particles))
                backup = r + self.RL_kwargs.gamma * (1 - d) * V_soft_
                self.debugger.add_scalars('Q_target/',  {'r ': r.mean(), 'V_soft': (self.RL_kwargs.gamma * (1 - d) * V_soft_).mean(), 'backup': backup.mean()}, itr)
            else:
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                backup = r + self.RL_kwargs.gamma * (1 - d) * (q_pi_targ.mean(-1) - self.RL_kwargs.alpha_c * logp_a2) 
                self.debugger.add_scalars('Q_target/',  {'r': r.mean(), 'Q': (self.RL_kwargs.gamma * (1 - d) * q_pi_targ.mean(-1)).mean(),\
                    'entropy': (self.RL_kwargs.gamma * (1 - d) * self.RL_kwargs.alpha_c * logp_a2).mean(), 'backup': backup.mean(), 'pure_entropy':logp_a2.mean()}, itr)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        
        self.debugger.add_scalars('Loss_q',  {'loss_q1 ': loss_q1, 'loss_q2': loss_q2, 'total': loss_q  }, itr)
        
        return loss_q


    def compute_loss_pi(self, data, itr):
        
        o = data['obs'].view(-1,1,self.obs_dim).repeat(1,self.ac.pi.num_particles,1).view(-1,self.obs_dim)
        
        a, logp_pi = self.ac(o, action_selection=None, with_logprob=True)

        # get the final action
        q1_pi = self.ac.q1(o, a).view(-1, self.ac.pi.num_particles)
        q2_pi = self.ac.q2(o, a).view(-1, self.ac.pi.num_particles)
        q_pi = torch.min(q1_pi, q2_pi).mean(-1)

        # Entropy-regularized policy loss
        if self.actor == 'svgd_sql' or (self.actor == 'svgd' and self.with_amor_infer):
            if self.actor == 'svgd_sql':
                a_updated, logp_pi = self.ac(o, action_selection=None, with_logprob=True)
            else:
                a_updated, logp_pi = self.ac(o, action_selection='amortized', with_logprob=True)
            # compte grad q wrt a
            grad_q = torch.autograd.grad((q_pi * self.ac.pi.num_particles).sum(), a)[0]
            grad_q = grad_q.view(-1, self.ac.pi.num_particles, self.act_dim).unsqueeze(2).detach() #(batch_size, num_svgd_particles, 1, act_dim)
            
            a = a.view(-1, self.ac.pi.num_particles, self.act_dim)
            a_updated = a_updated.view(-1, self.ac.pi.num_particles, self.act_dim)

            kappa, _, _, grad_kappa = self.ac.pi.Kernel(input_1=a, input_2=a_updated)
            a_grad = (1 / self.ac.pi.num_particles) * torch.sum(kappa.unsqueeze(-1) * grad_q + grad_kappa, dim=1) # (batch_size, num_svgd_particles, act_dim)

            loss_pi = -a_updated
            grad_loss_pi = a_grad
            # update the amortized policy network for efficient inference
            a_updated, logp_pi = self.ac(o, action_selection=None, with_logprob=True) 
            
        else:
            loss_pi = (self.RL_kwargs.alpha_a * logp_pi - q_pi).mean()

            grad_loss_pi = None
            self.debugger.add_scalars('Loss_pi',  {'logp_pi ': (self.RL_kwargs.alpha_a * logp_pi).mean(), 'q_pi': -q_pi.mean(), 'total': loss_pi  }, itr)
            
        return loss_pi, grad_loss_pi


    def update(self, data, itr):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data, itr)
        loss_q.backward()
        self.q_optimizer.step()
        
        if next(self.ac.pi.parameters(), None) is not None:
            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False
            # Compute the loss
            loss_pi, grad_loss_pi = self.compute_loss_pi(data, itr)
            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi.backward(gradient=grad_loss_pi)
            self.pi_optimizer.step()
            # Unfreeze Q-networks so you can optimize it at next step.
            for p in self.q_params:
                p.requires_grad = True
        
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.optim_kwargs.polyak)
                p_targ.data.add_((1 - self.optim_kwargs.polyak) * p.data)

    
    # This decorator should be used if MuJoCo environments are desired to be visualized during test phase. 
    # @render_browser
    def test_agent(self, itr=None, action_selection=None):
        robot_pic_rgb = None
        if action_selection:
            self.ac.pi.test_action_selection = action_selection

        if self.env_name in ['multigoal-max-entropy', 'Multigoal', 'multigoal-obstacles', 'multigoal-max-entropy-obstacles']:
            self.test_env.reset_rendering()

        for j in tqdm(range(self.RL_kwargs.num_test_episodes)):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0

            while not(d or (ep_len == self.RL_kwargs.max_steps)):
                o = torch.as_tensor(o, dtype=torch.float32).to(self.device).view(-1,self.obs_dim)
                o_ = o.view(-1,1,self.obs_dim).repeat(1,self.ac.pi.num_particles,1).view(-1,self.obs_dim) # move this inside pi.act
                a, log_p = self.ac(o_, action_selection=self.ac.pi.test_action_selection, with_logprob=False)
                a_detach = a.detach().cpu().numpy().squeeze()
                if self.need_q:
                    o_torch = torch.as_tensor(o, dtype=torch.float32).to(self.device)
                    q = self.ac.q1(o_torch, a).detach().cpu().numpy().squeeze()
                    o2, r, d, _ = self.test_env.step(a_detach,[q], [None])
                else:
                    o2, r, d, _ = self.test_env.step(a_detach)
                self.debugger.collect_data(o, a, o2, r, d, log_p, itr, ep_len, robot_pic_rgb=robot_pic_rgb)    
                
                ep_ret += r
                ep_len += 1
                
                o = o2
            print()
            print('####### --actor: ', self.actor, ' --ep_return: ', ep_ret, ' --ep_length: ', ep_len)
            self.evaluation_data[self.ac.pi.test_action_selection + 'test_episodes_return'].append(ep_ret)
            self.evaluation_data[self.ac.pi.test_action_selection + 'test_episodes_length'].append(ep_len)
            if not self.RL_kwargs.test_time:
                self.debugger.entropy_plot()
                self.debugger.init_dist_plots()
                
        if action_selection in ['max', 'random']:
            if self.env_name in ['multigoal-max-entropy', 'Multigoal', 'max-entropy-v0', 'multigoal-obstacles', 'multigoal-max-entropy-obstacles']:
            
                # self.test_env.render(itr=itr, fig_path=self.RL_kwargs.fig_path, ac=self.ac, paths=self.replay_buffer.paths)
                self.debugger.plot_policy(itr=itr, fig_path=self.RL_kwargs.fig_path) # For multigoal only


            self.debugger.log_to_tensorboard(itr=itr)


            if not self.RL_kwargs.test_time:
                # Save the model used in each test phase
                self.ac.save(itr)
        
    def save_data(self):
        pickle.dump(self.evaluation_data, open(self.RL_kwargs.evaluation_data_path + '/evaluation_data.pickle', "wb"))
 

    def forward(self):
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters(): 
            p.requires_grad = False
                
        # Prepare for interaction with environment
        o, ep_ret, ep_len = self.env.reset(), 0, 0 

        episode_itr = 0
        step_itr = 0
        
        EpRet = []
        EpLen = []

        # Main loop: collect experience in env and update/log each epoch
        for step_itr in tqdm(range(self.RL_kwargs.max_experiment_steps)):
           
            # Until exploration_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 

            if step_itr >= self.RL_kwargs.exploration_steps:
                o_torch = torch.as_tensor(o, dtype=torch.float32).to(self.device).view(-1,1,self.obs_dim).repeat(1,self.ac.pi.num_particles,1).view(-1,self.obs_dim)
                a_torch, logp = self.ac(o_torch, action_selection = self.RL_kwargs.train_action_selection, with_logprob=False, itr=step_itr)
                a = a_torch.detach().cpu().numpy().squeeze()

            else:
                a = self.env.action_space.sample()


            # Step the env
            if self.need_q:
                # Step in the medical env. Need to estimate and pass the q value of the action
                # action: 1 dimension
                o_torch = torch.as_tensor(o, dtype=torch.float32).to(self.device)
                a_torch = torch.as_tensor(a, dtype=torch.float32).to(self.device)
                q = self.ac.q1(o_torch, a_torch).detach().cpu().numpy()
                o2, r, d, info = self.env.step(a, [q], [None])
            else:
                o2, r, d, info = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.RL_kwargs.max_steps else d

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d, info, step_itr)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.RL_kwargs.max_steps):
                EpRet.append(ep_ret)
                EpLen.append(ep_len)
                self.evaluation_data['train_episodes_return'].append(ep_ret)
                self.evaluation_data['train_episodes_length'].append(ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0, 0
                episode_itr += 1
                d = True    
            
            # Update handling
            if step_itr >= self.RL_kwargs.update_after and step_itr % self.RL_kwargs.update_every == 0:
                if step_itr == self.RL_kwargs.update_after:
                    print('######################## Starting models update ########################')
                for j in range(self.RL_kwargs.update_every):
                    batch = self.replay_buffer.sample_batch(self.optim_kwargs.batch_size)
                    self.debugger.add_scalars('Batch_reward',  {'final_reward_num': int(np.sum(batch['rew'].detach().cpu().numpy()))}, step_itr)
                    self.update(data=batch, itr=step_itr)

            
            if ((step_itr+1)  >= self.RL_kwargs.collect_stats_after and (step_itr+1) % self.RL_kwargs.stats_steps_freq == 0) or step_itr == self.RL_kwargs.max_experiment_steps - 1:
                mean_return = []
                for action_selection in ['max', 'softmax']:
                    self.test_agent(step_itr, action_selection)
                    mean_return.append(np.mean(list(map(lambda x: x['expected_reward'], self.debugger.episodes_information))))
                self.debugger.tb_logger.add_scalars('Test_EpRet/return_mean_only',  {'max': mean_return[0], 'softmax': mean_return[1]}, step_itr)

                try:
                    self.debugger.add_scalars('EpRet/return_detailed',  {'Mean ': np.mean(EpRet), 'Min': np.min(EpRet), 'Max': np.max(EpRet)  }, step_itr)
                    self.debugger.add_scalars('EpRet/return_mean_only',  {'Mean ': np.mean(EpRet)}, step_itr)
                    self.debugger.add_scalar('EpLen',  np.mean(EpLen), step_itr)
                except:
                    print('Statistics collection frequency should be larger then the length of an episode!')
                    
                EpRet = []
                EpLen = []
            step_itr += 1
            self.save_data()
