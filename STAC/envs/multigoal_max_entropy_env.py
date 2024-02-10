from gym.utils import EzPickle
from gym import spaces
from gym import Env
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import torch
import matplotlib.patheffects as pe

class PointDynamics(object):
    """
    State: position.
    Action: velocity.
    """
    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state, action):
        mu_next = state + action
        state_next = mu_next + self.sigma * np.random.normal(size=self.s_dim)
        return state_next


class MultiGoalMaxEntropyEnv(Env, EzPickle): 
    """
    Move a 2D point mass to one of the goal positions. Cost is the distance to
    the closest goal.

    State: position.
    Action: velocity.
    """
    def __init__(self, goal_reward=10, actuation_cost_coeff=30.0, distance_cost_coeff=1.0, init_sigma=0.05, max_steps=None, plot_format=None, env_name=None):
        EzPickle.__init__(**locals())

        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.init_mu = np.zeros(2, dtype=np.float32)
        self.init_sigma = init_sigma
        self.max_steps = max_steps
        # goal
        
        
        self.goal_positions = np.array(((5, 0),(-4, 3), (-4, -3)), dtype=np.float32)
        self.goal_names_plotting = np.array(['$G_1$', '$G_2$', '$G_3$'])
        self.goal_names_positions = np.array([[0.5, -0.2], [-1.2, -0.15], [-1.2, -0.2]]) + self.goal_positions
        
        self.entropy_list = None
        self._obs_lst = np.array([
            [-3.5, -2.5],
            [-2.5, 0.5],
            [-1, 0],
            [0, 0],
            [1, 0],
            [4, 0]
        ])
       
        self.num_goals = len(self.goal_positions)
        self.goal_threshold = 0.5 #1.0
        self.goal_reward = goal_reward
        # reward
        self.action_cost_coeff = actuation_cost_coeff
        self.distance_cost_coeff = distance_cost_coeff
        # plotting
        self.xlim = (-7, 7)
        self.ylim = (-7, 7)
        self.vel_bound = 1.
        
        # logging
        self.episode_observations = [] 
        self.all_episode_observations = [] 
        self.ep_len = 0
        self.env_name = env_name

        self.entropy_obs_names = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
        self.entropy_obs_names_plotting = np.array(['$s_a$', '$s_b$', '$s_c$', '$s_d$', '$s_e$', '$s_f$'])

        self._n_samples = 10
        self.n_plots = len(self._obs_lst)
        self.x_size = (1.5 * self.n_plots + 1)
        self.y_size = 17 
        self.agent_failure = None
        self.plot_format = plot_format

        self.intersection = False

    def reset(self, init_state=None):
        if init_state:
            unclipped_observation = init_state
        else: 
            unclipped_observation = self.init_mu

        self.observation = np.clip(unclipped_observation, self.observation_space.low, self.observation_space.high)
        self.ep_len = 0
        self.all_episode_observations.append([list(self.observation)])
        return self.observation

    @property
    def observation_space(self):
        return spaces.Box(
            low=np.array((self.xlim[0], self.ylim[0])),
            high=np.array((self.xlim[1], self.ylim[1])),
            dtype=np.float32,
            shape=(2,))

    @property
    def action_space(self):
        return spaces.Box(
            low=-self.vel_bound,
            high=self.vel_bound,
            shape=(self.dynamics.a_dim, ),
            dtype=np.float32)


    def step(self, action):
        # compute action
        action = action.ravel()
        action = np.clip(action, self.action_space.low, self.action_space.high).ravel()

        # compute observation
        self.observation = self.dynamics.forward(self.observation, action)
        self.observation = np.clip(self.observation, self.observation_space.low,self.observation_space.high)
        self.ep_len += 1

        # compute reward
        reward = self.compute_reward(self.observation, action)

        # compute done
        distance_to_goals = [np.linalg.norm(self.observation - goal_position)for goal_position in self.goal_positions]
        self.min_dist_index = np.argmin(distance_to_goals)
        done = distance_to_goals[self.min_dist_index] < self.goal_threshold
        
        # reward at the goal
        if done:
            if self.env_name == 'test_env':
                self.number_of_hits_mode_acc[self.min_dist_index] += 1
            reward += self.goal_reward

        if done or self.ep_len == self.max_steps:
            self.episode_observations.append(self.observation)
        self.all_episode_observations[-1].append(list(self.observation))
        return self.observation, reward, done, None
    

    def compute_reward(self, observation, action): 
        # penalize the L2 norm of acceleration
        # noinspection PyTypeChecker
        action_cost = np.sum(action ** 2) * self.action_cost_coeff

        # penalize squared dist to goal
        cur_position = self.observation

        # noinspection PyTypeChecker
        goal_cost = self.distance_cost_coeff * np.amin([
            np.sum((cur_position - goal_position) ** 2)
            for goal_position in self.goal_positions
        ])

        # penalize staying with the log barriers
        costs = [action_cost, goal_cost]
        reward = -np.sum(costs)
        return reward
    
    def reset_rendering(self):
        self.episode_observations = []
        self.all_episode_observations = []
        self.number_of_hits_mode = np.zeros(self.num_goals)
        self.number_of_hits_mode_acc = np.zeros(self.num_goals)
        
    
    def render(self, itr, fig_path, ac=None, paths=None):
        paths = self.all_episode_observations
        self._init_plot(self.x_size, self.y_size)
        for a in range(len(self.all_episode_observations)):
            positions = np.array(paths[a])
            self._ax_lst[0].plot(positions[:, 0], positions[:, 1], color='blue')
            
        if ac.pi.actor!='svgd_sql':

            num_particles_tmp = ac.pi.num_particles
            ac.pi.num_particles = self._n_samples
            if not ac.pi.actor=='sac':
                ac.pi.Kernel.num_particles = ac.pi.num_particles
                ac.pi.identity = torch.eye(ac.pi.num_particles).to(ac.pi.device)

            self.entropy_list = []
            for i in range(len(self._obs_lst)):
                self.entropy_tmp = []
                o = torch.as_tensor(self._obs_lst[i], dtype=torch.float32).to(ac.pi.device).view(-1,1,self.observation_space.shape[0]).repeat(1,ac.pi.num_particles,1).view(-1,self.observation_space.shape[0])
                for _ in range(100): 
                    a, log_p = ac(o, action_selection=ac.pi.test_action_selection, with_logprob=True)
                    if ac.pi.actor=='sac':
                        log_p = log_p.view(-1,ac.pi.num_particles).mean(dim=1)
                    self.entropy_tmp.append(round(-log_p.detach().item(), 2))
                self.entropy_list.append(round(np.array(self.entropy_tmp).mean(), 2))
            
            
            ac.pi.num_particles = num_particles_tmp
            if not ac.pi.actor=='sac':
                ac.pi.Kernel.num_particles = ac.pi.num_particles
                ac.pi.identity = torch.eye(ac.pi.num_particles).to(ac.pi.device)

        for i in range(len(self._obs_lst)):
            self._ax_lst[0].scatter(self._obs_lst[i, 0], self._obs_lst[i, 1], color='white', edgecolors='#D13B00', marker='X', s=60, zorder=5)
            self._ax_lst[0].annotate(self.entropy_obs_names_plotting[i], (self._obs_lst[i,0] - 0.25, self._obs_lst[i,1] + 0.2), fontsize=20, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="#D13B00")], zorder=5)
            
        
        self._plot_level_curves(self._obs_lst, ac)
        self._plot_action_samples(ac)
        plt.plot()
        plt.savefig(fig_path+ '/env_' + str(itr) + '.' + 'png')   
        plt.close()

        modes_dist = (((positions).reshape(-1,1,2) - np.expand_dims(self.goal_positions,0))**2).sum(-1)
        ind = modes_dist[np.where(modes_dist.min(-1)<1)[0]].argmin(-1)
        self.number_of_hits_mode[ind]+=1
    
    
    def _init_plot(self, x_size, y_size, grid_size=(5,3), debugging=False):
        self._ax_lst = []
        self.fig_env = plt.figure(figsize=(x_size, y_size), constrained_layout=True) 
        self._ax_lst.append(plt.subplot2grid(grid_size, (0,0), colspan=3, rowspan=3))
        self._ax_lst[0].axis('equal')
        self._ax_lst[0].set_xlim(self.xlim)
        self._ax_lst[0].set_ylim(self.xlim)
        self._ax_lst[0].set_title('Multigoal Environment', fontsize=20)

        self._ax_lst[0].xaxis.set_tick_params(labelsize=15)
        self._ax_lst[0].yaxis.set_tick_params(labelsize=15)
        x_min, x_max = tuple(1.1 * np.array(self.xlim))
        y_min, y_max = tuple(1.1 * np.array(self.ylim))
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, 0.01),
            np.arange(y_min, y_max, 0.01)
        )
        goal_costs = np.amin([
            (X - goal_x) ** 2 + (Y - goal_y) ** 2
            for goal_x, goal_y in self.goal_positions
        ], axis=0)
        costs = goal_costs
        contours = self._ax_lst[0].contour(X, Y, costs, 20)
        self._ax_lst[0].clabel(contours, inline=1, fontsize=10, fmt='%.0f')
        self._ax_lst[0].set_xlim([x_min, x_max])
        self._ax_lst[0].set_ylim([y_min, y_max])
        self._ax_lst[0].plot(self.goal_positions[:, 0], self.goal_positions[:, 1], 'ro')

        
        for i in range(len(self.goal_positions)):
                self._ax_lst[0].annotate(self.goal_names_plotting[i], self.goal_names_positions[i], fontsize=19, color='red', zorder=2)


        if not debugging:
            for i in range(self.n_plots):
                ax = plt.subplot2grid(grid_size, (3 + i//3,i%3))
                ax.xaxis.set_tick_params(labelsize=15)
                ax.yaxis.set_tick_params(labelsize=15)
                ax.set_xlim((-1, 1))
                ax.set_ylim((-1, 1))
                ax.grid(True)
                self._ax_lst.append(ax)
            self._line_objects = list()
        return self._ax_lst[0]

    def _plot_level_curves(self, _obs_lst, ac):
        # Create mesh grid.
        xs = np.linspace(-1, 1, 50)
        ys = np.linspace(-1, 1, 50)
        xgrid, ygrid = np.meshgrid(xs, ys)
        a = np.concatenate((np.expand_dims(xgrid.ravel(), -1), np.expand_dims(ygrid.ravel(), -1)), -1)
        a = torch.from_numpy(a.astype(np.float32)).to(ac.pi.device)
        for i in range(len(_obs_lst)):
            o = torch.Tensor(_obs_lst[i]).repeat([a.shape[0],1]).to(ac.pi.device)
            with torch.no_grad():
                qs = ac.q1(o.to(ac.pi.device), a).cpu().detach().numpy()
            qs = qs.reshape(xgrid.shape)
            cs = self._ax_lst[i+1].contour(xgrid, ygrid, qs, 20)
            self._line_objects += cs.collections
            self._line_objects += self._ax_lst[i+1].clabel(
                cs, inline=1, fontsize=10, fmt='%.2f')

    def _plot_action_samples(self, ac):
        if ac.pi.actor in  ['svgd_nonparam', 'svgd_p0_param']:
            num_particles_tmp = ac.pi.num_particles
            ac.pi.num_particles = self._n_samples
            ac.pi.Kernel.num_particles = ac.pi.num_particles
            ac.pi.identity = torch.eye(ac.pi.num_particles).to(ac.pi.device)
        
        for i in range(len(self._obs_lst)):
            if ac.pi.actor in  ['svgd_nonparam', 'svgd_p0_param', 'svgd_sql']:

                o = torch.as_tensor(self._obs_lst[i], dtype=torch.float32).view(-1,1,self.observation_space.shape[0]).repeat(1,ac.pi.num_particles,1).view(-1,self.observation_space.shape[0]).to(ac.pi.device)
            else:
                o = torch.as_tensor(self._obs_lst[i], dtype=torch.float32).repeat([self._n_samples,1]).to(ac.pi.device)
            actions, _ = ac(o, action_selection=None, with_logprob=False)
            actions = actions.cpu().detach().numpy().squeeze()

            x, y = actions[:, 0], actions[:, 1]
            if ac.pi.actor not in  ['svgd_sql']:
                self._ax_lst[i+1].set_title('Entr(' + self.entropy_obs_names_plotting[i] + ')=' + str(self.entropy_list[i]), fontsize=15)
            self._line_objects += self._ax_lst[i+1].plot(x, y, 'b*')

        if ac.pi.actor in  ['svgd_nonparam', 'svgd_p0_param']:
            ac.pi.num_particles = num_particles_tmp
            ac.pi.Kernel.num_particles = ac.pi.num_particles
            ac.pi.identity = torch.eye(ac.pi.num_particles).to(ac.pi.device)





