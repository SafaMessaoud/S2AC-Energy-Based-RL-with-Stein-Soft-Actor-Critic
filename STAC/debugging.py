import numpy as np 
import torch as torch
import matplotlib.pyplot as plt


class Debugger():
    def __init__(self, tb_logger, ac, env_name, train_env, test_env, update_after, env_max_steps):
        self.ac = ac
        self.tb_logger = tb_logger
        self.env_name = env_name
        self.train_env = train_env
        self.test_env = test_env
        self.episodes_information = []
        self.episode_counter = 0
        self.episode_counter_2 = 0
        self.plot_cumulative_entropy = update_after + 5000
        self.env_max_steps = env_max_steps
        
        if self.env_name in ['Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles']:
            self.average_cumulative_entropy = np.zeros((self.test_env.num_goals))
            self.cumulative_entropy_coutner = np.zeros((self.test_env.num_goals))

    def collect_data(self, o, a, o2, r, d, log_p, itr, ep_len, robot_pic_rgb=None):
        
        if ep_len == 0:
            self.episodes_information.append({
                'observations':[],
                'action': [],
                'actions': [],
                # Debugging ##########
                'log_p': [],
                'logp_normal': [],
                'logp_svgd': [],
                'logp_tanh': [],
                'goal': None,
                'intersection': False,
                'next_state_rgb': [],

                'rewards': [],
                'expected_reward': None, 
                'episode_length': None,
                # p_0
                'mu': [],
                'sigma': [],
                # scores
                'q_score': [],
                'q_score_start': None, 
                'q_score_mid': None, 
                'q_score_end': None, 
                # hessian
                'q_hess' : [],
                'q_hess_mat' : [],
                'max_eigenval' : [],
                'q_hess_start': None, 
                'q_hess_mid': None, 
                'q_hess_end': None, 
                })

        self.episodes_information[-1]['observations'].append(o.detach().cpu().numpy().squeeze())

        if self.ac.pi.actor in ['svgd_nonparam', 'svgd_p0_pram']:
            self.episodes_information[-1]['actions'].append(self.ac.pi.a.detach().cpu().numpy().squeeze())
        self.episodes_information[-1]['action'].append(a.detach().cpu().numpy().squeeze())
        self.episodes_information[-1]['rewards'].append(r)
        self.test_env.entropy_list = None

        q1_value = self.ac.q1(o,a)
        q2_value = self.ac.q2(o,a)

        if log_p is not None:
            self.episodes_information[-1]['log_p'].append(-log_p.detach().item())
            self.episodes_information[-1]['logp_normal'].append(self.ac.pi.logp_normal_debug.detach().item())
            self.episodes_information[-1]['logp_svgd'].append(self.ac.pi.logp_svgd_debug.detach().item())
            self.episodes_information[-1]['logp_tanh'].append(self.ac.pi.logp_tanh_debug.detach().item())


        if self.env_name in ['Hopper-v2'] and robot_pic_rgb is not None:
            self.episodes_information[-1]['next_state_rgb'].append(robot_pic_rgb.tolist())


        if self.ac.pi.actor in ['sac', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            self.episodes_information[-1]['mu'].append(self.ac.pi.mu.detach().cpu().numpy())
            self.episodes_information[-1]['sigma'].append(self.ac.pi.sigma.detach().cpu().numpy())
        

        self.episodes_information[-1]['q1_values'] = q1_value.detach().cpu().numpy()
        self.episodes_information[-1]['q2_values'] = q2_value.detach().cpu().numpy()
        grad_q_ = torch.autograd.grad(torch.min(q1_value, q2_value), a, retain_graph=True, create_graph=True)[0].squeeze()
        self.episodes_information[-1]['q_score'].append(torch.abs(grad_q_).mean().detach().cpu().item())
        if self.env_name in ['Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles']:
            hess_q_mat = torch.stack([torch.autograd.grad(grad_q_[0], a, retain_graph=True)[0], torch.autograd.grad(grad_q_[1], a, retain_graph=True)[0]]).squeeze()
            hess_q = ((torch.abs(torch.autograd.grad(grad_q_[0], a, retain_graph=True)[0]) + torch.abs(torch.autograd.grad(grad_q_[1], a, retain_graph=True)[0])).sum()/4)
            self.episodes_information[-1]['q_hess'].append(hess_q.detach().cpu().item())
            self.episodes_information[-1]['q_hess_mat'].append(hess_q_mat.detach().cpu().numpy())
            self.episodes_information[-1]['max_eigenval'].append(torch.max(torch.linalg.eigvals(hess_q_mat).real).detach().cpu().item())


        if ((ep_len + 1) >= self.env_max_steps) or d: 
            self.episodes_information[-1]['observations'].append(o2.squeeze())
            self.episodes_information[-1]['expected_reward'] = np.sum(self.episodes_information[-1]['rewards'])
            self.episodes_information[-1]['episode_length'] = ep_len + 1
            if self.env_name in ['Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles']:
                self.episodes_information[-1]['intersection'] = self.test_env.intersection

                if d:
                    self.episodes_information[-1]['goal'] = self.test_env.min_dist_index
                if itr > self.plot_cumulative_entropy and d:
                    self.cumulative_entropy_coutner[self.test_env.min_dist_index] += 1
                    self.average_cumulative_entropy[self.test_env.min_dist_index] = ((self.cumulative_entropy_coutner[self.test_env.min_dist_index] - 1) * self.average_cumulative_entropy[self.test_env.min_dist_index] + np.array(self.episodes_information[-1]['log_p']).mean()) / (self.cumulative_entropy_coutner[self.test_env.min_dist_index])

            if ep_len >= 5:
                self.episodes_information[-1]['q_score_start'] = np.mean(self.episodes_information[-1]['q_score'][:5])
                self.episodes_information[-1]['q_hess_start'] = np.mean(self.episodes_information[-1]['q_hess'][:5])
            if ep_len >= 17:
                self.episodes_information[-1]['q_score_mid'] = np.mean(self.episodes_information[-1]['q_score'][12:17])
                self.episodes_information[-1]['q_hess_mid'] = np.mean(self.episodes_information[-1]['q_hess'][12:17])
            if ep_len >= 30:
                self.episodes_information[-1]['q_score_end'] = np.mean(self.episodes_information[-1]['q_score'][25:ep_len])
                self.episodes_information[-1]['q_hess_end'] = np.mean(self.episodes_information[-1]['q_hess'][25:ep_len])
    
         
    def plot_policy(self, itr, fig_path):
        if self.env_name in ['Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles']:
            ax = self.test_env._init_plot(x_size=7, y_size=7, grid_size=(1,1), debugging=True)

            cats_success = np.zeros((3,))
            cats_steps = np.zeros((3,))
            path = self.episodes_information[0]
            positions = np.stack(path['observations'])
            
            if not path['intersection']:
                cats_success[0]+= 1
                cats_steps[0]+= len(positions)
                ax.plot(positions[:, 0], positions[:, 1], color='blue')
            else:
                if path['goal'] is not None:
                    cats_success[1]+= 1
                    cats_steps[1]+= len(positions)
                    ax.plot(positions[:, 0], positions[:, 1], color='lime')
                else:
                    cats_success[2]+= 1
                    cats_steps[2]+= len(positions)
                    ax.plot(positions[:, 0], positions[:, 1], color='red')

            plt.savefig(fig_path + '/path_vis_'+ str(itr) + '.' + 'png')   
            plt.close()   


    def add_scalar(self, tb_path=None, value=None, itr=None):
            self.tb_logger.add_scalar(tb_path, value, itr)
    
    def add_scalars(self, tb_path=None, value=None, itr=None):
            self.tb_logger.add_scalars(tb_path, value, itr)

    def add_histogram(self, tb_path=None, value=None, itr=None):
            self.tb_logger.add_histogram(tb_path, value, itr)



    def init_dist_plots(self):
        if self.ac.pi.actor in ['svgd_p0_pram'] and self.env_name in ['Hopper-v2']:

            for i in range(10, 160, 10):
                if len(self.episodes_information[-1]['mu']) > i:
                    feed_dict_mu = {'x': self.episodes_information[-1]['mu'][i][0][0], 'y': self.episodes_information[-1]['mu'][i][0][1], 'z': self.episodes_information[-1]['mu'][i][0][2]}
                    feed_dict_sigma = {'x': self.episodes_information[-1]['sigma'][i][0][0], 'y': self.episodes_information[-1]['sigma'][i][0][1], 'z': self.episodes_information[-1]['sigma'][i][0][2]}
                    self.tb_logger.add_scalars('Entropy/mean_Hopper_' + str(i), feed_dict_mu, self.episode_counter_2)
                    self.tb_logger.add_scalars('Entropy/std_Hopper_' + str(i), feed_dict_sigma, self.episode_counter_2)
            self.episode_counter_2 += 1




    def log_to_tensorboard(self, itr):
        # related to the modes
        if self.env_name in ['Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles']:
            self.tb_logger.add_scalar('modes/num_modes',(self.test_env.number_of_hits_mode>0).sum(), itr)
            self.tb_logger.add_scalar('modes/total_number_of_hits_mode',self.test_env.number_of_hits_mode.sum(), itr)
            self.tb_logger.add_scalars('modes/hits_mode_accurate',{'mode_' + str(i): self.test_env.number_of_hits_mode_acc[i] for i in range(self.test_env.num_goals)}, itr)
            for ind in range(self.test_env.num_goals):
                self.tb_logger.add_scalar('modes/prob_mod_'+str(ind),self.test_env.number_of_hits_mode[ind]/self.test_env.number_of_hits_mode.sum() if self.test_env.number_of_hits_mode.sum() != 0 else 0.0, itr)

            
        if self.env_name in ['max-entropy-v0','multigoal-max-entropy', 'multigoal-max-entropy-obstacles'] and self.ac.pi.actor != 'svgd_sql' and self.test_env.entropy_list is not None:
            feed_dict = {str(self.test_env.entropy_obs_names[i]): self.test_env.entropy_list[i] for i in range(self.test_env.entropy_obs_names.shape[0])}
            self.tb_logger.add_scalars('Entropy/max_entropy_env_Entropies',  feed_dict, itr)

            feed_dict = {'goal_' + str(i): self.average_cumulative_entropy[i] for i in range(self.test_env.num_goals)}
            self.tb_logger.add_scalars('Entropy/max_entropy_env_CumulEntropies',  feed_dict, itr)
            
            if self.env_name in ['max-entropy-v0']:
                self.tb_logger.add_scalars('Entropy/max_entropy_env_Paths', {f'path_{i + 1}': self.test_env.paths[i] for i in range(len(self.test_env.paths))}, itr)
                self.tb_logger.add_scalars('Entropy/max_entropy_env_Failures', {'goal_1': self.train_env.failures[0], 'goal_2': self.train_env.failures[1]}, itr)

                if self.ac.pi.actor == 'sac':
                    feed_dict = {str(self.test_env.entropy_obs_names[i]): self.test_env.mean_list_x[i] for i in range(self.test_env.entropy_obs_names.shape[0])}
                    self.tb_logger.add_scalars('Entropy/max_entropy_env_Means_x',  feed_dict, itr)
                    feed_dict = {str(self.test_env.entropy_obs_names[i]): self.test_env.mean_list_y[i] for i in range(self.test_env.entropy_obs_names.shape[0])}
                    self.tb_logger.add_scalars('Entropy/max_entropy_env_Means_y',  feed_dict, itr)
                    feed_dict = {str(self.test_env.entropy_obs_names[i]): self.test_env.sigma_list_x[i] for i in range(self.test_env.entropy_obs_names.shape[0])}
                    self.tb_logger.add_scalars('Entropy/max_entropy_env_Sigmas_x',  feed_dict, itr)
                    feed_dict = {str(self.test_env.entropy_obs_names[i]): self.test_env.sigma_list_y[i] for i in range(self.test_env.entropy_obs_names.shape[0])}
                    self.tb_logger.add_scalars('Entropy/max_entropy_env_Sigmas_y',  feed_dict, itr)
            

        if self.ac.pi.actor in ['svgd_p0_pram'] and self.env_name in ['Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles']:
            feed_dict_mu_x = {}
            feed_dict_mu_y = {}
            feed_dict_sigma_x = {}
            feed_dict_sigma_y = {}

            for i in range(len(self.test_env._obs_lst)):
                
                o = torch.as_tensor(self.test_env._obs_lst[i], dtype=torch.float32).view(-1,1,self.test_env.observation_space.shape[0]).repeat(1,self.ac.pi.num_particles,1).view(-1,self.test_env.observation_space.shape[0]).to(self.ac.pi.device)
                actions, _ = self.ac(o, action_selection=None, with_logprob=False)
                mu, sigma = self.ac.pi.mu, self.ac.pi.sigma
                feed_dict_mu_x[self.test_env.entropy_obs_names[i]] = mu[0][0]
                feed_dict_mu_y[self.test_env.entropy_obs_names[i]] = mu[0][1]
                feed_dict_sigma_x[self.test_env.entropy_obs_names[i]] = sigma[0][0]
                feed_dict_sigma_y[self.test_env.entropy_obs_names[i]] = sigma[0][1]
            self.tb_logger.add_scalars('Entropy/mean_x',  feed_dict_mu_x, itr)
            self.tb_logger.add_scalars('Entropy/mean_y',  feed_dict_mu_y, itr)
            self.tb_logger.add_scalars('Entropy/std_x',  feed_dict_sigma_x, itr)
            self.tb_logger.add_scalars('Entropy/std_y',  feed_dict_sigma_y, itr)

        if self.ac.pi.actor in ['svgd_p0_pram'] and self.env_name in ['multigoal-max-entropy']:
            feed_dict_mu_x = {}
            feed_dict_mu_y = {}
            feed_dict_sigma_x = {}
            feed_dict_sigma_y = {}

            for i in range(len(self.test_env._obs_lst)):
                
                o = torch.as_tensor(self.test_env._obs_lst[i], dtype=torch.float32).view(-1,1,self.test_env.observation_space.shape[0]).repeat(1,self.ac.pi.num_particles,1).view(-1,self.test_env.observation_space.shape[0]).to(self.ac.pi.device)
                actions, _ = self.ac(o, action_selection=None, with_logprob=False)
                mu, sigma = self.ac.pi.mu, self.ac.pi.sigma
                feed_dict_mu_x[self.test_env.entropy_obs_names[i]] = mu[0][0]
                feed_dict_mu_y[self.test_env.entropy_obs_names[i]] = mu[0][1]
                feed_dict_sigma_x[self.test_env.entropy_obs_names[i]] = sigma[0][0]
                feed_dict_sigma_y[self.test_env.entropy_obs_names[i]] = sigma[0][1]
            self.tb_logger.add_scalars('Entropy/mean_x',  feed_dict_mu_x, itr)
            self.tb_logger.add_scalars('Entropy/mean_y',  feed_dict_mu_y, itr)
            self.tb_logger.add_scalars('Entropy/std_x',  feed_dict_sigma_x, itr)
            self.tb_logger.add_scalars('Entropy/std_y',  feed_dict_sigma_y, itr)


        q_score_ = list(map(lambda x: np.stack(x['q_score']), self.episodes_information))
        q_score_mean = list(map(lambda x: x.mean(), q_score_))
        q_score_min = list(map(lambda x: x.min(), q_score_))
        q_score_max = list(map(lambda x: x.max(), q_score_))
        self.tb_logger.add_scalars('smoothness/q_score_detailed',  {'Mean ': np.mean(q_score_mean), 'Min': np.mean(q_score_min), 'Max': np.mean(q_score_max)  }, itr)
        self.tb_logger.add_scalars('smoothness/q_score_mean_only',  {'Mean ': np.mean(q_score_mean)}, itr)
        q_score_averaged = []

        for i in ['_start', '_mid', '_end']:
            q_score_i = np.array(list(map(lambda x: x['q_score' + i], self.episodes_information)))
            q_score_averaged.append(np.mean(q_score_i[q_score_i != np.array(None)]))
        self.tb_logger.add_scalars('smoothness/q_score_averaged',  {'Start ': q_score_averaged[0], 'Mid': q_score_averaged[1], 'End': q_score_averaged[2] }, itr)

        if self.env_name in ['Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles']:

            q_hess_ = list(map(lambda x: np.stack(x['q_hess']), self.episodes_information))
            q_hess_mean = list(map(lambda x: x.mean(), q_hess_))
            q_hess_min = list(map(lambda x: x.min(), q_hess_))
            q_hess_max = list(map(lambda x: x.max(), q_hess_))

            q_eigenvals = list(map(lambda x: np.stack(x['max_eigenval']), self.episodes_information))
            q_eigenvals_min = list(map(lambda x: x.min(), q_eigenvals))
            q_eigenvals_mean = list(map(lambda x: x.mean(), q_eigenvals))
            q_eigenvals_max = list(map(lambda x: x.max(), q_eigenvals))
            q_eigenvals_abs_min = list(map(lambda x: np.absolute(x).min(), q_eigenvals))
            q_eigenvals_abs_mean = list(map(lambda x: np.absolute(x).mean(), q_eigenvals))
            q_eigenvals_abs_max = list(map(lambda x: np.absolute(x).max(), q_eigenvals))
        

            self.tb_logger.add_scalars('smoothness/q_hess', {'Mean ': np.mean(q_hess_mean), 'Min': np.mean(q_hess_min), 'Max': np.mean(q_hess_max)  }, itr)
            self.tb_logger.add_scalars('smoothness/q_eigenvals', {'Mean ': np.mean(q_eigenvals_mean), 'Min': np.mean(q_eigenvals_min), 'Max': np.mean(q_eigenvals_max)  }, itr)
            self.tb_logger.add_scalars('smoothness/q_eigenvals_abs', {'Mean ': np.mean(q_eigenvals_abs_mean), 'Min': np.mean(q_eigenvals_abs_min), 'Max': np.mean(q_eigenvals_abs_max)  }, itr)
            
            q_hess_averaged = []


            for i in ['_start', '_mid', '_end']:
                q_hess_i = np.array(list(map(lambda x: x['q_hess' + i], self.episodes_information)))
                q_hess_averaged.append(np.mean(q_hess_i[q_hess_i != np.array(None)]))
            self.tb_logger.add_scalars('smoothness/q_hess_averaged', {'Start ': q_hess_averaged[0], 'Mid': q_hess_averaged[1], 'End': q_hess_averaged[2] }, itr)

        
        episode_length = list(map(lambda x: x['episode_length'], self.episodes_information))

        self.tb_logger.add_scalar('Test_EpLen', np.mean(episode_length) , itr)
        
        
        
    def entropy_plot(self):
        if self.ac.pi.actor in ['svgd_p0_pram'] and self.env_name in ['Multigoal', 'multigoal-max-entropy']:
            log_p = []
            logp_normal = []
            logp_svgd = []
            logp_tanh = []
            mu = []
            sigma = []

            for indx, i in enumerate([0, 10, 25]):
                if len(self.episodes_information[-1]['log_p']) > i+1:
                    log_p.append(self.episodes_information[-1]['log_p'][i])
                    mu.append(np.absolute(self.episodes_information[-1]['mu'][i]).mean())
                    sigma.append(np.absolute(self.episodes_information[-1]['sigma'][i]).mean())
                    logp_normal.append(self.episodes_information[-1]['logp_normal'][i])
                    logp_svgd.append(self.episodes_information[-1]['logp_svgd'][i])
                    logp_tanh.append(self.episodes_information[-1]['logp_tanh'][i])

            if len(log_p) == 3:
                self.add_scalars(tb_path='Entropy/logp_normal', value={'step_0': logp_normal[0], 'step_10': logp_normal[1], 'step_25': logp_normal[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_svgd', value={'step_0': logp_svgd[0], 'step_10': logp_svgd[1], 'step_25': logp_svgd[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_tanh', value={'step_0': logp_tanh[0], 'step_10': logp_tanh[1], 'step_25': logp_tanh[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/mu', value={'step_0': mu[0], 'step_10': mu[1], 'step_25': mu[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/sigma', value={'step_0': sigma[0], 'step_10': sigma[1], 'step_25': sigma[2]}, itr=self.episode_counter)


            elif len(log_p) == 2:
                self.add_scalars(tb_path='Entropy/logp_normal', value={'step_0': logp_normal[0], 'step_10': logp_normal[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_svgd', value={'step_0': logp_svgd[0], 'step_10': logp_svgd[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_tanh', value={'step_0': logp_tanh[0], 'step_10': logp_tanh[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/mu', value={'step_0': mu[0], 'step_10': mu[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/sigma', value={'step_0': sigma[0], 'step_10': sigma[1]}, itr=self.episode_counter)

            elif len(log_p) == 1:

                self.add_scalars(tb_path='Entropy/logp_normal', value={'step_0': logp_normal[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_svgd', value={'step_0': logp_svgd[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_tanh', value={'step_0': logp_tanh[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/mu', value={'step_0': mu[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/sigma', value={'step_0': sigma[0]}, itr=self.episode_counter)


            self.episode_counter += 1


    def reset(self,):
        self.episodes_information = []

