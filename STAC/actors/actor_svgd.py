import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from networks import MLPSquashedGaussian, MLPFunction
from torch.distributions import Normal, Categorical
from actors.kernels import RBF


class ActorSvgd(torch.nn.Module):
    def __init__(self, actor, obs_dim, act_dim, act_limit, num_svgd_particles, svgd_sigma_p0, num_svgd_steps, svgd_lr, test_action_selection, batch_size, adaptive_sig,
    device, hidden_sizes, q1, q2, activation=torch.nn.ReLU, kernel_sigma=None, alpha=1,
    with_amor_infer=False):
        super().__init__()
        self.actor = actor
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.num_particles = num_svgd_particles
        self.num_svgd_steps = num_svgd_steps
        self.svgd_lr = svgd_lr
        self.device = device
        self.q1 = q1
        self.q2 = q2
        self.sigma_p0 = svgd_sigma_p0
        self.test_action_selection = test_action_selection
        self.batch_size = batch_size
        self.alpha = alpha
        self.with_amor_infer = with_amor_infer


        if actor == "svgd_nonparam":
            self.a0 = torch.normal(0, self.sigma_p0, size=(5 * batch_size * num_svgd_particles, self.act_dim)).to(self.device)
        elif actor == 'svgd_p0_pram':
            self.p0 = MLPSquashedGaussian(obs_dim, act_dim, hidden_sizes, activation)
        if self.with_amor_infer:
            self.amortized_net = MLPFunction(obs_dim, act_dim, act_dim, hidden_sizes, activation)

        if actor == "svgd_p0_kernel_pram":
            self.Kernel = RBF(True, act_dim, hidden_sizes, sigma=kernel_sigma, device=device)
        else:
            self.Kernel = RBF(num_particles=self.num_particles, sigma=kernel_sigma, adaptive_sig=adaptive_sig, device=device)
        
        # Identity
        self.identity = torch.eye(self.num_particles).to(self.device)
        self.identity_mat = torch.eye(self.act_dim).to(self.device)


    def svgd_optim(self, x, dx, dq): 
        dx = dx.view(x.size())
        x = x + self.svgd_lr * dx
        return x
    
    
    def sampler(self, obs, a, with_logprob=True):
        logp = 0
        q_s_a = None

        def phi(X):
            nonlocal logp
            X.requires_grad_(True)
            log_prob1 = self.q1(obs, X)
            log_prob2 = self.q2(obs, X)
            log_prob = torch.min(log_prob1, log_prob2)


            score_func = autograd.grad(log_prob.sum(), X, retain_graph=True, create_graph=True)[0]
           
            X = X.reshape(-1, self.num_particles, self.act_dim)
            score_func = score_func.reshape(X.size())
            K_XX, K_diff, K_gamma, K_grad = self.Kernel(X, X)
            phi = (K_XX.matmul(score_func) + K_grad.sum(1)) / self.num_particles 
            
            # compute the entropy
            if with_logprob:
                term1 = (K_grad * score_func.unsqueeze(1)).sum(-1).sum(2)/(self.num_particles-1)
                term2 = -2 * K_gamma.squeeze(-1).squeeze(-1) * ((K_grad.permute(0,2,1,3) * K_diff).sum(-1) - self.act_dim * (K_XX - self.identity)).sum(1) / (self.num_particles-1)

                logp = logp - self.svgd_lr * (term1 + term2) 
            
            return phi, log_prob, score_func 
        

        for t in range(self.num_svgd_steps):
            phi_, q_s_a, dq = phi(a)

            a = self.svgd_optim(a, phi_, dq)

        return a, logp, q_s_a

    def act(self, obs, action_selection=None, with_logprob=True, loss_q_=None, itr=None):
        logp_a = None
        
        if self.with_amor_infer:
            a_0 = torch.rand((len(obs), self.act_dim)).view(-1,self.act_dim).to(self.device)
            a_amor = torch.tanh(self.amortized_net(obs, a_0)) * self.act_limit

        if self.actor == "svgd_nonparam":
            a0 = self.a0[torch.randint(len(self.a0), (len(obs),))]
        elif self.actor == 'svgd_p0_pram':

            ns = int(2000/self.num_particles)
            obs_tmp = obs.view(-1, self.num_particles, self.obs_dim).repeat(1, ns, 1)
            self.mu, self.sigma = self.p0(obs_tmp)
            self.sigma = torch.clamp(self.sigma, 0.1, 1.0)
            
            self.init_dist_normal = Normal(self.mu, self.sigma)
            while True:
                a0 = self.init_dist_normal.rsample()
                a0 = torch.clamp(a0, self.mu-3*self.sigma, self.mu+3*self.sigma)

                indicies = torch.logical_and(a0 > self.mu - 3 * self.sigma, a0 < self.mu + 3 * self.sigma).all(-1)
                
                if (indicies.type(torch.float32).sum(-1) >= self.num_particles).all():
                    new_a0 = []
                    for i in range(a0.shape[0]):
                        new_a0.append(a0[i][indicies[i]][:self.num_particles])
                    new_a0 = torch.stack(new_a0)
                    a0 = new_a0.view(-1,self.act_dim)
                    self.mu, self.sigma = self.mu[:, :self.num_particles, :].reshape(-1, self.act_dim), self.sigma[:, :self.num_particles, :].reshape(-1, self.act_dim)
                    
                    self.init_dist_normal = Normal(self.mu, self.sigma)
                    break



        self.a0_debbug = a0.view(-1, self.num_particles, self.act_dim)
        # run svgd
        a, logp_svgd, q_s_a = self.sampler(obs, a0, with_logprob) 
        # compute the entropy 
        if with_logprob:
            if self.actor == "svgd_nonparam":
                logp_normal = - self.act_dim * 0.5 * np.log(2 * np.pi * self.sigma_p0) - (0.5 / self.sigma_p0) * (a0**2).sum(-1).view(-1,self.num_particles)
            elif self.actor == 'svgd_p0_pram':
                logp_normal = self.init_dist_normal.log_prob(a0).sum(axis=-1).view(-1,self.num_particles)
            logp_tanh_2 = - ( 2 * (np.log(2) - a - F.softplus(-2 * a))).sum(axis=-1).view(-1,self.num_particles)
            logp_a = (logp_normal + logp_svgd + logp_tanh_2).mean(-1)
            self.logp_normal_debug = logp_normal.mean()
            try:
                self.logp_svgd_debug = logp_svgd.mean()
            except:
                self.logp_svgd_debug = torch.tensor(0)
            self.logp_tanh_debug = logp_tanh_2.mean()
            
        a = self.act_limit * torch.tanh(a) 
        
        self.a =  a.view(-1, self.num_particles, self.act_dim)

        # at test time
        if action_selection is None:
            a = self.a
        elif action_selection == 'max':
            if self.num_svgd_steps == 0:
                q_s_a1 = self.q1(obs, a.view(-1, self.act_dim))
                q_s_a2 = self.q2(obs, a.view(-1, self.act_dim))
                q_s_a = torch.min(q_s_a1, q_s_a2)
                a_ = self.mu.view(-1, self.num_particles, self.act_dim)[:, 0, :]
            q_s_a = q_s_a.view(-1, self.num_particles)
            a = self.a[:,q_s_a.argmax(-1)]
        elif action_selection == 'softmax':
            if self.num_svgd_steps == 0:
                q_s_a1 = self.q1(obs, a.view(-1, self.act_dim))
                q_s_a2 = self.q2(obs, a.view(-1, self.act_dim))
                q_s_a = torch.min(q_s_a1, q_s_a2)
            q_s_a = q_s_a.view(-1, self.num_particles)
            soft_max_probs = torch.exp((q_s_a - q_s_a.max(dim=1, keepdim=True)[0]))
            dist = Categorical(soft_max_probs / torch.sum(soft_max_probs, dim=1, keepdim=True))
            a = self.a[:,dist.sample()]
        elif action_selection == 'random':
            a = self.a[:,np.random.randint(self.num_particles),:]
        elif action_selection == 'amortized':
            a = a_amor

        return a.view(-1, self.act_dim), logp_a