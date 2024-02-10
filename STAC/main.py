import argparse
from core import MaxEntrRL
import random
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from envs.multigoal_env import MultiGoalEnv
from envs.multigoal_env_obstacles import MultiGoalObstaclesEnv
from envs.multigoal_max_entropy_env import MultiGoalMaxEntropyEnv
from envs.multigoal_max_entropy_env_obstacles import MultiGoalMaxEntropyObstaclesEnv
import numpy as np
import gym
import mujoco_py
from datetime import datetime
from utils import AttrDict
import timeit
from tqdm import tqdm



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--gpu_id', type=int, default=2)
    # IMPORTANT: multigoal-max-entropy-obstacles and multigoal-obstacles should only be used at test time using a saved agent traned on the version of the environment without an obstacle.
    parser.add_argument('--env', type=str, default='multigoal-max-entropy', choices=['Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles', 'multigoal-obstacles', 'Hopper-v2', 'Ant-v2', 'Walker2d-v2', 'Humanoid-v2', 'HalfCheetah-v2', 'landmark', 'simple_landmark'])
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--actor', type=str, default='svgd_sql', choices=['sac', 'svgd_sql', 'svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram', 'diffusion'])

    ###### networks
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l_critic', type=int, default=2)
    parser.add_argument('--l_actor', type=int, default=3)
    parser.add_argument('--critic_activation', type=object, default=torch.nn.ELU)
    parser.add_argument('--actor_activation', type=object, default=torch.nn.ELU) 

    ###### RL 
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--a_c', type=float, default=0.2)
    parser.add_argument('--a_a', type=float, default=0.2)
    parser.add_argument('--replay_size', type=int, default=1e6)
    parser.add_argument('--load_replay', type=int, default=0)
    parser.add_argument('--max_experiment_steps', type=float, default=5e4)
    parser.add_argument('--exploration_steps', type=int, default=10000, help="pure exploration at the beginning of the training")
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--update_after', type=int, default=1000)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--critic_cnn', action='store_true')
    ###### optim 
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--lr_critic', type=float, default=1e-3)
    parser.add_argument('--lr_actor', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=100)
    
    ###### action selection
    parser.add_argument('--train_action_selection', type=str, default='random', choices=['random', 'max', 'softmax', 'adaptive_softmax', 'softmax_egreedy'])
    parser.add_argument('--test_action_selection', type=str, default='random', choices=['random', 'max', 'softmax', 'adaptive_softmax', 'softmax_egreedy', 'amortized'])
    parser.add_argument('--svgd_particles', type=int, default=10)
    parser.add_argument('--svgd_steps', type=int, default=5)
    parser.add_argument('--svgd_lr', type=float, default=0.1)
    parser.add_argument('--svgd_sigma_p0', type=float, default=0.5)
    parser.add_argument('--svgd_kernel_sigma', type=float, default=None)
    parser.add_argument('--kernel_sigma_adaptive', type=int, default=4)
    parser.add_argument('--with_amor_infer', action='store_true')
   
    # tensorboard
    parser.add_argument('--plot_format', type=str, default='pdf', choices=['png', 'jpeg', 'pdf', 'svg'])
    parser.add_argument('--stats_steps_freq', type=int, default=400) 
    parser.add_argument('--collect_stats_after', type=int, default=0)
    parser.add_argument('--model_path', type=str, default='./evaluation_data/z_after/svgd_nonparam_999999')

    
    
    
    ###################################################################################
    # A label to differantiate experiments based on their importance (primary, secondary, and debugging experiments)
    parser.add_argument('--experiment_importance', type=str, default='dbg', choices=['prm', 'scn', 'dbg']) 
    # Load one checkpoint from a previous experiment, and evaluate the agent in that checkpoint
    parser.add_argument('--test_time', type=int, default=0)
    # Evaluate each checkpoint of a previous experiment
    parser.add_argument('--all_checkpoints_test', type=int, default=0) 
    # Use the debugging hyperparameters (Used mainly to not edit the default parameters while debugging)
    parser.add_argument('--debugging', type=int, default=0) 
    ###################################################################################
        
    args = parser.parse_args()  
    args.debugging = bool(args.debugging)
    args.load_replay = bool(args.load_replay)
    args.test_time = bool(args.test_time)
    args.all_check_points_test = bool(args.all_checkpoints_test)


    if args.test_time:
        print('############################## TEST TIME ###################################')
        print('############################################################################')
        print('############################################################################')

    ################# Best parameters for a specific algorithm/environment #################
    if args.actor == 'svgd_sql':
        args.critic_activation = torch.nn.ReLU
        args.actor_activation = torch.nn.ReLU

    if args.actor in ['sac']:
        args.critic_activation = torch.nn.ReLU
        args.actor_activation = torch.nn.ReLU
    
    if args.env in ['Hopper-v2', 'Ant-v2', 'Walker2d-v2', 'Humanoid-v2', 'HalfCheetah-v2']:
        args.max_steps = 1000
    if args.env in ['multigoal-max-entropy', 'multigoal-max-entropy-obstacles']:
        args.num_test_episodes = 20

    
    if args.debugging:
        print('############################## DEBUGGING ###################################')
        args.exploration_steps = 1000
        args.max_experiment_steps = 30000
        args.num_test_episodes = 10
        print('############################################################################')
        
    # fix the seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # set number of thereads
    torch.set_num_threads(torch.get_num_threads())
    
    # get device
    device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    # actor arguments
    if (args.actor in ['svgd_nonparam','svgd_p0_pram','svgd_p0_kernel_pram']):
        actor_kwargs=AttrDict(num_svgd_particles=args.svgd_particles, num_svgd_steps=args.svgd_steps, 
            svgd_lr=args.svgd_lr, test_action_selection=args.test_action_selection, svgd_sigma_p0 = args.svgd_sigma_p0,
            batch_size=args.batch_size,  device=device, hidden_sizes=[args.hid]*args.l_actor, activation=args.actor_activation, 
            kernel_sigma=args.svgd_kernel_sigma, adaptive_sig=args.kernel_sigma_adaptive, 
            alpha=args.a_a, with_amor_infer=args.with_amor_infer)
    
    elif (args.actor == 'svgd_sql'):
        actor_kwargs=AttrDict(num_svgd_particles=args.svgd_particles, test_action_selection=args.test_action_selection, 
            batch_size=args.batch_size,  device=device, hidden_sizes=[args.hid]*args.l_actor, 
            activation=args.actor_activation, kernel_sigma=args.svgd_kernel_sigma, adaptive_sig=args.kernel_sigma_adaptive)
    elif (args.actor =='sac'):
        actor_kwargs=AttrDict(hidden_sizes=[args.hid]*args.l_actor, test_action_selection=args.test_action_selection, device=device, activation=args.actor_activation, batch_size=args.batch_size)
    
    # Logging
    project_name = args.experiment_importance + '_'
    if args.test_time:
        project_name +=  'test_'
    project_name += 'cnn' if args.critic_cnn else 'mlp'
    project_name +=  datetime.now().strftime("%b_%d_%Y_%H_%M_%S")+ '_' + args.actor + '_' + args.env + '_' + 'tnas_' + args.train_action_selection + '_ttas_' + args.test_action_selection  + '_a_c_'+str(args.a_c) + '_a_a_'+str(args.a_a) + '_bs_'+ str(args.batch_size) + '_gamma_' + str(args.gamma) + '_seed_' + str(args.seed) + '_ntep_' + str(args.num_test_episodes) + '_'
    if args.actor in ['svgd_nonparam']:
        project_name += 'ss_'+str(args.svgd_steps)+'_sp_'+str(args.svgd_particles)+'_slr_'+str(args.svgd_lr) + '_ssgm_p0_' + str(args.svgd_sigma_p0) + '_sks_' + str(args.svgd_kernel_sigma) + '_' + str(args.kernel_sigma_adaptive) + '_'
    elif args.actor in ['svgd_p0_pram', 'svgd_p0_kernel_pram']:
        project_name += 'ss_'+str(args.svgd_steps)+'_sp_'+str(args.svgd_particles)+'_slr_'+str(args.svgd_lr) + '_sks_' + str(args.svgd_kernel_sigma) + '_' + str(args.kernel_sigma_adaptive) + '_'
    elif args.actor in ['svgd_sql']:
        project_name += 'sparticles_'+str(args.svgd_particles)+'_slr_'+str(args.svgd_lr) + '_ssigma_p0_' + str(args.svgd_sigma_p0) + '_skernel_sigma_' + str(args.svgd_kernel_sigma) + '_' + str(args.kernel_sigma_adaptive) + '_'
    if args.test_time:
        project_name += 'PID_' + str(os.getpid())
    else:
        project_name += 'expr_' + str(args.max_experiment_steps) + '_explr_' + str(args.exploration_steps) + '_updt_' + str(args.update_after) + '_PID_' + str(os.getpid())


    # Handling Logging folders
    evaluation_data_path = 'evaluation_data/'
    tensorboard_path = 'tensorboard/'
    figures_path = 'figs/'
    replay_path = 'buffers/'
    data_path = os.path.join('./runs/', args.env, args.actor, project_name)
    if not os.path.exists(data_path):
        if not args.all_checkpoints_test:
            os.makedirs(data_path)
            os.makedirs(os.path.join(data_path, tensorboard_path))
            tb_logger = SummaryWriter(os.path.join(data_path, tensorboard_path))
            os.makedirs(os.path.join(data_path, figures_path))
            if not args.test_time:
                os.makedirs(os.path.join(data_path, evaluation_data_path))
                os.makedirs(os.path.join(data_path, replay_path))
    else:
        raise 'Error: Same file exists twice'



    # RL args
    RL_kwargs = AttrDict(stats_steps_freq=args.stats_steps_freq,gamma=args.gamma,
        alpha_c=args.a_c, alpha_a=args.a_a, replay_size=int(args.replay_size), exploration_steps=args.exploration_steps, update_after=args.update_after,
        update_every=args.update_every, num_test_episodes=args.num_test_episodes, max_steps = args.max_steps, 
        max_experiment_steps=int(args.max_experiment_steps), evaluation_data_path = os.path.join(data_path, evaluation_data_path), 
        debugging=args.debugging, plot_format=args.plot_format, load_replay= args.load_replay, replay_path=os.path.join(data_path, replay_path), fig_path=os.path.join(data_path, figures_path),
        collect_stats_after=args.collect_stats_after, test_time=args.test_time, all_checkpoints_test=args.all_checkpoints_test, model_path=args.model_path, train_action_selection=args.train_action_selection)

    # optim args
    optim_kwargs = AttrDict(polyak=args.polyak,lr_critic=args.lr_critic, lr_actor=args.lr_actor,batch_size=args.batch_size)

    # critic args
    critic_kwargs = AttrDict(hidden_sizes=[args.hid]*args.l_critic, activation=args.critic_activation,
                             critic_cnn=args.critic_cnn, gpu_id=args.gpu_id)

    # Choosing the env
    if args.env =='Multigoal':
        train_env = MultiGoalEnv(env_name='train_env', max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
        test_env = MultiGoalEnv(env_name='test_env', max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
    elif args.env == 'multigoal-max-entropy':
        train_env = MultiGoalMaxEntropyEnv(env_name='train_env', max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
        test_env = MultiGoalMaxEntropyEnv(env_name='test_env', max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
    elif args.env == 'multigoal-obstacles':
        train_env = MultiGoalObstaclesEnv(env_name='train_env', max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
        test_env = MultiGoalObstaclesEnv(env_name='test_env', max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
    elif args.env == 'multigoal-max-entropy-obstacles':
        train_env = MultiGoalMaxEntropyObstaclesEnv(env_name='train_env', max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
        test_env = MultiGoalMaxEntropyObstaclesEnv(env_name='test_env', max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
    else: 
        # MuJoCo environments
        train_env = gym.make(args.env)
        test_env = gym.make(args.env)

    # Logging the Hyperparameters used in the experiment
    print('########################################## Hyper-Parameters ##########################################')
    if not args.test_time:
        print('Debugging: ', args.debugging)
        print('GPU ID: ', args.gpu_id)
        print('Environment: ', args.env)
        print('Algorithm: ', args.actor)
        print('Hidden layer size: ', args.hid)
        print('Critic\'s Number of layers: ', args.l_critic)
        if args.actor not in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            print('Actor\'s Number of layers: ', args.l_actor)
        print('Critic\'s Activation: ', args.critic_activation)
        if args.actor not in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            print('Actor\'s Activation: ', args.actor_activation)
        print('Discount Factor (Gamma): ', args.gamma)
        print('Entropy coefficient (Alpha Critic): ', args.a_c)
        print('Entropy coefficient (Alpha Actor): ', args.a_a)
        print('Replay Buffer size: ', args.replay_size)
        print('Load Replay Buffer: ', args.load_replay)
        print('Experiment\'s steps: ', args.max_experiment_steps)
        print('Initial Exploration steps: ', args.exploration_steps)
        print('Number test episodes: ', args.num_test_episodes)
        print('Start Updating models after step: ', args.update_after)
        print('Update models every: ', args.update_every)
        print('Max Environment steps: ', args.max_steps)
        print('Polyak target update rate: ', args.polyak) 
        print('Critic\'s learning rate: ', args.lr_critic)
        if args.actor not in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            print('Actor\'s learning rate: ', args.lr_actor)
        print('Batch size: ', args.batch_size)

        print('Train action selection: ', args.train_action_selection)
        print('Test action selection: ', args.test_action_selection)

        if args.actor in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram', 'svgd_sql']:
            print('Number of particles for SVGD: ', args.svgd_particles)
            print('SVGD learning Rate: ', args.svgd_lr)
        if args.actor in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            print('Number of SVGD steps: ', args.svgd_steps)
            print('SVGD initial distribution\'s variance: ', args.svgd_sigma_p0)
            print('SVGD\'s kernel variance: ', args.svgd_kernel_sigma)
        print('Plot format: ', args.plot_format)
        print('Statistics Collection frequency: ', args.stats_steps_freq)
        print('Collect Statistics after: ', args.collect_stats_after)
        print('Seed: ', args.seed)
        print('Device: ', device)
    print('Project Name: ', project_name)
    print('Experiment Importance: ', args.experiment_importance)
    print('Experiment PID: ', os.getpid())
    print('######################################################################################################')


    stac=MaxEntrRL(train_env, test_env, env=args.env, actor=args.actor, device=device, 
        critic_kwargs=critic_kwargs, actor_kwargs=actor_kwargs,
        RL_kwargs=RL_kwargs, optim_kwargs=optim_kwargs,tb_logger=tb_logger,
        need_q=args.env in ["simple_landmark", "landmark"])

    start = timeit.default_timer()
    if args.test_time:
        # One checkpoint test
        stac.test_agent(0)
    else:
        # Training
        stac.forward()
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    print('Experiment Finished.') 
    print(project_name)














