from landmark.medical import MedicalPlayer
from gym import spaces
import numpy as np
from collections import (Counter, defaultdict, deque, namedtuple)
import copy
import gym

class CalibratedMedicalPlayer(MedicalPlayer):
    """
    Continuous, Single Agent landmark detection task that
    keeps as much of the original features as possible
    """
    
    def __init__(self, 
                 directory=None,
                 viz=False,
                 task=False,
                 files_list=None,
                 file_type="brain",
                 landmark_ids=None,
                 screen_dims=(45,45,45),
                 history_length=28,
                 multiscale=True,
                 max_num_frames=0,
                 saveGif=False,
                 saveVideo=False,
                 agents=1,
                 oscillations_allowed=4,
                 fixed_spawn=None,
                 logger=None,
                 nsteps=200):
        super().__init__(directory, viz, task, files_list, file_type, 
                         landmark_ids, screen_dims, history_length, multiscale,
                         max_num_frames, saveGif, saveVideo, agents,
                         oscillations_allowed, fixed_spawn, logger)
        self.action_shape = 3
        self.frame_stack = 4
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_shape,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(np.prod(screen_dims),), dtype=np.uint8)
        # self.n_steps = nsteps
        self.pixel_scale = 10
        # self.actions = None
        self.action_map = {
            # Z axis
            0: {
                # forward
                -1: 0,
                0: 0,
                # backward
                1: 5,
            },
            # Y axis
            1: {
                -1: 1,
                0: 1,
                1: 4,
            },
            # X axis
            2: {
                -1: 2,
                0: 2,
                1: 3,
            }
        }
        
    def reset(self, fixed_spawn=None):
        # self.steps = 0
        return super().reset(fixed_spawn).reshape(-1)

    def step(self, act, q_values, isOver=None):
        tot_reward = 0
        terminals = []
        for dim in range(3):
            act[dim] = np.clip(act[dim], -1, 1)
            move_pixels = np.abs(act[dim]) * self.pixel_scale + 1
            move_action = [self.action_map[dim][np.sign(act[dim])]]
            for _ in range(int(move_pixels)):
                state, reward, terminal, _ = super().step(move_action, q_values, None)
                print(reward, end=" ")
                print(terminal, end=" ")
                tot_reward += reward
                terminals.append(terminal)
        print()
        return state.reshape(-1), tot_reward, any(terminals), None
  
    def _calc_reward(self, current_loc, next_loc, agent):
        curr_dists = np.min([self.calcDistance(current_loc, target_loc, self.spacing) for target_loc in self._target_loc])
        next_dists = np.min([self.calcDistance(next_loc, target_loc, self.spacing) for target_loc in self._target_loc])
        return curr_dists - next_dists
    
    
class FrameStack(gym.Wrapper):
    """used when not training. wrapper for Medical Env"""

    def __init__(self, env, k):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.k = k  # history length
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape[0]
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp*k,),
                                            dtype=np.uint8)

    def reset(self, fixed_spawn=None):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset(fixed_spawn)
        for _ in range(self.k - 1):
            self.frames.append(np.zeros_like(ob))
        self.frames.append(ob)
        return self._observation()

    def step(self, acts, q_values, isOver):
        for i in range(self.agents):
            if isOver[i]:
                acts[i] = 15
        current_st, reward, terminal, info = self.env.step(
            acts, q_values, isOver)
        current_st = tuple(current_st)
        self.frames.append(current_st)
        return self._observation(), reward, terminal, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.stack(self.frames, axis=-1)
        # if self._base_dim == 2:
        #     return np.stack(self.frames, axis=-1)
        # else:
        #     return np.concatenate(self.frames, axis=2)
    
class SimpleMedicalPlayer(MedicalPlayer):
    """
    Simple Single Agent landmark detection task
    """
    def __init__(self, 
                 directory=None,
                 viz=False,
                 task=False,
                 files_list=None,
                 file_type="brain",
                 landmark_ids=None,
                 screen_dims=(27,27,27),
                 history_length=28,
                 multiscale=True,
                 max_num_frames=0,
                 saveGif=False,
                 saveVideo=False,
                 agents=1,
                 oscillations_allowed=4,
                 fixed_spawn=None,
                 logger=None,
                 nsteps=200):
        super().__init__(directory, viz, task, files_list, file_type, 
                         landmark_ids, screen_dims, history_length, multiscale,
                         max_num_frames, saveGif, saveVideo, agents,
                         oscillations_allowed, fixed_spawn, logger)
        self.action_shape = 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_shape,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(np.prod(screen_dims),), dtype=np.uint8)
        self.n_steps = nsteps
        
    def reset(self, fixed_spawn=None):
        self.steps = 0
        return super().reset(fixed_spawn).reshape(-1)
    
    def step(self, act, q_value):
        if self.action_shape == 1:
            return self.step_1dim(act, q_value)
        elif self.action_shape == 3:
            return self.step_3dim(act, q_value)
    
    def step_3dim(self, act, q_value):
        self.steps += 1
        rewards = []
        terminals = []
        for dim in range(3):
            act_dim = 2 * dim + 1 if  np.sign(act[dim]) > -1 else 0
            state, reward, terminal, info = super().step([act_dim], q_values=[q_value], isOver=0)
            rewards.append(reward)
            terminals.append(terminal[0])
        if self.steps >= self.n_steps:
            terminals.append(True)
        return state.reshape(-1), np.sum(rewards), np.any(terminals), info
        
    
    def step_1dim(self, act, q_value):
        self.steps += 1
        act = int((act+1)/(2/6))
        if act == 6:
            act = 5
        if act not in range(6):
            raise ValueError("Invalid action")
        state, reward, terminal, info = super().step([act], q_values=[q_value], isOver=0)
        if self.steps >= self.n_steps:
            terminal = [True]
        return state.reshape(-1), reward, terminal[0], info

    def _calc_reward(self, current_loc, next_loc, agent):
        curr_dists = np.min([self.calcDistance(current_loc, target_loc, self.spacing) for target_loc in self._target_loc])
        next_dists = np.min([self.calcDistance(next_loc, target_loc, self.spacing) for target_loc in self._target_loc])
        return curr_dists - next_dists

class SAMedicalPlayer(MedicalPlayer):
    """
    Single Agent landmark detection task
    """
    def __init__(self, 
                 directory=None,
                 viz=False,
                 task=False,
                 files_list=None,
                 file_type="brain",
                 landmark_ids=None,
                 screen_dims=(27,27,27),
                 history_length=28,
                 multiscale=True,
                 max_num_frames=0,
                 saveGif=False,
                 saveVideo=False,
                 agents=1,
                 oscillations_allowed=4,
                 fixed_spawn=None,
                 n_steps=200,
                 logger=None):
        super().__init__(directory, viz, task, files_list, file_type, 
                         landmark_ids, screen_dims, history_length, multiscale,
                         max_num_frames, saveGif, saveVideo, agents,
                         oscillations_allowed, fixed_spawn, logger)
        # todo: determine a proper way to define the continuous action space
        # 3/6 dimensions
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        # todo: add convnet to handle pixel input
        self.observation_space = spaces.Box(low=0, high=255, shape=(np.prod(screen_dims),), dtype=np.uint8)
        self.n_steps = n_steps
        
    def reset(self, fixed_spawn=None):
        self.steps = 0
        return super().reset(fixed_spawn).reshape(-1)
    
    # def step(self, act):
    #     self.steps += 1
    #     act = int((act+1)/(2/6))
    #     assert act in range(6)
    #     # todo: decide terminal properly
    #     # todo: rewrite the reward fn; OK
    #     state, reward, terminal, info = super().step([act], q_values=[[0]], isOver=0)
    #     if self.steps >= self.n_steps:
    #         terminal = [True]
    #     return state.reshape(-1), reward, terminal[0], info
    def step(self, act):
        """The environment's step function returns exactly what we need.
        Args:
          act: (3,)
        Returns:
          observation (object):
            an environment-specific object representing your observation of
            the environment. For example, pixel data from a camera, joint
            angles and joint velocities of a robot, or the board state in a
            board game.
          reward (float):
            amount of reward achieved by the previous action. The scale varies
            between environments, but the goal is always to increase your total
            reward.
          done (boolean):
            whether it's time to reset the environment again. Most (but not
            all) tasks are divided up into well-defined episodes, and done
            being True indicates the episode has terminated. (For example,
            perhaps the pole tipped too far, or you lost your last life.)
          info (dict):
            diagnostic information useful for debugging. It can sometimes be
            useful for learning (for example, it might contain the raw
            probabilities behind the environment's last state change). However,
            official evaluations of your agent are not allowed to use this for
            learning.
        """
        # Only one agent with index 0
        self.steps += 1
        current_loc = list(self._location[0])
        next_location = copy.deepcopy(current_loc)

        self.terminal = [False] * self.agents
        go_out = [False] * self.agents

        assert len(current_loc) == len(act)
        for dim in range(3):
            # agent movement
            # (0,1/3) -> move 1 pixel; (1/3,2/3) -> move 2 pixels; (2/3,1) -> move 3 pixels
            # next_location[dim] = current_loc[dim] + int(act[dim]*3+1)
            # (0,1) -> move 1 pixel; (-1,0) -> -1
            next_location[dim] = current_loc[dim] + int(np.sign(act[dim]))
            # check if reaches boundary
            if (next_location[dim] >= self._image_dims[dim]):
                next_location = current_loc
                go_out[0] = True

        # punish -1 reward if the agent tries to go out
        if self.task != 'play':
            if go_out[0]:
                self.reward[0] = -1
            else:
                self.reward[0] = self._calc_reward(
                    current_loc, next_location, agent=0)

        # update screen, reward ,location, terminal
        self._location = [next_location]
        self._screen = self._current_state()

        # terminate if the distance is less than 1 during trainig
        if self.task == 'train':
            for i in range(self.agents):
                if self.cur_dist[i] <= 1:
                    # self.logger.log(f"distance of agent {i} is <= 1")
                    print(f"distance of agent {0} is <= 1")
                    self.terminal[i] = True
                    self.num_success[i] += 1

        # terminate if maximum number of steps is reached
        if self.steps >= self.n_steps:
            self.terminal[0] = True

        # update history buffer with new location and qvalues
        if self.task != 'play':
            for i in range(self.agents):
                self.cur_dist[i] = self.calcDistance(self._location[i],
                                                     self._target_loc[i],
                                                     self.spacing)


        distance_error = self.cur_dist
        for i in range(self.agents):
            self.current_episode_score[i].append(self.reward[i])

        info = {}
        for i in range(self.agents):
            info[f"score_{i}"] = np.sum(self.current_episode_score[i])
            info[f"gameOver_{i}"] = self.terminal[i]
            info[f"distError_{i}"] = distance_error[i]
            info[f"filename_{i}"] = self.filename[i]
            info[f"agent_xpos_{i}"] = self._location[i][0]
            info[f"agent_ypos_{i}"] = self._location[i][1]
            info[f"agent_zpos_{i}"] = self._location[i][2]
            info[f"landmark_xpos_{i}"] = self._target_loc[i][0]
            info[f"landmark_ypos_{i}"] = self._target_loc[i][1]
            info[f"landmark_zpos_{i}"] = self._target_loc[i][2]
        # reshape the state into a 1d vector
        return self._current_state().reshape(-1), self.reward[0], self.terminal[0], info


    def _calc_reward_dist(self, current_loc, next_loc, agent):
        """
        Calculate the new reward based on the max decrease in euclidean 
        distance to the target location
        """     
        # minimum distance to target
        dists = [self.calcDistance(next_loc, loc, self.spacing) for loc in self._target_loc]
        return -min(dists)

    def _calc_reward(self, current_loc, next_loc, agent):
        curr_dists = np.min([self.calcDistance(current_loc, target_loc, self.spacing) for target_loc in self._target_loc])
        next_dists = np.min([self.calcDistance(next_loc, target_loc, self.spacing) for target_loc in self._target_loc])
        return curr_dists - next_dists
    
if __name__ == '__main__':
    player = SAMedicalPlayer()