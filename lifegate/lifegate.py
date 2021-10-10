import os
from copy import deepcopy
import pygame
import numpy as np
import click


# RGB colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
GREEN = (0, 255, 0)
WALL = (80, 80, 80)
YELLOW = (255, 255, 0)


class LifeGate(object):
    def __init__(self, state_mode, rng, death_drag, max_steps=100, fixed_life=True, rendering=False, image_saving=False, render_dir=None):
        self.rng = rng
        self.state_dtype = np.float32
        self.frame_skip = 1  # for env consistency
        self.fixed_life = fixed_life
        self.blue = BLUE
        self.death_drag = death_drag
        self.legal_actions = [0, 1, 2, 3, 4]
        self.action_meanings = ['no-op', 'up', 'down', 'left', 'right']
        self.reward_scheme = {'death': -1.0, 'recovery': +1.0, 'step': 0.0, 'barrier': 0.0}
        self.nb_actions = len(self.legal_actions)
        self.player_pos_x = None
        self.player_pos_y = None
        self.agent_init_pos = None
        self.state_mode = state_mode    # how the returned state look like ('pixel' or '1hot' or 'multi-head')
        # self.scr_w = None
        # self.scr_h = None
        # self.possible_recoveries = []
        self.recovery_observablity = True
        # self.observability_switch_point = None  # where to turn observability off
        # self.rendering_scale = None
        # self.barriers = None
        self.recoveries = None
        self.deaths = None
        # self.dead_ends = None
        self._rendering = rendering
        # self.state_shape = None
        self.init_subclass()
        if rendering:
            self._init_pygame()
        self.image_saving = image_saving
        self.render_dir_main = render_dir
        self.render_dir = None
        self.state = None
        self.step_id = 0
        self.game_over = False

        self.max_steps = max_steps

        self.reset()

    def init_subclass(self):
        # should implement sizes, barriers, recoveries, deaths, init_player(), and rendering_scale
        self.scr_w, self.scr_h = 10, 10
        self.tabular_state_shape = (self.scr_w, self.scr_h)
        self.state_shape = [24]
        self.rendering_scale = 30
        self.barriers = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [1, 5], [2, 5], [3, 5], [4, 5]]
        self.possible_recoveries = [[5, 0], [6, 0], [7, 0]]
        self.main_deaths = [[self.scr_w - 1, k] for k in range(self.scr_h)] + [[8,0]]
        self.dead_ends = [[x, y] for x in range(self.scr_w // 2, self.scr_w - 1) for y in range(self.scr_w // 2, self.scr_w)]
        self.observability_switch_point = [0, 5]

    @property
    def rendering(self):
        return self._rendering

    @rendering.setter
    def rendering(self, flag):
        if flag is True:
            if self._rendering is False:
                self._init_pygame()
                self._rendering = True
        else:
            self.close()
            self._rendering = False

    def _init_pygame(self):
        pygame.init()
        size = [self.rendering_scale * self.scr_w, self.rendering_scale * self.scr_h]
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("LifeGate")

    def _init_rendering_folder(self):
        if self.render_dir_main is None:
            self.render_dir_main = 'render'
        if not os.path.exists(os.path.join(os.getcwd(), self.render_dir_main)):
            os.mkdir(os.path.join(os.getcwd(), self.render_dir_main))
        i = 0
        while os.path.exists(os.path.join(os.getcwd(), self.render_dir_main, 'render' + str(i))):
            i += 1
        self.render_dir = os.path.join(os.getcwd(), self.render_dir_main, 'render' + str(i))
        os.mkdir(self.render_dir)
    
    def reset(self):
        if self.image_saving:
            self._init_rendering_folder()
        self.game_over = False
        self.step_id = 0
        self.recovery_observablity = True
        self.blue = BLUE
        state = self.init_episode()
        return state
    
    def init_episode(self):
        # should implement reconfigurations at the beginning of each episode
        self.player_pos_x, self.player_pos_y = 2, self.scr_h - 1
        targets = deepcopy(self.possible_recoveries)
        # if self.fixed_life == True:
        #     rec = targets.pop(2)  # fixed life-gate for DQN 
        # else:
        #     rec = targets.pop(self.rng.randint(len(targets)))
        self.recoveries = targets #[rec]
        self.deaths = self.main_deaths #+ targets
        return self.get_obs(self.state_mode)
    
    def render(self):
        if not self.rendering:
            return
        pygame.event.pump()
        self.screen.fill(BLACK)
        size = [self.rendering_scale, self.rendering_scale]
        for pos in self.dead_ends:
            p = [self.rendering_scale * pos[0], self.rendering_scale * pos[1]]
            rec1 = pygame.Rect(p[0], p[1], size[0], size[1])
            pygame.draw.rect(self.screen, YELLOW, rec1)
        player = pygame.Rect(self.rendering_scale * self.player_pos_x, self.rendering_scale * self.player_pos_y,
                             size[0], size[1])
        pygame.draw.rect(self.screen, WHITE, player)
        for pos in self.deaths:
            p = [self.rendering_scale * pos[0], self.rendering_scale * pos[1]]
            rec1 = pygame.Rect(p[0], p[1], size[0], size[1])
            pygame.draw.rect(self.screen, RED, rec1)
        for pos in self.recoveries:
            p = [self.rendering_scale * pos[0], self.rendering_scale * pos[1]]
            rec1 = pygame.Rect(p[0], p[1], size[0], size[1])
            pygame.draw.rect(self.screen, self.blue, rec1)  # self.blue will change if reach obs point
        for pos in self.barriers:
            p = [self.rendering_scale * pos[0], self.rendering_scale * pos[1]]
            rec1 = pygame.Rect(p[0], p[1], size[0], size[1])
            pygame.draw.rect(self.screen, WALL, rec1)
        pygame.display.flip()

        if self.image_saving:
            self.save_image()

    def save_image(self):
        if self.rendering and self.render_dir is not None:
            pygame.image.save(self.screen, self.render_dir + '/render' + str(self.step_id) + '.jpg')
        else:
            raise ValueError('env.rendering is False and/or environment has not been reset.')
    
    def close(self):
        if self.rendering:
            pygame.quit()
    
    def _move_player(self, action):
        x, y = (self.player_pos_x, self.player_pos_y)
        # dead-end:
        if [x, y] in self.dead_ends:
            if self.rng.binomial(1, 0.70):
                action = 4  # forceful right
            else:
                action = 0  # no-op
        else:
            # natural risk of death
            if self.rng.binomial(1, self.death_drag):  # say with 25% if death_drag==0.25
                action = 4
        
        if action == 4:    # right
            x += 1
        elif action == 3:  # left
            x -= 1
        elif action == 2:  # down
            y += 1
        elif action == 1:  # up
            y -= 1
        # updating the position
        if [x, y] in self.barriers or x < 0 or y < 0 or y >= self.scr_h:
            return
        else:
            self.player_pos_x, self.player_pos_y = x, y
        
    def _get_status(self):
        # check the current situation
        if [self.player_pos_x, self.player_pos_y] in self.deaths:
            return 'death'
        elif [self.player_pos_x, self.player_pos_y] in self.recoveries:
            return 'recovery'

    def step(self, action):
        assert action in self.legal_actions, 'Illegal action.'
        if self.step_id >= self.max_steps - 1:
            self.game_over = True
            return self.get_obs(self.state_mode), 0., self.game_over, {}
        self.step_id += 1
        self._move_player(action)
        if [self.player_pos_x, self.player_pos_y] == self.observability_switch_point and self.recovery_observablity == True:
            self.recovery_observablity = False
            self.blue = BLACK
        status = self._get_status()
        if status == 'death':
            self.game_over = True
            reward = self.reward_scheme['death']
        elif status == 'recovery':
            self.game_over = True
            reward = self.reward_scheme['recovery']
        else:
            reward = self.reward_scheme['step']
        return self.get_obs(self.state_mode), reward, self.game_over, {}
    
    def get_lives(self):
        if self.game_over == True:
            return 0
        else:
            return 1

    def get_state(self):
        return self.get_obs(self.state_mode)
    
    def get_obs(self, method):
        if method == 'vector':
            return self._get_vec_obs()
        elif method == 'pixel':
            return self._get_pixel_obs()
        elif method == 'tabular':
            return self._get_tabular_obs()
        else:
            raise ValueError('Unknown observation method.')
    
    def _get_vec_obs(self):
        x = np.zeros(self.scr_w + self.scr_h + len(self.possible_recoveries), dtype=self.state_dtype)
        x[self.player_pos_x] = 1.0
        x[self.player_pos_y + self.scr_w] = 1.0
        if self.recovery_observablity == True or self.fixed_life == True:
            for k in self.recoveries:
                x[k[0] - 5 + self.scr_w + self.scr_h] = 1.0
        return x

    def _get_tabular_obs(self):
        return np.array([self.player_pos_x, self.player_pos_y])

    def _get_pixel_obs(self):
        raise NotImplementedError


class LifeGatePivot(LifeGate):
    def init_subclass(self):
        self.scr_w, self.scr_h = 10, 10
        self.state_shape = [24]
        self.rendering_scale = 30
        self.barriers = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [1, 5], [2, 5], [3, 5], [4, 5]]
        self.possible_recoveries = [[5,0], [6, 0], [7, 0]]
        self.main_deaths = [[self.scr_w - 1, k] for k in range(self.scr_h)]+ [[8,0]]
        self.dead_ends = [[x, y] for x in range(self.scr_w // 2, self.scr_w - 1) for y in range(self.scr_w // 2, self.scr_w)]
        self.observability_switch_point = [0, 5]
    
    def init_episode(self):
        self.player_pos_x, self.player_pos_y = 2, self.scr_h - 1
        targets = deepcopy(self.possible_recoveries)
        # if self.fixed_life == True:
        #     rec = targets.pop(2)  # fixed life-gate for DQN 
        # else:
        #     rec = targets.pop(self.rng.randint(len(targets)))
        self.recoveries = targets #[rec]
        self.deaths = self.main_deaths #+ targets


@click.command()
@click.option('--save/--no-save', default=False, help='Saving rendering screen.')
def test(save):
    rng = np.random.RandomState(123)
    env = LifeGatePivot(rng=rng, state_mode='vector', death_drag=0.2, rendering=True, image_saving=save)
    print('state shape', env.state_shape)
    for _ in range(1):
        env.reset()
        env.render()
        while not env.game_over:
            action = input()
            action = int(action)
            obs, r, term, info = env.step(action)
            env.render()
            print("\033[2J\033[H\033[2J", end="")
            print()
            print('pos: ', env.player_pos_x, env.player_pos_y)
            print('reward: ', r)
            print('state:')
            print('─' * 30)
            print(obs)
            print('─' * 30)


if __name__ == '__main__':
    test()
