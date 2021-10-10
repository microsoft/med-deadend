import random
import os
import pickle
import click
import yaml
import numpy as np

from lifegate import LifeGate
# from lifegate_utils import Font
from ai import AI, AICount, Experiment

# np.set_printoptions(suppress=True, linewidth=200, precision=3)
floatX = 'float32'

class Font:
    purple = '\033[95m'
    cyan = '\033[96m'
    darkcyan = '\033[36m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bgblue = '\033[44m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'


def worker(params):
    np.random.seed(seed=params['random_seed'])
    random.seed(params['random_seed'])
    random_state = np.random.RandomState(params['random_seed'])

    if params['test'] is True:
        # remember to pass correct `folder_name` when call with test==True
        file_path = os.getcwd() + params['folder_location'] + params['folder_name'] + '/tabular_ai.pkl'
        with open(file_path, 'rb') as f:
            ai = pickle.load(f)
        env = LifeGate(max_steps=params['episode_max_len'], bridge_len=params['bridge_len'], state_mode='tabular',
                        rendering=True, image_saving=False, render_dir=None, rng=random_state)
        s = env.reset()
        env.render()
        term = False
        while not term:
            print('=' * 50)
            action = int(input('>>> Action: '))
            print(Font.bold + '>>> Q: ' + Font.end, [ai.get_q(s, a) for a in range(env.nb_actions)])
            s, r, term, info = env.step(action)
            print('reward: ', r, ' | term: ', term, ' | info: ', info)
            env.render()
    else:
        for ex in range(params['nb_experiments']):
            print('\n')
            print(Font.bold + Font.red + '>>>>> Experiment ', ex, ' >>>>>' + Font.end)
            print('\n')
            env = LifeGate(params['state_mode'], random_state, params['death_drag'], fixed_life=params['fixed_life'], 
                            rendering=False, image_saving=False, render_dir=None)
            # --- MAIN AI ---
            if params['explore_method'] in ['count', 'secure_count', 'secure']:
                ai = AICount(state_shape=env.tabular_state_shape, nb_actions=env.nb_actions,
                                init_q=params['init_q'], gamma=params['gamma'], alpha=params['alpha'],
                                learning_method=params['learning_method'], rng=random_state)
            else:
                ai = AI(state_shape=env.tabular_state_shape, nb_actions=env.nb_actions, init_q=params['init_q'],
                        gamma=params['gamma'], alpha=params['alpha'], learning_method=params['learning_method'],
                        rng=random_state)
            # --- Q_d and Q_r Networks ---
            q_d = AI(state_shape=env.tabular_state_shape, nb_actions=env.nb_actions, init_q=np.float32(0),
                            gamma=np.float32(1), alpha=params['alpha'], learning_method='ql', rng=random_state)
            
            q_r = AI(state_shape=env.tabular_state_shape, nb_actions=env.nb_actions, init_q=np.float32(0),
                            gamma=np.float32(1), alpha=params['alpha'], learning_method='ql', rng=random_state)

            # --- EXPERIMENT ---
            expt = Experiment(ai=ai, ai_explore=q_d, Q_R=q_r, env=env, saving_period=params['saving_period'],
                              printing_period=params['printing_period'], writing_period=params['writing_period'],
                              learning_method=params['learning_method'], explore_method=params['explore_method'],
                              epsilon=params['epsilon'], annealing_start_episode=params['annealing_start_episode'],
                              final_epsilon=params['final_epsilon'], annealing=params['annealing'],
                              annealing_episodes=params['annealing_episodes'], make_folder=True,
                              episode_max_len=params['episode_max_len'], rng=random_state,
                              folder_location=params['folder_location'], folder_name=params['folder_name'])
            with open(os.path.join(expt.folder_name, 'config.yaml'), 'w') as f:
                yaml.dump(params, f, default_flow_style=False)
            expt.run(nb_episodes=params['nb_episodes'], learning=True,
                     target_eval=params['target_eval'], nb_eval=params['nb_eval'])


@click.command()
@click.option('--options', '-o', multiple=True, nargs=2, type=click.Tuple([str, str]))
@click.option('--make/--no-make', '-m', default=False, help='Make results for paper.')  # NOTE THIS OPTION
def run(options, make):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cfg_file = os.path.join(dir_path, 'config.yaml')
    params = yaml.safe_load(open(cfg_file, 'r'))

    if make:  # making results for the paper
        params['nb_experiments'] = 10
        for l in [5, 7, 9, 11]:
            params['bridge_len'] = l
            params['folder_name'] = 'bridge_len' + str(l) + '_'
            worker(params)

    else:  # single run
        # replacing params with command line options
        for opt in options:
            assert opt[0] in params
            dtype = type(params[opt[0]])
            if dtype == bool:
                new_opt = False if opt[1] != 'True' else True
            else:
                new_opt = dtype(opt[1])
            params[opt[0]] = new_opt

        print('\n')
        print(Font.bold + Font.red + 'Parameters ' + Font.end)
        for key in params:
            print(key, params[key])
        print('\n')
        worker(params)


if __name__ == '__main__':
    run()
