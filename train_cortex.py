import os
import click
import yaml
import numpy as np
from cortex import Cortex
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  #  base folder on the running machine or VM
OUTPUT_DIR = ROOT_DIR


@click.command()
@click.option('--options', '-o', multiple=True, nargs=2, type=click.Tuple([str, str]))
def run(options):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cfg_file = os.path.join(dir_path, 'config_sepsis.yaml')
    params = yaml.safe_load(open(cfg_file, 'r'))

    # replacing params with command line options
    for opt in options:
        assert opt[0] in params
        dtype = type(params[opt[0]])
        if dtype == bool:
            new_opt = False if opt[1] != 'True' else True
        else:
            new_opt = dtype(opt[1])
        params[opt[0]] = new_opt

    print('Parameters ')
    for key in params:
        print(key, params[key])
    print('=' * 30)

    np.random.seed(params['random_seed'])
    torch.manual_seed(params['random_seed'])
    rng = np.random.RandomState(params['random_seed'])

    for ex in range(params["num_experiments"]):
        print('>>>>> Experiment ', ex, ' >>>>>')
        print('Initializing and training cortex')
        cortex = Cortex(train_data_file=params["train_data_file"], validation_data_file=params["validation_data_file"], 
                        minibatch_size=params["minibatch_size"], rng=rng, device=params["device"], perceptor=params["perceptor"], 
                        ais_size=params["ais_size"], perceptor_lr=params["perceptor_lr"], ais_gen_model=params["ais_gen_model"], 
                        ais_pred_model=params["ais_pred_model"], perception_neg_traj_ratio=params["perception_neg_traj_ratio"], 
                        folder_location=os.path.join(OUTPUT_DIR, params["folder_location"]), folder_name=params["folder_name"], 
                        num_actions=params["num_actions"], obs_dim=params["obs_dim"])
        
        with open(os.path.join(cortex.store_path, 'config.yaml'), 'w') as y:
            yaml.safe_dump(params, y)  # saving params for future reference
        
        cortex.load_mk_train_validation_data()
        cortex.train_perceptor(perceptor_num_epochs=params["perceptor_num_epochs"], saving_period=params["ais_saving_period"], resume=params["perceptor_resume"])
    print("Cortex training finished successfully")


if __name__ == '__main__':
    run()
