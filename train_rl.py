import os
import click
import yaml
import numpy as np
import pandas as pd
import torch
import re
from cortex import Cortex
from ai import AI
from experiment import DQNExperiment
from utils import DataLoader


def load_best_cortex(params, rng):
    # NOTE: returned cortex will need to either re-load data or load some new test data
    store_path = os.path.join(params["folder_location"], params["folder_name"])  # this is `cortex.store_path` if a cortex is loaded with params 
    checkpoint_dir = os.path.join(store_path, 'ais_checkpoints')
    f = os.listdir(checkpoint_dir)
    f = [k for k in f if k[-3:] == ".pt"]
    checkpoint_epochs = sorted([int(s) for k in f for s in re.findall(r'\d+', k)])
    if 'checkpoint.pt' in f:
        last_checkpoint_idx = ''
    else:
        last_checkpoint_idx = max(checkpoint_epochs)
    f = os.path.join(store_path, "ais_checkpoints", "checkpoint" + str(last_checkpoint_idx) + ".pt")
    c = torch.load(f)
    l = np.array(c["validation_loss"])
    best_cortex_idx = np.argmin(np.mean(l, axis=1))
    cortex = Cortex(train_data_file=params["train_data_file"], validation_data_file=params["validation_data_file"], 
                            minibatch_size=params["minibatch_size"], rng=rng, device=params["device"], perceptor=params["perceptor"], 
                            ais_size=params["ais_size"], perceptor_lr=params["perceptor_lr"], ais_gen_model=params["ais_gen_model"], 
                            ais_pred_model=params["ais_pred_model"], perception_neg_traj_ratio=params["perception_neg_traj_ratio"], 
                            folder_location=params["folder_location"], folder_name=params["folder_name"], 
                            num_actions=params["num_actions"], obs_dim=params["obs_dim"])
    cortex.reset_perceptor_networks()
    cortex.load_model_from_checkpoint(checkpoint_file_path=os.path.join(store_path, "ais_checkpoints", "checkpoint" + str(checkpoint_epochs[best_cortex_idx]) + ".pt"))
    return cortex, checkpoint_epochs[best_cortex_idx]


def make_data_loders(train_data_encoded, validation_data_encoded, rng, device):
    # Note that the loaders will be reset in Experiment
    loader_train = DataLoader(train_data_encoded, rng, 64, False, device, ": train data")
    loader_validation = DataLoader(validation_data_encoded, rng, 256, False, device, ": validation data")
    loader_train.make_transition_data(release=True)
    loader_validation.make_transition_data(release=True)
    return loader_train, loader_validation


def train(params, rng, loader_train, loader_validation):
    ai = AI(state_dim=params["ais_size"], nb_actions=params["num_actions"], gamma=params["gamma"], learning_rate=params["ai_lr"], 
                update_freq=params["update_freq"], use_ddqn=params["use_ddqn"], rng=rng, device=params["device"], 
                sided_Q=params["sided_Q"], network_size=params["ai_network_size"])
    print('Initializing Experiment')
    expt = DQNExperiment(data_loader_train=loader_train, data_loader_validation=loader_validation, ai=ai, ps=0, ns=2,
                        folder_location=params["folder_location"], folder_name=params["folder_name"], 
                        saving_period=params["exp_saving_period"], rng=rng, resume=params["ai_resume"])
    with open(os.path.join(expt.storage_ai, 'config_exp.yaml'), 'w') as y:
            yaml.safe_dump(params, y)  # saving new params for future reference
    print('Running experiment (training AI)')
    expt.do_epochs(number=params["exp_num_epochs"])
    print("Training AI finished successfully")


@click.command()
@click.option('--options', '-o', multiple=True, nargs=2, type=click.Tuple([str, str]))
@click.option('--folder', '-f', help="Main project folder that includes config.yaml")
@click.option('--save/--no-save', default=True, help=r"Use this flag to also save encoded states for post analysis.")
def run(options, folder, save):
    '''
    Run to train RL separately from Cortex, or alternatively call make_train() from a script.
    '''
    folder = os.path.abspath(folder)
    with open(os.path.join(folder, "config.yaml")) as f:
        params = yaml.safe_load(f)
    
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

    cortex, best_cortex_idx = load_best_cortex(params, rng)  # note that the loaded cortex has no data inside
    params["used_checkpoint_for_ai"] = "checkpoint" + str(best_cortex_idx) + ".pt"
    print(r">>> Best `cortex` found: ", best_cortex_idx)

    cortex.load_mk_train_validation_data()
    print("Train data ...")
    train_data_encoded = cortex.encode_data(cortex.train_data_trajectory)
    print("Validation data ...")
    validation_data_encoded = cortex.encode_data(cortex.validation_data_trajectory)

    for sided_Q in ["negative", "positive"]:
        print("AI: ", sided_Q.capitalize())
        params['sided_Q'] = sided_Q
        loader_train, loader_validation = make_data_loders(train_data_encoded, validation_data_encoded, rng, params['device'])
        train(params, rng, loader_train, loader_validation)

    if save:
        output_encoded_file_train = os.path.join(folder, 'encoded_train_data.csv')
        output_encoded_file_validation = os.path.join(folder, 'encoded_validation_data.csv')
        output_encoded_file_test = os.path.join(folder, 'encoded_test_data.csv')
        test_data = pd.read_csv(params['test_data_file'])
        test_data_trajectory = cortex.make_trajectory_data(test_data)
        test_data_encoded = cortex.encode_data(test_data_trajectory)

        cortex.encoded_trajectory_data_to_file(train_data_encoded, output_encoded_file_train)
        cortex.encoded_trajectory_data_to_file(validation_data_encoded, output_encoded_file_validation)
        cortex.encoded_trajectory_data_to_file(test_data_encoded, output_encoded_file_test)


if __name__ == '__main__':
    run()
