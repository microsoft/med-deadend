import os
import click
import yaml
import numpy as np
import pandas as pd
import torch
from state_construction import StateConstructor
from rl import RL
from experiment import DQNExperiment
from utils import DataLoader


def load_best_sc_network(params, rng):
    # NOTE: returned SC-Network will need to either re-load data or load some new test data
    store_path = os.path.join(params["folder_location"], params["folder_name"])  # this is `sc_network.store_path` if a SC-Network is loaded with params 
    # Initialize the SC-Network
    sc_network = StateConstructor(train_data_file=params["train_data_file"], validation_data_file=params["validation_data_file"], 
                            minibatch_size=params["minibatch_size"], rng=rng, device=params["device"], save_for_testing=params["save_all_checkpoints"],
                            sc_method=params["sc_method"], state_dim=params["embed_state_dim"], sc_learning_rate=params["sc_learning_rate"], 
                            ais_gen_model=params["ais_gen_model"], ais_pred_model=params["ais_pred_model"], sc_neg_traj_ratio=params["sc_neg_traj_ratio"], 
                            folder_location=params["folder_location"], folder_name=params["folder_name"], 
                            num_actions=params["num_actions"], obs_dim=params["obs_dim"])
    sc_network.reset_sc_networks()
    # Provide SC-Network with the pre-trained parameter set
    sc_network.load_model_from_checkpoint(checkpoint_file_path=os.path.join(store_path, "ais_checkpoints", "checkpoint_best.pt"))
    return sc_network


def make_data_loaders(train_data_encoded, validation_data_encoded, rng, device):
    # Note that the loaders will be reset in Experiment
    loader_train = DataLoader(train_data_encoded, rng, 64, False, device, ": train data")
    loader_validation = DataLoader(validation_data_encoded, rng, 256, False, device, ": validation data")
    loader_train.make_transition_data(release=True)
    loader_validation.make_transition_data(release=True)
    return loader_train, loader_validation


def train(params, rng, loader_train, loader_validation):
    qnet = RL(state_dim=params["embed_state_dim"], nb_actions=params["num_actions"], gamma=params["gamma"], learning_rate=params["rl_learning_rate"], 
                update_freq=params["update_freq"], use_ddqn=params["use_ddqn"], rng=rng, device=params["device"], 
                sided_Q=params["sided_Q"], network_size=params["rl_network_size"])
    print('Initializing Experiment')
    expt = DQNExperiment(data_loader_train=loader_train, data_loader_validation=loader_validation, q_network=qnet, ps=0, ns=2,
                        folder_location=params["folder_location"], folder_name=params["folder_name"], 
                        saving_period=params["exp_saving_period"], rng=rng, resume=params["rl_resume"])
    with open(os.path.join(expt.storage_rl, 'config_exp.yaml'), 'w') as y:
            yaml.safe_dump(params, y)  # saving new params for future reference
    print('Running experiment (training Q-Networks)')
    expt.do_epochs(number=params["exp_num_epochs"])
    print("Training Q-Networks finished successfully")


@click.command()
@click.option('--options', '-o', multiple=True, nargs=2, type=click.Tuple([str, str]))
@click.option('--folder', '-f', help="Main project folder that includes config.yaml")
@click.option('--save/--no-save', default=True, help=r"Use this flag to also save encoded states for post analysis.")
def run(options, folder, save):
    '''
    Run to train RL separately from SC-Network, or alternatively call make_train() from a script.
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
    
    # Initialize and load the pre-trained parameters for the SC-Network
    sc_network = load_best_sc_network(params, rng)  # note that the loaded SC-Network has no data inside
    params["used_checkpoint_for_rl"] = "checkpoint_best.pt"

    sc_network.load_mk_train_validation_data()
    print("Train data ...")
    train_data_encoded = sc_network.encode_data(sc_network.train_data_trajectory)
    print("Validation data ...")
    validation_data_encoded = sc_network.encode_data(sc_network.validation_data_trajectory)

    for sided_Q in ["negative", "positive"]:
        print("AI: ", sided_Q.capitalize())
        params['sided_Q'] = sided_Q
        loader_train, loader_validation = make_data_loaders(train_data_encoded, validation_data_encoded, rng, params['device'])
        train(params, rng, loader_train, loader_validation)

    if save:
        output_encoded_file_train = os.path.join(folder, 'encoded_train_data.csv')
        output_encoded_file_validation = os.path.join(folder, 'encoded_validation_data.csv')
        output_encoded_file_test = os.path.join(folder, 'encoded_test_data.csv')
        test_data = pd.read_csv(params['test_data_file'])
        test_data_trajectory = sc_network.make_trajectory_data(test_data)
        test_data_encoded = sc_network.encode_data(test_data_trajectory)

        sc_network.encoded_trajectory_data_to_file(train_data_encoded, output_encoded_file_train)
        sc_network.encoded_trajectory_data_to_file(validation_data_encoded, output_encoded_file_validation)
        sc_network.encoded_trajectory_data_to_file(test_data_encoded, output_encoded_file_test)


if __name__ == '__main__':
    run()
