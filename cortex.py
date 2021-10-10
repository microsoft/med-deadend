'''
Note: The standard trajectory dictionary has the following structure:
data['ob_cols'] = List with names of columns of observations
data['traj'] = nested dictionary - data['traj'][t]['obs'] = np.array with size length of trajectory t x |ob|
                                 - data['traj'][t]['actions'] = np.array with size length of trajectory t
                                 - data['traj'][t]['rewards'] = np.array with size length of trajectory t
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyprind
from utils import one_hot
import os


class AISGenerate_1(nn.Module):
    def __init__(self, ais_size, obs_dim, num_actions):
        super(AISGenerate_1, self).__init__()
        self.l1 = nn.Linear(obs_dim + num_actions, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.GRUCell(128, ais_size)
    def forward(self, x, h):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        h = self.l3(x, h)
        return h


class AISGenerate_2(nn.Module):
    def __init__(self, ais_size, obs_dim, num_actions):
        super(AISGenerate_2, self).__init__()
        self.l1 = nn.Linear(obs_dim + num_actions, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.GRUCell(64, ais_size)
    def forward(self, x, h):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        h = self.l4(x, h)
        return h


class AISPredict_1(nn.Module):
    def __init__(self, ais_size, obs_dim, num_actions):
        super(AISPredict_1, self).__init__()
        self.l1 = nn.Linear(ais_size + num_actions, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, obs_dim)
    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        obs = self.l3(x)
        return obs


class AISPredict_2(nn.Module):
    def __init__(self, ais_size, obs_dim, num_actions):
        super(AISPredict_2, self).__init__()
        self.l1 = nn.Linear(ais_size + num_actions, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, obs_dim)
    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        obs = self.l4(x)
        return obs


class Cortex(object):
    def __init__(self, train_data_file, validation_data_file, minibatch_size, rng, device, 
                perceptor, ais_size, perceptor_lr, ais_gen_model, ais_pred_model, perception_neg_traj_ratio, 
                folder_location, folder_name, num_actions, obs_dim):
        '''
        We assume discrete actions and scalar rewards!
        '''
        self.rng = rng
        self.device = device
        self.train_data_file = train_data_file
        self.validation_data_file = validation_data_file
        self.minibatch_size = minibatch_size
        self.ais_size = ais_size
        self.state_dim = ais_size
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.perceptor = perceptor
        self.perceptor_lr = perceptor_lr
        self.perception_neg_traj_ratio = perception_neg_traj_ratio
        store_path = os.path.join(folder_location, folder_name)
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        else:
            print("Folder " + store_path + " is found.")
        if not os.path.exists(os.path.join(store_path, 'ais')):
            os.mkdir(os.path.join(store_path, 'ais'))
        if not os.path.exists(os.path.join(store_path, 'ais_checkpoints')):
            os.mkdir(os.path.join(store_path, 'ais_checkpoints'))
        if not os.path.exists(os.path.join(store_path, 'ais_data')):
            os.mkdir(os.path.join(store_path, 'ais_data'))
        self.store_path = store_path
        self.checkpoint_file = os.path.join(store_path, 'ais_checkpoints/checkpoint.pt')
        self.ais_gen_file = os.path.join(store_path, 'ais_data/ais_gen.pt')
        self.ais_pred_file = os.path.join(store_path, 'ais_data/ais_pred.pt')
        self.ais_data_folder = os.path.join(store_path, 'ais_data')
        if ais_gen_model == 1:
            self.ais_gen_model = AISGenerate_1
        elif ais_gen_model == 2:
            self.ais_gen_model = AISGenerate_2
        if ais_pred_model == 1:
            self.ais_pred_model = AISPredict_1
        elif ais_pred_model == 2:
            self.ais_pred_model = AISPredict_2
    
    def reset(self):
        self.epoch_pos_finished = False
        self.epoch_neg_finished = False
        self.epoch_finished = False
        self.train_data_transition_head = 0
        self.train_data_transition_head_pos = 0
        self.train_data_transition_head_neg = 0
        self.train_data_transition_head_pos_last = 0
        self.train_data_transition_head_neg_last = 0
        self.rng.shuffle(self.train_data_transition_indices)
        self.rng.shuffle(self.train_data_transition_indices_pos)
        self.rng.shuffle(self.train_data_transition_indices_pos_last)
        self.rng.shuffle(self.train_data_transition_indices_neg)
        self.rng.shuffle(self.train_data_transition_indices_neg_last)
        return self.epoch_finished
    
    def reset_perceptor_networks(self):
        print('Cortex: reset perceptor')
        self.ais_gen = self.ais_gen_model(self.ais_size, self.obs_dim, self.num_actions).to(self.device)
        self.ais_pred = self.ais_pred_model(self.ais_size, self.obs_dim, self.num_actions).to(self.device)
    
    def load_model_from_checkpoint(self, checkpoint_file_path):
        checkpoint = torch.load(checkpoint_file_path)
        self.ais_gen.load_state_dict(checkpoint['gen_state_dict'])
        self.ais_pred.load_state_dict(checkpoint['pred_state_dict'])
        print("Cortex: generator and predictor models loaded.")

    def load_mk_train_validation_data(self):
        print("Cortex: loading raw data and making trajectory-level data")
        train_data = pd.read_csv(self.train_data_file)
        self.train_data_trajectory = self.make_trajectory_data(train_data)
        validation_data = pd.read_csv(self.validation_data_file)
        self.validation_data_trajectory = self.make_trajectory_data(validation_data)

    def make_trajectory_data(self, data):
        print('Cortex: making trajectory data')
        obs_cols = [i for i in data.columns if i[:2] == 'o:']
        ac_cols  = [i for i in data.columns if i[:2] == 'a:']
        rew_cols = [i for i in data.columns if i[:2] == 'r:']
        #Assuming discrete actions and scalar rewards:
        assert len(obs_cols) > 0, 'No observations present, or observation columns not prefixed with "o:"'
        assert len(ac_cols) > 0, 'No actions present, or actions column not prefixed with "a:"'
        assert len(rew_cols) > 0, 'No rewards present, or rewards column not prefixed with "r:"'
        assert len(ac_cols) == 1, 'Multiple action columns are present when a single action column is expected'
        assert len(rew_cols) == 1, 'Multiple reward columns are present when a single reward column is expected'
        ac_col = ac_cols[0]
        rew_col = rew_cols[0]
        data[ac_col] = data[ac_col]
        all_actions = data[ac_col].unique()
        all_actions.sort()
        try:
            all_actions = all_actions.astype(np.int32)
        except:
            raise ValueError('Actions are expected to be integers, but are not.')
        # if not all(all_actions == np.arange(self.num_actions, dtype=np.int32)):
        #     print(Font.red + 'Some actions are missing from data or all action space not properly defined.' + Font.end)
        print("Number of actions in the file: ", len(all_actions))
        trajectories = data['traj'].unique()
        data_trajectory = {}
        data_trajectory['obs_cols'] = obs_cols
        data_trajectory['ac_col']  = ac_col
        data_trajectory['rew_col'] = rew_col
        data_trajectory['num_actions'] = self.num_actions
        data_trajectory['obs_dim'] = len(obs_cols)
        data_trajectory['traj'] = {}
        data_trajectory['pos_traj'] = []
        data_trajectory['neg_traj'] = []
        bar = pyprind.ProgBar(len(trajectories))
        for i in trajectories:
            bar.update()
            traj_i = data[data['traj'] == i].sort_values(by='step')
            data_trajectory['traj'][i] = {}
            data_trajectory['traj'][i]['obs'] = torch.Tensor(traj_i[obs_cols].values).to(self.device)
            data_trajectory['traj'][i]['actions'] = torch.Tensor(traj_i[ac_col].values.astype(np.int32)).to(self.device).long()
            data_trajectory['traj'][i]['rewards'] = torch.Tensor(traj_i[rew_col].values).to(self.device)
            if sum(traj_i[rew_col].values) > 0:
                data_trajectory['pos_traj'].append(i)
            else:
                data_trajectory['neg_traj'].append(i)
        return data_trajectory

    def train_perceptor(self, perceptor_num_epochs, saving_period, resume):
        if self.perceptor == 'AIS':
            print('Cortex: training perceptor')
            device = self.device
            num_actions = self.train_data_trajectory['num_actions']
            obs_dim = self.train_data_trajectory['obs_dim']
            self.ais_gen = self.ais_gen_model(self.ais_size, obs_dim, num_actions).to(device)
            self.ais_pred = self.ais_pred_model(self.ais_size, obs_dim, num_actions).to(device)
            self.optimizer = torch.optim.Adam(list(self.ais_gen.parameters()) + list(self.ais_pred.parameters()), lr=self.perceptor_lr, amsgrad=True)
            self.perception_losses = []
            self.perception_losses_validation = []
            positive_trajectories = self.train_data_trajectory['pos_traj']
            negative_trajectories = self.train_data_trajectory['neg_traj']
            epoch_trajectories = list(self.train_data_trajectory['traj'].keys())
            if self.perception_neg_traj_ratio != 'NA':
                if len(negative_trajectories)/len(epoch_trajectories) > self.perception_neg_traj_ratio:
                    target_len_positive_trajectories = int(np.round((1-self.perception_neg_traj_ratio)*len(negative_trajectories)/self.perception_neg_traj_ratio))
                    epoch_trajectories = negative_trajectories + target_len_positive_trajectories//len(positive_trajectories)*positive_trajectories + positive_trajectories[:target_len_positive_trajectories%len(positive_trajectories)]
                else:
                    target_len_negative_trajectories = int(np.round(self.perception_neg_traj_ratio*len(negative_trajectories)/(1-self.perception_neg_traj_ratio)))
                    epoch_trajectories = positive_trajectories + target_len_negative_trajectories//len(negative_trajectories)*negative_trajectories + negative_trajectories[:target_len_negative_trajectories%len(negative_trajectories)]
            if resume:
                checkpoint = torch.load(self.checkpoint_file)
                self.ais_gen.load_state_dict(checkpoint['gen_state_dict'])
                self.ais_pred.load_state_dict(checkpoint['pred_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch_0 = checkpoint['epoch'] + 1
                self.perception_losses = checkpoint['loss']
                self.perception_losses_validation = checkpoint['validation_loss']
                print('Starting from epoch: {0} and continuing upto epoch {1}'.format(epoch_0, perceptor_num_epochs))
            else:
                epoch_0 = 0
            for epoch in range(epoch_0, perceptor_num_epochs):
                epoch_loss = []
                print("Cortex:Perceptor {0}: training Epoch = ".format(self.perceptor), epoch+1, 'out of', perceptor_num_epochs, 'epochs')
                bar = pyprind.ProgBar(len(epoch_trajectories))
                for traj in epoch_trajectories:
                    bar.update()
                    loss_pred = 0
                    h = torch.zeros(self.ais_size).to(device).view(1,-1)
                    obs = self.train_data_trajectory['traj'][traj]['obs']
                    actions = self.train_data_trajectory['traj'][traj]['actions'].view(-1,1)
                    rewards = self.train_data_trajectory['traj'][traj]['rewards'].view(-1,1)
                    ais = torch.zeros(obs.shape[0], self.ais_size).to(device)
                    action = torch.zeros(num_actions).to(device) #Initial action; all zeros
                    rew = torch.zeros(1).to(device) #Initial rewrad; zero
                    for step in range(obs.shape[0]-1):
                        h = self.ais_gen(torch.cat((obs[step,:], action)).view(1,-1), h)
                        ais[step,:] = h
                        action = one_hot(actions[step], num_actions, data_type='torch', device=device)
                        rew = rewards[step]
                        obs_pred_next_probs = self.ais_pred((torch.cat((ais[step,:], action))).view(1,-1))
                        # Loss in predicting distribution of next observation
                        loss_pred += -torch.distributions.MultivariateNormal(obs_pred_next_probs[0,:], torch.eye(obs_pred_next_probs[0,:].shape[0]).to(device)).log_prob(obs[step+1,:])
                    self.optimizer.zero_grad()
                    if obs.shape[0] > 1:
                        loss_pred.backward()
                        self.optimizer.step()
                        epoch_loss.append(loss_pred.detach().cpu().numpy())
                self.perception_losses.append(epoch_loss)
                if (epoch+1) % saving_period == 0:
                    #Computing validation loss
                    epoch_validation_loss = []
                    for traj in self.validation_data_trajectory['traj'].keys():
                        loss_val = 0
                        h_val = torch.zeros(self.ais_size).to(device).view(1, -1)
                        obs_val = self.validation_data_trajectory['traj'][traj]['obs']
                        actions_val = self.validation_data_trajectory['traj'][traj]['actions'].view(-1, 1)
                        rewards_val = self.validation_data_trajectory['traj'][traj]['rewards'].view(-1, 1)
                        ais_val = torch.zeros(obs.shape[0], self.ais_size).to(device)
                        action_val = torch.zeros(num_actions).to(device) #Initial action; all zeros
                        rew_val = torch.zeros(1).to(device) #Initial rewrad; zero
                        for step in range(obs_val.shape[0]-1):
                            with torch.no_grad():
                                h_val = self.ais_gen(torch.cat((obs_val[step,:], action_val)).view(1,-1), h_val)
                                ais_val[step,:] = h_val
                                action_val = one_hot(actions_val[step], num_actions, data_type='torch', device=device)
                                rew_val = rewards_val[step]
                                obs_pred_next_probs_val = self.ais_pred((torch.cat((ais_val[step,:], action_val))).view(1,-1))
                                # Loss in predicting distribution of next observation
                                loss_val += -torch.distributions.MultivariateNormal(obs_pred_next_probs_val[0,:], torch.eye(obs_pred_next_probs_val[0,:].shape[0]).to(device)).log_prob(obs_val[step+1,:])    
                        if obs_val.shape[0] > 1:
                            epoch_validation_loss.append(loss_val.detach().cpu().numpy())
                    self.perception_losses_validation.append(epoch_validation_loss)
                    # Save off checkpoint
                    try:
                        torch.save({
                            'epoch': epoch,
                            'gen_state_dict': self.ais_gen.state_dict(),
                            'pred_state_dict': self.ais_pred.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': self.perception_losses,
                            'validation_loss': self.perception_losses_validation,
                            }, self.checkpoint_file[:-3] + str(epoch) +'.pt')
                        np.save(self.ais_data_folder + '/ais_losses.npy', np.array(self.perception_losses))
                    except:
                        pass
                    # Save off validation losses
                    try:
                        np.save(self.ais_data_folder + '/ais_validation_losses.npy', np.array(self.perception_losses_validation))
                    except:
                        pass
            #Final epoch checkpoint
            try:
                torch.save(self.ais_gen.state_dict(), self.ais_gen_file)
                torch.save(self.ais_pred.state_dict(), self.ais_pred_file)
                torch.save({
                            'epoch': perceptor_num_epochs-1,
                            'gen_state_dict': self.ais_gen.state_dict(),
                            'pred_state_dict': self.ais_pred.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': self.perception_losses,
                            'validation_loss': self.perception_losses_validation,
                    }, self.checkpoint_file)
                np.save(self.ais_data_folder + '/ais_losses.npy', np.array(self.perception_losses))
            except:
                pass
            print('Cortex: Perception training finished successfully')
    
    def encode_data(self, data_trajectory):
        d = data_trajectory.copy()
        print("Cortex: encoding data")
        bar = pyprind.ProgBar(len(data_trajectory['traj'].keys()))
        for traj in data_trajectory['traj'].keys():
            bar.update()
            obs = data_trajectory['traj'][traj]['obs']
            actions = data_trajectory['traj'][traj]['actions'].view(-1,1)
            rewards = data_trajectory['traj'][traj]['rewards'].view(-1,1)
            ais = torch.zeros(obs.shape[0], self.ais_size).to(self.device)
            h = torch.zeros(self.ais_size).to(self.device).view(1,-1)
            a = torch.zeros(self.num_actions).to(self.device)
            # r = torch.zeros(1).to(self.device)
            with torch.no_grad():
                for step in range(obs.shape[0]):
                    h = self.ais_gen(torch.cat((obs[step,:], a)).view(1,-1), h)
                    ais[step,:] = h
                    # a = one_hot(actions[step], self.num_actions, data_type='torch', device=self.device)
                    # r = rewards[step]
            d['traj'][traj]['obs'] = ais.cpu().numpy()
            d['traj'][traj]['s'] = d['traj'][traj].pop('obs')  # switch to "s" (sinces it's state)
            d['traj'][traj]['actions'] = d['traj'][traj]['actions'].cpu().numpy()
            d['traj'][traj]['rewards'] = d['traj'][traj]['rewards'].cpu().numpy()
        s_cols = ['s:' + str(i) for i in range(self.ais_size)]
        d['s_cols'] = s_cols
        d['s_dim'] = len(s_cols)
        return d
    
    @staticmethod
    def encoded_trajectory_data_to_file(trajectory_data, filename):
        print('Cortex: Writing encoded trajectory data to file')
        col_names = ['traj', 'step']
        col_names.extend(['s:'+ i[2:] for i in trajectory_data['s_cols']])
        col_names.append('a:action')
        col_names.append('r:reward')
        all_data = []
        bar = pyprind.ProgBar(len(list(trajectory_data['traj'].keys())))
        for i in trajectory_data['traj'].keys():
            bar.update()
            for ctr in range(trajectory_data['traj'][i]['actions'].shape[0]):
                all_data.append([])
                all_data[-1].append(i)
                all_data[-1].append(ctr)
                for s_index in range(trajectory_data['traj'][i]['s'].shape[1]):
                    all_data[-1].append(trajectory_data['traj'][i]['s'][ctr, s_index])
                all_data[-1].append(int(trajectory_data['traj'][i]['actions'][ctr]))
                all_data[-1].append(trajectory_data['traj'][i]['rewards'][ctr])
        df = pd.DataFrame(all_data, columns=col_names)
        df.to_csv(filename, index=False)
