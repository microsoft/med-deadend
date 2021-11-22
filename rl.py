import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.)


class QNetwork_64(nn.Module):
    def __init__(self, state_dim=16, nb_actions=None):
        super(QNetwork_64, self).__init__()

        self.state_dim = state_dim
        self.nb_actions = nb_actions
        
        self.fc = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.nb_actions)
        )
        self.fc.apply(init_weights)
        
    def forward(self, x):
        x = self.fc(x)
        return x


class QNetwork_128(nn.Module):
    def __init__(self, state_dim=16, nb_actions=None):
        super(QNetwork_128, self).__init__()

        self.state_dim = state_dim
        self.nb_actions = nb_actions
        
        self.fc = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.nb_actions)
        )
        self.fc.apply(init_weights)
        
    def forward(self, x):
        x = self.fc(x)
        return x


class QNetwork_6464(nn.Module):
    def __init__(self, state_dim=16, nb_actions=None):
        super(QNetwork_6464, self).__init__()

        self.state_dim = state_dim
        self.nb_actions = nb_actions
        
        self.fc = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.nb_actions)
        )
        self.fc.apply(init_weights)
        
    def forward(self, x):
        x = self.fc(x)
        return x


class RL(object):
    def __init__(self, state_dim, nb_actions, gamma,
                 learning_rate, update_freq,use_ddqn,
                 rng, device, sided_Q, network_size):
        self.rng = rng
        self.state_dim = state_dim
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.update_freq = update_freq
        self.update_counter = 0
        self.use_ddqn = use_ddqn
        self.device = device
        self.network_size = network_size
        if self.network_size == 'small':
            QNetwork = QNetwork_64
        elif self.network_size == 'large':
            QNetwork = QNetwork_128
        elif self.network_size == '2layered':
            QNetwork = QNetwork_6464
        self.network = QNetwork(state_dim=self.state_dim, nb_actions=self.nb_actions)
        self.target_network = QNetwork(state_dim=self.state_dim, nb_actions=self.nb_actions)
        self.weight_transfer(from_model=self.network, to_model=self.target_network)
        self.network.to(self.device)
        self.target_network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, amsgrad=True)
        self.sided_Q = sided_Q

    def train_on_batch(self, s, a, r, s2, t):
        s  = torch.FloatTensor(np.float32(s)).to(self.device)
        s2 = torch.FloatTensor(np.float32(s2)).to(self.device)
        a  = torch.LongTensor(np.int64(a)).to(self.device)
        r  = torch.FloatTensor(np.float32(r)).to(self.device)
        t  = torch.FloatTensor(np.float32(t)).to(self.device)

        q = self.network(s)
        q2 = self.target_network(s2).detach()
        q_pred = q.gather(1, a.unsqueeze(1)).squeeze(1) 
        if self.use_ddqn:
            q2_net = self.network(s2).detach()
            q2_max = q2.gather(1, torch.max(q2_net, 1)[1].unsqueeze(1)).squeeze(1)
        else:
            q2_max = torch.max(q2, 1)[0]
        if self.sided_Q == 'negative':
            bellman_target = torch.clamp(r, max=0.0, min=-1.0) + self.gamma * torch.clamp(q2_max.detach(), max=0.0, min=-1.0) * (1 - t)
        elif self.sided_Q == 'positive':
            bellman_target = torch.clamp(r, max=1.0, min=0.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=0.0) * (1 - t)
        elif self.sided_Q == 'both':
            bellman_target = torch.clamp(r, max=1.0, min=-1.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=-1.0) * (1 - t)
        
        errs = (bellman_target - q_pred).unsqueeze(1)
        quad = torch.min(torch.abs(errs), 1)[0]
        lin = torch.abs(errs) - quad
        loss = torch.sum(0.5 * quad.pow(2) + lin)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def get_loss(self, s, a, r, s2, t):
        s  = torch.FloatTensor(np.float32(s)).to(self.device)
        s2 = torch.FloatTensor(np.float32(s2)).to(self.device)
        a  = torch.LongTensor(np.int64(a)).to(self.device)
        r  = torch.FloatTensor(np.float32(r)).to(self.device)
        t  = torch.FloatTensor(np.float32(t)).to(self.device)

        with torch.no_grad():
            q = self.network(s).detach()
            q2 = self.target_network(s2).detach()
        q_pred = q.gather(1, a.unsqueeze(1)).squeeze(1) 
        if self.use_ddqn:
            q2_net = self.network(s2).detach()
            q2_max = q2.gather(1, torch.max(q2_net, 1)[1].unsqueeze(1)).squeeze(1)
        else:
            q2_max = torch.max(q2, 1)[0]
        if self.sided_Q == 'negative':
            bellman_target = torch.clamp(r, max=0.0, min=-1.0) + self.gamma * torch.clamp(q2_max.detach(), max=0.0, min=-1.0) * (1 - t)
        elif self.sided_Q == 'positive':
            bellman_target = torch.clamp(r, max=1.0, min=0.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=0.0) * (1 - t)
        elif self.sided_Q == 'both':
            bellman_target = torch.clamp(r, max=1.0, min=-1.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=-1.0) * (1 - t)
        
        errs = (bellman_target - q_pred).unsqueeze(1)
        quad = torch.min(torch.abs(errs), 1)[0]
        lin = torch.abs(errs) - quad
        loss = torch.sum(0.5 * quad.pow(2) + lin)
        return loss.detach().cpu().numpy()

    def get_q(self, s):
        s = torch.FloatTensor(s).to(self.device)
        return self.network(s).detach().cpu().numpy()

    def get_max_action(self, s):
        s = torch.FloatTensor(s).to(self.device)
        q = self.network(s).detach()
        return q.max(1)[1].cpu().numpy()

    def get_action(self, states):
        return self.get_max_action(states)

    def learn(self, s, a, r, s2, term):
        """ Learning from one minibatch """
        loss = self.train_on_batch(s, a, r, s2, term)
        if self.update_counter == self.update_freq:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)
            self.update_counter = 0
        else:
            self.update_counter += 1
        return loss

    def dump_network(self, weights_file_path):
        try:
            torch.save(self.network.state_dict(), weights_file_path)
        except:
            pass

    def load_weights(self, weights_file_path, target=False):
        self.network.load_state_dict(torch.load(weights_file_path))
        if target:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)

    def resume(self, network_state_dict, target_network_state_dict, optimizer_state_dict):
        self.network.load_state_dict(network_state_dict)
        self.target_network.load_state_dict(target_network_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)

    @staticmethod
    def weight_transfer(from_model, to_model):
        to_model.load_state_dict(from_model.state_dict())

    def __getstate__(self):
        _dict = {k: v for k, v in self.__dict__.items()}
        del _dict['device']  # is not picklable
        return _dict