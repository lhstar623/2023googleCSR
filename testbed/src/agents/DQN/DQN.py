import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import json

class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.wins = []
        self.outcomes = []

    def append(self, state, action, reward, next_state, done, win, outcome):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.wins.append(win)
        self.outcomes.append(outcome)
    
    def numpy(self):
        return np.array(self.states), np.array(self.actions), np.array(self.rewards), np.array(self.next_states)
    
    def sample(self, batch_size):
        if batch_size > len(self.states):
            self.numpy()
        else:
            inds = np.random.choice(len(self.states), batch_size, replace = False)
            return np.array([self.states[idx] for idx in inds]), np.array([self.actions[idx] for idx in inds]), \
                np.array([self.rewards[idx] for idx in inds]), np.array([self.next_states[idx] for idx in inds]), np.array([self.dones[idx] for idx in inds], dtype=bool)

class Agent:
    def __init__(self, rng, name):
        self.rng = rng
        self.name = name
        self.context_dim = None
        self.step = 0

    def newdata(self, s, a, r, s_, done, win, outcome):
        '''
        This function is called at every step.

        Inputs
        s[numpy array] : state   a[float] : action  r[float] : reward
        s_[numpy array] : next state    done[bool] : the episode is done
        win[float; zero or one] : the agent won the auction at the step
        outcome[int; zero or one] : the conversion event occured at the step(this can be 1 even if the agent lost the auction)

        This function returns nothing
        '''
        raise NotImplementedError

    def bid(self, state):
        '''
        This function is called at every step, even if the agent has no budget.

        Input
        s[numpy array] : state

        Output : bidding[float] (bidding will be clipped in main.py not to exceed current budget)
        '''
        raise NotImplementedError
    
    def update(self):
        '''
        This function is called at the end of episodes.
        '''
        raise NotImplementedError

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim-3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim-3)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.fc4 = nn.Linear(input_dim, hidden_dim-3)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim-3)
        self.fc6 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        r = x[:,self.context_dim:]
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(torch.concat([q1, r], dim=1)))
        q1 = self.fc3(torch.concat([q1, r], dim=1))

        q2 = F.relu(self.fc4(x))
        q2 = F.relu(self.fc5(torch.concat([q2, r], dim=1)))
        q2 = self.fc6(torch.concat([q2, r], dim=1))

        return q1, q2
    
    def Q1(self, x):
        r = x[:,self.context_dim:]
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(torch.concat([q1, r], dim=1)))
        return self.fc3(torch.concat([q1, r], dim=1))
    

class DQN(Agent):
    def __init__(self, rng, name, context_dim):
        super().__init__(rng, name)
        self.context_dim = context_dim
        self.buffer = Buffer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open('src/agents/DQN/config.json') as f:
            # main file 위치 기준으로 상대경로 설정
            config = json.load(f)

        self.local_network = Critic(context_dim + 2 + 1, config['hidden_dim'], self.context_dim).to(self.device)
        self.target_network = copy.deepcopy(self.local_network)
        self.eps_init = self.eps = config['eps_init']
        self.eps_min = config['eps_min']
        self.eps_decay = config['eps_decay']

        self.optimizer = optim.Adam(self.local_network.parameters(), lr = config['lr'])
        self.batch_size = config['batch_size']
        self.tau = config['tau']
        self.num_grad_steps = config['num_grad_steps']
        self.episode = 0

    def newdata(self, s, a, r, s_, done, win, outcome):
        self.buffer.append(s,a,r,s_,done,win,outcome)

    def bid(self, state):
        self.step += 1
        n_values_search = 10
        b_grid = np.linspace(0, 1, n_values_search)
        x = torch.Tensor(np.concatenate([np.tile(state, (n_values_search, 1)),b_grid.reshape(-1,1)], axis=1)).to(self.device)
        with torch.no_grad():
            if self.rng.uniform(0, 1) < self.eps:
                bidding = self.rng.random()
            else:
                index = np.argmax(self.local_network.Q1(x).numpy(force=True))
                bidding = b_grid[index]
        return bidding

    def update(self):
        self.episode += 1
        self.eps = max(self.eps*self.eps_decay, self.eps_min)

        if len(self.buffer.states)<self.batch_size:
            return
        
        criterion = nn.MSELoss()
        self.local_network.train()
        self.target_network.eval()

        for i in range(self.num_grad_steps):
            states, biddings, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            q1, q2 = self.local_network(torch.Tensor(np.hstack([states, biddings.reshape(-1, 1)])).to(self.device))
            with torch.no_grad():
                n_values_search = 10
                b_grid = np.linspace(0, 1, n_values_search)
                x = torch.Tensor(np.concatenate([np.tile(next_states, (1, n_values_search)).reshape(-1,self.context_dim+2),
                                np.tile(b_grid.reshape(-1,1), (self.batch_size,1))], axis=1)).to(self.device)
                next_q1, next_q2  = self.target_network(x)
                target_q1 = torch.max(next_q1.reshape(self.batch_size,-1), dim=1, keepdim=False).values
                target_q2 = torch.max(next_q2.reshape(self.batch_size,-1), dim=1, keepdim=False).values
                target_q = torch.Tensor(rewards).to(self.device) + torch.Tensor(1.0-dones).to(self.device) * torch.min(target_q1, target_q2)
            loss = criterion(q1.squeeze(), target_q) + criterion(q2.squeeze(), target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.episode%2 == 0:
            for target_param, local_param in zip(self.target_network.parameters(), self.local_network.parameters()):
                target_param.data.copy_(self.tau * local_param + (1-self.tau) * target_param)