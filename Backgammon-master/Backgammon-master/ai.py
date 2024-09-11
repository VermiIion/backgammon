import os
from collections import deque

import gym
from gym import Env
from gym.spaces import Discrete, MultiDiscrete, Box, Dict, MultiBinary

import numpy as np
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from stable_baselines3.dqn.policies import QNetwork


class DQN(nn.Module):
    def __init__(self, lr, input_size, fc1_dims, fc2_dims, n_actions):
        super(DQN, self).__init__()
        self.input_size = input_size + 2  # Dodajemy miejsce na 2 kostki do wejścia
        self.fc1 = nn.Linear(self.input_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, dice_roll):
        x = T.cat((state, dice_roll), dim=1)  # Dodajemy kostki do stanu
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = self.fc3(x)
        return action


class Agent():
    def __init__(self, gamma, epsilon, lr, input_size, batch_size, n_actions, max_mem=100000, eps_end=0.01,
                 eps_decay=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_size = input_size
        self.n_actions = n_actions
        self.max_mem = max_mem
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.action_space = Discrete(self.n_actions)
        self.batch_size = batch_size
        self.mem_pointer = 0

        self.DQN_eval = DQN(self.lr, input_size=input_size, n_actions=n_actions, fc1_dims=256, fc2_dims=256)

        self.state_memory = deque(maxlen=self.max_mem, *input_size)
        self.new_state_memory = deque(maxlen=self.max_mem, *input_size)
        self.action_memory = deque(maxlen=self.max_mem)
        self.reward_memory = deque(maxlen=self.max_mem)
        self.terminal_memory = deque(maxlen=self.max_mem)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_pointer % self.max_mem
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_pointer += 1

    def observation(self, observation):
        if np.random.rand() > self.epsilon:
            state = T.tensor([observation]).to(self.DQN_eval.device)
            actions = self.DQN_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.n_actions)

        return action

    def learn(self):
        if self.mem_pointer < self.batch_size:
            return

        self.DQN_eval.optimizer.zero_grad()

        max_mem = min(self.max_mem, self.mem_pointer)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory).to(self.DQN_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.DQN_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.DQN_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.DQN_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.DQN_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.DQN_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.DQN_eval.loss(q_target, q_eval).to(self.DQN_eval.device)
        loss.backward()
        self.DQN_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_decay if self.epsilon > self.eps_end \
            else self.eps_end


def buildBoard():
    pass


class AIEnv(Env):
    def __init__(self):
        pass

    def calc_pip(self, board):
        white_pip = 0
        black_pip = 0

        for i in range(24):
            if board[i] > 0:  # Białe pionki
                white_pip += board[i] * (24 - i)  # Ilość pionków * odległość do domu
            elif board[i] < 0:  # Czarne pionki
                black_pip += abs(board[i]) * (i + 1)  # Ilość pionków * odległość do domu

        return white_pip, black_pip

    def predict(state, agent, dice_roll):
        state_tensor = T.tensor([state], dtype=T.float32).to(agent.DQN_eval.device)
        dice_roll_tensor = T.tensor([dice_roll], dtype=T.float32).to(agent.DQN_eval.device)
        actions = agent.DQN_eval.forward(state_tensor, dice_roll_tensor)
        action = T.argmax(actions).item()
        return action

    def move(board, from_pos, die_1, die_2, player):
        if player == 'white':
            if board[from_pos] > 0:
                # Pierwszy ruch zgodnie z rzutem pierwszej kostki
                to_pos_1 = from_pos - die_1
                if to_pos_1 >= 0:
                    board[from_pos] -= 1
                    board[to_pos_1] += 1
                # Drugi ruch zgodnie z rzutem drugiej kostki
                to_pos_2 = from_pos - die_2
                if to_pos_2 >= 0:
                    board[from_pos] -= 1
                    board[to_pos_2] += 1

                # Dublet - dodatkowe dwa ruchy
                if die_1 == die_2:
                    if from_pos - die_1 * 2 >= 0:
                        board[from_pos] -= 1
                        board[from_pos - die_1 * 2] += 1
                    if from_pos - die_1 * 3 >= 0:
                        board[from_pos] -= 1
                        board[from_pos - die_1 * 3] += 1

        elif player == 'black':
            if board[from_pos] < 0:
                # Pierwszy ruch zgodnie z rzutem pierwszej kostki
                to_pos_1 = from_pos + die_1
                if to_pos_1 <= 23:
                    board[from_pos] += 1
                    board[to_pos_1] -= 1
                # Drugi ruch zgodnie z rzutem drugiej kostki
                to_pos_2 = from_pos + die_2
                if to_pos_2 <= 23:
                    board[from_pos] += 1
                    board[to_pos_2] -= 1

                # Dublet - dodatkowe dwa ruchy
                if die_1 == die_2:
                    if from_pos + die_1 * 2 <= 23:
                        board[from_pos] += 1
                        board[from_pos + die_1 * 2] -= 1
                    if from_pos + die_1 * 3 <= 23:
                        board[from_pos] += 1
                        board[from_pos + die_1 * 3] -= 1

        return board

    def read(board):
        # Zakłada, że plansza jest listą o 28 elementach
        return np.array(board, dtype=np.float32)

    def roll_dice(self):
        die_1 = random.randint(1, 6)
        die_2 = random.randint(1, 6)
        return die_1, die_2



def buildBoard():
    board = [0] * 28

    # Przykład ustawienia pionków początkowych
    board[0] = 2  # 2 białe pionki na 1 trójkącie
    board[5] = -5  # 5 czarnych pionków na 6 trójkącie
    board[7] = -3  # 3 czarne pionki na 8 trójkącie
    board[11] = 5  # 5 białych pionków na 12 trójkącie
    board[12] = -5  # 5 czarnych pionków na 13 trójkącie
    board[16] = 3  # 3 białe pionki na 17 trójkącie
    board[18] = 5  # 5 białych pionków na 19 trójkącie
    board[23] = -2  # 2 czarne pionki na 24 trójkącie

    # Reprezentacja zbitych pionków (np. jeden biały pionek jest zbity)
    board[24] = 1

    # Reprezentacja pionków zakończonych grę (np. dwa czarne pionki zakończyły grę)
    board[27] = 2

agent = Agent(gamma=0.99, epsilon=1.0, lr=0.001, input_size=(28,), batch_size=64, n_actions=4)
board = buildBoard()

# Czytanie stanu planszy
state = read(board)

# Rzut kostkami
die_1, die_2 = roll_dice()

# Przewidywanie najlepszego ruchu
action = predict(state, agent, (die_1, die_2))

# Załóżmy, że akcja zwraca (from_pos, player)
from_pos, player = action

# Przesuwanie pionka zgodnie z wynikami rzutów kostkami
board = move(board, from_pos, die_1, die_2, player)

# Obliczanie pipów
white_pip, black_pip = calc_pip(board)

print(f"Białe pipy: {white_pip}, Czarne pipy: {black_pip}")
