import random
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# Definicja środowiska gry
class BackgammonEnv(gym.Env):
    def __init__(self, board_size=20, pawns_per_player=10):
        super(BackgammonEnv, self).__init__()
        self.board_size = board_size
        self.pawns_per_player = pawns_per_player
        self.board = [[0] * board_size for _ in range(2)]
        self.pawns = {'player1': [0] * pawns_per_player, 'player2': [board_size - 1] * pawns_per_player}
        self.turn = 'player1'

        # Przestrzeń stanów i akcji
        self.action_space = spaces.Discrete(pawns_per_player)
        self.observation_space = spaces.Box(low=0, high=board_size - 1, shape=(2, pawns_per_player), dtype=np.int32)

        self.reset()

    def reset(self):
        self.board = [[0] * self.board_size for _ in range(2)]
        self.pawns = {'player1': [0] * self.pawns_per_player, 'player2': [self.board_size - 1] * self.pawns_per_player}
        for i in range(self.pawns_per_player):
            self.board[0][0] += 1
            self.board[1][self.board_size - 1] += 1
        self.turn = 'player1'
        return self._get_observation()

    def _get_observation(self):
        return np.array([self.pawns['player1'], self.pawns['player2']])

    def step(self, action):
        player = self.turn
        roll = random.randint(1, 6)

        if not self._move_pawn(player, action, roll):
            reward = -1  # Kara za niepoprawny ruch
            done = False
        else:
            reward = 1 if self._check_win(player) else 0
            done = self._check_win(player)

        self.turn = 'player2' if self.turn == 'player1' else 'player1'
        return self._get_observation(), reward, done, {}

    def _move_pawn(self, player, pawn_index, roll):
        line = 0 if player == 'player1' else 1
        pos = self.pawns[player][pawn_index]
        new_pos = pos + roll if player == 'player1' else pos - roll
        if new_pos < 0 or new_pos >= self.board_size:
            return False
        self.board[line][pos] -= 1
        self.board[line][new_pos] += 1
        self.pawns[player][pawn_index] = new_pos
        return True

    def _check_win(self, player):
        final_pos = self.board_size - 1 if player == 'player1' else 0
        for pos in self.pawns[player]:
            if pos != final_pos:
                return False
        return True


# Definicja sieci neuronowej DQN
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Definicja agenta RL
class Agent:
    def __init__(self, state_dim, action_dim, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, lr=0.0005,
                 target_update=10):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_update = target_update
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=5000)
        self.steps = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.flatten()).unsqueeze(0)
        if random.random() < self.epsilon:
            return random.randint(0, self.model.fc3.out_features - 1)
        else:
            with torch.no_grad():
                q_values = self.model(state)
            return torch.argmax(q_values).item()

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state.flatten()).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()

            state = torch.FloatTensor(state.flatten()).unsqueeze(0)
            q_values = self.model(state)
            q_values[0][action] = target
            states.append(state)
            targets.append(q_values)

        # Batch update
        states = torch.cat(states)
        targets = torch.cat(targets)

        loss = nn.MSELoss()(self.model(states), targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Clip gradient
        self.optimizer.step()

        # Aktualizacja epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Aktualizacja sieci docelowej
        if self.steps % self.target_update == 0:
            self.update_target_model()
        self.steps += 1

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


# Trening agenta
def train_agent(episodes=1000):
    env = BackgammonEnv()
    agent = Agent(state_dim=2 * env.pawns_per_player, action_dim=env.pawns_per_player)

    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            agent.train_step()

        print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}")


# Start treningu
if __name__ == "__main__":
    train_agent(episodes=1000)
