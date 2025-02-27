import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt


# Klasa DQN
class DQN(nn.Module):
    def __init__(self, lr, input_size, fc1_dims, fc2_dims, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size + 2, fc1_dims)  # +2, aby dodać rzuty kostką
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, dice_roll):
        x = T.cat((state, dice_roll), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = self.fc3(x)
        return action


class Agent:
    def __init__(self, gamma, epsilon, lr, input_size, batch_size, n_actions, max_mem=100000, eps_end=0.01, eps_decay=1e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_size = input_size
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.max_mem = max_mem
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.mem_pointer = 0

        self.DQN_eval = DQN(lr=self.lr, input_size=input_size, fc1_dims=128, fc2_dims=64, n_actions=n_actions)
        self.DQN_target = DQN(lr=self.lr, input_size=input_size, fc1_dims=128, fc2_dims=64, n_actions=n_actions)

        self.state_memory = deque(maxlen=self.max_mem)
        self.new_state_memory = deque(maxlen=self.max_mem)
        self.action_memory = deque(maxlen=self.max_mem)
        self.reward_memory = deque(maxlen=self.max_mem)
        self.terminal_memory = deque(maxlen=self.max_mem)
        self.dice_memory = deque(maxlen=self.max_mem)

    def store_transition(self, state, action, reward, next_state, dice_roll, done):
        self.state_memory.append(state)
        self.new_state_memory.append(next_state)
        self.reward_memory.append(reward)
        self.action_memory.append(action)
        self.dice_memory.append(dice_roll)
        self.terminal_memory.append(done)
        self.mem_pointer += 1

    def observation(self, observation, dice_roll):
        if np.random.rand() > self.epsilon:
            state = T.tensor([observation], dtype=T.float32).to(self.DQN_eval.device)
            dice_roll = T.tensor([dice_roll], dtype=T.float32).to(self.DQN_eval.device)
            actions = self.DQN_eval.forward(state, dice_roll)
            from_pos = T.argmax(actions).item()
            action = np.array([from_pos])
        else:
            from_pos = np.random.choice(self.n_actions)
            action = np.array([from_pos])
        return action[0], dice_roll[0], dice_roll[1]

    def learn(self):
        if self.mem_pointer < self.batch_size:
            return

        self.DQN_eval.optimizer.zero_grad()

        max_mem = min(self.max_mem, self.mem_pointer)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch = T.tensor(np.array(self.state_memory)[batch], dtype=T.float32).to(self.DQN_eval.device)
        new_state_batch = T.tensor(np.array(self.new_state_memory)[batch], dtype=T.float32).to(self.DQN_eval.device)
        reward_batch = T.tensor(np.array(self.reward_memory)[batch], dtype=T.float32).to(self.DQN_eval.device)
        terminal_batch = T.tensor(np.array(self.terminal_memory)[batch], dtype=T.float32).to(self.DQN_eval.device)
        action_batch = T.tensor(np.array(self.action_memory)[batch], dtype=T.long).to(self.DQN_eval.device)
        dice_batch = T.tensor(np.array(self.dice_memory)[batch], dtype=T.float32).to(self.DQN_eval.device)

        # Przewidywane wartości Q dla bieżących stanów
        q_eval = self.DQN_eval.forward(state_batch, dice_batch)

        # Wybieramy przewidywaną wartość Q dla wybranej akcji (redukcja tensoru)
        q_eval = q_eval.gather(1, action_batch.unsqueeze(1)).squeeze(1)  # Ensure action_batch is (batch_size, 1)

        # Przewidywane wartości Q dla przyszłych stanów (Q(s', a'))
        q_next = self.DQN_target.forward(new_state_batch, dice_batch)
        q_next[terminal_batch.long()] = 0.0  # Zerujemy Q dla terminalnych stanów

        # Maksymalne wartości Q z przyszłych stanów
        q_target_max = T.max(q_next, dim=1)[0]

        # Ustalanie wartości docelowej Q (q_target)
        q_target = reward_batch + self.gamma * q_target_max  # Teraz q_target ma wymiar [64]

        # Sprawdzamy rozmiary tensorów (opcjonalnie, do debugowania)
        print(f"q_eval size: {q_eval.size()}")  # Oczekiwane: torch.Size([64])
        print(f"q_target size: {q_target.size()}")  # Oczekiwane: torch.Size([64])

        # Obliczanie straty
        loss = self.DQN_eval.loss(q_eval, q_target)
        loss.backward()
        self.DQN_eval.optimizer.step()

        # Aktualizacja epsilon (dla eksploracji)
        self.epsilon = self.epsilon - self.eps_decay if self.epsilon > self.eps_end else self.eps_end

        return loss.item()


# Środowisko AI dla Backgammona
class AIEnv:
    def __init__(self):
        self.board = self.build_board()

    def build_board(self):
        board = [0] * 28
        board[0] = 2
        board[5] = -5
        board[7] = -3
        board[11] = 5
        board[12] = -5
        board[16] = 3
        board[18] = 5
        board[23] = -2
        board[24] = 1
        board[27] = 2
        return board

    def get_state(self):
        return np.array(self.board, dtype=np.float32)

    def reset_game(self):
        self.board = self.build_board()

    def print_board(self):
        print("Board state:")
        print(self.board)

    def roll_dice(self):
        die_1 = random.randint(1, 6)
        die_2 = random.randint(1, 6)
        return [die_1, die_2]

    def move(self, from_pos, die_1, die_2, player):
        board = self.board.copy()
        if player == 'white' and board[from_pos] > 0:
            to_pos_1 = from_pos - die_1
            if 0 <= to_pos_1 < 24 and board[to_pos_1] >= -1:
                board[from_pos] -= 1
                board[to_pos_1] += 1

            to_pos_2 = from_pos - die_2
            if 0 <= to_pos_2 < 24 and board[to_pos_2] >= -1:
                board[from_pos] -= 1
                board[to_pos_2] += 1

            if die_1 == die_2:
                to_pos_3 = from_pos - die_1 * 2
                if 0 <= to_pos_3 < 24 and board[to_pos_3] >= -1:
                    board[from_pos] -= 1
                    board[to_pos_3] += 1

                to_pos_4 = from_pos - die_1 * 3
                if 0 <= to_pos_4 < 24 and board[to_pos_4] >= -1:
                    board[from_pos] -= 1
                    board[to_pos_4] += 1

        print("Board state after move:")
        self.board = board
        self.print_board()
        return board

    def check_done(self):
        white_pips = sum(1 for pip in self.board if pip > 0)
        black_pips = sum(1 for pip in self.board if pip < 0)

        return white_pips == 0 or black_pips == 0

    def read(self):
        return np.array(self.board, dtype=np.float32)

    def step(self, action):
        from_pos, die_1, die_2 = action
        done = False

        # Execute the move and update the board
        next_board = self.move(from_pos, die_1, die_2, 'white')

        # Check if the game is finished
        done = self.check_done()

        # Calculate the reward
        white_pips = self.count_pips('white')
        black_pips = self.count_pips('black')

        # Reward: fewer pips for the player is positive
        reward = white_pips - black_pips

        next_state = self.get_state()
        return next_state, reward, done

    def count_pips(self, player):
        """
        Liczy liczbę pipów gracza na planszy.
        """
        if player == 'white':
            return sum(1 for x in self.board if x > 0)
        elif player == 'black':
            return sum(1 for x in self.board if x < 0)
        return 0


# Trening agenta
def train_agent(env, agent, num_episodes):
    scores = []
    for episode in range(1, num_episodes + 1):
        state = env.get_state()  # Pobranie początkowego stanu planszy
        done = False
        total_reward = 0

        while not done:
            # Roll dice
            dice_roll = env.roll_dice()

            # Agent chooses an action (position and dice values)
            action = agent.observation(state, dice_roll)

            # Execute move in the environment
            next_state, reward, done = env.step(action)

            # Store the transition in the agent's memory
            agent.store_transition(state, action, reward, next_state, dice_roll, done)

            # Update the agent (learn from the experience)
            agent.learn()

            state = next_state
            total_reward += reward

        scores.append(total_reward)
        print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}")

    return scores






# Parametry
gamma = 0.99
epsilon = 1.0
lr = 0.001
input_size = 28  # Rozmiar stanu planszy + 2 kostki
batch_size = 64
n_actions = 24  # Liczba możliwych ruchów

env = AIEnv()
agent = Agent(gamma=gamma, epsilon=epsilon, lr=lr, input_size=input_size, batch_size=batch_size, n_actions=n_actions)

# Start training
scores = train_agent(env, agent, num_episodes=500)

