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
        # Zwiększ rozmiar wejściowy, aby uwzględnić zarówno stan gry, jak i rzut kostkami
        self.fc1 = nn.Linear(input_size + 2, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, dice_roll):
        # Łączenie state i dice_roll
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
        self.dice_memory = deque(maxlen=self.max_mem)  # Nowa pamięć dla rzutów kostką

    def store_transition(self, state, action, reward, next_state, dice_roll, done):
        index = self.mem_pointer % self.max_mem
        self.state_memory.append(state)
        self.new_state_memory.append(next_state)
        self.reward_memory.append(reward)
        self.action_memory.append(action)
        self.dice_memory.append(dice_roll)  # Dodajemy dice_roll do pamięci
        self.terminal_memory.append(done)
        self.mem_pointer += 1


    def observation(self, observation, dice_roll):
        """
        Function where the agent makes a decision based on the current board state and dice roll.
        """
        if np.random.rand() > self.epsilon:  # Exploitation: the agent chooses the best action
            state = T.tensor([observation], dtype=T.float32).to(self.DQN_eval.device)
            dice_roll = T.tensor([dice_roll], dtype=T.float32).to(self.DQN_eval.device)

            # Predicting actions based on the current state and dice roll
            actions = self.DQN_eval.forward(state, dice_roll)

            # Choosing the best action
            from_pos = T.argmax(actions).item()  # Choosing a position on the board

            # `action` should be a single index
            action = np.array([from_pos])
        else:
            # Exploration: the agent randomly chooses one of the possible actions
            from_pos = np.random.choice(self.n_actions)  # Randomly choosing a position

            # `action` should be a single index
            action = np.array([from_pos])

        # Returning the selected position along with the dice roll
        return action[0], dice_roll[0], dice_roll[1]

    def learn(self):
        if self.mem_pointer < self.batch_size:
            return

        self.DQN_eval.optimizer.zero_grad()

        max_mem = min(self.max_mem, self.mem_pointer)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        # Pobieramy partie danych z pamięci
        state_batch = T.tensor(np.array(self.state_memory)[batch], dtype=T.float32).to(self.DQN_eval.device)
        new_state_batch = T.tensor(np.array(self.new_state_memory)[batch], dtype=T.float32).to(self.DQN_eval.device)
        reward_batch = T.tensor(np.array(self.reward_memory)[batch], dtype=T.float32).to(self.DQN_eval.device)
        terminal_batch = T.tensor(np.array(self.terminal_memory)[batch], dtype=T.float32).to(self.DQN_eval.device)
        action_batch = T.tensor(np.array(self.action_memory)[batch], dtype=T.long).to(self.DQN_eval.device)

        # Musimy dodać wymiar do action_batch dla gather
        action_batch = action_batch.unsqueeze(1)  # Dodanie wymiaru dla gather

        # Przewidujemy wartości Q dla stanów i akcji
        dice_batch = T.zeros((state_batch.size(0), 2), dtype=T.float32).to(
            self.DQN_eval.device)  # Przykład rzutów kością
        q_eval = self.DQN_eval.forward(state_batch, dice_batch)

        # Sprawdzenie wymiarów q_eval i action_batch przed użyciem gather
        print(f"q_eval shape: {q_eval.shape}, action_batch shape: {action_batch.shape}")

        # Zbieramy wartości Q dla wybranych akcji
        q_eval = q_eval.gather(1, action_batch).squeeze(1)

        # Przewidujemy wartości Q dla nowych stanów
        q_next = self.DQN_eval.forward(new_state_batch, dice_batch)

        # Ustawiamy wartości Q na 0 dla stanów końcowych
        q_next[terminal_batch.long()] = 0.0

        # Obliczamy docelowe wartości Q
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        # Obliczamy stratę i aktualizujemy wagi
        loss = self.DQN_eval.loss(q_target, q_eval)
        loss.backward()
        self.DQN_eval.optimizer.step()

        # Aktualizacja epsilon
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
        """
        Return the current state of the board as a numpy array.
        """
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
        print(f"Player {player} moves from {from_pos} with dice {die_1}, {die_2}")

        # Klonujemy planszę dla debugowania
        board = self.board.copy()

        if player == 'white':
            if board[from_pos] > 0:
                # Pierwszy ruch zgodnie z rzutem pierwszej kostki
                to_pos_1 = from_pos - die_1
                if 0 <= to_pos_1 < 24:
                    if board[to_pos_1] >= -1:  # Można wykonać ruch, jeśli pole nie jest zablokowane
                        board[from_pos] -= 1
                        board[to_pos_1] += 1

                # Drugi ruch zgodnie z rzutem drugiej kostki
                to_pos_2 = from_pos - die_2
                if 0 <= to_pos_2 < 24:
                    if board[to_pos_2] >= -1:
                        board[from_pos] -= 1
                        board[to_pos_2] += 1

                # Dublet - dodatkowe dwa ruchy
                if die_1 == die_2:
                    if from_pos - die_1 * 2 >= 0:
                        to_pos_3 = from_pos - die_1 * 2
                        if board[to_pos_3] >= -1:
                            board[from_pos] -= 1
                            board[to_pos_3] += 1
                    if from_pos - die_1 * 3 >= 0:
                        to_pos_4 = from_pos - die_1 * 3
                        if board[to_pos_4] >= -1:
                            board[from_pos] -= 1
                            board[to_pos_4] += 1

        elif player == 'black':
            if board[from_pos] < 0:
                # Pierwszy ruch zgodnie z rzutem pierwszej kostki
                to_pos_1 = from_pos + die_1
                if 0 <= to_pos_1 < 24:
                    if board[to_pos_1] <= 1:  # Można wykonać ruch, jeśli pole nie jest zablokowane
                        board[from_pos] += 1
                        board[to_pos_1] -= 1

                # Drugi ruch zgodnie z rzutem drugiej kostki
                to_pos_2 = from_pos + die_2
                if 0 <= to_pos_2 < 24:
                    if board[to_pos_2] <= 1:
                        board[from_pos] += 1
                        board[to_pos_2] -= 1

                # Dublet - dodatkowe dwa ruchy
                if die_1 == die_2:
                    if from_pos + die_1 * 2 < 24:
                        to_pos_3 = from_pos + die_1 * 2
                        if board[to_pos_3] <= 1:
                            board[from_pos] += 1
                            board[to_pos_3] -= 1
                    if from_pos + die_1 * 3 < 24:
                        to_pos_4 = from_pos + die_1 * 3
                        if board[to_pos_4] <= 1:
                            board[from_pos] += 1
                            board[to_pos_4] -= 1

        print("Board state after move:")
        self.board = board  # Aktualizuj planszę
        self.print_board()
        return board

    def check_done(self):
        white_pips = sum(1 for i in range(0, 24) if self.board[i] > 0)
        black_pips = sum(1 for i in range(0, 24) if self.board[i] < 0)

        if white_pips == 0:
            print("Black wins!")
            return True
        if black_pips == 0:
            print("White wins!")
            return True
        return False

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

