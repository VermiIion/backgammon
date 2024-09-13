import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque


# Klasa DQN
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
        # Check dimensions of dice_roll and add a batch dimension if necessary
        if dice_roll.ndim < state.ndim:
            dice_roll = dice_roll.unsqueeze(0)  # Add batch dimension if dice_roll is 1D

        # Concatenate state and dice_roll along the second dimension (features)
        x = T.cat((state, dice_roll), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = self.fc3(x)
        return action

# Klasa agenta
class Agent():
    def __init__(self, gamma, epsilon, lr, input_size, batch_size, n_actions, max_mem=100000, eps_end=0.01,
                 eps_decay=5e-4):
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

        self.DQN_eval = DQN(self.lr, input_size=input_size, n_actions=n_actions, fc1_dims=256, fc2_dims=256)

        self.state_memory = deque(maxlen=self.max_mem)
        self.new_state_memory = deque(maxlen=self.max_mem)
        self.action_memory = deque(maxlen=self.max_mem)
        self.reward_memory = deque(maxlen=self.max_mem)
        self.terminal_memory = deque(maxlen=self.max_mem)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_pointer % self.max_mem
        self.state_memory.append(state)
        self.new_state_memory.append(next_state)
        self.reward_memory.append(reward)
        self.action_memory.append(action)
        self.terminal_memory.append(done)
        self.mem_pointer += 1

    def observation(self, observation, dice_roll):
        if np.random.rand() > self.epsilon:
            # Ensure observation and dice_roll are numpy arrays
            state = T.tensor(np.array([observation]), dtype=T.float32).to(self.DQN_eval.device)
            dice_roll = T.tensor(np.array([dice_roll]), dtype=T.float32).to(self.DQN_eval.device)

            # If dice_roll has fewer dimensions, add a batch dimension
            if dice_roll.ndim == 2:  # If dice_roll has batch dimension, ensure state has it too
                dice_roll = dice_roll.unsqueeze(0)  # Add batch dimension if necessary

            # Now the dimensions should be the same, and we can pass them to the DQN forward method
            actions = self.DQN_eval.forward(state, dice_roll)
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

        state_batch = T.tensor(np.array(self.state_memory)[batch], dtype=T.float32).to(self.DQN_eval.device)
        new_state_batch = T.tensor(np.array(self.new_state_memory)[batch], dtype=T.float32).to(self.DQN_eval.device)

        dice_roll_batch = T.tensor([env.roll_dice() for _ in range(self.batch_size)], dtype=T.float32).to(
            self.DQN_eval.device)
        reward_batch = T.tensor([self.reward_memory[i] for i in batch], dtype=T.float32).to(self.DQN_eval.device)
        terminal_batch = T.tensor([self.terminal_memory[i] for i in batch], dtype=T.bool).to(self.DQN_eval.device)

        action_batch = [self.action_memory[i] for i in batch]

        # Przewidywane wartości Q dla bieżących stanów
        q_eval = self.DQN_eval.forward(state_batch, dice_roll_batch)[batch_index, action_batch]

        # Nowy batch rzutów kostką dla następnych stanów
        dice_roll_batch_next = T.tensor([env.roll_dice() for _ in range(self.batch_size)], dtype=T.float32).to(
            self.DQN_eval.device)

        # Przewidywane wartości Q dla następnych stanów
        q_next = self.DQN_eval.forward(new_state_batch, dice_roll_batch_next)
        q_next[terminal_batch] = 0.0  # Jeśli epizod się zakończył, wartość Q jest zerowa

        # Obliczenie celu Q
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        # Obliczanie straty
        loss = self.DQN_eval.loss(q_target, q_eval).to(self.DQN_eval.device)
        loss.backward()
        self.DQN_eval.optimizer.step()

        # Zmniejszanie epsilonu (dla eksploracji)
        self.epsilon = self.epsilon - self.eps_decay if self.epsilon > self.eps_end else self.eps_end


# Środowisko AI dla Backgammona
class AIEnv():
    def __init__(self):
        self.board = self.build_board()

    def build_board(self):
        board = [0] * 28
        board[0] = 2  # 2 białe pionki na 1 trójkącie
        board[5] = -5  # 5 czarnych pionków na 6 trójkącie
        board[7] = -3  # 3 czarne pionki na 8 trójkącie
        board[11] = 5  # 5 białych pionków na 12 trójkącie
        board[12] = -5  # 5 czarnych pionków na 13 trójkącie
        board[16] = 3  # 3 białe pionki na 17 trójkącie
        board[18] = 5  # 5 białych pionków na 19 trójkącie
        board[23] = -2  # 2 czarne pionki na 24 trójkącie
        board[24] = 1  # 1 zbity biały pionek
        board[27] = 2  # 2 zakończone czarne pionki
        return board

    def roll_dice(self):
        die_1 = random.randint(1, 6)
        die_2 = random.randint(1, 6)

        # Zwracanie jako tensor
        dice_roll = np.array([die_1, die_2], dtype=np.float32)  # Tworzenie numpy array z rzutami kostką
        return dice_roll

    def move(self, from_pos, die_1, die_2, player):
        board = self.board
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

    def read(self):
        return np.array(self.board, dtype=np.float32)

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


# Zastosowanie:
env = AIEnv()
agent = Agent(gamma=0.99, epsilon=1.0, lr=0.001, input_size=env.board.__len__(), batch_size=64, n_actions=4)


# Czytanie stanu planszy
state = env.read()

# Rzut kostkami
dice_1, dice_2 = env.roll_dice()

# Przewidywanie najlepszego ruchu
action = agent.observation(state, (dice_1, dice_2))

# Załóżmy, że akcja zwraca (from_pos, player)
from_pos, player = action, 'white'  # Dla przykładu, można rozszerzyć logikę

# Przesuwanie pionka zgodnie z wynikami rzutów kostkami
env.move(from_pos, dice_1, dice_2, player)

# Aktualizacja stanu planszy
new_state = env.read()

# Parametry treningu
n_games = 1000  # Liczba gier do rozegrania
scores = []
epsilon_dec = 1e-3  # Szybkość zmniejszania epsilon
gamma = 0.99  # Współczynnik dyskonta

# Pętla treningowa
for i in range(n_games):
    done = False
    score = 0

    # Resetowanie środowiska na początek każdej gry
    state = env.read()

    while not done:
        # Rzut kostkami
        dice_1, dice_2 = env.roll_dice()

        # Agent wybiera akcję na podstawie aktualnego stanu i rzutu kostkami
        action = agent.observation(state, (dice_1, dice_2))

        # Podział akcji na pozycję początkową i gracza
        from_pos = action  # Można rozwinąć logikę do wyboru gracza, na razie np. 'white'
        player = 'white'  # Można dodać logikę wyboru gracza

        # Wykonanie ruchu w środowisku
        new_board = env.move(from_pos, dice_1, dice_2, player)

        # Odczyt nowego stanu
        new_state = env.read()

        # Załóżmy, że nagroda to np. +1 za poprawny ruch
        reward = 1

        # Sprawdzenie, czy gra się skończyła (można rozwinąć logikę)
        done = False  # Tu należy dodać logikę warunku końca gry

        # Przechowywanie doświadczeń
        agent.store_transition(state, action, reward, new_state, done)

        # Agent uczy się z doświadczeń
        agent.learn()

        # Aktualizacja stanu
        state = new_state
        score += reward

    # Epsilon decyduje o eksploracji - zmniejsza się z każdą grą
    agent.epsilon = max(agent.eps_end, agent.epsilon - epsilon_dec)

    # Przechowywanie wyniku gry
    scores.append(score)

    # Wyświetlanie wyników co 100 gier
    if (i + 1) % 100 == 0:
        avg_score = np.mean(scores[-100:])
        print(f'Gra {i + 1}/{n_games}, Średni wynik z ostatnich 100 gier: {avg_score}, Epsilon: {agent.epsilon}')
