import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque


# Definicja klasy DQN (Deep Q-Network) w celu modelowania polityki agenta.
class DQN(nn.Module):
    def __init__(self, lr, input_size, fc1_dims, fc2_dims, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, fc1_dims)  # Pierwsza warstwa ukryta, przyjmuje state i rzut kostką.
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)  # Druga warstwa ukryta.
        self.fc3 = nn.Linear(fc2_dims, n_actions)  # Warstwa wyjściowa.
        self.optimizer = optim.Adam(self.parameters(), lr=lr)  # Optymalizator (Adam).
        self.loss = nn.MSELoss()  # Funkcja straty (błąd średniokwadratowy).
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")  # Wybór urządzenia.
        self.to(self.device)  # Przesłanie modelu na GPU lub CPU.

    def forward(self, state, dice_roll):
        # Połączenie stanu (state) z rzutem kostką (dice_roll) jako wejście do sieci.
        print(f"State shape: {state.shape}, Dice roll shape: {dice_roll.shape}")
        x = T.cat((state, dice_roll), dim=1)
        print(f"Combined input shape: {x.shape}")
        x = F.relu(self.fc1(x))  # Aktywacja ReLU na pierwszej warstwie ukrytej.
        x = F.relu(self.fc2(x))  # Aktywacja ReLU na drugiej warstwie ukrytej.
        x = self.fc3(x)  # Wyjście sieci (wartości Q dla każdej akcji).
        return x


# Definicja agenta zarządzającego polityką i uczeniem.
class Agent:
    def __init__(self, gamma, epsilon, lr, input_size, batch_size, n_actions, max_mem=100000, eps_end=0.01,
                 eps_decay=1e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_size = input_size
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.max_mem = max_mem
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.mem_pointer = 0  # Wskaźnik na aktualną ilość zapisanych danych w pamięci.

        # Inicjalizacja sieci głównej (eval) i docelowej (target).
        self.DQN_eval = DQN(lr=self.lr, input_size=input_size, fc1_dims=128, fc2_dims=64, n_actions=n_actions)
        self.DQN_target = DQN(lr=self.lr, input_size=input_size, fc1_dims=128, fc2_dims=64, n_actions=n_actions)

        # Bufory pamięci do przechowywania doświadczeń (stan, akcja, nagroda, itp.).
        self.state_memory = deque(maxlen=self.max_mem)
        self.new_state_memory = deque(maxlen=self.max_mem)
        self.action_memory = deque(maxlen=self.max_mem)
        self.reward_memory = deque(maxlen=self.max_mem)
        self.terminal_memory = deque(maxlen=self.max_mem)
        self.dice_memory = deque(maxlen=self.max_mem)

    def observation(self, state, dice_roll, player):
        """
        Wykonuje predykcję akcji na podstawie obecnego stanu, rzutu kości i gracza.
        """
        self.DQN_eval.eval()  # Ustawienie sieci w tryb oceny (testowy)

        # Przetwarzanie danych wejściowych
        state_tensor = T.FloatTensor(state).unsqueeze(0).to(self.DQN_eval.device)  # Dodanie wymiaru batch
        dice_tensor = T.FloatTensor(dice_roll).view(1, -1).to(self.DQN_eval.device)

        # Predykcja wartości Q dla każdej akcji
        with T.no_grad():
            q_values = self.DQN_eval(state_tensor, dice_tensor)

        # Wybór akcji: eksploracja (random) lub eksploatacja (argmax)
        if np.random.rand() < self.epsilon:  # Eksploracja
            action_idx = np.random.randint(0, self.n_actions)
            action_value = q_values[0, action_idx].item()
        else:  # Eksploatacja
            action_idx = q_values.argmax(dim=1).item()
            action_value = q_values.max(dim=1).values.item()

        return action_idx, action_value

    def store_transition(self, state, action, reward, next_state, dice_roll, done):
        # Przechowywanie pojedynczego przejścia w buforze pamięci.
        self.state_memory.append(state)
        self.new_state_memory.append(next_state)
        self.reward_memory.append(reward)
        self.action_memory.append(action)
        self.dice_memory.append(dice_roll)
        self.terminal_memory.append(done)
        self.mem_pointer += 1

    def choose_action(self, state, dice_roll):
        # Zamiana stanu i rzutu kostką na tensory PyTorch.
        state_tensor = T.tensor(state, dtype=T.float32).unsqueeze(0).to(self.DQN_eval.device)
        dice_tensor = T.tensor(dice_roll, dtype=T.float32).view(1, -1).to(self.DQN_eval.device)

        # Zastosowanie eksploracji lub eksploatacji w wyborze akcji.
        if np.random.random() > self.epsilon:
            with T.no_grad():
                # Wybór akcji maksymalizującej wartość Q.
                assert state.shape[1] == 24, "State dimension mismatch!"
                assert dice_roll.shape[1] == 2, "Dice roll dimension mismatch!"
                q_values = self.DQN_eval.forward(state_tensor, dice_tensor)
                action = T.argmax(q_values).item()
        else:
            # Wybór losowej akcji (eksploracja).
            action = np.random.choice(self.n_actions)

        return action

    def learn(self):
        if self.mem_pointer < self.batch_size:
            return  # Ucz się tylko, gdy w pamięci jest wystarczająco dużo danych.

        self.DQN_eval.optimizer.zero_grad()

        # Losowy wybór batcha przejść z pamięci.
        max_mem = min(self.mem_pointer, self.max_mem)
        batch_indices = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch = T.tensor(np.array(self.state_memory)[batch_indices], dtype=T.float32).to(self.DQN_eval.device)
        new_state_batch = T.tensor(np.array(self.new_state_memory)[batch_indices], dtype=T.float32).to(
            self.DQN_eval.device)
        reward_batch = T.tensor(np.array(self.reward_memory)[batch_indices], dtype=T.float32).to(self.DQN_eval.device)
        terminal_batch = T.tensor(np.array(self.terminal_memory)[batch_indices], dtype=T.float32).to(
            self.DQN_eval.device)
        action_batch = T.tensor(np.array(self.action_memory)[batch_indices], dtype=T.long).to(self.DQN_eval.device)
        dice_batch = T.tensor(np.array(self.dice_memory)[batch_indices], dtype=T.float32).to(
            self.DQN_eval.device)

        # Obliczenie Q(s, a) za pomocą sieci "eval".
        q_eval = self.DQN_eval.forward(state_batch, dice_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Obliczenie max(Q(s', a')) za pomocą sieci "target".
        q_next = self.DQN_target.forward(new_state_batch, dice_batch)
        q_next[terminal_batch == 1] = 0.0  # Zerowanie wartości Q dla stanów końcowych.
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        # Obliczenie straty i optymalizacja sieci.
        loss = self.DQN_eval.loss(q_eval, q_target)
        loss.backward()
        self.DQN_eval.optimizer.step()

        # Stopniowa redukcja eksploracji (zmniejszanie epsilon).
        self.epsilon = max(self.eps_end, self.epsilon - self.eps_decay)

    def update_target_network(self):
        # Kopiowanie wag z sieci "eval" do sieci "target".
        self.DQN_target.load_state_dict(self.DQN_eval.state_dict())


# Definicja środowiska Backgammon (plansza, ruchy, logika gry).
class BackgammonEnv:
    def __init__(self):
        self.board = self.build_board()  # Inicjalizacja planszy.
        self.turn = 'white'  # Który gracz zaczyna.

    def build_board(self):
        # Tworzenie planszy gry w Backgammon w postaci tablicy z wartościami początkowymi.
        board = [0] * 24
        board[0] = 2
        board[5] = -5
        board[7] = -3
        board[11] = 5
        board[12] = -5
        board[16] = 3
        board[18] = 5
        board[23] = -2
        return board

    def reset(self):
        # Resetowanie planszy do stanu początkowego.
        self.board = self.build_board()
        self.turn = 'white'
        return np.array(self.board, dtype=np.float32)

    def roll_dice(self):
        # Rzut dwiema kostkami.
        return random.randint(1, 6), random.randint(1, 6)

    def step(self, action):
        from_pos, dice_roll = action  # dice_roll to krotka z dwoma wartościami
        for die_roll in dice_roll:  # Iteruj przez każdy rzut kostką
            success = self.move(from_pos, die_roll, self.turn)
            if not success:
                return self.get_state(), -1, True  # Nielegalny ruch = kara i koniec gry

        # Sprawdzenie warunków końca gry
        if sum(self.board) == 0:
            return self.get_state(), 1, True  # Wygrana białych
        if sum(self.board) == 0:
            return self.get_state(), -1, True  # Wygrana czarnych

        return self.get_state(), 0, False

    def move(self, from_pos, die_roll, player):
        to_pos = from_pos + die_roll if player == 'white' else from_pos - die_roll
        if self._is_valid_move(from_pos, to_pos, player):
            self._update_board(from_pos, to_pos, player)
            return True
        return False


    def _update_board(self, from_pos, to_pos, player):
        opponent = 'black' if player == 'white' else 'white'
        if player == 'white':
            self.board[from_pos] -= 1
            if self.board[to_pos] == -1:  # Bic przeciwnika
                self.bar[opponent] += 1
                self.board[to_pos] = 0
            self.board[to_pos] += 1
        else:
            self.board[from_pos] += 1
            if self.board[to_pos] == 1:  # Bic przeciwnika
                self.bar[opponent] += 1
                self.board[to_pos] = 0
            self.board[to_pos] -= 1

    def _is_valid_move(self, from_pos, to_pos, player):
        if to_pos < 0 or from_pos < 0:
            return False
        if player == 'white' and self.board[from_pos] <= 0:
            return False
        if player == 'black' and self.board[from_pos] >= 0:
            return False

        if player == 'white' and (to_pos < 0 or to_pos > 23):
            return False
        if player == 'black' and (to_pos > 23 or to_pos < 0):
            return False

        opponent = 'black' if player == 'white' else 'white'
        if player == 'white' and self.board[to_pos] < -1:
            return False
        if player == 'black' and self.board[to_pos] > 1:
            return False

        return True

    def get_state(self):
        return np.array(self.board, dtype=np.float32)



# Kod główny do treningu agenta

num_episodes = 1000

# Przykład uruchomienia treningu
def train_agent(env, agent, num_episodes, target_update=10):
    """
    Trenuje agenta w środowisku Backgammona z uwzględnieniem zasad gry i wzmocnień (Reinforcement Learning).

    Args:
        env: Obiekt środowiska (BackgammonEnv).
        agent: Obiekt agenta (Agent).
        num_episodes: Liczba epizodów do treningu.
        target_update: Co ile epizodów aktualizować sieć docelową (DQN_target).

    Returns:
        scores: Lista nagród uzyskanych w każdym epizodzie.
    """
    scores = []  # Lista nagród dla każdego epizodu

    for episode in range(1, num_episodes + 1):
        # Resetowanie środowiska przed każdym epizodem
        state = env.reset()
        done = False
        total_reward = 0

        # Zmienna pomocnicza do przechowywania aktualnego gracza ("white" lub "black")
        current_player = 'white'

        while not done:
            for _ in range(2):  # Każdy gracz wykonuje dwa rzuty z rzędu
                if done:  # Sprawdź, czy gra już się zakończyła po pierwszym ruchu
                    break

                # Rzut dwiema kostkami dla aktualnego ruchu
                dice_roll = env.roll_dice()

                # Agent wybiera najlepszą akcję na podstawie aktualnego stanu i rzutu kością
                action_idx, _ = agent.observation(state, dice_roll, player=current_player)

                # Mapowanie akcji (indeksu) na konkretne ruchy w grze
                # Zależy od szczegółowej implementacji środowiska, tutaj załóżmy (od pozycji, wartość rzutu kością)
                from_pos = action_idx  # Zakładamy, że akcja wskazuje pozycję początkową
                action = (from_pos, dice_roll)  # Akcja to pozycja początkowa + wartość kostki

                # Wykonanie ruchu w środowisku
                next_state, reward, done = env.step(action)

                # Przechowywanie przejścia w pamięci doświadczeń agenta
                agent.store_transition(state, action_idx, reward, next_state, dice_roll, done)

                # Nauka agenta (trening sieci DQN)
                loss = agent.learn()

                # Przechodzenie do następnego stanu
                state = next_state

                # Sumowanie nagród dla całego epizodu
                total_reward += reward

            # Przełączanie gracza na następny ruch ("white" <-> "black")
            current_player = 'black' if current_player == 'white' else 'white'

        # Po zakończeniu epizodu dodajemy wynik do listy wyników
        scores.append(total_reward)

        # Aktualizujemy sieć docelową co target_update epizodów
        if episode % target_update == 0:
            agent.DQN_target.load_state_dict(agent.DQN_eval.state_dict())

        # Wypisanie postępu co 10 epizodów
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    return scores

# Przykład uruchomienia treningu
# Parametry treningu
num_episodes = 10000

gamma = 0.99
epsilon = 1.0
lr = 0.001
input_size = 24  # Liczba cech wejściowych
n_dice = 2  # Liczba kostek w Backgammonie
batch_size = 24
n_actions = 24

# Definicja środowiska i agenta
env = BackgammonEnv()
agent = Agent(gamma, epsilon, lr, input_size + n_dice, batch_size, n_actions)

# Trening agenta
scores = train_agent(env, agent, num_episodes=num_episodes, target_update=50)

# Wizualizacja wyników treningu
import matplotlib.pyplot as plt
plt.plot(scores)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Training Performance')
plt.show()
