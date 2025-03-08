import re
import os
import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque


class DQN(nn.Module):
    def __init__(self, lr, input_size, fc1_dims, fc2_dims, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, fc1_dims)  # +1, bo dodajemy rzut kostką jako osobną cechę
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, move_dice_pairs):
        x = T.cat((state, move_dice_pairs), dim=1)  # Doklejamy dwa ruchy + kostki
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    # def forward(self, state, die):
    #     if not isinstance(die, T.Tensor):
    #         die = T.tensor([die], dtype=T.float32).to(self.device)  # Konwersja rzutu na tensor
    #
    #     if die.dim() == 1:
    #         die = die.unsqueeze(1)
    #
    #     x = T.cat((state, die), dim=1)  # Konkatenacja stanu z rzutem kostką
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x


class Agent:
    def __init__(self, gamma, epsilon, lr, input_size, batch_size, n_actions, max_mem=10000, eps_end=0.01,
                 eps_decay=1e-2, exploration_type='epsilon_greedy', temp=1.0):
        self.exploration_type = exploration_type
        self.temp = temp
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

        # Oddzielne sieci dla graczy
        self.DQN_eval_white = DQN(lr=self.lr, input_size=input_size, fc1_dims=128, fc2_dims=256, n_actions=n_actions)
        self.DQN_eval_white = DQN(lr=self.lr, input_size=input_size, fc1_dims=128, fc2_dims=256, n_actions=n_actions)
        # self.exploration_type_white = 'softmax'
        self.DQN_target_white = DQN(lr=self.lr, input_size=input_size, fc1_dims=128, fc2_dims=256, n_actions=n_actions)

        self.DQN_eval_black = DQN(lr=self.lr, input_size=input_size, fc1_dims=128, fc2_dims=256, n_actions=n_actions)
        self.DQN_eval_black = DQN(lr=self.lr, input_size=input_size, fc1_dims=128, fc2_dims=256, n_actions=n_actions)
        # self.exploration_type_black = 'epsilon_greedy'
        self.DQN_target_black = DQN(lr=self.lr, input_size=input_size, fc1_dims=128, fc2_dims=256, n_actions=n_actions)

        self.state_memory = deque(maxlen=self.max_mem)
        self.new_state_memory = deque(maxlen=self.max_mem)
        self.action_memory = deque(maxlen=self.max_mem)
        self.reward_memory = deque(maxlen=self.max_mem)
        self.terminal_memory = deque(maxlen=self.max_mem)
        self.dice_memory = deque(maxlen=self.max_mem)

    def store_transition(self, state, action1, dice1, action2, dice2, reward, next_state, done):
        move_dice_pairs = np.array([action1, dice1, action2, dice2], dtype=np.float32)
        self.state_memory.append(state)
        self.action_memory.append(move_dice_pairs)
        self.new_state_memory.append(next_state)
        self.reward_memory.append(reward)
        self.terminal_memory.append(done)
        self.mem_pointer += 1

    def get_possible_fields(self, board, player):

        if player == 'white':
            # Jeśli biali mają pionki na barze, to mogą je tylko ściągnąć z baru na planszę
            if env.bar['white'] > 0:
                possible_moves = [0]  # [i for i in range(24) if board[i] >= 0]  # Pola, na których nie ma przeciwnika
            else:
                # Zwróć pola, z których gracz może wykonać ruch, jeśli nie ma pionków na barze
                possible_moves = [i for i, value in enumerate(board) if value > 0]  # Pionki białe
        else:
            # Jeśli czarni mają pionki na barze, to mogą je tylko ściągnąć z baru na planszę
            if env.bar['black'] > 0:
                possible_moves = [23]  # [i for i in range(24) if board[i] <= 0]  # Pola, na których nie ma przeciwnika
            else:
                # Zwróć pola, z których gracz może wykonać ruch, jeśli nie ma pionków na barze
                possible_moves = [i for i, value in enumerate(board) if value < 0]  # Pionki czarne

        return possible_moves

    def choose_action(self, state, dice_roll, player, forbidden_actions=None):
        # if sum(max(0, x) for x in env.board) > 15 or sum(max(0, -x) for x in env.board) < -15:
        #     print(state)

        if forbidden_actions is None:
            forbidden_actions = set()
        if self.exploration_type == 'softmax':
            return self.choose_action_softmax(state, dice_roll, player, forbidden_actions)
        return self.choose_action_epsilon_greedy(state, dice_roll, player, forbidden_actions)

    def choose_action_epsilon_greedy(self, state, dice_roll, player, forbidden_actions=None):
        if forbidden_actions is None:
            forbidden_actions = set()

        eval_net = self.DQN_eval_white if player == 'white' else self.DQN_eval_black

        state_tensor = T.tensor(state, dtype=T.float32).unsqueeze(0).to(eval_net.device)

        possible_fields = self.get_possible_fields(state, player)
        if not possible_fields:
            return -1, -1  # ❌ Brak możliwych ruchów

        # 🔹 **Pierwszy ruch**
        mask = T.zeros(self.n_actions).to(eval_net.device)
        mask[possible_fields] = 1  # Dozwolone ruchy
        move_dice_pairs = T.zeros((1, 4)).to(eval_net.device)  # Inicjalizacja pustych ruchów (2x [ruch, kostka])

        if np.random.random() < self.epsilon:
            action1 = np.random.choice(possible_fields)
        else:
            with T.no_grad():
                q_values = eval_net.forward(state_tensor, move_dice_pairs)
                q_values = q_values * mask  # Filtrujemy ruchy niedozwolone
                action1 = T.argmax(q_values).item()

        # 🔄 **Symulacja pierwszego ruchu**
        moved, _, done, _ = env.step((action1, dice_roll[0]), player, movement=False)

        if not moved:
            forbidden_actions.add(action1)
            env.board = state
            return -1, -1

        if done:
            env.board = state
            return action1, -1

        # 🔹 **Drugi ruch po wykonaniu pierwszego**
        new_state_tensor = T.tensor(env.get_state(), dtype=T.float32).unsqueeze(0).to(eval_net.device)
        possible_fields_2 = self.get_possible_fields(env.get_state(), player)

        if not possible_fields_2:
            env.board = state
            return action1, -1

        mask2 = T.zeros(self.n_actions).to(eval_net.device)
        mask2[possible_fields_2] = 1

        move_dice_pairs[0, :2] = T.tensor([action1, dice_roll[0]], dtype=T.float32).to(
            eval_net.device)  # Zapisujemy ruch 1

        if np.random.random() < self.epsilon:
            action2 = np.random.choice(possible_fields_2)
        else:
            with T.no_grad():
                q_values_2 = eval_net.forward(new_state_tensor, move_dice_pairs)
                q_values_2 = q_values_2 * mask2  # Filtrujemy niedozwolone
                action2 = T.argmax(q_values_2).item()
        env.board = state
        return action1, action2

    def choose_action_softmax(self, state, dice_roll, player, forbidden_actions=None):
        if forbidden_actions is None:
            forbidden_actions = set()

        eval_net = self.DQN_eval_white if player == 'white' else self.DQN_eval_black

        state_tensor = T.tensor(state, dtype=T.float32).unsqueeze(0).to(eval_net.device)
        dice_tensor = T.tensor(dice_roll, dtype=T.float32).view(1, -1).to(eval_net.device)

        possible_fields = self.get_possible_fields(state, player)
        possible_fields = [f for f in possible_fields if f not in forbidden_actions]  # Usuwamy zakazane akcje

        if not possible_fields:
            return -1  # Specjalna wartość oznaczająca brak ruchu

        with T.no_grad():
            q_values = eval_net.forward(state_tensor, dice_tensor).cpu().numpy().flatten()

        #  Maska dla dozwolonych pól
        q_values_masked = np.full_like(q_values, -np.inf)  # Domyślnie ustawiamy bardzo niskie wartości
        q_values_masked[possible_fields] = q_values[
            possible_fields]  # Przypisujemy wartości Q tylko dla dozwolonych pól

        # Zabezpieczenie przed przepełnieniem i NaN
        q_values_masked -= np.max(q_values_masked)  # Normalizacja do uniknięcia overflow
        exp_values = np.exp(q_values_masked / self.temp)

        if np.sum(exp_values) == 0 or np.isnan(exp_values).any():
            return np.random.choice(possible_fields) if possible_fields else -1  # Unikamy błędów

        probs = exp_values / np.sum(exp_values)  # Normalizacja do sumy 1

        return np.random.choice(possible_fields, p=probs[possible_fields] / np.sum(probs[possible_fields]))

    def learn(self, player):
        if self.mem_pointer < self.batch_size:
            return

        eval_net = self.DQN_eval_white if player == 'white' else self.DQN_eval_black
        target_net = self.DQN_target_white if player == 'white' else self.DQN_target_black

        eval_net.optimizer.zero_grad()

        max_mem = min(self.mem_pointer, self.max_mem)
        batch_indices = np.random.choice(max_mem, self.batch_size, replace=False)

        move_dice_batch = T.tensor(np.array(self.action_memory)[batch_indices], dtype=T.float32).to(eval_net.device)
        state_batch = T.tensor(np.array(self.state_memory)[batch_indices], dtype=T.float32).to(eval_net.device)
        new_state_batch = T.tensor(np.array(self.new_state_memory)[batch_indices], dtype=T.float32).to(eval_net.device)
        reward_batch = T.tensor(np.array(self.reward_memory)[batch_indices], dtype=T.float32).to(eval_net.device)
        terminal_batch = T.tensor(np.array(self.terminal_memory)[batch_indices], dtype=T.float32).to(eval_net.device)

        q_eval = eval_net(state_batch, move_dice_batch).max(1)[0]
        q_next = target_net(new_state_batch, move_dice_batch).max(1)[0]
        q_next[terminal_batch == 1] = 0.0

        q_target = reward_batch + self.gamma * q_next
        loss = eval_net.loss(q_eval, q_target)
        loss.backward()
        self.epsilon = max(self.eps_end, self.epsilon - self.eps_decay)
        eval_net.optimizer.step()

    def update_target_network(self, player):
        """
        Aktualizuje sieć docelową wybranego gracza.
        """
        if player == 'white':
            self.DQN_target_white.load_state_dict(self.DQN_eval_white.state_dict())
        else:
            self.DQN_target_black.load_state_dict(self.DQN_eval_black.state_dict())

    def clean_line(self, line):
        """
        Usuwa znaki specjalne, pozostawiając tylko rzuty kostką i ruchy graczy.
        """
        line = re.sub(r'[^0-9a-zA-Z\s/()-]', '', line)  # Usunięcie znaków specjalnych
        return line.strip()

    def parse_moves(self, file_path):
        """
        Parsuje ruchy obu graczy z pliku tekstowego zapisu gry w Backgammon.
        """

        def transform_position_white(pos):
            if pos == "Bar":
                return "bar"
            elif pos == "25":  # Bear Off dla białego
                return "bar_off"
            elif pos == "0":  # Bear Off dla czarnego
                return "bar_off"
            else:
                return str(24 - int(pos))  # Odwrócenie numeracji dla białych

        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [self.clean_line(line) for line in file.readlines()]

        moves = []
        for line in lines:
            line = line.replace("Off", "")  # Usunięcie znaku '*'
            match = re.match(r'\s*(\d+)\)\s*(\d)\s*(\d):\s*(.*?)\s*(\d)\s*(\d):\s*(.*?)$', line)
            if match:
                move_num, p1_dice1, p1_dice2, p1_move, p2_dice1, p2_dice2, p2_move = match.groups()

                # Transformacja numeracji dla białych
                # Transformacja numeracji dla białych, obsługa 'Bar' i 'Bar Off'
                p1_transformed = " ".join([
                    f"{transform_position_white(m.split('/')[0])}/{transform_position_white(m.split('/')[1])}"
                    for m in p1_move.split() if '/' in m
                ])

                moves.append(((int(p1_dice1), int(p1_dice2), p1_transformed), (int(p2_dice1), int(p2_dice2), p2_move)))

        return moves

    def encode_move(self, move):
        """
        Konwertuje ruch na reprezentację numeryczną dla modelu DQN.
        """
        encoded = []
        actions = move.split()
        for action in actions:
            match = re.match(r'(\d+|bar|bear_off)/(\d+|bar|bear_off)', action)
            if match:
                from_pos, to_pos = match.groups()
                encoded.append((from_pos, to_pos))
        return encoded

    def prepare_training_data(self, moves):
        """
        Przygotowuje dane treningowe w formacie nadającym się do sieci DQN.
        """
        training_data = []
        for (p1_dice1, p1_dice2, p1_move), (p2_dice1, p2_dice2, p2_move) in moves:
            encoded_p1 = self.encode_move(p1_move)
            encoded_p2 = self.encode_move(p2_move)
            training_data.append(((p1_dice1, p1_dice2), encoded_p1, (p2_dice1, p2_dice2), encoded_p2))

        return training_data

    def process_all_files(self, folder_path):
        """
        Odczytuje i przetwarza wszystkie pliki w podanym folderze.
        """
        all_training_data = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith(".txt"):
                print(f"Przetwarzanie pliku: {file_name}")
                moves = self.parse_moves(file_path)
                training_data = self.prepare_training_data(moves)
                all_training_data.extend(training_data)

        return all_training_data


class BackgammonEnv:
    def __init__(self):
        self.bar = {'white': 0, 'black': 0}  # Bar graczy: biali i czarni
        self.board = self.build_board()

    def build_board(self):
        board = [0] * 24
        # Ustawienie białych pionków
        board[0] = 2  # 2 pionki na polu 1
        board[11] = 5  # 5 pionków na polu 12
        board[16] = 3  # 3 pionki na polu 17
        board[18] = 5  # 5 pionków na polu 19

        # Ustawienie czarnych pionków
        board[23] = -2  # 2 pionki na polu 24
        board[12] = -5  # 5 pionków na polu 13
        board[7] = -3  # 3 pionki na polu 8
        board[5] = -5  # 5 pionków na polu 6

        return board

    def reset(self):
        self.board = self.build_board()
        self.bar = {'white': 0, 'black': 0}  # Resetujemy bar przy nowej grze
        self.turn = 'white'
        return np.array(self.board, dtype=np.float32)

    def roll_dice(self):
        # Rzut dwiema kostkami.
        return random.randint(1, 6), random.randint(1, 6)

    def count_negative_fields(self):
        return sum(1 for field in self.board if field < 0)

    def count_positive_fields(self):

        return sum(1 for field in self.board if field > 0)

    def calculate_pips(self, player):
        if player == 'white':
            return sum(pos * abs(self.board[pos]) for pos in range(24) if self.board[pos] > 0)
        else:
            return

    def step(self, action, player, movement, save_file=False):
        from_pos, dice_roll = action
        total_reward = 0  # Całkowita nagroda za akcję
        done = False
        state = self.board
        if self.count_negative_fields() == 1 or self.count_positive_fields() == 1:
        #print('')
            i=0
        to_pos = from_pos + dice_roll if player == 'white' else from_pos - dice_roll

        if (player == 'white' and to_pos >= 25) or (player == 'black' and to_pos <= 0):
            total_reward += 10

        success = self.move(from_pos, dice_roll, player)

        if not success:
            # Kara za niepoprawny ruch
            if movement:
                return True, -3, True, ''
            else:
                return False, 0, False, ''
        if 0 <= to_pos <= 23 and abs(state[to_pos]) == 1 and (
                (player == 'white' and state[to_pos] < 0) or
                (player == 'black' and state[to_pos] > 0)
        ):
            total_reward += 5

        # Sprawdzenie warunków końca gry
        if player == 'white':
            if self.count_positive_fields() == 0:
                return True, total_reward + 10, True, 'white'  # Nagroda za wygraną

        elif player == 'black':
            if self.count_negative_fields() == 0:
                return True, total_reward + 10, True, 'black'  # Nagroda za wygraną

        # Dodaj stałą nagrodę za poprawny ruch, aby zachęcać do działania
        total_reward += 1
        if save_file:
            with open("game_log1.txt", "a") as log_file:
                log_file.write(f"{player} | {dice_roll} {from_pos} {to_pos} {total_reward}\n")
        return True, total_reward, done, ''

    def move(self, from_pos, dice_roll, player,
             agent=Agent(gamma=0.99, epsilon=1, lr=0.0001, input_size=28, batch_size=16, n_actions=24)):
        state = self.get_state()
        """
        Wyjście z baru
        """
        if player == 'white' and self.bar['white'] > 0:
            # Gracz biały ma pionki na barze, więc musi je najpierw ściągnąć na planszę
            to_pos = 0 + (dice_roll - 1)
            if self._is_valid_move(from_pos, to_pos, player):
                self._update_board(from_pos, to_pos, player)
                # self.bar['white'] -= 1
                if sum(max(0, x) for x in env.board) + self.bar['white'] > 15 or sum(max(0, -x) for x in env.board) - \
                        self.bar['black'] < -15:
                    agent.get_possible_fields(env.get_state(), player)
                return True
            else:
                return False
        elif player == 'black' and self.bar['black'] > 0:
            # Gracz czarny ma pionki na barze, więc musi je najpierw ściągnąć na planszę
            to_pos = 23 - (dice_roll - 1)
            if self._is_valid_move(from_pos, to_pos, player):
                self._update_board(from_pos, to_pos, player)
                if sum(max(0, x) for x in env.board) + self.bar['white'] > 15 or sum(max(0, -x) for x in env.board) - \
                        self.bar['black'] < -15:
                    agent.get_possible_fields(env.get_state(), player)
                return True
            else:
                return False

        if player == 'white':
            to_pos = from_pos + dice_roll
        else:
            to_pos = from_pos - dice_roll

        """
        Wyjście z planszy
        """

        # Sprawdzamy, czy pionek wychodzi poza planszę (powyżej 23 lub poniżej 0)
        if player == 'white' and to_pos > 23:
            # Pionek wychodzi z planszy (biały)
            if self.board[from_pos] > 0:  # Białe pionki
                self.board[from_pos] -= 1  # Usuwamy pionek z planszy
                # print(f"White piece has exited the board from position {from_pos}.")
                if sum(max(0, x) for x in env.board) + self.bar['white'] > 15 or sum(max(0, -x) for x in env.board) - \
                        self.bar['black'] < -15:
                    agent.get_possible_fields(env.get_state(), player)
                return True

        elif player == 'black' and to_pos < 0:
            # Pionek wychodzi z planszy (czarny)
            if self.board[from_pos] < 0:  # Czarne pionki
                self.board[from_pos] += 1  # Usuwamy pionek z planszy
                # print(f"Black piece has exited the board from position {from_pos}.")
                if sum(max(0, x) for x in env.board) + self.bar['white'] > 15 or sum(max(0, -x) for x in env.board) - \
                        self.bar['black'] < -15:
                    agent.get_possible_fields(env.get_state(), player)
                return True

        """
        Normalny ruch
        """
        if self._is_valid_move(from_pos, to_pos, player):
            self._update_board(from_pos, to_pos, player)
            # print(f"ruch {player}")
            if sum(max(0, x) for x in env.board) + self.bar['white'] > 15 or sum(max(0, -x) for x in env.board) - \
                    self.bar['black'] < -15:
                agent.get_possible_fields(env.get_state(), player)
            return True
        return False

    def _update_board(self, from_pos, to_pos, player):
        """Aktualizuje stan planszy po wykonanym ruchu."""
        # opponent = 'black' if player == 'white' else 'white'

        # if not self._is_valid_move(from_pos, to_pos, player):  # Sprawdzamy legalność ruchu
        #     return  # Jeśli ruch jest nielegalny, nie aktualizujemy planszy

        if player == 'white':
            # if self.board[from_pos] > 0:
            if self.bar['white'] > 0:
                self.bar['white'] -= 1
                # print(f"White piece has exited the bar")
            else:
                self.board[from_pos] -= 1

            # Sprawdzamy, czy można bić pionek przeciwnika
            if self.board[to_pos] == -1:  # Jeśli na to_pos jest dokładnie 1 czarny pionek
                self.bar['black'] += 1  # Przeciwnik trafia na bar
                self.board[to_pos] = 0  # Pole staje się puste
            # print('zbity czarny')

            self.board[to_pos] += 1  # Dodajemy pionek na nową pozycję

        else:  # Dla czarnych
            if self.bar['black'] > 0:
                self.bar['black'] -= 1
                # print(f"Black piece has exited the bar")
            else:
                self.board[from_pos] += 1  # Usuwamy pionek z pola startowego

            # Sprawdzamy, czy można bić pionek przeciwnika
            if self.board[to_pos] == 1:  # Jeśli na to_pos jest dokładnie 1 biały pionek
                self.bar['white'] += 1  # Przeciwnik trafia na bar
                self.board[to_pos] = 0  # Pole staje się puste
                # print('zbity bialy')

            self.board[to_pos] -= 1  # Dodajemy pionek czarny na nową pozycję

    def _is_valid_move(self, from_pos, to_pos, player):
        if player == 'white':
            if self.board[from_pos] == 0:
                if self.bar['white'] > 0 and from_pos == 0:
                    return True
                else:
                    return False
            if self.board[from_pos] < 0:
                return False
        if player == 'black':
            if self.board[from_pos] == 0:
                if self.bar['black'] > 0 and from_pos == 23:
                    return True
                else:
                    return False
            if self.board[from_pos] > 0:
                return False
        if 0 <= to_pos <= 23:
            if player == 'white':
                if self.board[to_pos] < -1:
                    return False

            if player == 'black' and self.board[to_pos] > 1:
                return False

            # return True
        return True

    def get_state(self):
        return np.array(self.board, dtype=np.float32)


# def train_movement(env, agent, num_episodes, max_steps=50):
#     """
#     Trening podstawowego poruszania się pionkiem w uproszczonym środowisku.
#     """
#     for episode in range(1, num_episodes + 1):
#         state = env.reset()
#         done = False
#         step_count = 0
#         total_reward = 0
#         player = 'white'
#
#         while not done and step_count < max_steps:
#             step_count += 1
#
#             # Rzut kostką
#             dice_roll = env.roll_dice()
#             for die in dice_roll:
#                 state = env.get_state()
#                 success = False
#                 tried_actions = set()  # Zbiór przechowujący sprawdzone błędne akcje
#
#                 # Decyzja agenta o ruchu
#                 while not success:
#                     action_idx = agent.choose_action(env.get_state(), die, player=player,
#                                                      forbidden_actions=tried_actions)
#                     if action_idx == -1:
#                         break
#                     from_pos = action_idx
#                     action = (from_pos, die)
#
#                     success, reward, done, winner = env.step(action, player=player, movement=True)
#
#                     if not success:
#                         reward -= 0.5  # Kara za nielegalny ruch
#                         tried_actions.add(action_idx)  # Dodanie akcji do zakazanych
#                     else:
#                         agent.store_transition(state, action_idx, reward, env.board, die, done)
#                         total_reward += reward
#                         tried_actions.add(action_idx)
#                 # if env.board is None:
#                 #     print('state none')
#             # Nauka agenta
#             if player == 'white':
#                 agent.learn('white')
#             else:
#                 agent.learn('black')
#
#             # Aktualizacja stanu
#             player = 'white' if player == 'black' else 'white'
#         # Aktualizacja sieci co epizod
#         agent.update_target_network('white')
#         agent.update_target_network('black')
#
#         # Wyświetlenie postępu
#         if episode % 100 == 0:
#             print(
#                 f"Pre-training Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")


def train_agent(env, agent_white, agent_black, num_episodes):
    temp = 0
    scores = []

    for episode in range(1, num_episodes + 1):
        print("Episode ", episode)
        env.reset()
        done = False
        total_reward_white = 0
        total_reward_black = 0
        reward_white = 0
        reward_black = 0
        total_reward = 0
        current_player = random.choice(['black', 'white'])
        wmove1 = [0, 0]
        wmove2 = [0, 0]
        bmove1 = [0, 0]
        bmove2 = [0, 0]
        transitions_white = []
        transitions_black = []

        winner = ''

        while not done:
            # for i in range(2):  # Każdy gracz wykonuje ruch
            dice_roll = env.roll_dice()
            current_agent = agent_white if current_player == 'white' else agent_black
            if current_player == 'white':
                reward_white = 0
            else:
                reward_black = 0
            # i=0
            success1 = False
            success2 = False
            forbidden_moves1 = set()
            forbidden_moves2 = set()
            if current_player == 'white':
                white_prev_state = env.get_state()
            else:
                black_prev_state = env.get_state()
            while not success1 and not success2:
                action_idx1, action_idx2 = current_agent.choose_action(env.get_state(), dice_roll,
                                                                       player=current_player,
                                                                       forbidden_actions=forbidden_moves1)
                if action_idx1 == -1:
                    break
                from_pos = action_idx1
                action = (from_pos, dice_roll[0])

                success1, reward, done, winner = env.step(action, player=current_player, movement=False)
                if not success1:
                    reward -= 0.5
                    forbidden_moves1.add(from_pos)
                else:
                    if current_player == 'white':
                        wmove1 = (from_pos, dice_roll[0])
                        reward_white += reward
                    else:
                        bmove1 = (from_pos, dice_roll[0])
                        reward_black += reward
                if done:
                    break

                if action_idx2 == -1:
                    break
                from_pos = action_idx2
                action = (from_pos, dice_roll[1])

                success2, reward, done, winner = env.step(action, player=current_player, movement=False)
                if not success2:
                    reward -= 0.5
                    forbidden_moves2.add(from_pos)
                else:
                    if current_player == 'white':
                        wmove2 = (from_pos, dice_roll[1])
                        reward_white += reward
                    else:
                        bmove2 = (from_pos, dice_roll[1])
                        reward_black += reward

                    total_reward += reward

            if current_player == 'black' and wmove1 != [0, 0] and wmove2 != [0, 0]:
                transitions_white.append(
                    (black_prev_state, wmove1[0], wmove1[1], wmove2[0], wmove2[1], reward_white + reward_white - reward_black, env.board, done))
                total_reward_white += reward_white - reward_black
            elif current_player == 'white' and bmove1 != [0, 0] and bmove2 != [0, 0]:
                transitions_black.append(
                    (white_prev_state, bmove1[0], bmove1[1], bmove2[0], bmove2[1], reward_black + reward_black - reward_white, env.board, done))
                total_reward_white += reward_black - reward_white
            # Nauka po dwóch ruchach dla białego

            # Jeśli gra się skończyła, nie zmieniamy tury
            if not done:
                current_player = 'black' if current_player == 'white' else 'white'
            temp += 1

        for transition in transitions_white:
            agent_white.store_transition(*transition)
            agent_white.learn('white')
        transitions_white.clear()

        for transition in transitions_black:
            agent_black.store_transition(*transition)
            agent_black.learn('black')
        transitions_black.clear()

        scores.append((total_reward, total_reward_white, total_reward_black))

        # Aktualizacja sieci zwycięzcy
        if winner == 'white':
            agent_white.update_target_network('white')
        elif winner == 'black':
            agent_black.update_target_network('black')

        # Wyświetlenie wyników co 100 epizodów
        if episode % 100 == 0:
            if len(scores) >= 100:
                rolling_avg_white = np.mean([score[0] for score in scores[-100:]])  # Ostatnie 100 epizodów dla białego
                rolling_avg_black = np.mean([score[1] for score in scores[-100:]])  # Ostatnie 100 epizodów dla czarnego
            else:
                rolling_avg_white = np.mean([score[0] for score in scores])
                rolling_avg_black = np.mean([score[1] for score in scores])

            print(
                f"Episode {episode}/{num_episodes}, "
                f"Total Reward White: {total_reward_white:.2f}, "
                f"Total Reward Black: {total_reward_black:.2f}, "
                f"Rolling Avg Reward White (Last 100 Episodes): {rolling_avg_white:.2f}, "
                f"Rolling Avg Reward Black (Last 100 Episodes): {rolling_avg_black:.2f}, "
                f"Epsilon White: {agent_white.epsilon:.3f}, "
                f"Epsilon Black: {agent_black.epsilon:.3f}"
            )

    return scores


# def train_agent_from_data(agent_white, agent_black, training_data, env, num_epochs=10):
#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch + 1}/{num_epochs}")
#
#         for (p1_dice, p1_moves, p2_dice, p2_moves) in training_data:
#             state = env.reset()  # Resetujemy planszę przed każdym epizodem
#             p1_dice = np.array(p1_dice, dtype=np.float32)  # Konwersja rzutów na float32
#             p2_dice = np.array(p2_dice, dtype=np.float32)
#
#             # Gracz biały (white)
#             for move in p1_moves:
#                 from_pos, to_pos = move
#
#                 action = (from_pos, p1_dice)  # Akcja jako (z pola, rzut kostką)
#                 success, reward, done, _ = env.step(action, player="white", movement=True)
#
#                 agent_white.store_transition(state, from_pos, reward, env.board, p1_dice, done)
#                 agent_white.learn("white") # Aktualizacja stanu dla kolejnego ruchu
#
#             # Gracz czarny (black)
#             for move in p2_moves:
#                 from_pos, to_pos = move
#
#                 action = (from_pos, p2_dice)  # Akcja jako (z pola, rzut kostką)
#                 success, reward, done, _ = env.step(action, player="black", movement=True)
#
#                 agent_black.store_transition(state, from_pos, reward, next_state, p2_dice, done)
#                 agent_black.learn("black")
#                 state = next_state  # Aktualizacja stanu dla kolejnego ruchu
#
#     # Po zakończeniu epok aktualizujemy sieci docelowe
#     agent_white.update_target_network("white")
#     agent_black.update_target_network("black")


def random_player_action(env, die, agent):
    """
    Losowy ruch gracza (dla losowego przeciwnika) na podstawie możliwych ruchów.
    Jeśli agent jest podany, używa jego metody get_possible_fields, aby określić dostępne ruchy.
    """
    state = env.get_state()
    player = env.turn
    # dice_rolls = env.roll_dice()  # Zakładamy, że dice_rolls to lista dwóch rzutów

    # Jeśli agent jest podany, użyj jego metody get_possible_fields

    possible_fields = agent.get_possible_fields(state, player)

    random.shuffle(possible_fields)

    # Jeśli brak dostępnych pól, pomijamy ruch i przechodzimy do następnej tury
    if not possible_fields:
        # print(f"Warning: No possible moves for player {player}. Skipping turn.")
        return None  # Zwracamy None, aby oznaczyć brak ruchu

    # Próbujemy wykonać ruchy na podstawie możliwych pól i rzutów kostką
    for from_pos in possible_fields:
        if env.move(from_pos, die, player):  # Użycie funkcji move
            return True  # Zwracamy pierwszy znaleziony poprawny ruch

    # Jeśli nie znaleziono żadnego poprawnego ruchu, pomijamy turę
    # print(f"Warning: No valid moves for player {player}. Skipping turn.")
    return False


def play_with_random_opponent(env, agent, num_episodes):
    """
    Pozwala botowi grać przeciwko losowemu graczowi przez określoną liczbę epizodów.
    """
    agent_wins = 0
    bot_wins = 0

    for episode in range(1, num_episodes + 1):
        # print(f"Episode")
        state = env.reset()
        done = False
        winner = ''
        while not done:
            dice_roll = env.roll_dice()
            # print(f"Current turn: {env.turn}")  # Debug: sprawdzanie zmiany tury
            for die in dice_roll:
                success = False
                if env.turn == 'white':
                    forbidden_moves = set()
                    while not success:
                        action_idx = agent.choose_action(env.get_state(), die, player='white',
                                                         forbidden_actions=forbidden_moves)
                        if action_idx == -1:
                            break
                        from_pos = action_idx
                        action = (from_pos, die)

                        success, reward, done, winner = env.step(action, player='white', movement=False)
                        if not success:
                            reward -= 0.5
                            forbidden_moves.add(from_pos)
                        # print("White's move executed")  # Debug: Potwierdzenie ruchu białych
                else:
                    if not random_player_action(env, die, agent):
                        continue
            env.turn = 'black' if env.turn == 'white' else 'white'
            # if env.turn == 'black':
            # #print("Black's move executed")  # Debug: Potwierdzenie ruchu czarnych
            #     env.turn = 'white'
            # else:
            # #print("Black skipped the turn.")  # Debug: Czarny gracz pomija turę
            #     env.turn = 'white'

        if winner == 'white':  # Wygrana białych (agenta)
            agent_wins += 1

        if winner == 'black':
            bot_wins += 1

        # Wyświetl wynik co 10 gier
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}, Agent Wins: {agent_wins}, Bot Wins:{bot_wins}")

    win_rate = (agent_wins / num_episodes) * 100
    print(f"Agent Win Rate: {win_rate:.2f}%")
    return win_rate


def test_network_architectures(env, input_size, n_actions, lr=0.001, num_episodes=100):
    architectures = [
        {'fc1': 64, 'fc2': 128},
        {'fc1': 128, 'fc2': 256},
        {'fc1': 256, 'fc2': 512},
        {'fc1': 128, 'fc2': 128, 'fc3': 128}
    ]

    results = {}
    for idx, arch in enumerate(architectures):
        print(f"\nTesting architecture {idx + 1}: {arch}")

        # Modyfikacja klasy DQN dla różnych architektur
        class CustomDQN(DQN):
            def __init__(self, lr, input_size, n_actions):
                super().__init__(lr, input_size, arch['fc1'], arch['fc2'], n_actions)
                if 'fc3' in arch:
                    self.fc3 = nn.Linear(arch['fc2'], arch['fc3'])
                    self.fc4 = nn.Linear(arch['fc3'], n_actions)

        agent = Agent(gamma=0.99, epsilon=1.0, lr=lr,
                      input_size=input_size, batch_size=32, n_actions=n_actions)

        scores = train_agent(env, agent, agent, num_episodes)
        results[f"Arch_{idx + 1}"] = np.mean(scores[-10:])

    print("\nArchitecture Comparison:")
    for arch, score in results.items():
        print(f"{arch}: {score:.2f}")


def test_exploration_strategies(env, num_episodes=100):
    """5.2 Porównanie strategii eksploracji"""

    class ModifiedAgent(Agent):
        def __init__(self, exploration_type='epsilon_greedy', temp=1.0, **kwargs):
            super().__init__(**kwargs)
            self.exploration_type = exploration_type
            self.temp = temp

        def choose_action(self, state, dice_roll, player, forbidden_actions=None):
            eval_net = self.DQN_eval_white if player == 'white' else self.DQN_eval_black
            state_tensor = T.tensor(state, dtype=T.float32).unsqueeze(0).to(eval_net.device)
            dice_tensor = T.tensor(dice_roll, dtype=T.float32).view(1, -1).to(eval_net.device)
            possible_fields = self.get_possible_fields(state, player)

            if not possible_fields:
                return 0  # Brak możliwych ruchów

            with T.no_grad():
                q_values = eval_net.forward(state_tensor, dice_tensor).cpu().numpy().flatten()

            if self.exploration_type == 'softmax':
                # 🛠 Normalizacja wartości Q, aby uniknąć przepełnienia
                q_values -= np.max(q_values)

                exp_values = np.exp(q_values / self.temp)

                #  Obsługa błędów NaN i Inf
                if np.any(np.isnan(exp_values)) or np.any(np.isinf(exp_values)):
                    print("NaN lub Inf w softmax! Wybór losowej akcji.")
                    return np.random.choice(possible_fields)

                probs = exp_values / np.sum(exp_values)

                # 🛠 Sprawdzenie poprawności prawdopodobieństw
                if np.isnan(probs).any() or np.sum(probs) == 0:
                    print("Niepoprawne prawdopodobieństwa w softmax! Wybór losowej akcji.")
                    return np.random.choice(possible_fields)

                action = np.random.choice(range(self.n_actions), p=probs)
            else:  # ε-greedy
                if np.random.random() > self.epsilon:
                    action = np.argmax(q_values)
                else:
                    action = np.random.choice(possible_fields)

            return action if action in possible_fields else np.random.choice(possible_fields)

    strategies = [
        {'type': 'epsilon_greedy', 'params': {'eps_decay': 1e-4}},
        {'type': 'softmax', 'params': {'temp': 0.5}}
    ]

    results = {}
    for strategy in strategies:
        agent = ModifiedAgent(exploration_type=strategy['type'],
                              temp=strategy['params'].get('temp', 1.0),
                              gamma=0.99, epsilon=1.0, lr=0.001,
                              input_size=28, batch_size=32, n_actions=24)

        scores = train_agent(env, agent, agent, num_episodes)
        results[strategy['type']] = np.mean(scores[-10:])

    print("\n **Porównanie strategii eksploracji:**")
    for strategy, score in results.items():
        print(f"{strategy}: {score:.2f}")


def hyperparameter_tuning(env):
    """5.3 Testowanie różnych kombinacji hiperparametrów"""
    param_grid = {
        'gamma': [0.9, 0.95, 0.99],
        'lr': [0.0001, 0.001, 0.01],
        'batch_size': [16, 32, 64]
    }

    best_score = -np.inf
    best_params = {}

    for gamma in param_grid['gamma']:
        for lr in param_grid['lr']:
            for batch_size in param_grid['batch_size']:
                agent = Agent(gamma=gamma, epsilon=1.0, lr=lr,
                              input_size=28, batch_size=batch_size, n_actions=24)

                scores = train_agent(env, agent, agent, 50)
                mean_score = np.mean(scores[-10:])

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {'gamma': gamma, 'lr': lr, 'batch_size': batch_size}

    print(f"\nBest params: {best_params} with score: {best_score:.2f}")


class HeuristicBot:
    def choose_action(self, state, dice_roll, player):
        possible_fields = [i for i, val in enumerate(state) if (val > 0 if player == 'white' else val < 0)]
        if not possible_fields:
            return None  # Brak możliwych ruchów
        return min(possible_fields) if player == 'white' else max(possible_fields)


def play_with_heuristic_opponent(env, agent, num_episodes):
    heuristic_bot = HeuristicBot()
    agent_wins = 0
    bot_wins = 0

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        current_player = 'white'

        while not done:
            dice_roll = env.roll_dice()
            forbidden_actions = set()
            winner = ''
            for dice in dice_roll:
                success = False
                forbidden_actions.clear()
                if current_player == 'white':
                    while not success:
                        action_idx = agent.choose_action(env.get_state(), dice, player=current_player,
                                                         forbidden_actions=forbidden_actions)
                        if action_idx == -1:
                            next_state = state
                            break
                        action = (action_idx, dice)
                        success, reward, done, winner = env.step(action, player=current_player, movement=False)
                        if not success:
                            forbidden_actions.add(action_idx)
                else:
                    action_idx = heuristic_bot.choose_action(env.get_state(), dice, player=current_player)
                    if action_idx == None:
                        continue
                    action = (action_idx, dice)
                    success, reward, done, winner = env.step(action, player=current_player, movement=False)
                #     if not success:
                #         next_state = state
                # state = next_state

            if done:
                if winner == 'white':
                    agent_wins += 1
                elif winner == 'black':
                    bot_wins += 1
            current_player = 'black' if current_player == 'white' else 'white'

    win_rate = (agent_wins / num_episodes) * 100
    print(f"Win Rate vs Heuristic Bot: {win_rate:.2f}%")
    return win_rate


def test_against_different_opponents(env, trained_agent, num_episodes=100):
    """5.4 Test przeciwko różnym przeciwnikom"""
    opponents = {
        'random': None,
        'heuristic': HeuristicBot()
    }

    results = {}
    for opponent_type, opponent in opponents.items():
        if opponent_type == 'random':
            win_rate = play_with_random_opponent(env, trained_agent, num_episodes)
        else:  # if opponent_type == 'heuristic':
            win_rate = play_with_heuristic_opponent(env, trained_agent, num_episodes)
        results[opponent_type] = win_rate

    print("\nOpponent Comparison:")
    for opponent, rate in results.items():
        print(f"Win rate vs {opponent}: {rate:.2f}%")


def agent_vs_agent(env, agent_white, agent_black, num_episodes=100):
    white_wins = 0
    black_wins = 0

    for episode in range(1, num_episodes + 1):
        with open("game_log1.txt", "a") as log_file:
            log_file.write(f"Game: {episode}\n")
        state = env.reset()
        done = False
        winner = ''
        current_player = 'white'

        while not done:
            dice_roll = env.roll_dice()
            forbidden_actions = set()
            winner = ''
            for dice in dice_roll:
                success = False
                current_agent = agent_white if current_player == 'white' else agent_black
                while not success:
                    action_idx = current_agent.choose_action(env.get_state(), dice, player=current_player,
                                                             forbidden_actions=forbidden_actions)
                    if action_idx == -1:
                        break
                    action = (action_idx, dice)
                    success, reward, done, winner = env.step(action, player=current_player, movement=False,
                                                             save_file=True)
                    if not success:
                        forbidden_actions.add(action_idx)
            # Zmiana gracza na przeciwnika
            current_player = 'black' if current_player == 'white' else 'white'

        # Rejestrowanie zwycięzcy
        if winner == 'white':
            white_wins += 1
        elif winner == 'black':
            black_wins += 1

        # Wyświetlanie postępu co 10 epizodów
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}, White Wins: {white_wins}, Black Wins: {black_wins}")

    # Obliczenie wskaźnika wygranych
    win_rate_white = (white_wins / num_episodes) * 100
    win_rate_black = (black_wins / num_episodes) * 100

    print(f"\n Wyniki pojedynku agentów:")
    print(f" Biały Agent - Win Rate: {win_rate_white:.2f}%")
    print(f" Czarny Agent - Win Rate: {win_rate_black:.2f}%")

    return win_rate_white, win_rate_black


# Konfiguracja i trening agentów
num_episodes = 1000

gamma = 0.99
epsilon = 1
lr = 0.0001
input_size = 28
batch_size = 64
n_actions = 24

env = BackgammonEnv()
agent_white = Agent(gamma, epsilon, lr, input_size, batch_size, n_actions, eps_decay=1e-5)
agent_black = Agent(gamma, epsilon, lr, input_size, batch_size, n_actions, eps_decay=1e-5)

# Ścieżka do folderu z plikami
folder_path = "./training_games"

# Parsowanie wszystkich plików w folderze
all_training_data = Agent.process_all_files(agent_white, folder_path)

# train_agent_from_data(agent_white, agent_black, all_training_data,env)

# Konwersja do tensora PyTorch
# tensor_data = T.tensor(all_training_data, dtype=T.float32)
# print("Dane wejściowe do modelu:", tensor_data.shape)


# 5.1 Test architektur sieci
# test_network_architectures(env, input_size=26, n_actions=24, num_episodes=250)

# 5.2 Test strategii eksploracji
# test_exploration_strategies(env, num_episodes=150)
# 5.3 Strojenie hiperparametrów
# hyperparameter_tuning(env)
# 5.5 Analiza jakości
# enhanced_step_analysis(env)
# Pre-trening poruszania się pionkami
print("Starting pre-training for white agent...")
# train_movement(env, agent_white, num_episodes=1000)

print("Starting pre-training for black agent...")
# train_movement(env, agent_black, num_episodes=1000)
# Gra na wstępnie nauczonych sieciach
print("Playing with pre-trained agents...")
# play_with_random_opponent(env, agent_white, num_episodes=100)
print("After Playing with pre-trained agents...")
scores = train_agent(env, agent_white, agent_black, num_episodes=num_episodes)

# Wybór lepszego bota i gra przeciwko losowemu przeciwnikowi
#white_win_rate = play_with_random_opponent(env, agent_white, num_episodes=100)
#black_win_rate = play_with_random_opponent(env, agent_black, num_episodes=100)

#better_agent = agent_white if white_win_rate > black_win_rate else agent_black
#agent_vs_agent(env, agent_white, agent_black)
#print(f"The better agent is: {'White' if better_agent == agent_white else 'Black'}")

# Lepszy bot gra przeciwko losowemu przeciwnikowi
#test_against_different_opponents(env, better_agent, num_episodes=1000)

import matplotlib.pyplot as plt

# Obliczenie średnich kroczących dla nagród białych i czarnych
rolling_avg_white = np.convolve([score[0] for score in scores], np.ones(10) / 10, mode='valid')
rolling_avg_black = np.convolve([score[1] for score in scores], np.ones(10) / 10, mode='valid')

plt.figure(figsize=(10, 6))
plt.plot(rolling_avg_white, label="White Agent (Last 10 Episodes)", color='blue')
plt.plot(rolling_avg_black, label="Black Agent (Last 10 Episodes)", color='red')

plt.xlabel('Episodes')
plt.ylabel('Total Reward (Last 10 Episodes)')
plt.title('Training Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()


def save_agent(agent, filename="agent_white.pth"):
    T.save(agent.DQN_eval_white.state_dict(), filename)
    print(f" Model zapisany jako {filename}")


save_agent(agent_white, "agent_white.pth")
save_agent(agent_black, "agent_black.pth")

"""
poprzednie choose action

     def choose_action_softmax(self, state, dice_roll, player, forbidden_actions=None):
    #     if forbidden_actions is None:
    #         forbidden_actions = set()
    #     eval_net = self.DQN_eval_white if player == 'white' else self.DQN_eval_black
    #
    #     state_tensor = T.tensor(state, dtype=T.float32).unsqueeze(0).to(eval_net.device)
    #     dice_tensor = T.tensor(dice_roll, dtype=T.float32).view(1, -1).to(eval_net.device)
    #
    #     possible_fields = self.get_possible_fields(state, player)
    #
    #     if not possible_fields:
    #         return -1  # Brak możliwych ruchów
    #
    #     with T.no_grad():
    #         q_values = eval_net.forward(state_tensor, dice_tensor).cpu().numpy().flatten()
    #
    #     # 🔹 Zabezpieczenie przed przepełnieniem
    #     q_values -= np.max(q_values)  # Normalizacja do uniknięcia overflow
    #     exp_values = np.exp(q_values / self.temp)
    #
    #     if np.sum(exp_values) == 0 or np.isnan(exp_values).any():
    #         return np.random.choice(possible_fields)  # 🔹 Unikamy błędu NaN
    #
    #     probs = exp_values / np.sum(exp_values)  # Normalizacja do sumy 1
    #
    #     return np.random.choice(range(self.n_actions), p=probs) if np.sum(probs) > 0 else np.random.choice(
    #         possible_fields)
    #
    # def choose_action_epsilon_greedy(self, state, dice_roll, player, forbidden_actions=None):
    #     if forbidden_actions is None:
    #         forbidden_actions = set()
    #     eval_net = self.DQN_eval_white if player == 'white' else self.DQN_eval_black
    #
    #     state_tensor = T.tensor(state, dtype=T.float32).unsqueeze(0).to(eval_net.device)
    #     dice_tensor = T.tensor(dice_roll, dtype=T.float32).view(1, -1).to(eval_net.device)
    #
    #     possible_fields = self.get_possible_fields(state, player)
    #     # Usuwanie zakazanych akcji
    #     possible_fields = [f for f in possible_fields if f not in forbidden_actions]
    #
    #     if not possible_fields:
    #         return -1  # Brak możliwych ruchów
    #
    #     mask = T.zeros(self.n_actions).to(eval_net.device)
    #     mask[possible_fields] = 1
    #
    #     if np.random.random() > self.epsilon:
    #         with T.no_grad():
    #             q_values = eval_net.forward(state_tensor, dice_tensor)
    #             q_values = q_values * mask  # Filtrujemy Q-wartości tylko dla możliwych ruchów
    #             action = T.argmax(q_values).item()
    #     else:
    #         action = np.random.choice(possible_fields)
    #     return action
    #
"""

# def enhanced_step_analysis(env):
#     """5.5 Rozszerzona analiza rozgrywki"""
#
#     class InstrumentedEnv(BackgammonEnv):
#         def __init__(self):
#             super().__init__()
#             self.metrics = {
#                 'moves': 0,
#                 'hits': 0,
#                 'errors': 0,
#                 'bear_offs': 0
#             }
#
#         def step(self, action, player, movement):
#             result = super().step(action, player, movement)
#             self.metrics['moves'] += 1
#             if result[1] > 0.5: self.metrics['hits'] += 1
#             if result[1] < 0: self.metrics['errors'] += 1
#             if 'bar_off' in str(action): self.metrics['bear_offs'] += 1
#             return result
#
#     instrumented_env = InstrumentedEnv()
#     agent = Agent(gamma=0.99, epsilon=0.1, lr=0.001,
#                   input_size=24, batch_size=32, n_actions=24)
#
#     play_with_random_opponent(instrumented_env, agent, 100)
#
#     print("\nGame Quality Metrics:")
#     for metric, value in instrumented_env.metrics.items():
#         print(f"{metric}: {value}")
