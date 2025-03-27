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
        self.fc1 = nn.Linear(input_size, fc1_dims, bias=False)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims, bias=False)
        self.fc3 = nn.Linear(fc2_dims, n_actions, bias=False)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

        # Zapis wag do pliku TXT
        #self.save_initial_weights_txt("initial_weights.txt")

    def save_initial_weights_txt(self, file_path):
        """Zapisuje wagi fc1, fc2 i fc3 do pliku tekstowego"""
        with open(file_path, "w") as f:
            f.write("Wagi fc1:\n")
            np.savetxt(f, self.fc1.weight.detach().cpu().numpy(), fmt="%.6f")

            f.write("\nWagi fc2:\n")
            np.savetxt(f, self.fc2.weight.detach().cpu().numpy(), fmt="%.6f")

            f.write("\nWagi fc3:\n")
            np.savetxt(f, self.fc3.weight.detach().cpu().numpy(), fmt="%.6f")

        print(f"Initial weights saved to {file_path}")

    def forward(self, state, die):
        if not isinstance(die, T.Tensor):
            die = T.tensor([die], dtype=T.float32).to(self.device)  # Konwersja rzutu na tensor

        if die.dim() == 1:
            die = die.unsqueeze(1)

        x = T.cat((state, die), dim=1)  # Konkatenacja stanu z rzutem kostką
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent:
    def __init__(self, gamma, epsilon, lr, input_size, batch_size, n_actions, max_mem=10000, eps_end=0.01,
                 eps_decay=1e-4, exploration_type='epsilon_greedy', temp=1.0):
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
        # self.exploration_type_white = 'softmax'
        self.DQN_target_white = DQN(lr=self.lr, input_size=input_size, fc1_dims=128, fc2_dims=256, n_actions=n_actions)

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
        # print(
        #     f"Storing: state={state}, actions={move_dice_pairs}, reward={reward}, next_state={next_state}, done={done}")

        self.state_memory.append(state)
        self.action_memory.append(move_dice_pairs)
        self.new_state_memory.append(next_state)
        self.reward_memory.append(np.clip(reward, -1, 1))  # Normalizacja nagrody
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

    def choose_action(self, state, dice_roll, player, forbidden_actions=None, trained=False):
        # if sum(max(0, x) for x in env.board) > 15 or sum(max(0, -x) for x in env.board) < -15:
        #     print(state)

        if forbidden_actions is None:
            forbidden_actions = set()
        if trained == 0:
            if self.exploration_type == 'softmax':
                return self.choose_action_softmax(state, dice_roll, player, forbidden_actions)
            return self.choose_action_epsilon_greedy(state, dice_roll, player, forbidden_actions)
        else:
            return self.choose_action_trained(state, dice_roll, player, forbidden_actions)

    def choose_action_epsilon_greedy(self, state, dice_roll, player, forbidden_actions=None):
        if forbidden_actions is None:
            forbidden_actions = set()

        prev_state = state
        prev_bar_w = env.bar['white']
        prev_bar_b = env.bar['black']
        eval_net = self.DQN_eval_white if player == 'white' else self.DQN_eval_black

        state_tensor = T.tensor(state, dtype=T.float32).unsqueeze(0).to(eval_net.device)

        possible_fields = self.get_possible_fields(state, player)
        possible_fields = [f for f in possible_fields if f not in forbidden_actions]
        if not possible_fields:
            env.board = prev_state
            env.bar['white'] = prev_bar_w
            env.bar['black'] = prev_bar_b
            return -1, -1

        mask = T.full((self.n_actions,), -T.inf, device=eval_net.device)  # Ustawiamy -inf zamiast 0
        mask[possible_fields] = 0
        move_dice_pair = T.zeros((1, 2)).to(eval_net.device)

        if np.random.random() < self.epsilon:
            action1 = np.random.choice(possible_fields)
        else:
            with T.no_grad():
                q_values = eval_net.forward(state_tensor, move_dice_pair)
                q_values = q_values + mask  # Filtrujemy ruchy niedozwolone
                action1 = T.argmax(q_values).item()

        moved, _, done, _ = env.step((action1, dice_roll[0]), player, movement=False)

        if not moved:
            forbidden_actions.add(action1)
            env.board = prev_state
            env.bar['white'] = prev_bar_w
            env.bar['black'] = prev_bar_b
            return -1, -1

        if done:
            env.board = prev_state
            env.bar['white'] = prev_bar_w
            env.bar['black'] = prev_bar_b
            return action1, -1

        new_state_tensor = T.tensor(env.get_state(), dtype=T.float32).unsqueeze(0).to(eval_net.device)
        possible_fields_2 = self.get_possible_fields(env.get_state(), player)

        if not possible_fields_2:
            env.board = prev_state
            env.bar['white'] = prev_bar_w
            env.bar['black'] = prev_bar_b
            return action1, -1

        mask2 = T.full((self.n_actions,), -T.inf, device=eval_net.device)  # Ustawiamy -inf zamiast 0
        mask[possible_fields] = 0

        move_dice_pair[0, :2] = T.tensor([action1, dice_roll[0]], dtype=T.float32).to(
            eval_net.device)

        if np.random.random() < self.epsilon:
            action2 = np.random.choice(possible_fields_2)
        else:
            with T.no_grad():
                q_values_2 = eval_net.forward(new_state_tensor, move_dice_pair)
                q_values_2 = q_values_2 + mask2
                action2 = T.argmax(q_values_2).item()
        env.board = prev_state
        env.bar['white'] = prev_bar_w
        env.bar['black'] = prev_bar_b
        return action1, action2

    def choose_action_softmax(self, state, dice_roll, player, forbidden_actions=None):
        if forbidden_actions is None:
            forbidden_actions = set()

        eval_net = self.DQN_eval_white if player == 'white' else self.DQN_eval_black

        state_tensor = T.tensor(state, dtype=T.float32).unsqueeze(0).to(eval_net.device)
        dice_tensor = T.tensor(dice_roll, dtype=T.float32).view(1, -1).to(eval_net.device)

        possible_fields = self.get_possible_fields(state, player)
        possible_fields = [f for f in possible_fields if f not in forbidden_actions]

        if not possible_fields:
            return -1

        with T.no_grad():
            q_values = eval_net.forward(state_tensor, dice_tensor).cpu().numpy().flatten()

        q_values_masked = np.full_like(q_values, -np.inf)
        q_values_masked[possible_fields] = q_values[
            possible_fields]

        q_values_masked -= np.max(q_values_masked)
        exp_values = np.exp(q_values_masked / self.temp)

        if np.sum(exp_values) == 0 or np.isnan(exp_values).any():
            return np.random.choice(possible_fields) if possible_fields else -1

        probs = exp_values / np.sum(exp_values)

        return np.random.choice(possible_fields, p=probs[possible_fields] / np.sum(probs[possible_fields]))

    def choose_action_trained(self, state, dice_roll, player, forbidden_actions = None):
        moved = False
        done = False
        if forbidden_actions is None:
            forbidden_actions = set()
        prev_state = state.copy()
        prev_bar_w = env.bar['white']
        prev_bar_b = env.bar['black']
        eval_net = self.DQN_eval_white if player == 'white' else self.DQN_eval_black

        state_tensor = T.tensor(state, dtype=T.float32).unsqueeze(0).to(eval_net.device)

        possible_fields = self.get_possible_fields(state, player)
        possible_fields = [f for f in possible_fields if f not in forbidden_actions]
        if not possible_fields:
            env.board = prev_state
            env.bar['white'] = prev_bar_w
            env.bar['black'] = prev_bar_b
            return -1, -1

        mask = T.full((self.n_actions,), float('-inf'), device=eval_net.device)
        mask[possible_fields] = 0

        move_dice_pair = T.zeros((1, 2)).to(eval_net.device)

        with T.no_grad():
            q_values = eval_net.forward(state_tensor, move_dice_pair)
            q_values = q_values + mask
            if q_values.max() == float('-inf'):
                env.board = prev_state
                env.bar['white'] = prev_bar_w
                env.bar['black'] = prev_bar_b
                return -1, -1
            action1 = T.argmax(q_values).item()

        moved, _, done, _ = env.step((action1, dice_roll[0]), player, movement=False)

        if not moved:
            env.board = prev_state
            env.bar['white'] = prev_bar_w
            env.bar['black'] = prev_bar_b
            forbidden_actions.add(action1)
            return -1, -1

        if done:
            env.board = prev_state
            env.bar['white'] = prev_bar_w
            env.bar['black'] = prev_bar_b
            return action1, -1

        new_state_tensor = T.tensor(env.get_state(), dtype=T.float32).unsqueeze(0).to(eval_net.device)
        possible_fields_2 = self.get_possible_fields(env.get_state(), player)

        if not possible_fields_2:
            env.board = prev_state
            env.bar['white'] = prev_bar_w
            env.bar['black'] = prev_bar_b
            return action1, -1

        mask2 = T.full((self.n_actions,), float('-inf'), device=eval_net.device)
        mask2[possible_fields_2] = 0

        move_dice_pair[0, :2] = T.tensor([action1, dice_roll[0]], dtype=T.float32).to(eval_net.device)

        with T.no_grad():
            q_values_2 = eval_net.forward(new_state_tensor, move_dice_pair)
            q_values_2 = q_values_2 + mask2
            if q_values_2.max() == float('-inf'):
                env.board = prev_state
                env.bar['white'] = prev_bar_w
                env.bar['black'] = prev_bar_b
                return action1, -1
            action2 = T.argmax(q_values_2).item()

        env.board = prev_state
        env.bar['white'] = prev_bar_w
        env.bar['black'] = prev_bar_b
        return action1, action2

    def learn(self, player):
        if self.mem_pointer < self.batch_size:
            return

        eval_net = self.DQN_eval_white if player == 'white' else self.DQN_eval_black
        target_net = self.DQN_target_white if player == 'white' else self.DQN_target_black

        eval_net.optimizer.zero_grad()

        max_mem = min(self.mem_pointer, self.max_mem)
        batch_indices = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch = T.tensor(np.array(self.state_memory)[batch_indices], dtype=T.float32).to(eval_net.device)
        new_state_batch = T.tensor(np.array(self.new_state_memory)[batch_indices], dtype=T.float32).to(eval_net.device)
        reward_batch = T.tensor(np.array(self.reward_memory)[batch_indices], dtype=T.float32).to(eval_net.device)
        terminal_batch = T.tensor(np.array(self.terminal_memory)[batch_indices], dtype=T.float32).to(eval_net.device)

        move_dice_batch_1 = state_batch[:, -4:-2]  # Pierwszy ruch
        move_dice_batch_2 = state_batch[:, -2:]  # Drugi ruch

        q_eval_1 = eval_net(state_batch, move_dice_batch_1).max(1)[0]
        q_eval_2 = eval_net(state_batch, move_dice_batch_2).max(1)[0]

        q_next_1 = target_net(new_state_batch, move_dice_batch_1).max(1)[0]
        q_next_2 = target_net(new_state_batch, move_dice_batch_2).max(1)[0]

        q_next_1[terminal_batch == 1] = 0.0
        q_next_2[terminal_batch == 1] = 0.0

        q_target_1 = reward_batch + self.gamma * q_next_1
        q_target_2 = reward_batch + self.gamma * q_next_2

        loss_1 = eval_net.loss(q_eval_1, q_target_1)
        loss_2 = eval_net.loss(q_eval_2, q_target_2)

        loss = (loss_1 + loss_2) / 2
        loss.backward()

        eval_net.optimizer.step()
        self.epsilon = max(self.eps_end, self.epsilon - self.eps_decay)

    def update_target_network(self, player):

        t = 0.01
        if player == 'white':
            for target_param, eval_param in zip(self.DQN_target_white.parameters(), self.DQN_eval_white.parameters()):
                target_param.data.copy_(t * eval_param.data + (1 - t) * target_param.data)
        else:
            for target_param, eval_param in zip(self.DQN_target_black.parameters(), self.DQN_eval_black.parameters()):
                target_param.data.copy_(t * eval_param.data + (1 - t) * target_param.data)

    def clean_line(self, line):

        line = re.sub(r'[^0-9a-zA-Z\s/()-]', '', line)  # Usunięcie znaków specjalnych
        return line.strip()

    def parse_moves(self, file_path):

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

        encoded = []
        actions = move.split()
        for action in actions:
            match = re.match(r'(\d+|bar|bear_off)/(\d+|bar|bear_off)', action)
            if match:
                from_pos, to_pos = match.groups()
                encoded.append((from_pos, to_pos))
        return encoded

    def prepare_training_data(self, moves):

        training_data = []
        for (p1_dice1, p1_dice2, p1_move), (p2_dice1, p2_dice2, p2_move) in moves:
            encoded_p1 = self.encode_move(p1_move)
            encoded_p2 = self.encode_move(p2_move)
            training_data.append(((p1_dice1, p1_dice2), encoded_p1, (p2_dice1, p2_dice2), encoded_p2))

        return training_data

    def process_all_files(self, folder_path):

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
            return sum(pos * abs(self.board[pos]) for pos in range(24) if self.board[pos] > 0) + 25 * self.bar['white']
        else:
            return sum(pos * abs(self.board[pos]) for pos in range(24) if self.board[pos] < 0) + 25 * self.bar['black']

    def step(self, action, player, movement, save_file=False):
        from_pos, dice_roll = action
        total_reward = 0
        done = False
        state = self.board
        prev_pips = env.calculate_pips(player)
        to_pos = from_pos + dice_roll if player == 'white' else from_pos - dice_roll

        # Nagroda za wejście na ostatnie pola planszy
        if (player == 'white' and to_pos >= 25) or (player == 'black' and to_pos <= 0):
            total_reward += 0.5  # Zwiększona nagroda

        success = self.move(from_pos, dice_roll, player)

        if not success:
            return False, 0, False, ''

        if 0 <= to_pos <= 23:
            if abs(state[to_pos]) == 1 and (
                    (player == 'white' and state[to_pos] < 0) or
                    (player == 'black' and state[to_pos] > 0)
            ):
                total_reward += 0.5  # Większa nagroda za bicie pionka przeciwnika

            if abs(env.board[to_pos]) == 1:
                total_reward -= 0.05  # Zmniejszona kara

            #total_reward += 0.1  # Zwiększona nagroda za ruch

        # Nagroda za zmniejszenie pipów (bardziej motywuje do przesuwania pionków)
        total_reward += 0.02 * (prev_pips - env.calculate_pips(player))

        # Nagroda za wygraną
        if player == 'white' and self.count_positive_fields() == 0:
            return True, total_reward + 3, True, 'white'  # Większa nagroda za wygraną

        elif player == 'black' and self.count_negative_fields() == 0:
            return True, total_reward + 3, True, 'black'

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
                #self.bar['white'] -= 1
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
                if from_pos == 0 and self.bar['white'] > 0:
                    return True
                return False
        if player == 'black':
            if self.board[from_pos] == 0:
                if self.bar['black'] > 0 and from_pos == 23:
                    return True
                else:
                    return False
            if self.board[from_pos] > 0:
                if from_pos == 23 and self.bar['black'] > 0:
                    return True
                return False
        if 0 <= to_pos <= 23:
            if player == 'white':
                if self.board[to_pos] < -1:
                    return False

            if player == 'black':
                if self.board[to_pos] > 1:
                    return False

            # return True
        return True

    def get_state(self):
        return np.array(self.board, dtype=np.float32)


def train_agent(env, agent_white, agent_black, num_episodes):
    temp = 0
    scores = []

    for episode in range(1, num_episodes + 1):
        # print("Episode ", episode)
        env.reset()
        done = False
        total_reward_white = 0
        total_reward_black = 0
        reward_white = 0
        reward_black = 0
        total_reward = 0
        moves_white = 0
        moves_black = 0
        current_player = random.choice(['black', 'white'])
        wmove1 = [0, 0]
        wmove2 = [0, 0]
        bmove1 = [0, 0]
        bmove2 = [0, 0]
        transitions_white = []
        transitions_black = []
        white_prev_state = env.get_state().copy()
        black_prev_state = env.get_state().copy()
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
            barw = False
            barb = False
            if current_player == 'white':
                white_prev_state = env.get_state().copy()
            else:
                black_prev_state = env.get_state().copy()
            if current_player == 'white' and env.bar['white'] > 0:
                barw = True
            if current_player == 'black' and env.bar['black'] > 0:
                barb = True
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
                    # reward -= 0.5
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
                #if barw == False:
                    transitions_white.append(
                        (black_prev_state, wmove1[0], wmove1[1], wmove2[0], wmove2[1], reward_white - reward_black,
                         env.board, done))
                    total_reward_black += reward_black
            elif current_player == 'white' and bmove1 != [0, 0] and bmove2 != [0, 0]:
                #if barb == False:
                    transitions_black.append(
                        (white_prev_state, bmove1[0], bmove1[1], bmove2[0], bmove2[1], reward_black - reward_white,
                         env.board, done))
                    total_reward_white += reward_white
            # Nauka po dwóch ruchach dla białego

            # Jeśli gra się skończyła, nie zmieniamy tury
            if not done:
                current_player = 'black' if current_player == 'white' else 'white'
                if current_player == 'white':
                    moves_white += 1
                else:
                    moves_black += 1
            temp += 1
        # print(moves_white)
        # print(moves_black)
        for transition in transitions_white:
            agent_white.store_transition(*transition)
            agent_white.learn('white')
        transitions_white.clear()

        for transition in transitions_black:
            agent_black.store_transition(*transition)
            agent_black.learn('black')
        transitions_black.clear()

        scores.append((total_reward, total_reward_white, total_reward_black))

        if episode % 10 == 0:
            agent_white.update_target_network('white')
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
                f"Moves white: {moves_white}, "
                f"Moves black: {moves_black}"
            )
    return scores


def random_player_action(env, die, agent):
    """
    Losowy ruch gracza (dla losowego przeciwnika) na podstawie możliwych ruchów.
    Jeśli agent jest podany, używa jego metody get_possible_fields, aby określić dostępne ruchy.
    """
    state = env.get_state()
    player = env.turn

    possible_fields = agent.get_possible_fields(state, player)

    random.shuffle(possible_fields)

    if not possible_fields:
        return None

    for from_pos in possible_fields:
        if env.move(from_pos, die, player):
            return True
        else:
            possible_fields.remove(from_pos)

    return False


def play_with_random_opponent(env, agent, num_episodes, start):
    """
    Pozwala botowi grać przeciwko losowemu graczowi przez określoną liczbę epizodów.
    """
    agent_wins = 0
    bot_wins = 0
    for episode in range(1, num_episodes + 1):
        # print(f"Episode")
        state = env.reset()
        env.turn = start
        done = False
        winner = ''
        while not done:
            dice_roll = env.roll_dice()
            success1 = False
            success2 = False
            forbidden_moves1 = set()
            forbidden_moves2 = set()
            # print(f"Current turn: {env.turn}")  # Debug: sprawdzanie zmiany tury
            if env.turn == 'white':
                while not success1 and not success2:
                    action_idx1, action_idx2 = agent.choose_action(env.get_state(), dice_roll,
                                                                   player="white",
                                                                   forbidden_actions=forbidden_moves1, trained=True)
                    if action_idx1 == -1:
                        break
                    from_pos = action_idx1
                    action = (from_pos, dice_roll[0])

                    success1, reward, done, winner1 = env.step(action, player='white', movement=False)
                    if not success1:
                        forbidden_moves1.add(from_pos)
                    if done:
                        winner = 'white'
                        break

                    if action_idx2 == -1:
                        break
                    from_pos = action_idx2
                    action = (from_pos, dice_roll[1])

                    success2, reward, done, winner1 = env.step(action, player='white', movement=False)
                    if not success2:
                        # reward -= 0.5
                        forbidden_moves2.add(from_pos)
                    if done:
                        winner = 'white'
            else:
                for die in dice_roll:
                    if not random_player_action(env, die, agent):
                        break
                if env.count_negative_fields() == 0:
                    done = True
                    winner = 'black'
            env.turn = 'black' if env.turn == 'white' else 'white'

        if winner == 'white':  # Wygrana białych (agenta)
            agent_wins += 1

        if winner == 'black':
            bot_wins += 1

        # Wyświetl wynik co 10 gier
        # if episode % 10 == 0:
        #     print(f"Episode {episode}/{num_episodes}, Agent Wins: {agent_wins}, Bot Wins:{bot_wins}")

    win_rate = (agent_wins / num_episodes) * 100
    #print(f"Agent Win Rate: {win_rate:.2f}%")
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

        def choose_action(self, state, dice_roll, player, forbidden_actions=None, trained=False):
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
                              input_size=26, batch_size=32, n_actions=24)

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
                              input_size=26, batch_size=batch_size, n_actions=24)

                scores = train_agent(env, agent, agent, 50)
                mean_score = np.mean(scores[-10:])

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {'gamma': gamma, 'lr': lr, 'batch_size': batch_size}

    print(f"\nBest params: {best_params} with score: {best_score:.2f}")


class HeuristicBot:
    def choose_action(self, state, player, forbidden_actions=None):

        if forbidden_actions is None:
            forbidden_actions = set()
        possible_fields = None
        if env.bar[player] > 0:
            if player == 'white':
                possible_fields = [0]
            elif player == 'black':
                possible_fields = [23]
        else:
            possible_fields = [i for i, val in enumerate(state) if (val > 0 if player == 'white' else val < 0)]
        move = None
        while possible_fields:  # Sprawdzamy, czy possible_fields NIE jest puste
            if player == 'white':
                move = min(possible_fields)
            else:
                move = max(possible_fields)

            if move in forbidden_actions:
                possible_fields.remove(move)
            else:
                return move

        return None


def play_with_heuristic_opponent(env, agent, num_episodes,start):
    heuristic_bot = HeuristicBot()
    agent_wins = 0
    bot_wins = 0

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        current_player = start

        while not done:
            dice_roll = env.roll_dice()
            forbidden_moves1 = set()
            forbidden_moves2 = set()
            success1 = False
            success2 = False
            winner = ''
            if current_player == 'white':
                while not success1 and not success2:
                    action_idx1, action_idx2 = agent.choose_action(env.get_state(), dice_roll,
                                                                   player=current_player,
                                                                   forbidden_actions=forbidden_moves1, trained=True)
                    if action_idx1 == -1:
                        break
                    from_pos = action_idx1
                    action = (from_pos, dice_roll[0])

                    success1, reward, done, winner1 = env.step(action, player=current_player, movement=False)
                    if not success1:
                        forbidden_moves1.add(from_pos)
                    if done:
                        winner = current_player
                        break

                    if action_idx2 == -1:
                        break
                    from_pos = action_idx2
                    action = (from_pos, dice_roll[1])

                    success2, reward, done, winner1 = env.step(action, player=current_player, movement=False)
                    if not success2:
                        # reward -= 0.5
                        forbidden_moves2.add(from_pos)
                    if winner1:
                        winner = 'white'
            else:
                for dice in dice_roll:
                    success = False
                    forbidden_moves = set()
                    while not success:
                        action_idx = heuristic_bot.choose_action(env.get_state(), player=current_player,
                                                                 forbidden_actions=forbidden_moves)
                        if action_idx == None:
                            break
                        else:
                            action = (action_idx, dice)
                            success, reward, done, winner = env.step(action, player=current_player, movement=False)
                            if not success:
                                forbidden_moves.add(action_idx)

            if done:
                if winner == 'white':
                    agent_wins += 1
                elif winner == 'black':
                    bot_wins += 1
            current_player = 'black' if current_player == 'white' else 'white'

    win_rate = (agent_wins / num_episodes) * 100
    #print(f"Win Rate vs Heuristic Bot: {win_rate:.2f}%")
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
            win_rate = play_with_random_opponent(env, trained_agent, num_episodes,'white')
            print(f"White start win rate vs {opponent_type}: {win_rate:.2f}%")
            win_rate = play_with_random_opponent(env, trained_agent, num_episodes,'black')
            print(f"Black start win rate vs {opponent_type}: {win_rate:.2f}%")

        else:  # if opponent_type == 'heuristic':
            win_rate = play_with_heuristic_opponent(env, trained_agent, num_episodes,'white')
            print(f"White start win rate vs {opponent_type}: {win_rate:.2f}%")
            win_rate = play_with_heuristic_opponent(env, trained_agent, num_episodes, 'black')
            print(f"Black start win rate vs {opponent_type}: {win_rate:.2f}%")

        results[opponent_type] = win_rate

    # print("\nOpponent Comparison:")
    # for opponent, rate in results.items():
    #     print(f"Win rate vs {opponent}: {rate:.2f}%")


def play_random_vs_heuristic(env, num_episodes):
    """
    Pozwala losowemu botowi grać przeciwko botowi heurystycznemu przez określoną liczbę epizodów.
    """
    heuristic_bot = HeuristicBot()
    random_wins = 0
    heuristic_wins = 0

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        current_player = 'black'#random.choice(['white', 'black'])  # Losowo wybiera, kto zaczyna
        winner = ''

        while not done:
            dice_roll = env.roll_dice()
            forbidden_actions = set()
            for dice in dice_roll:
                success = False
                forbidden_actions.clear()

                if current_player == 'white':
                    # Losowy bot wybiera ruchy
                    if current_player == 'white':
                        # Jeśli biali mają pionki na barze, to mogą je tylko ściągnąć z baru na planszę
                        if env.bar['white'] > 0:
                            possible_moves = [
                                0]  # [i for i in range(24) if board[i] >= 0]  # Pola, na których nie ma przeciwnika
                        else:
                            # Zwróć pola, z których gracz może wykonać ruch, jeśli nie ma pionków na barze
                            possible_moves = [i for i, value in enumerate(env.board) if value > 0]  # Pionki białe
                    else:
                        # Jeśli czarni mają pionki na barze, to mogą je tylko ściągnąć z baru na planszę
                        if env.bar['black'] > 0:
                            possible_moves = [
                                23]  # [i for i in range(24) if board[i] <= 0]  # Pola, na których nie ma przeciwnika
                        else:
                            # Zwróć pola, z których gracz może wykonać ruch, jeśli nie ma pionków na barze
                            possible_moves = [i for i, value in enumerate(env.board) if value < 0]  # Pionki czarne

                    if not possible_moves:
                         continue

                    random.shuffle(possible_moves)
                    for from_pos in possible_moves:
                        if env.move(from_pos, dice, current_player):
                            continue
                        else:
                            possible_moves.remove(from_pos)

            else:
                for dice in dice_roll:
                    # Heurystyczny bot wykonuje ruch
                    action_idx = heuristic_bot.choose_action(env.get_state(), player=current_player)
                    if action_idx is not None:
                        action = (action_idx, dice)
                        success, _, done, winner = env.step(action, player=current_player, movement=False)

            # Przełączanie gracza
            current_player = 'black' if current_player == 'white' else 'white'

        # Aktualizacja wyników
        if winner == 'white':
            random_wins += 1
        elif winner == 'black':
            heuristic_wins += 1

        # Co 10 gier pokazujemy postęp
        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1}/{num_episodes} - Random Bot Wins: {random_wins}, Heuristic Bot Wins: {heuristic_wins}")

    win_rate = (random_wins / num_episodes) * 100
    print(f"Random Bot Win Rate vs Heuristic Bot: {win_rate:.2f}%")
    return win_rate


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
            # for i in range(2):  # Każdy gracz wykonuje ruch
            dice_roll = env.roll_dice()
            current_agent = agent_white if current_player == 'white' else agent_black
            # i=0
            success1 = False
            success2 = False
            forbidden_moves1 = set()
            forbidden_moves2 = set()
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
                    forbidden_moves1.add(from_pos)

                if done:
                    break

                if action_idx2 == -1:
                    break
                from_pos = action_idx2
                action = (from_pos, dice_roll[1])

                success2, reward, done, winner = env.step(action, player=current_player, movement=False)
                if not success2:
                    # reward -= 0.5
                    forbidden_moves2.add(from_pos)
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


def print_board(board: list) -> str:
    board = list(board)
    def format_one_number(i: int) -> str:
        """Return a string of length 3. Spaces only if 0. Number with sign (+/-) otherwise."""
        if i == 0:
            return "   "
        elif i < 0:
            return f"{int(i):3}"
        else:
            return ("+" + str(int(i))).rjust(3)

    assert type(board) == list
    assert len(board) == 24

    bar = "     "

    first_line = "".join(f"{i:3}" for i in range(12, 18)) + bar + "".join(f"{i:3}" for i in range(18, 24))
    # print(f"first_line = {first_line}")

    part1, part2 = board[12:18], board[18:24]
    part3, part4 = board[11:5:-1], board[5::-1]
    part1str = "".join(format_one_number(i) for i in part1)
    part2str = "".join(format_one_number(i) for i in part2)
    second_line = part1str + bar + part2str

    third_line = ""

    part3str = "".join(format_one_number(i) for i in part3)
    part4str = "".join(format_one_number(i) for i in part4)
    fourth_line = part3str + bar + part4str

    fifth_line = "".join(f"{i:3}" for i in range(11, 5, -1)) + bar + "".join(f"{i:3}" for i in range(5, -1, -1))

    whole_string = "\n".join([first_line, second_line, third_line, fourth_line, fifth_line]) + "\n"
    return whole_string


def play_game(agent):
    done = False
    env.reset()
    dice_roll = env.roll_dice()
    print(f"Rzut kośćmi: {dice_roll}")

    if dice_roll[0] >= dice_roll[1]:
        player = 'white'
    else:
        player = 'black'

    winner = ''

    while not done:
        dice_roll = env.roll_dice()
        print('----------------------------------------------')


        if player == 'white':
            success1 = False
            success2 = False
            print(f"Tura BIAŁYCH")
            print(f"Rzut kośćmi: {dice_roll}")
            print(print_board(env.board))
            if env.bar['white'] > 0:
                possible_moves = [0]
            else:
                possible_moves = [i for i, value in enumerate(env.board) if value > 0]

            print(f"Możliwe pola do ruchu: {possible_moves}")

            while not success1:
                try:
                    from_pos = int(input(f"Kostka: {dice_roll[0]} ➝ Wpisz pole dla 1. ruchu: "))
                    if from_pos not in possible_moves:
                        print("Ruch niedozwolony, spróbuj ponownie.")
                        continue
                    success1 = env.move(from_pos, dice_roll[0], 'white')
                except ValueError:
                    print("Wprowadź poprawny numer pola!")

            print(print_board(env.board))
            possible_moves = [i for i, value in enumerate(env.board) if value > 0]
            print(f"Możliwe pola po ruchu 1: {possible_moves}")

            if env.count_positive_fields() == 0:
                done = True
                winner = 'white'
                break

            success2 = False

            while not success2:
                try:
                    from_pos = int(input(f"Kostka: {dice_roll[1]} ➝ Wpisz pole dla 2. ruchu: "))
                    if from_pos not in possible_moves:
                        print("Ruch niedozwolony, spróbuj ponownie.")
                        continue
                    success2 = env.move(from_pos, dice_roll[1], 'white')
                except ValueError:
                    print("Wprowadź poprawny numer pola!")

            print(print_board(env.board))
            # possible_moves = [i for i, value in enumerate(env.board) if value > 0]
            # print(f"Możliwe pola po ruchu 2: {possible_moves}")

            if env.count_positive_fields() == 0:
                done = True
                winner = 'white'
                break

        else:  # CZARNY (Bot)
            print(f"Tura CZARNYCH (Bot)")
            print(f"Rzut kośćmi: {dice_roll}")
            print(print_board(env.board))
            if env.bar['black'] > 0:
                possible_moves = [23]
            else:
                possible_moves = [i for i, value in enumerate(env.board) if value < 0]

            print(f"Bot: Możliwe pola do ruchu: {possible_moves}")

            success1 = False
            success2 = False
            forbidden_moves1 = set()
            forbidden_moves2 = set()

            while not success1 and not success2:
                action_idx1, action_idx2 = agent.choose_action(env.get_state(), dice_roll,
                                                               player=player,
                                                               forbidden_actions=forbidden_moves1, trained=True)
                if action_idx1 == -1:
                    break
                success1, reward, done, winner1 = env.step((action_idx1, dice_roll[0]), player=player, movement=False)
                if not success1:
                    forbidden_moves1.add(action_idx1)
                else:
                    print(f"Bot ruch 1: z {action_idx1} o {dice_roll[0]} pól")

                print(print_board(env.board))
                possible_moves = [i for i, value in enumerate(env.board) if value < 0]
                print(f"Bot Możliwe pola po ruchu 1: {possible_moves}")

                if action_idx2 == -1:
                    break
                success2, reward, done, winner1 = env.step((action_idx2, dice_roll[1]), player=player, movement=False)
                if not success2:
                    forbidden_moves2.add(action_idx2)
                else:
                    print(f"Bot ruch 2 z {action_idx2} o {dice_roll[1]} pól")

                #print(print_board(env.board))  # Wyświetl planszę po ruchu 2
                # possible_moves = [i for i, value in enumerate(env.board) if value < 0]
                # print(f"Bot Możliwe pola po ruchu 2: {possible_moves}")

                if winner1:
                    winner = player

        player = 'white' if player == 'black' else 'black'

    print(f"Zwycięzca: {winner}!")


def random_vs_random(env, agent, num_episodes):
    """
    Pozwala botowi grać przeciwko losowemu graczowi przez określoną liczbę epizodów.
    """
    randomw = 0
    randomb = 0

    for episode in range(1, num_episodes + 1):
        # print(f"Episode")
        state = env.reset()
        done = False
        winner = ''
        while not done:
            dice_roll = env.roll_dice()
            # print(f"Current turn: {env.turn}")  # Debug: sprawdzanie zmiany tury
            if env.turn == 'white':
                for die in dice_roll:
                    if not random_player_action(env, die, agent):
                        continue
                if env.count_positive_fields() == 0:
                    done = True
                    winner = 'white'
            else:
                for die in dice_roll:
                    if not random_player_action(env, die, agent):
                        continue
                if env.count_negative_fields() == 0:
                    done = True
                    winner = 'black'
            env.turn = 'black' if env.turn == 'white' else 'white'

        if winner == 'white':  # Wygrana białych (agenta)
            randomw += 1

        if winner == 'black':
            randomb += 1

        # Wyświetl wynik co 10 gier
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}, White Wins: {randomw}, Black Wins:{randomb}")

    #win_rate = (agent_wins / num_episodes) * 100
    #print(f"Agent Win Rate: {win_rate:.2f}%")
    #return win_rate


def play_with_random_opponent_agent_forget(env, num_episodes):
    """
    Pozwala botowi grać przeciwko losowemu graczowi przez określoną liczbę epizodów.
    """
    agent_wins = 0
    bot_wins = 0

    for episode in range(1, num_episodes + 1):
        agent = Agent(0.95, 1, 0.001, 26, 16, 24, eps_decay=1e-5)
        # print(f"Episode")
        state = env.reset()
        done = False
        winner = ''
        while not done:
            dice_roll = env.roll_dice()
            success1 = False
            success2 = False
            forbidden_moves1 = set()
            forbidden_moves2 = set()
            # print(f"Current turn: {env.turn}")  # Debug: sprawdzanie zmiany tury
            if env.turn == 'white':
                while not success1 and not success2:
                    action_idx1, action_idx2 = agent.choose_action(env.get_state(), dice_roll,
                                                                   player="white",
                                                                   forbidden_actions=forbidden_moves1, trained=True)
                    if action_idx1 == -1:
                        break
                    from_pos = action_idx1
                    action = (from_pos, dice_roll[0])

                    success1, reward, done, winner1 = env.step(action, player='white', movement=False)
                    if not success1:
                        forbidden_moves1.add(from_pos)
                    if done:
                        winner = 'white'
                        break

                    if action_idx2 == -1:
                        break
                    from_pos = action_idx2
                    action = (from_pos, dice_roll[1])

                    success2, reward, done, winner1 = env.step(action, player='white', movement=False)
                    if not success2:
                        # reward -= 0.5
                        forbidden_moves2.add(from_pos)
                    if done:
                        winner = 'white'
            else:
                for die in dice_roll:
                    if not random_player_action(env, die, agent):
                        continue
                if env.count_negative_fields() == 0:
                    done = True
                    winner = 'black'
            env.turn = 'black' if env.turn == 'white' else 'white'

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

def random_player_action_file(env, die, agent,file):
    """
    Losowy ruch gracza (dla losowego przeciwnika) na podstawie możliwych ruchów.
    Jeśli agent jest podany, używa jego metody get_possible_fields, aby określić dostępne ruchy.
    """
    state = env.get_state()
    player = env.turn

    possible_fields = agent.get_possible_fields(state, player)

    random.shuffle(possible_fields)

    if not possible_fields:
        return None

    for from_pos in possible_fields:
        if env.move(from_pos, die, player):
            file.write(f"Black move: from {from_pos} by {die} steps\n")
            return True
        else:
            possible_fields.remove(from_pos)

    return False


def play_with_random_opponent_file(env, agent, num_episodes, filename="game_log_full.txt"):
    """
    Pozwala botowi grać przeciwko losowemu graczowi przez określoną liczbę epizodów.
    Zapisywane są wszystkie ruchy w formacie funkcji print_board do pliku.
    """
    agent_wins = 0
    bot_wins = 0

    with open(filename, "w") as file:  # Otwieramy plik w trybie zapisu (nadpisuje poprzednią zawartość)
        for episode in range(1, num_episodes + 1):
            file.write(f"\n===== Episode {episode} =====\n")  # Nagłówek epizodu
            env.reset()
            done = False
            winner = ''

            while not done:
                dice_roll = env.roll_dice()
                file.write("\nRzut kośćmi: " + str(dice_roll) + "\n")  # Zapis rzutów kości
                file.write(print_board(env.board))  # Zapis aktualnej planszy
                file.write("\nBar białe: " + str(env.bar['white']))
                file.write("\nBar czarne: " + str(env.bar['black'])+"\n")
                success1 = False
                success2 = False
                forbidden_moves1 = set()
                forbidden_moves2 = set()

                if env.turn == 'white':
                    while not success1 and not success2:
                        action_idx1, action_idx2 = agent.choose_action(env.get_state(), dice_roll,
                                                                       player="white",
                                                                       forbidden_actions=forbidden_moves1, trained=True)
                        if action_idx1 == -1:
                            break
                        from_pos = action_idx1
                        action = (from_pos, dice_roll[0])

                        success1, reward, done, winner1 = env.step(action, player='white', movement=False)
                        if not success1:
                            forbidden_moves1.add(from_pos)
                        else:
                            file.write(f"White move 1: from {action_idx1} by {dice_roll[0]} steps\n")

                        if done:
                            winner = 'white'
                            break

                        if action_idx2 == -1:
                            break
                        from_pos = action_idx2
                        action = (from_pos, dice_roll[1])

                        success2, reward, done, winner1 = env.step(action, player='white', movement=False)
                        if not success2:
                            forbidden_moves2.add(from_pos)
                        else:
                            file.write(f"White move 2: from {action_idx2} by {dice_roll[1]} steps\n")

                        if done:
                            winner = 'white'
                else:
                    for die in dice_roll:
                        if not random_player_action_file(env, die, agent,file):
                            continue
                    #file.write("Black moves (Random Player)\n")
                    if env.count_negative_fields() == 0:
                        done = True
                        winner = 'black'

                env.turn = 'black' if env.turn == 'white' else 'white'

            if winner == 'white':  # Wygrana białych (agenta)
                agent_wins += 1
                file.write("White wins!\n")
            elif winner == 'black':
                bot_wins += 1
                file.write("Black wins!\n")

            # Wyświetl wynik co 10 gier
            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes}, Agent Wins: {agent_wins}, Bot Wins:{bot_wins}")

    win_rate = (agent_wins / num_episodes) * 100
    print(f"Agent Win Rate: {win_rate:.2f}%")
    return win_rate

def play_with_heuristic_opponent_file(env, agent, num_episodes, filename="heuristic_game_log.txt"):
    """
    Pozwala botowi grać przeciwko heurystycznemu graczowi przez określoną liczbę epizodów.
    Zapisywane są wszystkie ruchy w formacie funkcji print_board do pliku.
    """
    heuristic_bot = HeuristicBot()
    agent_wins = 0
    bot_wins = 0

    with open(filename, "w") as file:  # Otwieramy plik w trybie zapisu (nadpisuje poprzednią zawartość)
        for episode in range(1, num_episodes + 1):
            file.write(f"\n===== Episode {episode} =====\n")  # Nagłówek epizodu
            env.reset()
            done = False
            winner = ''

            while not done:
                dice_roll = env.roll_dice()
                file.write("\nRzut kośćmi: " + str(dice_roll) + "\n")  # Zapis rzutów kości
                file.write(print_board(env.board))  # Zapis aktualnej planszy
                file.write("\nBar białe: " + str(env.bar['white']))
                file.write("\nBar czarne: " + str(env.bar['black']) + "\n")

                success1 = False
                success2 = False
                forbidden_moves1 = set()
                forbidden_moves2 = set()

                if env.turn == 'white':
                    while not success1 and not success2:
                        action_idx1, action_idx2 = agent.choose_action(env.get_state(), dice_roll,
                                                                       player="white",
                                                                       forbidden_actions=forbidden_moves1, trained=True)
                        if action_idx1 == -1:
                            break
                        from_pos = action_idx1
                        action = (from_pos, dice_roll[0])

                        success1, reward, done, winner1 = env.step(action, player='white', movement=False)
                        if not success1:
                            forbidden_moves1.add(from_pos)
                        else:
                            file.write(f"White move 1: from {action_idx1} by {dice_roll[0]} steps\n")

                        if done:
                            winner = 'white'
                            break

                        if action_idx2 == -1:
                            break
                        from_pos = action_idx2
                        action = (from_pos, dice_roll[1])

                        success2, reward, done, winner1 = env.step(action, player='white', movement=False)
                        if not success2:
                            forbidden_moves2.add(from_pos)
                        else:
                            file.write(f"White move 2: from {action_idx2} by {dice_roll[1]} steps\n")

                        if done:
                            winner = 'white'
                else:
                    for die in dice_roll:
                        success = False
                        forbidden_moves = set()
                        while not success:
                            action_idx = heuristic_bot.choose_action(env.get_state(), player="black",
                                                                     forbidden_actions=forbidden_moves)
                            if action_idx is None:
                                break
                            else:
                                action = (action_idx, die)
                                success, reward, done, winner = env.step(action, player="black", movement=False)
                                if not success:
                                    forbidden_moves.add(action_idx)
                                else:
                                    file.write(f"Black move: from {action_idx} by {die} steps\n")

                env.turn = 'black' if env.turn == 'white' else 'white'

            if winner == 'white':  # Wygrana białych (agenta)
                agent_wins += 1
                file.write("White wins!\n")
            elif winner == 'black':
                bot_wins += 1
                file.write("Black wins!\n")

            # Wyświetl wynik co 10 gier
            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes}, Agent Wins: {agent_wins}, Bot Wins: {bot_wins}")

    win_rate = (agent_wins / num_episodes) * 100
    print(f"Agent Win Rate vs Heuristic Bot: {win_rate:.2f}%")
    return win_rate

# Konfiguracja i trening agentów
num_episodes = 5000

gamma = 0.95
epsilon = 1
lr = 0.001
input_size = 26
batch_size = 64
n_actions = 24

env = BackgammonEnv()
#play_random_vs_heuristic(env, 1000)
agent_white = Agent(gamma, epsilon, lr, input_size, batch_size, n_actions, eps_decay=1e-5)
agent_black = Agent(gamma, epsilon, lr, input_size, batch_size, n_actions, eps_decay=1e-5)

# 5.1 Test architektur sieci
#test_network_architectures(env, input_size=26, n_actions=24, num_episodes=250)

# 5.2 Test strategii eksploracji
# test_exploration_strategies(env, num_episodes=150)
# 5.3 Strojenie hiperparametrów
#hyperparameter_tuning(env)
# 5.5 Analiza jakości
# enhanced_step_analysis(env)
# Pre-trening poruszania się pionkami
print("Starting pre-training for white agent...")
# train_movement(env, agent_white, num_episodes=1000)
print("Starting pre-training for black agent...")
# train_movement(env, agent_black, num_episodes=1000)
# Gra na wstępnie nauczonych sieciach
# print("Playing with pre-trained agent...")
print("white")
test_against_different_opponents(env, agent_white, num_episodes=1000)
print("black")
test_against_different_opponents(env, agent_black, num_episodes=1000)
# print("After Playing with pre-trained agents...")
scores = train_agent(env, agent_white, agent_black, num_episodes=num_episodes)

# Wybór lepszego bota i gra przeciwko losowemu przeciwnikowi
print('white')
white_win_rate = test_against_different_opponents(env, agent_white, num_episodes=1000)
white_win_rate = test_against_different_opponents(env, agent_white, num_episodes=1000)
print('black')
black_win_rate = test_against_different_opponents(env, agent_black, num_episodes=1000)
agw, agb = agent_vs_agent(env, agent_white, agent_black, 1000)
better_agent = agent_white if agw > agb else agent_black
print(f"The better agent is: {'White' if better_agent == agent_white else 'Black'}")

#Lepszy bot gra przeciwko losowemu przeciwnikowi
test_against_different_opponents(env, better_agent, num_episodes=1000)

import matplotlib.pyplot as plt


# Nagrody dla każdego epizodu
# Nagrody dla każdego epizodu
white_rewards = [score[0] for score in scores]
black_rewards = [score[1] for score in scores]

# Skumulowane sumy nagród (kumulatywna suma)
cumulative_white_rewards = np.cumsum(white_rewards)
cumulative_black_rewards = np.cumsum(black_rewards)

# Średnie kroczące (ostatnie 10 epizodów)
rolling_avg_white = np.convolve(white_rewards, np.ones(10) / 10, mode='valid')
rolling_avg_black = np.convolve(black_rewards, np.ones(10) / 10, mode='valid')

# Różnica nagród (biały - czarny)
rolling_avg_difference = rolling_avg_white - rolling_avg_black

# 🔹 Wykres 1: Średnie kroczące (ostatnie 10 epizodów)
plt.figure(figsize=(10, 6))
plt.plot(rolling_avg_white, label="White Agent (Last 10 Episodes)", color='blue')
plt.plot(rolling_avg_black, label="Black Agent (Last 10 Episodes)", color='red')
plt.plot(rolling_avg_difference, label="Difference (White - Black)", color='green', linestyle='dashed')

plt.xlabel('Episodes')
plt.ylabel('Total Reward (Last 10 Episodes)')
plt.title('Training Performance (Short-Term)')
plt.legend()
plt.grid(True)
plt.show()

# 🔹 Wykres 2: Skumulowane sumy nagród
plt.figure(figsize=(10, 6))
plt.plot(cumulative_white_rewards, label="White Agent (Cumulative)", color='blue', linestyle='solid')
plt.plot(cumulative_black_rewards, label="Black Agent (Cumulative)", color='red', linestyle='solid')

plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.title('Training Performance (Cumulative Sum)')
plt.legend()
plt.grid(True)
plt.show()


def save_agent(agent, filename="agent_white.pth"):
    T.save(agent.DQN_eval_white.state_dict(), filename)
    print(f" Model zapisany jako {filename}")


# save_agent(agent_white, "agent_white.pth")
# save_agent(agent_black, "agent_black.pth")
#random_vs_random(env, agent_white, 1000)
#play_with_random_opponent_agent_forget(env, 1000)
#play_random_vs_heuristic(env, 1000)
#play_game(agent_white)
play_with_random_opponent_file(env,agent_white,10)
play_with_heuristic_opponent_file(env, agent_white,10)

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
