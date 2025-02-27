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
        self.fc1 = nn.Linear(input_size, fc1_dims)  # Pierwsza warstwa ukryta.
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)  # Druga warstwa ukryta.
        self.fc3 = nn.Linear(fc2_dims, n_actions)  # Warstwa wyjściowa.
        self.optimizer = optim.Adam(self.parameters(), lr=lr)  # Optymalizator (Adam).
        self.loss = nn.MSELoss()  # Funkcja straty (błąd średniokwadratowy).
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")  # Wybór urządzenia.
        self.to(self.device)  # Przesłanie modelu na GPU lub CPU.

    def forward(self, state, dice_roll):
        x = T.cat((state, dice_roll), dim=1)
        x = F.relu(self.fc1(x))  # Aktywacja ReLU na pierwszej warstwie ukrytej.
        x = F.relu(self.fc2(x))  # Aktywacja ReLU na drugiej warstwie ukrytej.
        x = self.fc3(x)  # Wyjście sieci (wartości Q dla każdej akcji).
        return x


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
        self.mem_pointer = 0

        # Oddzielne sieci dla graczy
        self.DQN_eval_white = DQN(lr=self.lr, input_size=input_size, fc1_dims=128, fc2_dims=256, n_actions=n_actions)
        self.DQN_target_white = DQN(lr=self.lr, input_size=input_size, fc1_dims=128, fc2_dims=256, n_actions=n_actions)

        self.DQN_eval_black = DQN(lr=self.lr, input_size=input_size, fc1_dims=128, fc2_dims=256, n_actions=n_actions)
        self.DQN_target_black = DQN(lr=self.lr, input_size=input_size, fc1_dims=128, fc2_dims=256, n_actions=n_actions)

        self.state_memory = deque(maxlen=self.max_mem)
        self.new_state_memory = deque(maxlen=self.max_mem)
        self.action_memory = deque(maxlen=self.max_mem)
        self.reward_memory = deque(maxlen=self.max_mem)
        self.terminal_memory = deque(maxlen=self.max_mem)
        self.dice_memory = deque(maxlen=self.max_mem)

    def observation(self, state, dice_roll, player):
        """
        Wybiera akcję w zależności od gracza.
        """
        eval_net = self.DQN_eval_white if player == 'white' else self.DQN_eval_black
        eval_net.eval()

        state_tensor = T.FloatTensor(state).unsqueeze(0).to(eval_net.device)
        dice_tensor = T.FloatTensor(dice_roll).view(1, -1).to(eval_net.device)

        with T.no_grad():
            q_values = eval_net(state_tensor, dice_tensor)

        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(0, self.n_actions)
        else:
            action_idx = q_values.argmax(dim=1).item()

        return action_idx

    def store_transition(self, state, action, reward, next_state, dice_roll, done):
        self.state_memory.append(state)
        self.new_state_memory.append(next_state)
        self.reward_memory.append(reward)
        self.action_memory.append(action)
        self.dice_memory.append(dice_roll)
        self.terminal_memory.append(done)
        self.mem_pointer += 1

    def choose_action(self, state, dice_roll, player):
        eval_net = self.DQN_eval_white if player == 'white' else self.DQN_eval_black

        state_tensor = T.tensor(state, dtype=T.float32).unsqueeze(0).to(eval_net.device)
        dice_tensor = T.tensor(dice_roll, dtype=T.float32).view(1, -1).to(eval_net.device)

        if np.random.random() > self.epsilon:
            with T.no_grad():
                q_values = eval_net.forward(state_tensor, dice_tensor)
                action = T.argmax(q_values).item()
        else:
            action = np.random.choice(self.n_actions)

        return action

    def learn(self, player):
        """
        Nauka oparta na mini-batchu przejść.
        """
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
        action_batch = T.tensor(np.array(self.action_memory)[batch_indices], dtype=T.long).to(eval_net.device)
        dice_batch = T.tensor(np.array(self.dice_memory)[batch_indices], dtype=T.float32).to(eval_net.device)

        q_eval = eval_net(state_batch, dice_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        q_next = target_net(new_state_batch, dice_batch)
        q_next[terminal_batch == 1] = 0.0
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = eval_net.loss(q_eval, q_target)
        loss.backward()
        eval_net.optimizer.step()

        self.epsilon = max(self.eps_end, self.epsilon - self.eps_decay)

    def update_target_network(self, player):
        """
        Aktualizuje sieć docelową wybranego gracza.
        """
        if player == 'white':
            self.DQN_target_white.load_state_dict(self.DQN_eval_white.state_dict())
        else:
            self.DQN_target_black.load_state_dict(self.DQN_eval_black.state_dict())


class BackgammonEnv:
    def build_board(self):
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
        self.board = self.build_board()
        self.turn = 'white'
        return np.array(self.board, dtype=np.float32)

    def roll_dice(self):
        # Rzut dwiema kostkami.
        return random.randint(1, 6), random.randint(1, 6)

    def count_negative_fields(self):
        """
        Liczy liczbę pól z wartością ujemną w planszy.
        """
        return sum(1 for field in self.board if field < 0)

    def count_positive_fields(self):
        """
        Liczy liczbę pól z wartością dodatnią w planszy.
        """
        return sum(1 for field in self.board if field > 0)

    def calculate_pips(self, player):
        """Oblicza sumę PIP dla gracza."""
        if player == 'white':
            return sum(pos * abs(self.board[pos]) for pos in range(24) if self.board[pos] > 0)
        else:
            return

    def step(self, action, player):
        from_pos, dice_roll = action
        for die_roll in dice_roll:
            success = self.move(from_pos, die_roll, self.turn)
            if not success:
                return self.get_state(), -1, True

        # Sprawdzenie warunków końca gry
        if player == 'white':
            if self.count_positive_fields() == 0:
                return self.get_state(), 1, True
        elif player == 'black':
            if self.count_positive_fields() == 0:
                return self.get_state(), -1, True

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


def train_movement(env, agent, num_episodes, max_steps=50):
    """
    Trening podstawowego poruszania się pionkiem w uproszczonym środowisku.
    """
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        step_count = 0
        total_reward = 0

        while not done and step_count < max_steps:
            step_count += 1

            # Rzut kostką
            dice_roll = env.roll_dice()

            # Decyzja agenta o ruchu
            action_idx = agent.observation(state, dice_roll, player='white')
            from_pos = action_idx
            action = (from_pos, dice_roll)

            # Wyświetl aktualny stan planszy i planowany ruch
            print(f"Current Board State (White): {state}")
            print(f"Dice Roll: {dice_roll}, Selected Action: Move from {from_pos} by {dice_roll}")

            # Wykonanie ruchu
            next_state, reward, done = env.step(action, player='white')

            # Wyświetl stan planszy po ruchu i wynik
            print(f"Next Board State: {next_state}")
            print(f"Reward: {reward}, Done: {done}")
            print("-" * 50)

            # Zapis przejścia do pamięci
            agent.store_transition(state, action_idx, reward, next_state, dice_roll, done)

            # Nauka agenta
            agent.learn('white')

            # Aktualizacja stanu
            state = next_state
            total_reward += reward

        # Aktualizacja sieci co epizod
        agent.update_target_network('white')

        # Wyświetlenie postępu
        if episode % 100 == 0:
            print(
                f"Pre-training Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")


def train_agent(env, agent_white, agent_black, num_episodes):
    scores = []
    total_reward_white = 0  # Suma nagród dla białego
    total_reward_black = 0  # Suma nagród dla czarnego

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        current_player = 'white'

        transitions_white = []  # Przechowuje doświadczenia dla białego
        transitions_black = []  # Przechowuje doświadczenia dla czarnego

        while not done:
            for _ in range(2):
                if done:
                    break

                dice_roll = env.roll_dice()
                current_agent = agent_white if current_player == 'white' else agent_black
                action_idx = current_agent.observation(state, dice_roll, player=current_player)

                from_pos = action_idx
                action = (from_pos, dice_roll)

                next_state, reward, done = env.step(action, player=current_player)

                # Zapamiętaj przejście dla odpowiedniego gracza
                if current_player == 'white':
                    transitions_white.append((state, action_idx, reward, next_state, dice_roll, done))
                    total_reward_white += reward
                else:
                    transitions_black.append((state, action_idx, reward, next_state, dice_roll, done))
                    total_reward_black += reward

                state = next_state
                total_reward += reward

            # Nauka po dwóch ruchach dla białego
            for transition in transitions_white:
                agent_white.store_transition(*transition)
                agent_white.learn(current_player)
            transitions_white.clear()

            # Nauka po dwóch ruchach dla czarnego
            for transition in transitions_black:
                agent_black.store_transition(*transition)
                agent_black.learn(current_player)
            transitions_black.clear()

            # Zmiana gracza
            current_player = 'black' if current_player == 'white' else 'white'

        scores.append(total_reward)

        # Aktualizacja sieci zwycięzcy
        winner = 'white' if total_reward > 0 else 'black'
        if winner == 'white':
            agent_white.update_target_network(winner)
        else:
            agent_black.update_target_network(winner)

        # Wyświetlenie wyników co 100 epizodów
        if episode % 100 == 0:
            rolling_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(
                f"Episode {episode}/{num_episodes}, "
                f"Total Reward White: {total_reward_white}, "
                f"Total Reward Black: {total_reward_black}, "
                f"Rolling Avg Reward (Last 100 Episodes): {rolling_avg:.2f}, "
                f"Epsilon White: {agent_white.epsilon:.3f}, "
                f"Epsilon Black: {agent_black.epsilon:.3f}"
            )

    return scores


def random_player_action(env):
    """Gracz losowy wykonuje ruch."""
    while True:
        from_pos = random.randint(0, 23)
        dice_roll = env.roll_dice()
        for die in dice_roll:
            to_pos = from_pos + die if env.turn == 'white' else from_pos - die
            if env._is_valid_move(from_pos, to_pos, env.turn):
                return from_pos, dice_roll


def play_with_random_opponent(env, agent, num_episodes):
    """
    Pozwala botowi grać przeciwko losowemu graczowi przez określoną liczbę epizodów.
    """
    agent_wins = 0

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False

        while not done:
            if env.turn == 'white':
                dice_roll = env.roll_dice()
                action_idx = agent.observation(state, dice_roll, player='white')
                from_pos = action_idx
                action = (from_pos, dice_roll)

                next_state, reward, done = env.step(action, player='white')

            else:
                while True:
                    action = random_player_action(env)
                    if reward != -1:
                        break  # Jeśli ruch był poprawny, przerwij

                next_state, reward, done = env.step(action, player='black')

            state = next_state

        if reward == 1:  # Wygrana białych (agenta)
            agent_wins += 1

        # Wyświetl wynik co 10 gier
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}, Agent Wins: {agent_wins}")

    win_rate = (agent_wins / num_episodes) * 100
    print(f"Agent Win Rate: {win_rate:.2f}%")
    return win_rate


# Konfiguracja i trening agentów
num_episodes = 5000

gamma = 0.99
epsilon = 1
lr = 0.001
input_size = 24
n_dice = 2
batch_size = 32
n_actions = 24

env = BackgammonEnv()
agent_white = Agent(gamma, epsilon, lr, input_size + n_dice, batch_size, n_actions)
agent_black = Agent(gamma, epsilon, lr, input_size + n_dice, batch_size, n_actions)

# Pre-trening poruszania się pionkami
print("Starting pre-training for white agent...")
train_movement(env, agent_white, num_episodes=1000)

print("Starting pre-training for black agent...")
train_movement(env, agent_black, num_episodes=1000)

# Trening gry
print("Starting full training for both agents...")
scores = train_agent(env, agent_white, agent_black, num_episodes=5000)

# Gra na nauczonych sieciach
print("Playing with trained agents...")
play_with_random_opponent(env, agent_white, num_episodes=100)

scores = train_agent(env, agent_white, agent_black, num_episodes=num_episodes)

# Wybór lepszego bota i gra przeciwko losowemu przeciwnikowi
white_win_rate = play_with_random_opponent(env, agent_white, num_episodes=100)
black_win_rate = play_with_random_opponent(env, agent_black, num_episodes=100)

better_agent = agent_white if white_win_rate > black_win_rate else agent_black
print(f"The better agent is: {'White' if better_agent == agent_white else 'Black'}")

# Lepszy bot gra przeciwko losowemu przeciwnikowi
play_with_random_opponent(env, better_agent, num_episodes=1000)

import matplotlib.pyplot as plt

rolling_avg = np.convolve(scores, np.ones(100) / 100, mode='valid')

plt.figure(figsize=(10, 6))
plt.plot(rolling_avg, label="Rolling Average (100 episodes)")
plt.xlabel('Episodes')
plt.ylabel('Average of Last 100 Rewards')
plt.title('Training Performance (Rolling Average)')
plt.legend()
plt.grid(True)
plt.show()
win_rate = play_with_random_opponent(env, better_agent, num_episodes=1000)