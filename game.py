import random


class Game:
    def __init__(self, board_size=20, pawns_per_player=10):
        self.board_size = board_size
        self.board = [[0] * board_size for _ in range(2)]  # 0 oznacza puste pole
        self.pawns_per_player = pawns_per_player
        self.pawns = {
            'player1': [0] * pawns_per_player,  # Gracz 1 zaczyna na pozycji 0
            'player2': [board_size - 1] * pawns_per_player  # Gracz 2 zaczyna na pozycji końcowej
        }
        self.turn = 'player1'  # Gracz zaczynający

    def roll_dice(self):
        # Rzut kostką (1-6)
        return random.randint(1, 6)

    def move_pawn(self, player, pawn_index, roll):
        if pawn_index >= self.pawns_per_player:
            print("Nieprawidłowy numer pionka.")
            return False

        # Ustal kierunek ruchu i aktualną pozycję pionka
        line = 0 if player == 'player1' else 1
        pos = self.pawns[player][pawn_index]

        # Określenie nowej pozycji na podstawie gracza
        new_pos = pos + roll if player == 'player1' else pos - roll

        # Sprawdzenie, czy nowa pozycja jest na planszy
        if new_pos < 0 or new_pos >= self.board_size:
            print("Ruch wychodzi poza planszę.")
            return False

        # Zasady zbijania i blokowania ruchu
        if self.board[1 - line][new_pos] == 1:  # Przeciwnik ma jednego pionka - można zbić
            print(f"{player} zbija pionek przeciwnika na pozycji {new_pos}")
            self.board[1 - line][new_pos] = 0
            opponent = 'player2' if player == 'player1' else 'player1'
            self.pawns[opponent].remove(new_pos)
            self.pawns[opponent].append(
                self.board_size - 1 if opponent == 'player1' else 0)  # Przeciwnik wraca na start

        elif self.board[1 - line][new_pos] > 1:  # Przeciwnik ma więcej niż jeden pionek - ruch zablokowany
            print("Nie można wejść na pole przeciwnika, gdy jest tam więcej niż jeden pionek.")
            return False

        # Aktualizacja pozycji pionka
        self.board[line][pos] -= 1
        self.board[line][new_pos] += 1
        self.pawns[player][pawn_index] = new_pos

        return True

    def check_win(self, player):
        # Sprawdzenie, czy wszystkie pionki gracza są na końcu planszy
        final_pos = self.board_size - 1 if player == 'player1' else 0
        for pos in self.pawns[player]:
            if pos != final_pos:
                return False
        return True

    def play_turn(self):
        # Ruch gracza
        player = self.turn
        roll = self.roll_dice()
        print(f"{player} wyrzucił {roll}")

        if len(self.pawns[player]) == 0:
            print(f"{player} nie ma dostępnych pionków do ruchu.")
            self.turn = 'player2' if self.turn == 'player1' else 'player1'
            return

        # Wybór pionka do ruchu
        pawn_index = 0  # Na potrzeby przykładu: zawsze wybieramy pierwszy pionek
        if not self.move_pawn(player, pawn_index, roll):
            print("Ruch niedozwolony. Tracisz kolejkę.")

        # Sprawdzenie, czy gracz wygrał
        if self.check_win(player):
            print(f"{player} wygrywa grę!")
            return True

        # Zmiana tury
        self.turn = 'player2' if self.turn == 'player1' else 'player1'
        return False

    def start_game(self):
        # Inicjalizacja pionków
        for i in range(self.pawns_per_player):
            self.board[0][0] += 1  # Pionki gracza 1 na pozycji początkowej
            self.board[1][self.board_size - 1] += 1  # Pionki gracza 2 na pozycji końcowej

        # Pętla główna gry
        print("Rozpoczynamy grę!")
        while True:
            if self.play_turn():
                break
