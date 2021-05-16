from copy import deepcopy
import random
import numpy as np


class Sudoku:
    def __init__(self, initial_board):
        self.sudoku_numbers = deepcopy(initial_board)
        self.possible_values = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        self.changeable_numbers = [[initial_board[y][x] not in self.possible_values for x in range(9)] for y in
                                   range(9)]
        self.solved_sudoku_numbers = deepcopy(initial_board)

    # We index sudoku squares by the least coordinate: for example first square (left upper corner) consists of
    # fields with coordinates: (0,0) (0,1) (0,2) (1,0) (1,1) (1,2) (2,0) (2,1) (2,2) -> and for each of these points
    # square-index is equal to (0,0) = (min_x, min_y)
    def get_square(self, pos_x, pos_y):
        sq_x = 3 * (pos_x // 3)
        sq_y = 3 * (pos_y // 3)
        return sq_x, sq_y

    # When solving sudoku we are looking for next free field:
    def next_pos(self, pos_x, pos_y):
        # First we try to find it within points with greater coordinates:
        for x in range(pos_x, 9):
            for y in range(pos_y, 9):
                if self.solved_sudoku_numbers[x][y] not in self.possible_values:
                    return x, y
        # We try to find it wherever and if we can't we return None point:
        return next(((x, y) for x in range(9) for y in range(9) if
                     self.solved_sudoku_numbers[x][y] not in self.possible_values), (-1, -1))

    def check_move(self, i, j, value):
        check_row = all([self.solved_sudoku_numbers[i][x] != value for x in range(9)])
        check_col = all([self.solved_sudoku_numbers[x][j] != value for x in range(9)])
        sq_x, sq_y = self.get_square(i, j)
        check_square = all(
            [self.solved_sudoku_numbers[x][y] != value for x in range(sq_x, sq_x + 3) for y in range(sq_y, sq_y + 3)])
        return check_row and check_col and check_square

    def solve_sudoku(self, pos_x=0, pos_y=0):
        pos_x, pos_y = self.next_pos(pos_x, pos_y)
        if pos_x == -1 and pos_y == -1:
            return True
        for value in self.possible_values:
            if self.check_move(pos_x, pos_y, value):
                self.solved_sudoku_numbers[pos_x][pos_y] = value
                if self.solve_sudoku(pos_x, pos_y):
                    return True
                self.solved_sudoku_numbers[pos_x][pos_y] = None
        return False

    def check_squares(self):
        # We iterate through possible square indexes:
        for sq_x in range(0, 9, 3):
            for sq_y in range(0, 9, 3):
                square = [self.sudoku_numbers[i][j] for i in range(sq_x, sq_x + 3) for j in range(sq_y, sq_y + 3)]
                square_values = list(filter(lambda x: x in self.possible_values, square))
                # Tricky one: we check if there are some duplicates:
                if len(square_values) != len(set(square_values)):
                    return False
        return True

    def get_hint(self):
        self.solved_sudoku_numbers = deepcopy(self.sudoku_numbers)
        if self.solve_sudoku():
            free_positions = [(i, j) for i in range(9) for j in range(9) if
                              self.sudoku_numbers[i][j] not in self.possible_values]
            if not free_positions:
                return None
            i, j = random.choice(free_positions)
            self.sudoku_numbers[i][j] = self.solved_sudoku_numbers[i][j]
            return i, j, self.solved_sudoku_numbers[i][j]
        return None