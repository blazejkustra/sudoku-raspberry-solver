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

    def check_rows_columns(self):
        for i in range(9):
            row = self.sudoku_numbers[i]
            column = [row[i] for row in self.sudoku_numbers]
            row_values = list(filter(lambda x: x in self.possible_values, row))
            col_values = list(filter(lambda x: x in self.possible_values, column))
            # Tricky one: we check if there are some duplicates:
            check_row = (len(row_values) == len(set(row_values)))
            check_col = (len(col_values) == len(set(col_values)))
            if not check_row or not check_col:
                return False
        return True

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

    def check_sudoku(self):
        if (not self.check_rows_columns()) or (not self.check_squares()):
            return "wrong"
        valid_values = [[self.sudoku_numbers[x][y] in self.possible_values for x in range(9)] for y in range(9)]
        count_valid_values = 0
        for i in range(9):
            count_valid_values += sum(valid_values[i])
        if count_valid_values == 81:
            return "all"
        return "ok"

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

    def generate_sudoku(self, mask_rate=0.6):
        while True:
            # Start with empty sudoku:
            sudoku = np.zeros((9, 9), np.int)
            possible_values_numpy = np.arange(1, 9 + 1)
            # First row is a random permutation of [1,2,3,4,5,6,7,8,9]
            sudoku[0, :] = np.random.choice(possible_values_numpy, 9, replace=False)
            try:
                for row in range(1, 9):
                    for column in range(9):
                        possible_column_vals = np.setdiff1d(possible_values_numpy, sudoku[:row, column])
                        possible_row_vals = np.setdiff1d(possible_values_numpy, sudoku[row, :column])
                        possible_vals = np.intersect1d(possible_column_vals, possible_row_vals)
                        square_r, square_c = row // 3, column // 3
                        possible_square_vals = np.setdiff1d(np.arange(0, 9 + 1),
                                                            sudoku[square_r * 3:(square_r + 1) * 3,
                                                            square_c * 3:(square_c + 1) * 3].ravel())
                        possible = np.intersect1d(possible_vals, possible_square_vals)
                        sudoku[row, column] = np.random.choice(possible, size=1)
                break
            except ValueError:
                pass
        mask_sudoku = sudoku.copy()
        mask_sudoku[np.random.choice([True, False], size=sudoku.shape, p=[mask_rate, 1 - mask_rate])] = 0
        self.sudoku_numbers = mask_sudoku.tolist()
        self.changeable_numbers = [[mask_sudoku[y][x] not in self.possible_values for x in range(9)] for y in
                                   range(9)]

    def reset_sudoku(self):
        self.sudoku_numbers = deepcopy(
            [[self.sudoku_numbers[y][x] if not self.changeable_numbers[y][x] else None for x in range(9)] for y in
             range(9)])

    def change_value(self, x, y, value):
        self.sudoku_numbers[x][y] = value
