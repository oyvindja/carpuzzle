from calendar import c
from tabnanny import check
from this import d
from tkinter import HORIZONTAL
from xmlrpc.client import Boolean
import numpy as np
from enum import Enum
from copy import deepcopy
from typing import Tuple
from dataclasses import dataclass
from bisect import insort

BOARD_SIZE = 6
RED_CAR_ID = 1
RED_CAR_LENGTH = 2


class Direction(Enum):
    HORIZONTAL = 1
    VERTICAL = 2


@dataclass
class Move:
    """Store a move of car with 'car_id" to position 'pos'"""

    car_id: int
    delta: int
    pos: int


class Car:
    def __init__(
        self,
        car_id: int,
        lane: int,
        length: int,
        direction: Direction,
        redcar=False,
    ):
        if not (car_id > 0):
            raise ValueError("Car must have car_id > 0")

        self.car_id = car_id
        self.direction = direction
        self.redcar = redcar

        if direction == Direction["HORIZONTAL"]:

            def get_blocks(p):
                return lane, slice(p, p + length)

        elif direction == Direction["VERTICAL"]:

            def get_blocks(p):
                return slice(p, p + length), lane

        else:
            raise ValueError("Unsupported direction")

        self.blocks = dict()
        startpos = 0

        while True:
            self.blocks[startpos] = get_blocks(startpos)
            if (startpos + length) >= BOARD_SIZE:
                break

            startpos += 1
        self.possible_positions = sorted(list(self.blocks.keys()))
        self.last_position = self.possible_positions[-1]

    def __repr__(self) -> str:
        rc = "RED " if self.redcar else ""
        return f"{rc}Car ID {self.car_id}, pos {self.pos}, direction {self.direction}, len {self.length}"


class Cars:
    def __init__(self):
        self.cars = dict()

    def add_car(self, car):
        if car.car_id in self.cars:
            raise ValueError(f"Car {car.car_id} already added")
        self.cars[car.car_id] = car

    def __len__(self) -> int:
        return len(self.cars)

    def __getitem__(self, car_id) -> Car:
        return self.cars[car_id]


class CarPositions:
    def __init__(self):
        self.positions = dict()

    def add_position(self, car_id, pos):
        if car_id in self.positions:
            raise ValueError(f"Id {car_id} is already in CarPositions")
        self.positions[car_id] = pos

    def get_positions(self):
        return self.positions

    def move_car(self, car_id, pos):
        self.positions[car_id] = pos

    def __iter__(self):
        for k, v in self.positions.items():
            yield k, v

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, car_id) -> int:
        return self.positions[car_id]

    def __eq__(self, other):
        return self.positions == other.positions


class CarCollision(Exception):
    pass


def board_string_to_array(board_string) -> np.ndarray:
    rows = []
    for line in board_string.splitlines():
        row = np.array(line.split()).astype(int)
        if len(row) > 0:
            rows.append(row)

    board_array = np.array(rows)
    return board_array


class Board:
    def __init__(self, minimize=None, verbose=False):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
        self.car_positions = CarPositions()
        self.cars = Cars()
        self.previous_hash = None
        self.car_swaps = 0
        self.move_count = 0
        self.move_to_get_here = None
        if minimize is None:
            minimize = "car_swaps"

        if minimize == "move_count":
            self.max_move_distance = 1
        elif minimize == "car_swaps":
            self.max_move_distance = np.inf
        else:
            raise ValueError(
                "minimize parameter must be either move_count or car_swaps"
            )

    @classmethod
    def board_from_numpy(cls, board_array: np.ndarray, **kwargs):
        board = cls(**kwargs)

        for car, pos in get_cars(board_array):
            board.place_car(car, pos)
            if car.car_id == RED_CAR_ID:
                if car.direction != Direction.HORIZONTAL:
                    raise RuntimeError(f"Red car (ID {RED_CAR_ID}) must be horizontal)")

        return board

    @classmethod
    def board_from_str(cls, board_string: str, **kwargs):
        return cls.board_from_numpy(board_string_to_array(board_string), **kwargs)

    def place_car(self, car, pos):
        self.car_positions.add_position(car.car_id, pos)

        blocks = car.blocks[pos]
        if np.sum(self.board[blocks]) > 0:
            # Another car was already placed here
            raise CarCollision()

        self.board[blocks] = car.car_id
        self.cars.add_car(car)

    def move_car(self, car_id, pos):
        car = self.cars[car_id]
        old_pos = self.car_positions[car_id]
        old_blocks = car.blocks[old_pos]
        self.board[old_blocks] = 0

        new_blocks = car.blocks[pos]
        self.board[new_blocks] = car_id
        self.car_positions.move_car(car_id, pos)

        if (self.move_to_get_here is None) or (car_id != self.move_to_get_here.car_id):
            self.car_swaps += 1

        delta = pos - old_pos
        self.move_count += 1
        self.move_to_get_here = Move(car_id=car_id, delta=delta, pos=pos)

    def is_solved(self):
        return self.car_positions[RED_CAR_ID] == (BOARD_SIZE - RED_CAR_LENGTH)

    def get_next_boards(self):
        next_boards = list()

        for car_id, pos in self.car_positions:
            car = self.cars[car_id]
            # How far we can move in either direction depends on
            min_pos = max(pos - self.max_move_distance, 0)
            max_pos = min(pos + self.max_move_distance, car.last_position)

            for move_pos in [
                range(pos - 1, min_pos - 1, -1),
                range(pos + 1, max_pos + 1),
            ]:
                for next_pos in move_pos:
                    new_blocks = car.blocks[next_pos]
                    if len(set(self.board[new_blocks]) - {0, car.car_id}) > 0:
                        # If we would move to a position that has other blocks
                        # than the empty block or the existing car car_id, we couldn't
                        # Try moving in the other direction instead
                        break

                    # Avoid making copy of the .cars object, as
                    # these are constant for all boards
                    memo = {}
                    memo[id(self.cars)] = self.cars
                    b = deepcopy(self, memo)
                    b.previous_hash = hash(self)
                    b.move_car(car_id, next_pos)
                    next_boards.append(b)
        return next_boards

    def __lt__(self, other) -> Boolean:
        """Compare two Boards, based on move-count first, car-swaps secondly"""
        if self.move_count == other.move_count:
            return self.car_swaps < other.car_swaps
        else:
            return self.move_count < other.move_count

    def __repr__(self) -> str:
        r = "\n".join(
            [" ".join([f"{car_id:2d}" for car_id in row]) for row in self.board]
        )
        return r.replace(" 0", "  ")

    def __hash__(self):
        return hash(self.board.tobytes())


class SortedBoards:
    def __init__(self):
        self.boards = []

    def insert(self, board: Board):
        """Insert into list, keeping self.boards sorted in decreasing move count order"""
        insort(self.boards, board)

    def pop(self) -> Board:
        """Pop the board with the lowest move count, i.e. the first element"""
        # TODO : Refactor, to avoid costly pop(0)
        return self.boards.pop(0)

    def __len__(self) -> int:
        return len(self.boards)


#  2  3  4  4  5  0
#  2  3  6  0  5  7
#  2  0  6  1  1  7
#  8  8  8  9  0  7
#  0  0 10  9 11 11
# 12 12 10 13 13  0


def get_cars(board: np.ndarray):
    ysize, xsize = board.shape
    assert xsize == BOARD_SIZE, "X-size must match BOARD_SIZE"
    assert ysize == BOARD_SIZE, "Y-size must match BOARD_SIZE"

    for direction, b in [(Direction.HORIZONTAL, board), (Direction.VERTICAL, board.T)]:
        for rowpos, row in enumerate(b):
            car_id = 0
            length = 0
            startpos = 0

            for colpos in range(xsize + 1):
                if colpos < xsize:
                    val = row[colpos]
                else:
                    val = 0

                if val == 0:
                    # Empty spot
                    if (car_id > 0) and (length > 1):
                        yield Car(
                            car_id, rowpos, length, direction, car_id == 1
                        ), startpos
                    length = 0

                elif val == car_id:
                    # Car continues
                    length += 1

                else:
                    # New car ?
                    if (car_id > 0) and (length > 1):
                        yield Car(
                            car_id, rowpos, length, direction, car_id == 1
                        ), startpos
                    length = 1
                    startpos = colpos

                car_id = val


class Solver:
    def __init__(self, initial_board: Board):
        self.initial_board = initial_board
        self.boards = dict()
        self.solution_car_swap_count = np.inf
        self.solved_board = None
        self.check_boards = SortedBoards()

        if self.initial_board.is_solved():
            raise RuntimeError("Initial board is already solved")

        self.add_board(self.initial_board)

    def add_board(self, board: Board):
        board_hash = hash(board)
        if board_hash in self.boards.keys():
            existing_board = self.boards[board_hash]
            if board < existing_board:
                # Found shorter path
                self.boards[board_hash] = board
        else:
            # Found new board position
            self.boards[board_hash] = board

            # Check out its neighbors later
            self.check_boards.insert(board)

        if board.is_solved():
            if (self.solved_board is None) or (board < self.solved_board):
                self.solved_board = board
                self.solution_car_swap_count = board.car_swaps

                print(
                    f"Found solution of with {board.move_count} moves, {board.car_swaps} car swaps"
                )
                print(f"Have looked at {len(self.boards)} board positions so far")

    def solve(self):
        self.check_boards.insert(self.initial_board)

        while len(self.check_boards) > 0:
            # Pick a board to check
            check_board = self.check_boards.pop()
            for new_board in check_board.get_next_boards():
                self.add_board(new_board)

        board = self.solved_board
        if board is not None:
            solution_boards = [board]
            while True:
                previous_hash = board.previous_hash
                if previous_hash is None:
                    break
                else:
                    board = self.boards[previous_hash]
                solution_boards.append(board)

            return solution_boards[::-1]


if __name__ == "__main__":
    # fmt: off
    board_array = np.array([
        [  2,  3,  4,  4,  5,  0],
        [  2,  3,  6,  0,  5,  7],
        [  2,  0,  6,  1,  1,  7],
        [  8,  8,  8,  9,  0,  7],
        [  0,  0, 10,  9, 11, 11],
        [ 12, 12, 10, 13, 13,  0],
    ])
    # fmt: on

    # fmt: off
    # board_array = np.array([
    #     [  0,  4,  4,  0,  0,  0],
    #     [  0,  0,  0,  0,  0,  7],
    #     [  0,  0,  5,  1,  1,  7],
    #     [  0,  0,  5,  9,  0,  7],
    #     [  0,  0,  0,  9,  0,  0],
    #     [ 12, 12,  0, 13, 13,  0],
    # ])
    # fmt: on

    s = Solver(Board.board_from_numpy(board_array))
    solution = s.solve()

    if len(solution):
        print(f"Solution length {len(solution)}")
        print(f"Studied {len(s.boards)} board positions")

        for i, board in enumerate(solution):
            print(f"****** Board position {i}  :  {board.move_to_get_here}")
            print(f"****** Board hash {hash(board)}")
            print(board)
