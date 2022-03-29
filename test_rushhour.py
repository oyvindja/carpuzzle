from rushhour import (
    Board,
    Solver,
    board_string_to_array,
)
import numpy as np
import pytest
from copy import deepcopy

# fmt: off
BOARD_38 =  np.array([
    [  2,  3,  4,  4,  5,  0],
    [  2,  3,  6,  0,  5,  7],
    [  2,  0,  6,  1,  1,  7],
    [  8,  8,  8,  9,  0,  7],
    [  0,  0, 10,  9, 11, 11],
    [ 12, 12, 10, 13, 13,  0],
])
# fmt: on


BOARD_38_STR = """
  2  3  4  4  5  0
  2  3  6  0  5  7
  2  0  6  1  1  7
  8  8  8  9  0  7
  0  0 10  9 11 11
 12 12 10 13 13  0
"""


BOARD_X_STR = """
  2  3  4  4  5  0
  2  3  0  0  5  7
  2  0  1  1  5  7
  0  0  0  0  0  7
  0  0  0  0  0  0
  0  0 13 13 13 13
"""


BOARD_X_SOLUTION = """
  2  3  4  4  0  0
  2  3  0  0  0  0
  2  0  0  0  1  1
  0  0  0  0  5  7
  0  0  0  0  5  7
 13 13 13 13  5  7
"""


def test_board_init():
    board = Board.board_from_str(BOARD_38_STR)
    assert len(board.car_positions) == 13
    assert len(board.cars) == 13


@pytest.mark.parametrize("minimize", ["move_count", "car_swaps"])
def test_board_38(minimize):
    initial_board = Board.board_from_numpy(BOARD_38, minimize=minimize)
    s = Solver(initial_board)
    solution = s.solve()

    solution_length = s.solved_board.move_count
    car_swaps = s.solved_board.car_swaps
    move_distance = sum([abs(b.move_to_get_here.delta) for b in solution[1:]])

    print(f"Minimized the {minimize}")
    print(
        f"Found solution with {solution_length} moves, swapping car {car_swaps} times, total move distance {move_distance}"
    )
    print(f"Have looked at total {len(s.boards)} board positions")

    # for i, board in enumerate(solution):
    #    print(f"****** Board position {i}  :  {board.move_to_get_here}")
    #    print(f"****** Car swap {board.car_swaps}")
    #    print(board)

    assert car_swaps <= 60
    assert solution_length <= 84


@pytest.mark.parametrize(
    "minimize,expected_solution_length", [("move_count", 9), ("car_swaps", 4)]
)
def test_board_x(minimize, expected_solution_length):
    initial_board = Board.board_from_str(BOARD_X_STR, minimize=minimize)
    s = Solver(initial_board)
    solution = s.solve()

    solution_length = s.solved_board.move_count
    car_swaps = s.solved_board.car_swaps
    print(f"Minimized the {minimize}")
    print(
        f"Found solution with {solution_length} moves, swapping car {car_swaps} times"
    )
    print(f"Have looked at total {len(s.boards)} board positions")

    for i, board in enumerate(solution):
        print(f"****** Board position {i}  :  {board.move_to_get_here}")
        print(f"****** Board hash {hash(board)}")
        print(board)

    assert solution_length == expected_solution_length


def check_unique_boards(boards):
    hashes = set()
    cars_objects = set()
    for board in boards:
        hashes.add(hash(board))
        cars_objects.add(id(board.cars))

    assert len(boards) == len(hashes), "Not all boards were unique"
    assert len(cars_objects) == 1, "Found multiple copies of the board.cars object"


def test_next_boards():
    board = Board.board_from_str(BOARD_X_STR, minimize="move_count")
    initial_board = deepcopy(board)

    next_boards = board.get_next_boards()
    assert len(next_boards) == 7
    check_unique_boards(next_boards)

    board.move_car(13, 1)
    next_boards = board.get_next_boards()
    assert len(next_boards) == 8
    check_unique_boards(next_boards)

    board.move_car(13, 0)
    next_boards = board.get_next_boards()
    assert len(next_boards) == 7
    check_unique_boards(next_boards)

    board.move_car(7, 0)
    next_boards = board.get_next_boards()
    assert len(next_boards) == 6
