import random
import sys
from typing import Callable, List, Tuple, Union

import metaheuristics as meta

Matrix = List[List[int]]

VALUES = [0, 32, 64, 128, 160, 192, 223, 255]


class Block:
    def __init__(self, row, column, height, width, value):
        self.row: int = row
        self.column: int = column
        self.height: int = height
        self.width: int = width
        self.value: int = value


class BlockMatrix:
    def __init__(self, blocks, height, width):
        self.blocks: List[Block] = blocks
        self.height: int = height
        self.width: int = width

    def __getitem__(self, pos):
        row, column = pos
        for block in self.blocks:
            if row in range(block.row, block.row + block.height) and column in range(
                block.column, block.column + block.width
            ):
                return block.value
        raise IndexError

    def __str__(self):
        return self.to_str()

    def to_str(self, pretty=False):
        result = ""
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append("{:3}".format(self[i, j]) if pretty else f"{self[i, j]}")
            result += " ".join(row) + "\n"
        return result


def mse_distance(M1: Matrix, n, m) -> Callable[[BlockMatrix], float]:
    def quality(M2: BlockMatrix) -> float:
        return sum(sum((M1[i][j] - M2[i, j]) ** 2 for j in range(m)) for i in range(n)) / (n * m)

    return quality


def tweak_factory(k: int) -> Callable[[BlockMatrix], BlockMatrix]:
    def intensity_disruption(bm: BlockMatrix) -> BlockMatrix:
        block = random.choice(bm.blocks)
        block.value = random.choice(VALUES)
        return bm

    def split(value) -> Tuple[int, int]:
        v1 = v2 = k
        rest = value - 2 * k
        while rest > 0:
            if random.random() > 0.5:
                v1 += 1
            else:
                v2 += 1
            rest -= 1
        return v1, v2

    def merge_then_split(bm: BlockMatrix) -> BlockMatrix:
        try:
            big_block = random.choice([b for b in bm.blocks if b.height > k or b.width > k])
        except IndexError:
            return intensity_disruption(bm)

        neighbor: Union[Block, None] = None
        direction: Union[str, None] = None

        for b in bm.blocks:
            if b is big_block:
                continue

            if (
                b.row == big_block.row
                and b.height == big_block.height
                and (b.column + b.width == big_block.column or big_block.column + big_block.width == b.column)
            ):
                neighbor, direction = b, "horizontal"
                break

            if (
                b.column == big_block.column
                and b.width == big_block.width
                and (b.row + b.height == big_block.row or big_block.row + big_block.height == b.row)
            ):
                neighbor, direction = b, "vertical"
                break

        if neighbor:
            if direction == "horizontal":
                w1, w2 = split(big_block.width + neighbor.width)

                neighbor.column = min(big_block.column, neighbor.column)
                neighbor.width = w1

                big_block.column = neighbor.column + w1
                big_block.width = w2
            else:
                h1, h2 = split(big_block.height + neighbor.height)

                neighbor.row = min(big_block.row, neighbor.row)
                neighbor.height = h1

                big_block.row = neighbor.row + h1
                big_block.height = h2
        else:
            big_block.value = random.choice(VALUES)

        return bm

    def block_swap(bm: BlockMatrix) -> BlockMatrix:
        b1 = random.choice(bm.blocks)
        b2 = random.choice([b for b in bm.blocks if b is not b1])
        b1.value, b2.value = b2.value, b1.value
        return bm

    def tweak(bm: BlockMatrix):
        tweak_technique = random.choice([intensity_disruption, merge_then_split, block_swap])

        return tweak_technique(bm)

    return tweak


def random_block_matrix(n, m, k) -> BlockMatrix:
    blocks: List[Block] = []
    n_blocks_count = n // k
    m_blocks_count = m // k

    extend_last_column = m % k
    extend_last_row = n % k

    for i in range(n_blocks_count):
        for j in range(m_blocks_count):
            block = Block(row=i * k, column=j * k, height=k, width=k, value=random.choice(VALUES))
            blocks.append(block)

    if extend_last_column:
        for block in blocks:
            if block.column == (m_blocks_count - 1) * k:
                block.width += extend_last_column

    if extend_last_row:
        for block in blocks:
            if block.row == (n_blocks_count - 1) * k:
                block.height += extend_last_row

    return BlockMatrix(blocks, n, m)


def zeros(n, m) -> BlockMatrix:
    return BlockMatrix(blocks=[Block(0, 0, n, m, 0)], height=n, width=m)


def main():
    _input = input().split()
    t = float(_input[0])
    n, m, k = map(int, _input[1:])
    matrix = [list(map(int, input().split())) for _ in range(n)]

    best_solution, best_result = meta.simulated_annealing(
        function=mse_distance(matrix, n, m),
        initial_solution=random_block_matrix(n, m, k + 1),
        initial_temperature=1.52,
        cooling_schedule=lambda t, _: t * 0.85,
        tweak=tweak_factory(k),
        timeout=t,
        deep_input=True,
    )

    # print(mse_distance(matrix, random_block_matrix(n, m, k)))
    # print(mse_distance(matrix, zeros(n, m)))

    print(best_result)
    print(best_solution, file=sys.stderr)


if __name__ == "__main__":
    main()
