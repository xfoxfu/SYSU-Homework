use crate::board::{Board, CellState};
use once_cell::sync::Lazy;
use std::collections::HashMap;

#[allow(clippy::type_complexity)]
static SCORE_MAP: Lazy<HashMap<(i8, i8, i8, i8, i8), isize>> = Lazy::new(|| {
    vec![
        ((0, 1, 1, 0, 0), 1),
        ((0, 0, 1, 1, 0), 1),
        ((1, 1, 0, 1, 0), 4),
        ((0, 0, 1, 1, 1), 10),
        ((1, 1, 1, 0, 0), 10),
        ((0, 1, 1, 1, 0), 100),
        ((1, 1, 1, 0, 1), 100),
        ((1, 1, 0, 1, 1), 100),
        ((1, 0, 1, 1, 1), 100),
        ((1, 1, 1, 1, 0), 100),
        ((0, 1, 1, 1, 1), 100),
        ((1, 1, 1, 1, 1), 10000),
    ]
    .into_iter()
    .collect()
});

fn map_cell(cell: CellState, target: CellState) -> i8 {
    if cell == target {
        1 // placed
    } else if cell == CellState::Null {
        0 // placeable
    } else {
        -1 // unplaceable
    }
}

fn eval_line(
    cells: (CellState, CellState, CellState, CellState, CellState),
    target: CellState,
) -> isize {
    let cells = (
        map_cell(cells.0, target),
        map_cell(cells.1, target),
        map_cell(cells.2, target),
        map_cell(cells.3, target),
        map_cell(cells.4, target),
    );
    SCORE_MAP.get(&cells).copied().unwrap_or(0)
}

pub fn evaluate(board: &Board, target: CellState) -> isize {
    let mut score = 0;
    for row in 0..board.size {
        for col in 0..board.size {
            if row + 4 < board.size {
                score += eval_line(
                    (
                        board.get(row, col),
                        board.get(row + 1, col),
                        board.get(row + 2, col),
                        board.get(row + 3, col),
                        board.get(row + 4, col),
                    ),
                    target,
                );
            }
            if col + 4 < board.size {
                score += eval_line(
                    (
                        board.get(row, col),
                        board.get(row, col + 1),
                        board.get(row, col + 2),
                        board.get(row, col + 3),
                        board.get(row, col + 4),
                    ),
                    target,
                );
            }
            if row + 4 < board.size && col + 4 < board.size {
                score += eval_line(
                    (
                        board.get(row, col),
                        board.get(row + 1, col + 1),
                        board.get(row + 2, col + 2),
                        board.get(row + 3, col + 3),
                        board.get(row + 4, col + 4),
                    ),
                    target,
                );
            }
            if row + 4 < board.size && col >= 4 {
                score += eval_line(
                    (
                        board.get(row, col),
                        board.get(row + 1, col - 1),
                        board.get(row + 2, col - 2),
                        board.get(row + 3, col - 3),
                        board.get(row + 4, col - 4),
                    ),
                    target,
                );
            }
        }
    }
    score
}
