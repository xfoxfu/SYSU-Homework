use crate::board::{Board, CellState};
use indicatif::ProgressBar;

pub fn search_strategy(board: &mut Board, threshold: u8) -> (usize, usize) {
    let mut bar = ProgressBar::new(
        (1..=threshold)
            .map(|v| (board.size * board.size).pow(v as u32) as u64)
            .sum(),
    );
    let (row, col, _) = search(board, 0, threshold, &mut bar, (-isize::MAX, isize::MAX));
    bar.abandon();

    (row, col)
}

fn search(
    board: &mut Board,
    depth: u8,
    threshold: u8,
    bar: &mut ProgressBar,
    (mut alpha, beta): (isize, isize),
) -> (usize, usize, isize) {
    let (mut max_row, mut max_col) = (usize::MAX, usize::MAX);
    if depth < threshold {
        for row in 0..board.size {
            for col in 0..board.size {
                if board.get(row, col) != CellState::Null {
                    continue;
                }
                if !board.has_neighbor(row, col) {
                    continue;
                }
                let recover = board.current_try_place(row, col).unwrap();
                let nmax = -search(board, depth + 1, threshold, bar, (-beta, -alpha)).2;
                board.current_recover(recover).unwrap();
                if nmax > alpha {
                    if nmax >= beta {
                        return (max_row, max_col, beta);
                    }
                    alpha = nmax;
                    max_row = row;
                    max_col = col;
                }
            }
            bar.inc(board.size as u64);
        }
    } else {
        let factor = if depth % 2 == 0 { 1 } else { -1 };
        let score = crate::eval::evaluate(board, board.machine_color())
            - crate::eval::evaluate(board, board.human_color());
        return (usize::MAX, usize::MAX, score * factor);
    }

    (max_row, max_col, alpha)
}
