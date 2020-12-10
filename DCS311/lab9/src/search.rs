use crate::board::{Board, BoardState, CellState};
use indicatif::ProgressBar;

pub fn search_strategy(board: &mut Board, threshold: u8) -> (usize, usize) {
    let mut bar = ProgressBar::new((board.size * board.size).pow(threshold as u32) as u64);
    let (row, col, _) = search(board, 0, threshold, &mut bar);
    bar.finish();

    (row, col)
}

fn search(
    board: &mut Board,
    depth: u8,
    threshold: u8,
    bar: &mut ProgressBar,
) -> (usize, usize, isize) {
    let (mut max, mut max_row, mut max_col) = (isize::MIN, usize::MAX, usize::MAX);
    let factor = if board.is_human_turn() { -1 } else { 1 };
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
                let nmax = search(board, depth + 1, threshold, bar).2 * factor;
                if nmax > max {
                    max = nmax;
                    max_row = row;
                    max_col = col;
                }
                board.current_recover(recover).unwrap();
            }
            bar.inc(board.size as u64);
        }
    } else {
        let score = crate::eval::evaluate(board, board.machine_color())
            - crate::eval::evaluate(board, board.human_color());
        return (usize::MAX, usize::MAX, score);
    }

    (max_row, max_col, max * factor)
}
