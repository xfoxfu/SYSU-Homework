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
                // 对于每一个棋盘
                // 若不能放置，跳过
                if board.get(row, col) != CellState::Null {
                    continue;
                }
                // 若附近没有棋子，跳过
                if !board.has_neighbor(row, col) {
                    continue;
                }
                // 尝试放置
                let recover = board.current_try_place(row, col).unwrap();
                let nmax = -search(board, depth + 1, threshold, bar, (-beta, -alpha)).2; // 估价
                board.current_recover(recover).unwrap(); // 还原

                // 若需要更新最大值
                if nmax > alpha {
                    // 需要剪枝的情形
                    if nmax >= beta {
                        return (max_row, max_col, beta);
                    }
                    // 否则，更新
                    alpha = nmax;
                    max_row = row;
                    max_col = col;
                }
            }
            bar.inc(board.size as u64);
        }
    } else {
        // 叶节点，计算估价
        let factor = if depth % 2 == 0 { 1 } else { -1 }; // 负数因子
        let score = crate::eval::evaluate(board, board.machine_color())
            - crate::eval::evaluate(board, board.human_color()); // 估价
        return (usize::MAX, usize::MAX, score * factor);
    }

    (max_row, max_col, alpha)
}
