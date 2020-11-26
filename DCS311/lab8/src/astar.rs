use super::{Cell, Map};
use embedded_graphics::{pixelcolor::Rgb888, DrawTarget};
use priority_queue::PriorityQueue;
use std::cmp::Reverse;
use std::collections::HashMap;

pub struct AStarSearch<'a> {
    map: &'a mut Map,
    frontier: PriorityQueue<(usize, usize), Reverse<usize>>,
    parent: HashMap<(usize, usize), (usize, usize)>,
}

impl<'a> AStarSearch<'a> {
    pub fn new(map: &'a mut Map) -> Self {
        let mut frontier = PriorityQueue::new();
        frontier.push(map.current().clone(), Reverse(0));
        let parent = HashMap::new();

        Self {
            map,
            frontier,
            parent,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum SearchError {
    #[error("failed to reach target")]
    Failure,
}

impl<'a> AStarSearch<'a> {
    pub fn iterate(&mut self) -> Result<Option<()>, SearchError> {
        let ((u, v), Reverse(pri)) = self.frontier.pop().ok_or(SearchError::Failure)?;
        let cell = self.map.get_mut(u, v);
        if *cell == Cell::End {
            return Ok(Some(()));
        }
        self.map.explore(u, v);
        for (i, j) in self.map.adjacent(u, v).into_iter() {
            if !self.map.get(i, j).reachable() {
                continue;
            }
            if let Some(Reverse(npri)) = self.frontier.get_priority(&(i, j)).copied() {
                if pri + 1 < npri {
                    self.frontier.change_priority(&(i, j), Reverse(pri + 1));
                    *self.parent.get_mut(&(i, j)).unwrap() = (u, v);
                }
            } else {
                self.frontier.push((i, j), Reverse(pri + 1));
                self.parent.insert((i, j), (u, v));
            }
        }

        Ok(None)
    }
}

impl<'a> AStarSearch<'a> {
    pub fn draw<D: DrawTarget<Rgb888>>(&self, display: &mut D) -> Result<(), D::Error> {
        use embedded_graphics::prelude::*;

        self.map.draw(display)?;

        let (mut i, mut j) = *self.map.current();
        self.map
            .draw_cell(display, i as i32, j as i32, Rgb888::CYAN)?;
        while let Some((u, v)) = self.parent.get(&(i, j)).copied() {
            i = u;
            j = v;
            self.map
                .draw_cell(display, i as i32, j as i32, Rgb888::BLUE)?;
        }

        Ok(())
    }
}
