use super::{Cell, Map};
use embedded_graphics::{pixelcolor::Rgb888, DrawTarget};
use embedded_graphics_simulator::SimulatorDisplay;
use priority_queue::PriorityQueue;
use std::collections::HashMap;
use std::{cmp::Reverse, marker::PhantomData};

#[derive(thiserror::Error, Debug)]
pub enum SearchError {
    #[error("failed to reach target")]
    Failure,
}

pub trait Search {
    fn iterate(&mut self) -> Result<Option<Vec<(usize, usize)>>, SearchError>;
    fn get_route(&self) -> Vec<(usize, usize)>;
    fn draw(
        &self,
        display: &mut SimulatorDisplay<Rgb888>,
    ) -> Result<(), <SimulatorDisplay<Rgb888> as DrawTarget<Rgb888>>::Error>;
}

pub trait CostFunction {
    fn g(prev: usize, dist: usize) -> usize {
        prev + dist
    }
    fn h(cur: (usize, usize), target: (usize, usize)) -> usize;
    fn f(prev: usize, dist: usize, cur: (usize, usize), target: (usize, usize)) -> Reverse<usize> {
        Reverse(Self::g(prev, dist) + Self::h(cur, target))
    }
}

pub struct GenericSearch<'a, C: CostFunction> {
    map: &'a mut Map,
    frontier: PriorityQueue<(usize, usize), Reverse<usize>>,
    parent: HashMap<(usize, usize), (usize, usize)>,
    _c: PhantomData<C>,
}

impl<'a, C: CostFunction> GenericSearch<'a, C> {
    pub fn new(map: &'a mut Map) -> Self {
        let mut frontier = PriorityQueue::new();
        frontier.push(map.current(), Reverse(0));
        let parent = HashMap::new();

        Self {
            map,
            frontier,
            parent,
            _c: Default::default(),
        }
    }
}

impl<'a, C: CostFunction> Search for GenericSearch<'a, C> {
    fn iterate(&mut self) -> Result<Option<Vec<(usize, usize)>>, SearchError> {
        let ((u, v), Reverse(pri)) = self.frontier.pop().ok_or(SearchError::Failure)?;
        let cell = *self.map.get(u, v);
        self.map.explore(u, v);
        if cell == Cell::End {
            return Ok(Some(self.get_route()));
        }
        for (i, j) in self.map.adjacent(u, v).into_iter() {
            if !self.map.get(i, j).reachable() {
                continue;
            }
            if let Some(Reverse(npri)) = self.frontier.get_priority(&(i, j)).copied() {
                if pri + 1 < npri {
                    self.frontier
                        .change_priority(&(i, j), C::f(pri, 1, (u, v), self.map.target()));
                    *self.parent.get_mut(&(i, j)).unwrap() = (u, v);
                }
            } else {
                self.frontier
                    .push((i, j), C::f(pri, 1, (u, v), self.map.target()));
                self.parent.insert((i, j), (u, v));
            }
        }

        Ok(None)
    }

    fn get_route(&self) -> Vec<(usize, usize)> {
        let mut r = Vec::new();
        let (mut i, mut j) = self.map.current();
        r.push((i, j));
        while let Some((u, v)) = self.parent.get(&(i, j)).copied() {
            i = u;
            j = v;
            r.push((i, j));
        }
        r
    }

    fn draw(
        &self,
        display: &mut SimulatorDisplay<Rgb888>,
    ) -> Result<(), <SimulatorDisplay<Rgb888> as DrawTarget<Rgb888>>::Error> {
        use embedded_graphics::prelude::*;

        self.map.draw(display)?;

        for (k, (i, j)) in self.get_route().into_iter().enumerate() {
            self.map.draw_cell(
                display,
                i as i32,
                j as i32,
                if k == 0 { Rgb888::CYAN } else { Rgb888::BLUE },
            )?;
        }

        Ok(())
    }
}
