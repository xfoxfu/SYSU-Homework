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
        // 获得下一次的节点
        let ((u, v), Reverse(pri)) = self.frontier.pop().ok_or(SearchError::Failure)?; // 若无法达到目的节点（队列为空），返回错误
        let cell = *self.map.get(u, v);
        self.map.explore(u, v);
        // 若达到边界，返回路径信息
        if cell == Cell::End {
            return Ok(Some(self.get_route()));
        }
        for (i, j) in self.map.adjacent(u, v).into_iter() {
            if !self.map.get(i, j).reachable() {
                continue;
            }
            let dist = C::f(pri, 1, (u, v), self.map.target());
            if let Some(Reverse(npri)) = self.frontier.get_priority(&(i, j)).copied() {
                // 如果已经存在队列中，且需要更新距离，更新距离信息
                if dist.0 < npri {
                    self.frontier.change_priority(&(i, j), dist);
                    *self.parent.get_mut(&(i, j)).unwrap() = (u, v);
                }
            } else {
                // 否则直接插入
                self.frontier.push((i, j), dist);
                self.parent.insert((i, j), (u, v));
            }
        }

        // 需要继续迭代
        Ok(None)
    }

    fn get_route(&self) -> Vec<(usize, usize)> {
        let mut r = Vec::new();
        let (mut i, mut j) = self.map.current();
        // 加入终点
        r.push((i, j));
        // 持续迭代来追加前序节点
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
