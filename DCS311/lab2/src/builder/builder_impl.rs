use super::BinaryTree;
use crate::case::*;
use crate::selector::Selector;
use std::marker::PhantomData;

pub struct Builder<S> {
    _sel: PhantomData<S>,
    data: Vec<Case>,
}

impl<'a, S> Builder<S>
where
    S: Selector,
{
    pub fn with_data(data: Vec<Case>) -> Self {
        Self {
            data,
            _sel: PhantomData,
        }
    }
}

static FNS: &[CasePredicateFn] = &[
    |c: &Case| c.buying == Buying::High,
    |c: &Case| c.buying == Buying::Low,
    |c: &Case| c.buying == Buying::Med,
    |c: &Case| c.buying == Buying::Vhigh,
    |c: &Case| c.maint == Maint::High,
    |c: &Case| c.maint == Maint::Low,
    |c: &Case| c.maint == Maint::Med,
    |c: &Case| c.maint == Maint::Vhigh,
    |c: &Case| c.doors == Doors::Two,
    |c: &Case| c.doors == Doors::Three,
    |c: &Case| c.doors == Doors::Four,
    |c: &Case| c.doors == Doors::FiveOrMore,
    |c: &Case| c.persons == Persons::Two,
    |c: &Case| c.persons == Persons::Four,
    |c: &Case| c.persons == Persons::More,
    |c: &Case| c.lug_boot == LugBoot::Big,
    |c: &Case| c.lug_boot == LugBoot::Med,
    |c: &Case| c.lug_boot == LugBoot::Small,
    |c: &Case| c.safety == Safety::High,
    |c: &Case| c.safety == Safety::Low,
    |c: &Case| c.safety == Safety::Med,
];

impl<S> Builder<S>
where
    S: Selector,
{
    pub fn best_fn(&self) -> Option<CasePredicateFn> {
        S::best_fn(&self.data, FNS.iter()).cloned()
    }

    pub fn vote_majority(&self) -> Label {
        [Label::True, Label::False, Label::Unlabeled]
            .iter()
            .map(|l| (*l, self.data.iter().filter(|c| c.label == *l).count()))
            .max_by_key(|(_, c)| *c)
            .unwrap()
            .0
    }

    pub fn into_tree(self) -> BinaryTree<CasePredicateFn, Label> {
        let f = self.best_fn();
        if f.is_none() {
            return BinaryTree::new_leaf(self.vote_majority());
        }
        let f = f.unwrap();

        // 左子树是真值树
        let mut left = Vec::new();
        let mut right = Vec::new();

        for data in self.data.into_iter() {
            if f(&data) {
                left.push(data)
            } else {
                right.push(data)
            }
        }

        let left_tree = Self::with_data(left).into_tree().into_boxed();
        let right_tree = Self::with_data(right).into_tree().into_boxed();

        BinaryTree::new_vertex(f, left_tree, right_tree)
    }
}
