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

#[cfg(debug_assertions)]
pub(crate) fn _debug_print_fn<F>(f: F) -> &'static str
where
    F: Fn(&Case) -> bool,
{
    let mut v = Vec::new();
    for b in [Buying::High, Buying::Low, Buying::Med, Buying::Vhigh]
        .iter()
        .copied()
    {
        for m in [Maint::High, Maint::Low, Maint::Med, Maint::Vhigh]
            .iter()
            .copied()
        {
            for d in [Doors::Two, Doors::Three, Doors::Four, Doors::FiveOrMore]
                .iter()
                .copied()
            {
                for p in [Persons::Two, Persons::Four, Persons::More].iter().copied() {
                    for l in [LugBoot::Big, LugBoot::Med, LugBoot::Small].iter().copied() {
                        for s in [Safety::High, Safety::Low, Safety::Med].iter().copied() {
                            v.push(Case::new(b, m, d, p, l, s, Label::True));
                        }
                    }
                }
            }
        }
    }
    let id = FNS
        .iter()
        .enumerate()
        .find(|t| v.iter().filter(|c| f(c)).all(|c| t.1(c)))
        .unwrap()
        .0;
    match id {
        0 => "Buying::High",
        1 => "Buying::Low",
        2 => "Buying::Med",
        3 => "Buying::Vhigh",
        4 => "Maint::High",
        5 => "Maint::Low",
        6 => "Maint::Med",
        7 => "Maint::Vhigh",
        8 => "Doors::Two",
        9 => "Doors::Three",
        10 => "Doors::Four",
        11 => "Doors::FiveOrMore",
        12 => "Persons::Two",
        13 => "Persons::Four",
        14 => "Persons::More",
        15 => "LugBoot::Big",
        16 => "LugBoot::Med",
        17 => "LugBoot::Small",
        18 => "Safety::High",
        19 => "Safety::Low",
        20 => "Safety::Med",
        _ => "UNEXPECTED",
    }
}

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
        // 边界条件 1，结果属于同一类别
        if self.data.iter().all(|p| p.label == Label::True) {
            return BinaryTree::new_leaf(Label::True);
        } else if self.data.iter().all(|p| p.label == Label::False) {
            return BinaryTree::new_leaf(Label::False);
        }

        let f = self.best_fn();
        // 边界条件 2，无可划分特征
        // 此处通过保证特征选取不能是所有数据完全一致的特征，
        // 避免了边界条件 3 的存在
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
