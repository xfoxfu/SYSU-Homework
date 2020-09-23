use crate::case::{Case, Label};

pub enum BinaryTree<V, L> {
    Vertex {
        predicate: V,
        left_child: Box<BinaryTree<V, L>>,
        right_child: Box<BinaryTree<V, L>>,
    },
    Leaf {
        value: L,
    },
}

impl<V, L> BinaryTree<V, L> {
    pub fn new_vertex(
        predicate: V,
        left_child: Box<BinaryTree<V, L>>,
        right_child: Box<BinaryTree<V, L>>,
    ) -> Self {
        BinaryTree::Vertex {
            predicate,
            left_child,
            right_child,
        }
    }

    pub fn new_leaf(value: L) -> Self {
        BinaryTree::Leaf { value }
    }

    pub fn into_boxed(self) -> Box<Self> {
        Box::new(self)
    }
}

impl BinaryTree<fn(&Case) -> bool, Label> {
    pub fn traverse(&self, case: &Case) -> Label {
        match self {
            BinaryTree::Vertex {
                predicate: f,
                left_child: l,
                right_child: r,
            } => {
                if f(case) {
                    l.traverse(case)
                } else {
                    r.traverse(case)
                }
            }
            BinaryTree::Leaf { value } => *value,
        }
    }
}
