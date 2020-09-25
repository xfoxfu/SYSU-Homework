use super::BinaryTree;
use crate::case::*;
use std::sync::atomic::{AtomicUsize, Ordering};

static ID: AtomicUsize = AtomicUsize::new(0);

fn print_tree_node<W, V>(w: &mut W, tree: &BinaryTree<V, Label>) -> usize
where
    W: std::io::Write,
    V: Fn(&Case) -> bool,
{
    match tree {
        BinaryTree::Vertex {
            predicate,
            left_child,
            right_child,
        } => {
            let id = ID.fetch_add(1, Ordering::Relaxed);
            writeln!(
                w,
                r#"{} [label="{}"];"#,
                id,
                super::builder_impl::_debug_print_fn(predicate)
            )
            .unwrap();
            let left = print_tree_node(w, left_child);
            let right = print_tree_node(w, right_child);
            // ww,riteln!("{} -> {};", id, left);
            writeln!(w, "{} -> {};", id, right).unwrap();
            writeln!(
                w,
                r#"{} -> {} [labeldistance=2.5, labelangle=45, headlabel="T"] ;"#,
                id, left
            )
            .unwrap();
            id
        }
        BinaryTree::Leaf { value } => {
            let id = ID.fetch_add(1, Ordering::Relaxed);
            writeln!(w, r#"{} [label="{:?}"];"#, id, value).unwrap();
            id
        }
    }
}

pub fn print_tree<W, V>(w: &mut W, tree: &BinaryTree<V, Label>)
where
    W: std::io::Write,
    V: Fn(&Case) -> bool,
{
    writeln!(w, "digraph Tree {{").unwrap();
    writeln!(w, "node [shape=box] ;").unwrap();
    print_tree_node(w, tree);
    writeln!(w, "}}").unwrap();
}
