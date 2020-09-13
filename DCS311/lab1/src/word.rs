use std::{collections::HashMap, sync::atomic::AtomicUsize};

pub struct WordContext(HashMap<String, usize>, AtomicUsize);

impl WordContext {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn get_id<'a>(&'a mut self, s: &str) -> Word<'a> {
        // self.0.contains(value)
    }
}

pub struct Word<'a>(usize, &'a WordContext);

impl<'a> Word<'a> {
    pub fn new(ctx: &'a WordContext, id: usize) -> Self {
        Self(id, ctx)
    }
}
