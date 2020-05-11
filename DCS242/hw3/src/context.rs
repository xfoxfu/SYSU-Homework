use std::collections::HashMap;

pub struct Context {
    pub fn_is_safe: HashMap<String, bool>,
}

impl Context {
    pub fn new() -> Self {
        Context {
            fn_is_safe: HashMap::new(),
        }
    }

    pub fn set_fn(&mut self, name: String, is_safe: bool) {
        self.fn_is_safe.insert(name, is_safe);
    }

    pub fn get_fn(&mut self, name: &str) -> Option<bool> {
        self.fn_is_safe.get(name).copied()
    }
}
