wit_bindgen::generate!({
    path: "wit/world.wit"
});

export!(Component);

use std::cell::RefCell;

use exports::test::guest::foo::{Guest, GuestBar};
use test::guest::log::log;

struct Component {
    val: RefCell<i32>,
}

impl Guest for Component {
    type Bar = Self;
}

impl GuestBar for Component {
    fn new(val: i32) -> Self {
        log(&format!("Creating new Component with value: {}", val));
        Component {
            val: RefCell::new(val),
        }
    }

    fn value(&self) -> i32 {
        *self.val.borrow()
    }
}
