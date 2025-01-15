#![allow(dead_code)]
#[allow(warnings)]
#[cfg_attr(rustfmt, rustfmt_skip)]
mod bindings;

use std::cell::RefCell;

use bindings::exports::test::guest::foo::{Guest, GuestBar};
use bindings::test::guest::log::log;

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

bindings::export!(Component with_types_in bindings);
