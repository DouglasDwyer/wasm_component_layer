wit_bindgen::generate!({
    path: "wit/component.wit",
    exports: {
        "test:guest/foo": Foo
    }
});

struct Foo;

impl exports::test::guest::foo::Guest for Foo {
    fn select_nth(x: Vec<String>, i: u32) -> String {
        x.into_iter().nth(i as usize).expect("Could not get value.")
    }
}