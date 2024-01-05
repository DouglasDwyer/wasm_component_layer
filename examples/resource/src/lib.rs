wit_bindgen::generate!({
    path: "wit/component.wit",
    exports: {
        "test:guest/foo": Foo
    }
});

struct Foo;

impl exports::test::guest::foo::Guest for Foo {
    fn use_resource() {
        let resource = test::guest::bar::Myresource::new(42);
        resource.print_a();
    }

    fn consume_resource(_: test::guest::bar::Myresource) { }
}