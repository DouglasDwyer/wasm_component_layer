wit_bindgen::generate!({
    path: "../wit/component.wit",
    world: "outer-guest",
});
export!(Outer);

struct Outer;

impl exports::test::guest::outer::Guest for Outer {
    fn make_resource(a: i32) -> test::guest::inner::Myresource {
        test::guest::inner::Myresource::new(a)
    }

    fn consume_resource(r: test::guest::inner::Myresource) -> String {
        test::guest::inner::consume_resource(r)
    }

    fn borrow_resource(r: &test::guest::inner::Myresource) -> String {
        test::guest::inner::borrow_resource(r)
    }
}
