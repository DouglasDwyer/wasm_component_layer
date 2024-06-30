wit_bindgen::generate!({
    path: "../wit/component.wit",
    world: "inner-guest",
});
export!(Inner);

struct Inner;

impl exports::test::guest::inner::Guest for Inner {
    type Myresource = MyResource;

    fn consume_resource(r: crate::exports::test::guest::inner::Myresource) -> String {
        let r: MyResource = r.into_inner();
        exports::test::guest::inner::GuestMyresource::display(&r)
    }

    fn borrow_resource(r: crate::exports::test::guest::inner::MyresourceBorrow) -> String {
        exports::test::guest::inner::GuestMyresource::display(r.get::<MyResource>())
    }
}

struct MyResource(i32);

impl exports::test::guest::inner::GuestMyresource for MyResource {
    fn new(a: i32) -> Self {
        Self(a)
    }

    fn display(&self) -> String {
        format!("{}", self.0)
    }
}
