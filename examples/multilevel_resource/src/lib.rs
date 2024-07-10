wit_bindgen::generate!({
    path: "wit/component.wit",
    exports: {
        "test:guest/run": Run
    }
});

struct Run;

impl exports::test::guest::run::Guest for Run {
    fn start() {
        let error = test::guest::error::Error::new();
        test::guest::types::borrow_error(&error);

        let streams_error = test::guest::streams::StreamsError::new();
        test::guest::types::borrow_streams_error(&streams_error);
    }
}