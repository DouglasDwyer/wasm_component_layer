wit_bindgen::generate!({
    path: "wit/component.wit",
    exports: {
        "test:guest/run": Run
    }
});

struct Run;
impl exports::test::guest::run::Guest for Run {
    fn start() {
        test::guest::host::param_list(&[1, 2, 3], &[3, 4, 5]);
    }
}
