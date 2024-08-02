wit_bindgen::generate!({
    path: "wit/component.wit",
    exports: {
        "test:guest/run": Run
    }
});

struct Run;
impl exports::test::guest::run::Guest for Run {
    fn start() {
        test::guest::host::param_list(&[11, 22, 33]);

        test::guest::host::param_record(test::guest::host::Event::Open);
        test::guest::host::param_record(test::guest::host::Event::Close(44));
        test::guest::host::param_record(test::guest::host::Event::Click(
            test::guest::host::ClickType::Press(8),
        ));
        
        test::guest::host::param_option(Some(66));
        test::guest::host::param_option(None);

        test::guest::host::param_result_all(Ok(71));
        test::guest::host::param_result_all(Err(72));
        test::guest::host::param_result_ok(Ok(88));
        test::guest::host::param_result_ok(Err(()));
        test::guest::host::param_result_err(Ok(()));
        test::guest::host::param_result_err(Err(99));
        test::guest::host::param_result_none(Ok(()));
        test::guest::host::param_result_none(Err(()));

        test::guest::host::param_mult(
            &[
                "param-list".to_string(),
                "index-1".to_string(),
                "index-2".to_string(),
            ],
            test::guest::host::Event::Close(6),
            Some("param-option"),
            Ok("result-all-ok"),
        );
        test::guest::host::param_mult(
            &[],
            test::guest::host::Event::Click(test::guest::host::ClickType::Press(8)),
            None,
            Err("result-all-err"),
        );
    }
}
