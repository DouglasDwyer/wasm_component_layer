wit_bindgen::generate!({
    path: "wit/component.wit",
    exports: {
        "test:guest/run": Run
    }
});

struct Run;
impl exports::test::guest::run::Guest for Run {
    fn start() {
        // result-option
        test::guest::host::log(&format!(
            "`result-option` result should Some(1) is {:?}",
            test::guest::host::result_option(true)
        ));
        test::guest::host::log(&format!(
            "`result-option` result should None is {:?}",
            test::guest::host::result_option(false)
        ));

        // result-result
        test::guest::host::log(&format!(
            "`result-result` result should Ok(1) is {:?}",
            test::guest::host::result_result(true)
        ));
        test::guest::host::log(&format!(
            "`result-result` result should Err(0) is {:?}",
            test::guest::host::result_result(false)
        ));

        // result-result-ok
        test::guest::host::log(&format!(
            "`result-result-ok` result should Ok(1) is {:?}",
            test::guest::host::result_result_ok(true)
        ));
        test::guest::host::log(&format!(
            "`result-result-ok` result should Err(()) is {:?}",
            test::guest::host::result_result_ok(false)
        ));

        // result-result-err
        test::guest::host::log(&format!(
            "`result-result-err` result should Ok(()) is {:?}",
            test::guest::host::result_result_err(true)
        ));
        test::guest::host::log(&format!(
            "`result-result-err` result should Err(0) is {:?}",
            test::guest::host::result_result_err(false)
        ));

        // result-result-none
        test::guest::host::log(&format!(
            "`result-result-none` result should Ok(()) is {:?}",
            test::guest::host::result_result_none(true)
        ));
        test::guest::host::log(&format!(
            "`result-result-none` result should Err(()) is {:?}",
            test::guest::host::result_result_none(false)
        ));

        // params-option
        test::guest::host::log(&format!(
            "`params-option` param should \"Some(S32(1))\" is {:?}",
            test::guest::host::params_option(Some(1))
        ));
        test::guest::host::log(&format!(
            "`params-option` param should \"None\" is {:?}",
            test::guest::host::params_option(None)
        ));

        // // params-result
        // test::guest::host::log(&format!(
        //     "`params-result` param should \"Ok(S32(1))\" is {:?}",
        //     test::guest::host::params_result(Ok(1))
        // ));
        // test::guest::host::log(&format!(
        //     "`params-result` param should \"Err(S32(0))\" is {:?}",
        //     test::guest::host::params_result(Err(0))
        // ));
    }
}
