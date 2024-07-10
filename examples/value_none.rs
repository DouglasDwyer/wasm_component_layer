use anyhow::*;
use wasm_component_layer::*;

// The bytes of the component.
const WASM: &[u8] = include_bytes!("value_none/component.wasm");

pub fn main() {
    let engine = Engine::new(wasmi_runtime_layer::Engine::default());
    let mut store = Store::new(&engine, ());
    let component = Component::new(&engine, WASM).unwrap();
    let mut linker = Linker::default();

    let host_interface = linker
        .define_instance("test:guest/host".try_into().unwrap())
        .unwrap();
    host_interface
        .define_func(
            "log",
            Func::new(
                &mut store,
                FuncType::new([ValueType::String], []),
                move |_, params, _| {
                    if let Value::String(message) = &params[0] {
                        println!("[GuestLog] {message}");
                    }
                    Ok(())
                },
            ),
        )
        .unwrap();

    let ty_result_option = OptionType::new(ValueType::S32);
    host_interface
        .define_func(
            "result-option",
            Func::new(
                &mut store,
                FuncType::new(
                    [ValueType::Bool],
                    [ValueType::Option(ty_result_option.clone())],
                ),
                move |_, params, results| {
                    if let Value::Bool(is_some) = params[0] {
                        println!("[HostLog]: result-option.params ( is_some: {:?} )", is_some);
                        let result = if is_some { Some(Value::S32(1)) } else { None };
                        results[0] = Value::Option(
                            OptionValue::new(ty_result_option.clone(), result).unwrap(),
                        );
                    };
                    Ok(())
                },
            ),
        )
        .unwrap();

    let ty_result_result = ResultType::new(Some(ValueType::S32), Some(ValueType::S32));
    host_interface
        .define_func(
            "result-result",
            Func::new(
                &mut store,
                FuncType::new(
                    [ValueType::Bool],
                    [ValueType::Result(ty_result_result.clone())],
                ),
                move |_, params, results| {
                    if let Value::Bool(is_ok) = params[0] {
                        println!("[HostLog]: result-result.params ( is_ok {:?} )", is_ok);
                        let result = if is_ok {
                            Result::Ok(Some(Value::S32(1)))
                        } else {
                            Result::Err(Some(Value::S32(0)))
                        };
                        results[0] = Value::Result(
                            ResultValue::new(ty_result_result.clone(), result).unwrap(),
                        );
                    };
                    Ok(())
                },
            ),
        )
        .unwrap();

    let ty_result_result_ok = ResultType::new(Some(ValueType::S32), None);
    host_interface
        .define_func(
            "result-result-ok",
            Func::new(
                &mut store,
                FuncType::new(
                    [ValueType::Bool],
                    [ValueType::Result(ty_result_result_ok.clone())],
                ),
                move |_, params, results| {
                    if let Value::Bool(is_ok) = params[0] {
                        println!("[HostLog]: result-result-ok.params ( is_ok {:?} )", is_ok);
                        let result = if is_ok {
                            Result::Ok(Some(Value::S32(1)))
                        } else {
                            Result::Err(None)
                        };
                        results[0] = Value::Result(
                            ResultValue::new(ty_result_result_ok.clone(), result).unwrap(),
                        );
                    };
                    Ok(())
                },
            ),
        )
        .unwrap();

    let ty_result_result_err = ResultType::new(None, Some(ValueType::S32));
    host_interface
        .define_func(
            "result-result-err",
            Func::new(
                &mut store,
                FuncType::new(
                    [ValueType::Bool],
                    [ValueType::Result(ty_result_result_err.clone())],
                ),
                move |_, params, results| {
                    if let Value::Bool(is_ok) = params[0] {
                        println!("[HostLog]: result-result-err.params ( is_ok {:?} )", is_ok);
                        let result = if is_ok {
                            Result::Ok(None)
                        } else {
                            Result::Err(Some(Value::S32(0)))
                        };
                        results[0] = Value::Result(
                            ResultValue::new(ty_result_result_err.clone(), result).unwrap(),
                        );
                    };
                    Ok(())
                },
            ),
        )
        .unwrap();

    let ty_result_result_none = ResultType::new(None, None);
    host_interface
        .define_func(
            "result-result-none",
            Func::new(
                &mut store,
                FuncType::new(
                    [ValueType::Bool],
                    [ValueType::Result(ty_result_result_none.clone())],
                ),
                move |_, params, results| {
                    if let Value::Bool(is_ok) = params[0] {
                        println!("[HostLog]: result-result-none.params ( is_ok {:?} )", is_ok);
                        let result = if is_ok {
                            Result::Ok(None)
                        } else {
                            Result::Err(None)
                        };
                        results[0] = Value::Result(
                            ResultValue::new(ty_result_result_none.clone(), result).unwrap(),
                        );
                    };
                    Ok(())
                },
            ),
        )
        .unwrap();

    let ty_params_option = OptionType::new(ValueType::S32);
    host_interface
        .define_func(
            "params-option",
            Func::new(
                &mut store,
                FuncType::new(
                    [ValueType::Option(ty_params_option.clone())],
                    [ValueType::String],
                ),
                move |_, params, results| {
                    if let Value::Option(param) = &params[0] {
                        let param = match **param {
                            Some(ref value) => Some(value),
                            None => None,
                        };
                        println!("[HostLog]: params-option.params {:?}", param);
                        results[0] = Value::String(format!("{:?}", param).into());
                    }
                    Ok(())
                },
            ),
        )
        .unwrap();

    // let ty_params_result = ResultType::new(Some(ValueType::S32), Some(ValueType::S32));
    // host_interface
    //     .define_func(
    //         "params-result",
    //         Func::new(
    //             &mut store,
    //             FuncType::new(
    //                 [ValueType::Result(ty_params_result.clone())],
    //                 [ValueType::String],
    //             ),
    //             move |_, params, results| {
    //                 if let Value::Result(param) = &params[0] {
    //                     let param = match **param {
    //                         std::result::Result::Ok(ref value) => {
    //                             std::result::Result::Ok(value.clone().unwrap())
    //                         }
    //                         std::result::Result::Err(ref value) => {
    //                             std::result::Result::Err(value.clone().unwrap())
    //                         }
    //                     };
    //                     println!("[HostLog]: params-result.params {:?}", param);
    //                     results[0] = Value::String(format!("{:?}", param).into());
    //                 }
    //                 Ok(())
    //             },
    //         ),
    //     )
    //     .unwrap();

    let instance = linker.instantiate(&mut store, &component).unwrap();
    let interface = instance
        .exports()
        .instance(&"test:guest/run".try_into().unwrap())
        .unwrap();
    let start = interface.func("start").unwrap().typed::<(), ()>().unwrap();
    start.call(&mut store, ()).unwrap();
}
