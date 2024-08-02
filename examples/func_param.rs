use anyhow::*;
use wasm_component_layer::*;

// The bytes of the component.
const WASM: &[u8] = include_bytes!("func_param/component.wasm");

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
            "param-list",
            Func::new(
                &mut store,
                FuncType::new([ValueType::List(ListType::new(ValueType::S16))], []),
                move |_, params, _results| {
                    println!("[HostLog] param-list.params");
                    println!(" └ {:?}", params);
                    Ok(())
                },
            ),
        )
        .unwrap();

    host_interface
        .define_func(
            "param-record",
            Func::new(
                &mut store,
                FuncType::new(
                    [ValueType::Variant(
                        VariantType::new(
                            None,
                            [
                                VariantCase::new("open", None),
                                VariantCase::new("close", Some(ValueType::U64)),
                                VariantCase::new(
                                    "click",
                                    Some(ValueType::Variant(
                                        VariantType::new(
                                            None,
                                            [
                                                VariantCase::new("up", None),
                                                VariantCase::new("press", Some(ValueType::U8)),
                                                VariantCase::new("down", None),
                                            ],
                                        )
                                        .unwrap(),
                                    )),
                                ),
                            ],
                        )
                        .unwrap(),
                    )],
                    [],
                ),
                move |_, params, _| {
                    println!("[HostLog] param-record.params");
                    println!(" └ {:?}", params);
                    Ok(())
                },
            ),
        )
        .unwrap();

    host_interface
        .define_func(
            "param-option",
            Func::new(
                &mut store,
                FuncType::new([ValueType::Option(OptionType::new(ValueType::U16))], []),
                move |_, params, _results| {
                    println!("[HostLog] param-option.params");
                    println!(" └ {:?}", params);
                    Ok(())
                },
            ),
        )
        .unwrap();

    host_interface
        .define_func(
            "param-result-all",
            Func::new(
                &mut store,
                FuncType::new(
                    [ValueType::Result(ResultType::new(
                        Some(ValueType::U8),
                        Some(ValueType::U8),
                    ))],
                    [],
                ),
                move |_, params, _results| {
                    println!("[HostLog] param-result-all.params");
                    println!(" └ {:?}", params);
                    Ok(())
                },
            ),
        )
        .unwrap();

    host_interface
        .define_func(
            "param-result-ok",
            Func::new(
                &mut store,
                FuncType::new(
                    [ValueType::Result(ResultType::new(
                        Some(ValueType::U8),
                        None,
                    ))],
                    [],
                ),
                move |_, params, _results| {
                    println!("[HostLog] param-result-ok.params");
                    println!(" └ {:?}", params);
                    Ok(())
                },
            ),
        )
        .unwrap();

    host_interface
        .define_func(
            "param-result-err",
            Func::new(
                &mut store,
                FuncType::new(
                    [ValueType::Result(ResultType::new(
                        None,
                        Some(ValueType::U8),
                    ))],
                    [],
                ),
                move |_, params, _results| {
                    println!("[HostLog] param-result-err.params");
                    println!(" └ {:?}", params);
                    Ok(())
                },
            ),
        )
        .unwrap();

    host_interface
        .define_func(
            "param-result-none",
            Func::new(
                &mut store,
                FuncType::new([ValueType::Result(ResultType::new(None, None))], []),
                move |_, params, _results| {
                    println!("[HostLog] param-result-none.params");
                    println!(" └ {:?}", params);
                    Ok(())
                },
            ),
        )
        .unwrap();

    host_interface
        .define_func(
            "param-mult",
            Func::new(
                &mut store,
                FuncType::new(
                    [
                        ValueType::List(ListType::new(ValueType::String)),
                        ValueType::Variant(
                            VariantType::new(
                                None,
                                [
                                    VariantCase::new("open", None),
                                    VariantCase::new("close", Some(ValueType::U64)),
                                    VariantCase::new(
                                        "click",
                                        Some(ValueType::Variant(
                                            VariantType::new(
                                                None,
                                                [
                                                    VariantCase::new("up", None),
                                                    VariantCase::new("press", Some(ValueType::U8)),
                                                    VariantCase::new("down", None),
                                                ],
                                            )
                                            .unwrap(),
                                        )),
                                    ),
                                ],
                            )
                            .unwrap(),
                        ),
                        ValueType::Option(OptionType::new(ValueType::String)),
                        ValueType::Result(ResultType::new(
                            Some(ValueType::String),
                            Some(ValueType::String),
                        )),
                    ],
                    [],
                ),
                move |_, params, _results| {
                    println!("[HostLog] param-mult.params");
                    println!(" └ {:?}", params);
                    Ok(())
                },
            ),
        )
        .unwrap();

    linker
        .instantiate(&mut store, &component)
        .unwrap()
        .exports()
        .instance(&"test:guest/run".try_into().unwrap())
        .unwrap()
        .func("start")
        .unwrap()
        .typed::<(), ()>()
        .unwrap()
        .call(&mut store, ())
        .unwrap();
}
