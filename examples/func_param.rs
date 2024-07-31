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
                FuncType::new(
                    [
                        ValueType::List(ListType::new(ValueType::U8)),
                        ValueType::List(ListType::new(ValueType::S16)),
                    ],
                    [],
                ),
                move |_, params, _| {
                    println!("[HostLog] param-list.params");
                    println!("          ∟ {:?}", params);
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
