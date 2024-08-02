use anyhow::*;
use wasm_component_layer::*;

// The bytes of the component.
const WASM: &[u8] = include_bytes!("multilevel_resource/component.wasm");

pub fn main() {
    let engine = Engine::new(wasmi_runtime_layer::Engine::default());
    let mut store = Store::new(&engine, ());
    let component = Component::new(&engine, WASM).unwrap();
    let mut linker = Linker::default();

    let errory_ty = ResourceType::new::<Error>(None);
    let errory_ty_clone = errory_ty.clone();
    let error_interface = linker
        .define_instance("test:guest/error".try_into().unwrap())
        .unwrap();
    error_interface
        .define_resource("error", errory_ty.clone())
        .unwrap();
    error_interface
        .define_func(
            "[constructor]error",
            Func::new(
                &mut store,
                FuncType::new([], [ValueType::Own(errory_ty.clone())]),
                move |ctx, _, results| {
                    results[0] = Value::Own(ResourceOwn::new(ctx, Error(), errory_ty.clone())?);
                    Ok(())
                },
            ),
        )
        .unwrap();

    let streams_error_ty = ResourceType::new::<StreamsError>(None);
    let streams_error_ty_clone = streams_error_ty.clone();
    let streams_interface = linker
        .define_instance("test:guest/streams".try_into().unwrap())
        .unwrap();
    streams_interface
        .define_resource("streams-error", streams_error_ty.clone())
        .unwrap();
    streams_interface
        .define_func(
            "[constructor]streams-error",
            Func::new(
                &mut store,
                FuncType::new([], [ValueType::Own(streams_error_ty.clone())]),
                move |ctx, _, results| {
                    results[0] = Value::Own(ResourceOwn::new(
                        ctx,
                        StreamsError(),
                        streams_error_ty.clone(),
                    )?);
                    Ok(())
                },
            ),
        )
        .unwrap();

    let types_interface = linker
        .define_instance("test:guest/types".try_into().unwrap())
        .unwrap();
    types_interface
        .define_func(
            "borrow-error",
            Func::new(
                &mut store,
                FuncType::new([ValueType::Borrow(errory_ty_clone)], []),
                move |_, _, _| {
                    println!("borrow-error");
                    Ok(())
                },
            ),
        )
        .unwrap();

    types_interface
        .define_func(
            "borrow-streams-error",
            Func::new(
                &mut store,
                FuncType::new([ValueType::Borrow(streams_error_ty_clone)], []),
                move |_, _, _| {
                    println!("borrow-streams-error");
                    Ok(())
                },
            ),
        )
        .unwrap();

    let instance = linker.instantiate(&mut store, &component).unwrap();
    let interface = instance
        .exports()
        .instance(&"test:guest/run".try_into().unwrap())
        .unwrap();
    let start = interface.func("start").unwrap().typed::<(), ()>().unwrap();
    start.call(&mut store, ()).unwrap();
}

#[derive(Debug)]
struct Error();

#[derive(Debug)]
struct StreamsError();
