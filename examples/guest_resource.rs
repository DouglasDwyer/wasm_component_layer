use anyhow::*;
use wasm_component_layer::*;

// The bytes of the component.
const WASM: &[u8] =
    include_bytes!("./guest_resource/target/wasm32-unknown-unknown/debug/guest_resource.wasm");

pub fn main() {
    // Create a new engine for instantiating a component.
    let engine = Engine::new(wasmi_runtime_layer::Engine::default());

    // Create a store for managing WASM data and any custom user-defined state.
    let mut store = Store::new(&engine, ());

    // Parse the component bytes and load its imports and exports.
    let component = Component::new(&engine, WASM).unwrap();

    // Create a linker that will be used to resolve the component's imports, if any.
    let mut linker = Linker::default();

    // Create a host interface that will be used to define the host functions that the component can call.
    let host_interface = linker
        .define_instance("test:guest/log".try_into().unwrap())
        .unwrap();

    // Define the host function that the component can call.
    // This is our `log` function that will be called by the guest component.
    host_interface
        .define_func(
            "log",
            Func::new(
                &mut store,
                FuncType::new([ValueType::String], []),
                move |_, params, _results| {
                    let params = match &params[0] {
                        Value::String(s) => s,
                        _ => panic!("Unexpected parameter type"),
                    };

                    println!("[HostLog] log");
                    println!(" └ {}", params.to_string());
                    Ok(())
                },
            ),
        )
        .unwrap();

    // Create an instance of the component using the linker.
    let instance = linker.instantiate(&mut store, &component).unwrap();

    // Get the interface that the interface exports.
    let interface = instance
        .exports()
        .instance(&"test:guest/foo".try_into().unwrap())
        .unwrap();

    // Get the function for creating and using a resource.
    let resource_type = interface.resource("bar").unwrap();

    // Call the resource constructor for 'bar' using a direct function call
    let resource_constructor = interface.func("[constructor]bar").unwrap();

    // We need to provide a mutable reference to store the results.
    // This can be any Value type, as it will get overwritten by the result.
    // It is a Value::Bool here but will be overwritten by a Value::Own(ResourceOwn)
    // after we call the constructor.
    let mut results = vec![Value::Bool(false)];

    // Construct the resource with the argument `42`
    resource_constructor
        .call(&mut store, &[Value::S32(42)], &mut results)
        .unwrap();

    let resource = match results[0] {
        Value::Own(ref resource) => resource.clone(),
        _ => panic!("Unexpected result type"),
    };

    let borrow_res = resource.borrow(store.as_context_mut()).unwrap();
    let arguments = vec![Value::Borrow(borrow_res)];

    let mut results = vec![Value::S32(0)];

    // Get the `value` method of the `bar` resource
    let method_bar_value = interface.func("[method]bar.value").unwrap();

    // Call the method `bar.value`, mutate the result
    method_bar_value
        .call(&mut store, &arguments, &mut results)
        .unwrap();

    match results[0] {
        Value::S32(v) => {
            println!("[ResultLog]");
            println!(" └ bar.value() = {}", v);
            assert_eq!(v, 42);
        }
        _ => panic!("Unexpected result type"),
    }
}
