use wasm_component_layer::*;

// The bytes of the component.
const WASM: &[u8] = include_bytes!("transfer/component.wasm");

pub fn main() {
    // Create a new engine for instantiating a component.
    let engine = Engine::new(wasmtime_runtime_layer::Engine::default());

    // Create a store for managing WASM data and any custom user-defined state.
    let mut store = Store::new(&engine, ());

    // Parse the component bytes and load its imports and exports.
    let component = Component::new(&engine, WASM).unwrap();
    // Create a linker that will be used to resolve the component's imports, if any.
    let linker = Linker::default();
    // Create an instance of the component using the linker.
    let instance = linker.instantiate(&mut store, &component).unwrap();

    // Get the interface that the interface exports.
    let interface = instance
        .exports()
        .instance(&"test:guest/outer".try_into().unwrap())
        .unwrap();
    // Get the function for creating a resource.
    let make_resource = interface
        .func("make-resource")
        .unwrap();
    let borrow_resource = interface
        .func("borrow-resource")
        .unwrap();
    let consume_resource = interface
        .func("consume-resource")
        .unwrap();

    let mut results = [Value::U8(0)];
    make_resource.call(&mut store, &[Value::S32(42)], &mut results).unwrap();
    let [Value::Own(resource)] = results else {
        panic!("make-resource returned {results:?}, expected resource");
    };
    println!("Calling make-resource(42) == {resource:?}");

    let resource_borrow = resource.borrow(&mut store).unwrap();

    let mut results = [Value::U8(0)];
    borrow_resource.call(&mut store, &[Value::Borrow(resource_borrow)], &mut results).unwrap();
    let [Value::String(result)] = results else {
        panic!("borrow-resource returned {results:?}, expected string");
    };
    println!("Calling borrow-resource(&<..>) == {result:?}");

    let mut results = [Value::U8(0)];
    consume_resource.call(&mut store, &[Value::Own(resource)], &mut results).unwrap();
    let [Value::String(result)] = results else {
        panic!("consume-resource returned {results:?}, expected string");
    };
    println!("Calling consume-resource(<..>) == {result:?}");
}
