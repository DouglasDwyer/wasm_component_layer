use wasm_component_layer::*;

// The bytes of the component.
const WASM: &[u8] = include_bytes!("single_component/component.wasm");

pub fn main() {
    // Create a new engine for instantiating a component.
    let engine = Engine::new(wasmi_runtime_layer::Engine::default());

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
        .instance(&"test:guest/foo".try_into().unwrap())
        .unwrap();
    // Get the function for selecting a list element.
    let select_nth = interface
        .func("select-nth")
        .unwrap()
        .typed::<(Vec<String>, u32), String>()
        .unwrap();

    // Create an example list to test upon.
    let example = ["a", "b", "c"]
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    println!(
        "Calling select-nth({example:?}, 1) == {}",
        select_nth.call(&mut store, (example.clone(), 1)).unwrap()
    );
    // Prints 'Calling select-nth(["a", "b", "c"], 1) == b'
}
