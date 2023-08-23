use wasm_component_layer::*;

const WASM: &[u8] = include_bytes!("single_component/component.wasm");

pub fn main() {
    let engine = Engine::new(wasmi::Engine::default());
    let mut store = Store::new(&engine, ());
    
    let component = Component::new(&engine, WASM).unwrap();
    let instance = Linker::default().instantiate(&mut store, &component).unwrap();

    let interface = instance.exports().instance(&"test:guest/foo".try_into().unwrap()).unwrap();
    let select_nth = interface.func("select-nth").unwrap().typed::<(Vec<String>, u32), (String,)>().unwrap();

    let example = ["a", "b", "c"].iter().map(ToString::to_string).collect::<Vec<_>>();
    println!("Calling select-nth({example:?}, 1) == {}", select_nth.call(&mut store, (example.clone(), 1)).unwrap().0);
    // Prints 'Calling select-nth(["a", "b", "c"], 1) == b'
}