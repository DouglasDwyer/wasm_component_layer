# wasm_component_layer

[![Crates.io](https://img.shields.io/crates/v/wasm_component_layer.svg)](https://crates.io/crates/wasm_component_layer)
[![Docs.rs](https://docs.rs/wasm_component_layer/badge.svg)](https://docs.rs/wasm_component_layer)
[![Unsafe Forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

`wasm_component_layer` is a runtime agnostic implementation of the [WebAssembly component model](https://github.com/WebAssembly/component-model).
It supports loading and linking WASM components, inspecting and generating component interface types at runtime, and more atop any WebAssembly backend. The implementation is based upon the [`wasmtime`](https://github.com/bytecodealliance/wasmtime), [`js-component-bindgen`](https://github.com/bytecodealliance/jco), and [`wit-parser`](https://github.com/bytecodealliance/wasm-tools/tree/main) crates.

## Usage

To use `wasm_component_layer`, a runtime is required. The [`wasm_runtime_layer`](https://github.com/DouglasDwyer/wasm_runtime_layer) crate provides the common interface used for WebAssembly runtimes, so when using this crate it must also be added to the `Cargo.toml` file with the appropriate runtime selected. For instance, the examples in this repository use the [`wasmi_runtime_layer`](https://crates.io/crates/wasmi_runtime_layer) runtime:

```toml
wasm_component_layer = "0.1.16"
wasmi_runtime_layer = "0.31.0"
# wasmtime_runtime_layer = "21.0.0"
# js_wasm_runtime_layer = "0.4.0"
```

The following is a small overview of `wasm_component_layer`'s API. The complete example may be found in the [examples folder](/examples). Consider a WASM component with the following WIT:

```wit
package test:guest

interface foo {
    // Selects the item in position n within list x
    select-nth: func(x: list<string>, n: u32) -> string
}

world guest {
    export foo
}
```

The component can be loaded into `wasm_component_layer` and invoked as follows:

```rust
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
    let interface = instance.exports().instance(&"test:guest/foo".try_into().unwrap()).unwrap();
    // Get the function for selecting a list element.
    let select_nth = interface.func("select-nth").unwrap().typed::<(Vec<String>, u32), String>().unwrap();

    // Create an example list to test upon.
    let example = ["a", "b", "c"].iter().map(ToString::to_string).collect::<Vec<_>>();

    println!("Calling select-nth({example:?}, 1) == {}", select_nth.call(&mut store, (example.clone(), 1)).unwrap());
    // Prints 'Calling select-nth(["a", "b", "c"], 1) == b'
}
```

## Supported capabilities

`wasm_component_layer` supports the following major capabilities:

- Parsing and instantiating WASM component binaries
- Runtime generation of component interface types
- Specialized list types for faster lifting/lowering
- Structural equality of component interface types, as mandated by the spec
- Support for guest resources
- Support for strongly-typed host resources with destructors

The following things have yet to be implemented:

- String transcoders
- A macro for generating host bindings
- More comprehensive tests
- Subtyping

## Optional features

**serde** - Allows for the serialization of identifiers, types, and values. Note that serializing resources is not allowed, because resources may be tied to specific instances.

## Examples
```shell
# EXAMPLE_NAME: [single_component|resource|multilevel_resource|...]

# build example wasm file
cd examples/EXAMPLE_NAME # cd examples/single_component
rustup toolchain install nightly
rustup override set nightly
cargo build
wasm-tools component new target/wasm32-unknown-unknown/debug/component_example.wasm -o component.wasm
wasm-tools print component.wasm -o component.wat

# run example in host implementation
cd ../../
cargo run --example EXAMPLE_NAME # cargo run --example single_component
```