use anyhow::*;
use wasm_component_layer::*;

// The bytes of the component.
const WASM: &[u8] = include_bytes!("resource/component.wasm");

pub fn main() {
    // Create a new engine for instantiating a component.
    let engine = Engine::new(wasmi_runtime_layer::Engine::default());

    // Create a store for managing WASM data and any custom user-defined state.
    let mut store = Store::new(&engine, ());

    // Create a type to represent the host-defined resource.
    let resource_ty = ResourceType::new::<MyResource>(None);

    // Create the resource constructor function.
    let resource_ty_clone = resource_ty.clone();
    let resource_constructor = Func::new(
        &mut store,
        FuncType::new([ValueType::S32], [ValueType::Own(resource_ty.clone())]),
        move |ctx, args, results| {
            let Value::S32(a) = args[0] else {
                bail!("Incorrect input type.")
            };
            results[0] = Value::Own(ResourceOwn::new(
                ctx,
                MyResource(a),
                resource_ty_clone.clone(),
            )?);
            Ok(())
        },
    );

    // Create the resource print function.
    let resource_print = Func::new(
        &mut store,
        FuncType::new([ValueType::Borrow(resource_ty.clone())], []),
        |ctx, args, _| {
            let Value::Borrow(res) = &args[0] else {
                bail!("Incorrect input type.")
            };
            println!(
                "Called print with value {:?}",
                res.rep::<MyResource, _, _>(&ctx.as_context()).unwrap()
            );
            Ok(())
        },
    );

    // Parse the component bytes and load its imports and exports.
    let component = Component::new(&engine, WASM).unwrap();
    // Create a linker that will be used to resolve the component's imports, if any.
    let mut linker = Linker::default();

    // Create the interface containing the resource.
    let resource_interface = linker
        .define_instance("test:guest/bar".try_into().unwrap())
        .unwrap();

    // Defines the necessary host-side functions for using the resource.
    resource_interface
        .define_resource("myresource", resource_ty)
        .unwrap();
    resource_interface
        .define_func("[constructor]myresource", resource_constructor)
        .unwrap();
    resource_interface
        .define_func("[method]myresource.print-a", resource_print)
        .unwrap();

    // Create an instance of the component using the linker.
    let instance = linker.instantiate(&mut store, &component).unwrap();

    // Get the interface that the interface exports.
    let interface = instance
        .exports()
        .instance(&"test:guest/foo".try_into().unwrap())
        .unwrap();

    // Get the function for creating and using a resource.
    let use_resource = interface
        .func("use-resource")
        .unwrap()
        .typed::<(), ()>()
        .unwrap();

    // Prints 'Called print with value MyResource(42)'
    use_resource.call(&mut store, ()).unwrap();
}

#[derive(Debug)]
pub struct MyResource(pub i32);
