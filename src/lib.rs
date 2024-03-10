#![deny(warnings)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

//! `wasm_component_layer` is a runtime agnostic implementation of the [WebAssembly component model](https://github.com/WebAssembly/component-model).
//! It supports loading and linking WASM components, inspecting and generating component interface types at runtime, and more atop any WebAssembly backend. The implementation is based upon the [`wasmtime`](https://github.com/bytecodealliance/wasmtime), [`js-component-bindgen`](https://github.com/bytecodealliance/jco), and [`wit-parser`](https://github.com/bytecodealliance/wasm-tools/tree/main) crates.
//!
//! ## Usage
//!
//! To use `wasm_component_layer`, a runtime is required. The [`wasm_runtime_layer`](https://github.com/DouglasDwyer/wasm_runtime_layer) crate provides the common interface used for WebAssembly runtimes, so when using this crate it must also be added to the `Cargo.toml` file with the appropriate runtime selected. For instance, the examples in this repository use the [`wasmi`](https://github.com/paritytech/wasmi) runtime:
//!
//! ```toml
//! wasm_component_layer = "0.1.0"
//! wasm_runtime_layer = { version = "0.1.1", features = [ "backend_wasmi" ] }
//! ```
//!
//! The following is a small overview of `wasm_component_layer`'s API. The complete example may be found in the [examples folder](/examples). Consider a WASM component with the following WIT:
//!
//! ```wit
//! package test:guest
//!
//! interface foo {
//!     // Selects the item in position n within list x
//!     select-nth: func(x: list<string>, n: u32) -> string
//! }
//!
//! world guest {
//!     export foo
//! }
//! ```
//!
//! The component can be loaded into `wasm_component_layer` and invoked as follows:
//!
//! ```ignore
//! use wasm_component_layer::*;
//!
//! // The bytes of the component.
//! const WASM: &[u8] = include_bytes!("single_component/component.wasm");
//!
//! pub fn main() {
//!     // Create a new engine for instantiating a component.
//!     let engine = Engine::new(wasmi::Engine::default());
//!
//!     // Create a store for managing WASM data and any custom user-defined state.
//!     let mut store = Store::new(&engine, ());
//!     
//!     // Parse the component bytes and load its imports and exports.
//!     let component = Component::new(&engine, WASM).unwrap();
//!     // Create a linker that will be used to resolve the component's imports, if any.
//!     let linker = Linker::default();
//!     // Create an instance of the component using the linker.
//!     let instance = linker.instantiate(&mut store, &component).unwrap();
//!
//!     // Get the interface that the interface exports.
//!     let interface = instance.exports().instance(&"test:guest/foo".try_into().unwrap()).unwrap();
//!     // Get the function for selecting a list element.
//!     let select_nth = interface.func("select-nth").unwrap().typed::<(Vec<String>, u32), String>().unwrap();
//!
//!     // Create an example list to test upon.
//!     let example = ["a", "b", "c"].iter().map(ToString::to_string).collect::<Vec<_>>();
//!     
//!     println!("Calling select-nth({example:?}, 1) == {}", select_nth.call(&mut store, (example.clone(), 1)).unwrap());
//!     // Prints 'Calling select-nth(["a", "b", "c"], 1) == b'
//! }
//! ```
//!
//! ## Features
//!
//! `wasm_component_layer` supports the following major features:
//!
//! - Parsing and instantiating WASM component binaries
//! - Runtime generation of component interface types
//! - Specialized list types for faster
//! - Structural equality of component interface types, as mandated by the spec
//! - Support for guest resources
//! - Support for strongly-typed host resources with destructors
//!
//! The following features have yet to be implemented:
//!
//! - A macro for generating host bindings
//! - More comprehensive tests
//! - Subtyping

/// Implements the Canonical ABI conventions for converting between guest and host types.
mod abi;

/// Provides the ability to create and call component model functions.
mod func;

/// Defines identifiers for component packages and interfaces.
mod identifier;

/// Defines a macro that will either pattern-match results or throw an error.
mod require_matches;

/// Defines all types related to the component model.
mod types;

/// Provides the ability to instantiate component model types.
mod values;

use std::any::*;
use std::sync::atomic::*;
use std::sync::*;

use anyhow::*;
use fxhash::*;
use id_arena::*;

use slab::*;
pub use wasm_runtime_layer::Engine;
use wasm_runtime_layer::*;
use wasmtime_environ::component::*;
use wit_component::*;
use wit_parser::*;

pub use crate::func::Func;
pub use crate::func::*;
pub use crate::identifier::PackageName;
pub use crate::identifier::*;
use crate::require_matches::*;
pub use crate::types::*;
pub use crate::types::{FuncType, ValueType, VariantCase};
pub use crate::values::*;
pub use crate::values::{Enum, Flags, Record, Tuple, Value, Variant};

/// A parsed and validated WebAssembly component, which may be used to instantiate [`Instance`]s.
#[derive(Clone, Debug)]
pub struct Component(Arc<ComponentInner>);

impl Component {
    /// Creates a new component with the given engine and binary data.
    pub fn new<E: backend::WasmEngine>(engine: &Engine<E>, bytes: &[u8]) -> Result<Self> {
        let (inner, types) = Self::generate_component(engine, bytes)?;
        Ok(Self(Arc::new(Self::generate_resources(
            Self::load_exports(Self::extract_initializers(inner, &types)?, &types)?,
        )?)))
    }

    /// The types and interfaces exported by this component.
    pub fn exports(&self) -> &ComponentTypes {
        &self.0.export_types
    }

    /// The types and interfaces imported by this component. To instantiate
    /// the component, all of these imports must be satisfied by the [`Linker`].
    pub fn imports(&self) -> &ComponentTypes {
        &self.0.import_types
    }

    /// The root package of this component.
    pub fn package(&self) -> &PackageIdentifier {
        &self.0.package
    }

    /// Parses the given bytes into a component, and creates an uninitialized component backing.
    fn generate_component<E: backend::WasmEngine>(
        engine: &Engine<E>,
        bytes: &[u8],
    ) -> Result<(ComponentInner, wasmtime_environ::component::ComponentTypes)> {
        /// A counter that uniquely identifies components.
        static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

        let decoded = wit_component::decode(bytes)
            .context("Could not decode component information from bytes.")?;

        let (mut resolve, world_id) = match decoded {
            DecodedWasm::WitPackage(..) => bail!("Cannot instantiate WIT package as module."),
            DecodedWasm::Component(resolve, id) => (resolve, id),
        };

        let adapter_vec = wasmtime_environ::ScopeVec::new();
        let (translation, module_data, component_types) =
            Self::translate_modules(bytes, &adapter_vec)?;

        let export_mapping = Self::generate_export_mapping(&module_data);
        let mut modules =
            FxHashMap::with_capacity_and_hasher(module_data.len(), Default::default());

        for (id, module) in module_data {
            modules.insert(
                id,
                ModuleTranslation {
                    module: Module::new(engine, std::io::Cursor::new(module.wasm))?,
                    translation: module.module,
                },
            );
        }

        let mut size_align = SizeAlign::default();
        size_align.fill(&resolve);

        let package = (&resolve.packages[resolve.worlds[world_id]
            .package
            .context("No package associated with world.")?]
        .name)
            .into();

        let package_identifiers = Self::generate_package_identifiers(&resolve)?;
        let interface_identifiers =
            Self::generate_interface_identifiers(&resolve, &package_identifiers)?;

        let type_identifiers =
            Self::generate_type_identifiers(&mut resolve, &interface_identifiers);

        Ok((
            ComponentInner {
                export_mapping,
                export_names: FxHashMap::default(),
                import_types: ComponentTypes::new(),
                export_types: ComponentTypes::new(),
                export_info: ExportTypes::default(),
                extracted_memories: FxHashMap::default(),
                extracted_reallocs: FxHashMap::default(),
                extracted_post_returns: FxHashMap::default(),
                id: ID_COUNTER.fetch_add(1, Ordering::AcqRel),
                generated_trampolines: FxHashMap::default(),
                instance_modules: wasmtime_environ::PrimaryMap::default(),
                interface_identifiers,
                type_identifiers,
                modules,
                resource_map: vec![
                    TypeResourceTableIndex::from_u32(u32::MAX - 1);
                    resolve.types.len()
                ],
                resolve,
                size_align,
                translation,
                world_id,
                package,
            },
            component_types,
        ))
    }

    /// Generates type identifiers for all types in the resolve.
    fn generate_type_identifiers(
        resolve: &mut Resolve,
        interface_ids: &[InterfaceIdentifier],
    ) -> Vec<Option<TypeIdentifier>> {
        let mut ids = Vec::with_capacity(resolve.types.len());

        for (_, def) in &mut resolve.types {
            if let Some(name) = std::mem::take(&mut def.name) {
                ids.push(Some(TypeIdentifier::new(
                    name,
                    match &def.owner {
                        TypeOwner::Interface(x) => Some(interface_ids[x.index()].clone()),
                        _ => None,
                    },
                )));
            } else {
                ids.push(None);
            }
        }

        ids
    }

    /// Creates a mapping from module index to entities, used to resolve component exports at link-time.
    fn generate_export_mapping(
        module_data: &wasmtime_environ::PrimaryMap<
            StaticModuleIndex,
            wasmtime_environ::ModuleTranslation,
        >,
    ) -> FxHashMap<StaticModuleIndex, FxHashMap<wasmtime_environ::EntityIndex, String>> {
        let mut export_mapping =
            FxHashMap::with_capacity_and_hasher(module_data.len(), Default::default());

        for (idx, module) in module_data {
            let entry: &mut FxHashMap<wasmtime_environ::EntityIndex, String> =
                export_mapping.entry(idx).or_default();
            for (name, index) in module.module.exports.clone() {
                entry.insert(index, name);
            }
        }

        export_mapping
    }

    /// Fills in the abstract resource types for the given component.
    fn generate_resources(mut inner: ComponentInner) -> Result<ComponentInner> {
        for (_key, item) in &inner.resolve.worlds[inner.world_id].imports {
            match item {
                WorldItem::Type(x) => {
                    if inner.resolve.types[*x].kind == TypeDefKind::Resource {
                        if let Some(name) = &inner.resolve.types[*x].name {
                            ensure!(
                                inner
                                    .import_types
                                    .root
                                    .resources
                                    .insert(
                                        name.as_str().into(),
                                        ResourceType::from_resolve(
                                            inner.type_identifiers[x.index()].clone(),
                                            *x,
                                            &inner,
                                            None
                                        )?
                                    )
                                    .is_none(),
                                "Duplicate resource import."
                            );
                        }
                    }
                }
                WorldItem::Interface(x) => {
                    for (name, ty) in &inner.resolve.interfaces[*x].types {
                        if inner.resolve.types[*ty].kind == TypeDefKind::Resource {
                            let ty = ResourceType::from_resolve(
                                inner.type_identifiers[ty.index()].clone(),
                                *ty,
                                &inner,
                                None,
                            )?;
                            let entry = inner
                                .import_types
                                .instances
                                .entry(inner.interface_identifiers[x.index()].clone())
                                .or_insert_with(ComponentTypesInstance::new);
                            ensure!(
                                entry.resources.insert(name.as_str().into(), ty).is_none(),
                                "Duplicate resource import."
                            );
                        }
                    }
                }
                _ => {}
            }
        }

        for (_key, item) in &inner.resolve.worlds[inner.world_id].exports {
            match item {
                WorldItem::Type(x) => {
                    if inner.resolve.types[*x].kind == TypeDefKind::Resource {
                        if let Some(name) = &inner.resolve.types[*x].name {
                            ensure!(
                                inner
                                    .export_types
                                    .root
                                    .resources
                                    .insert(
                                        name.as_str().into(),
                                        ResourceType::from_resolve(
                                            inner.type_identifiers[x.index()].clone(),
                                            *x,
                                            &inner,
                                            None
                                        )?
                                    )
                                    .is_none(),
                                "Duplicate resource export."
                            );
                        }
                    }
                }
                WorldItem::Interface(x) => {
                    for (name, ty) in &inner.resolve.interfaces[*x].types {
                        if inner.resolve.types[*ty].kind == TypeDefKind::Resource {
                            let ty = ResourceType::from_resolve(
                                inner.type_identifiers[ty.index()].clone(),
                                *ty,
                                &inner,
                                None,
                            )?;
                            let entry = inner
                                .export_types
                                .instances
                                .entry(inner.interface_identifiers[x.index()].clone())
                                .or_insert_with(ComponentTypesInstance::new);
                            ensure!(
                                entry.resources.insert(name.as_str().into(), ty).is_none(),
                                "Duplicate resource export."
                            );
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(inner)
    }

    /// Parses all package identifiers.
    fn generate_package_identifiers(resolve: &Resolve) -> Result<Vec<PackageIdentifier>> {
        let mut res = Vec::with_capacity(resolve.packages.len());

        for (_, pkg) in &resolve.packages {
            res.push(PackageIdentifier::from(&pkg.name));
        }

        Ok(res)
    }

    /// Generates a mapping from interface ID to parsed interface identifier.
    fn generate_interface_identifiers(
        resolve: &Resolve,
        packages: &[PackageIdentifier],
    ) -> Result<Vec<InterfaceIdentifier>> {
        let mut res = Vec::with_capacity(resolve.interfaces.len());

        for (_, iface) in &resolve.interfaces {
            let pkg = iface
                .package
                .context("Interface did not have associated package.")?;
            res.push(InterfaceIdentifier::new(
                packages[pkg.index()].clone(),
                iface
                    .name
                    .as_deref()
                    .context("Exported interface did not have valid name.")?,
            ));
        }

        Ok(res)
    }

    /// Fills in all initialization data for the component.
    fn extract_initializers(
        mut inner: ComponentInner,
        types: &wasmtime_environ::component::ComponentTypes,
    ) -> Result<ComponentInner> {
        let lowering_options = Self::get_lowering_options_and_extract_trampolines(
            &inner.translation.trampolines,
            &mut inner.generated_trampolines,
        )?;
        let mut imports = FxHashMap::default();
        for (key, _) in &inner.resolve.worlds[inner.world_id].imports {
            let name = inner.resolve.name_world_key(key);
            imports.insert(name, key.clone());
        }

        let _root_name = Arc::<str>::from("$root");

        let mut destructors = FxHashMap::default();

        for initializer in &inner.translation.component.initializers {
            match initializer {
                GlobalInitializer::InstantiateModule(InstantiateModule::Static(idx, _def)) => {
                    inner.instance_modules.push(*idx);
                }
                GlobalInitializer::ExtractMemory(ExtractMemory { index, export }) => {
                    ensure!(
                        inner
                            .extracted_memories
                            .insert(*index, export.clone())
                            .is_none(),
                        "Extracted the same memory more than once."
                    );
                }
                GlobalInitializer::ExtractRealloc(ExtractRealloc { index, def }) => {
                    if let CoreDef::Export(export) = def {
                        ensure!(
                            inner
                                .extracted_reallocs
                                .insert(*index, export.clone())
                                .is_none(),
                            "Extracted the same memory more than once."
                        );
                    } else {
                        bail!("Unexpected post return definition type.");
                    }
                }
                GlobalInitializer::ExtractPostReturn(ExtractPostReturn { index, def }) => {
                    if let CoreDef::Export(export) = def {
                        ensure!(
                            inner
                                .extracted_post_returns
                                .insert(*index, export.clone())
                                .is_none(),
                            "Extracted the same memory more than once."
                        );
                    } else {
                        bail!("Unexpected post return definition type.");
                    }
                }
                GlobalInitializer::LowerImport { index, import } => {
                    let (idx, lowering_opts, index_ty) = lowering_options[*index];
                    let (import_index, path) = &inner.translation.component.imports[*import];
                    let (import_name, _) = &inner.translation.component.import_types[*import_index];
                    let world_key = &imports[import_name];

                    let imp = match &inner.resolve.worlds[inner.world_id].imports[world_key] {
                        WorldItem::Function(func) => {
                            assert_eq!(path.len(), 0);
                            ComponentImport {
                                instance: None,
                                name: import_name.as_str().into(),
                                func: func.clone(),
                                options: lowering_opts.clone(),
                            }
                        }
                        WorldItem::Interface(i) => {
                            assert_eq!(path.len(), 1);
                            let iface = &inner.resolve.interfaces[*i];
                            let func = &iface.functions[&path[0]];

                            ComponentImport {
                                instance: Some(inner.interface_identifiers[i.index()].clone()),
                                name: path[0].as_str().into(),
                                func: func.clone(),
                                options: lowering_opts.clone(),
                            }
                        }
                        WorldItem::Type(_) => unreachable!(),
                    };

                    let ty = crate::types::FuncType::from_component(&imp.func, &inner, None)?;
                    let inst = if let Some(inst) = &imp.instance {
                        inner
                            .import_types
                            .instances
                            .entry(inst.clone())
                            .or_insert_with(ComponentTypesInstance::new)
                    } else {
                        &mut inner.import_types.root
                    };

                    Self::update_resource_map(
                        &inner.resolve,
                        types,
                        &imp.func,
                        index_ty,
                        &mut inner.resource_map,
                    );

                    ensure!(
                        inst.functions.insert(imp.name.clone(), ty).is_none(),
                        "Attempted to insert duplicate import."
                    );

                    ensure!(
                        inner
                            .generated_trampolines
                            .insert(idx, GeneratedTrampoline::ImportedFunction(imp))
                            .is_none(),
                        "Attempted to insert duplicate import."
                    );
                }
                GlobalInitializer::Resource(x) => {
                    if let Some(destructor) = x.dtor.clone() {
                        ensure!(
                            destructors.insert(x.index, destructor).is_none(),
                            "Attempted to define duplicate resource."
                        );
                    }
                }
                _ => bail!("Not yet implemented {initializer:?}."),
            }
        }

        for trampoline in inner.generated_trampolines.values_mut() {
            if let GeneratedTrampoline::ResourceDrop(x, destructor) = trampoline {
                let resource = &types[*x];
                if let Some(resource_idx) = inner
                    .translation
                    .component
                    .defined_resource_index(resource.ty)
                {
                    *destructor = destructors.remove(&resource_idx);
                }
            }
        }

        Ok(inner)
    }

    /// Creates a mapping from lowered functions to trampoline data,
    /// and records any auxiliary trampolines in the map.
    fn get_lowering_options_and_extract_trampolines<'a>(
        trampolines: &'a wasmtime_environ::PrimaryMap<TrampolineIndex, Trampoline>,
        output_trampolines: &mut FxHashMap<TrampolineIndex, GeneratedTrampoline>,
    ) -> Result<
        wasmtime_environ::PrimaryMap<
            LoweredIndex,
            (TrampolineIndex, &'a CanonicalOptions, TypeFuncIndex),
        >,
    > {
        let mut lowers = wasmtime_environ::PrimaryMap::default();
        for (idx, trampoline) in trampolines {
            match trampoline {
                Trampoline::LowerImport {
                    index,
                    lower_ty,
                    options,
                } => assert!(
                    lowers.push((idx, options, *lower_ty)) == *index,
                    "Indices did not match."
                ),
                Trampoline::Transcoder {
                    op,
                    from,
                    from64,
                    to,
                    to64,
                } => {
                    if *from64 || *to64 {
                        bail!("Trampoline::Transcoder is not implemented for memory64");
                    }
                    match op {
                        Transcode::Copy(FixedEncoding::Utf8) => {
                            output_trampolines.insert(
                                idx,
                                GeneratedTrampoline::Utf8CopyTranscoder {
                                    from: *from,
                                    to: *to,
                                },
                            );
                        }
                        transcode => {
                            bail!("Trampoline::Transcoder is not implemented for {transcode:?}")
                        }
                    }
                }
                Trampoline::AlwaysTrap => {
                    // FIXME: this trampoline should be de-duplicated
                    output_trampolines.insert(idx, GeneratedTrampoline::AlwaysTrap);
                }
                Trampoline::ResourceNew(x) => {
                    output_trampolines.insert(idx, GeneratedTrampoline::ResourceNew(*x));
                }
                Trampoline::ResourceRep(x) => {
                    output_trampolines.insert(idx, GeneratedTrampoline::ResourceRep(*x));
                }
                Trampoline::ResourceDrop(x) => {
                    output_trampolines.insert(idx, GeneratedTrampoline::ResourceDrop(*x, None));
                }
                Trampoline::ResourceTransferOwn => {
                    // FIXME: this trampoline should be de-duplicated
                    output_trampolines.insert(idx, GeneratedTrampoline::ResourceTransferOwn);
                }
                Trampoline::ResourceTransferBorrow => {
                    // FIXME: this trampoline should be de-duplicated
                    output_trampolines.insert(idx, GeneratedTrampoline::ResourceTransferBorrow);
                }
                Trampoline::ResourceEnterCall => {
                    // FIXME: this trampoline should be de-duplicated
                    output_trampolines.insert(idx, GeneratedTrampoline::ResourceEnterCall);
                }
                Trampoline::ResourceExitCall => {
                    // FIXME: this trampoline should be de-duplicated
                    output_trampolines.insert(idx, GeneratedTrampoline::ResourceExitCall);
                }
            }
        }
        Ok(lowers)
    }

    /// Translates the given bytes into component data and a set of core modules.
    fn translate_modules<'a>(
        bytes: &'a [u8],
        scope: &'a wasmtime_environ::ScopeVec<u8>,
    ) -> Result<(
        ComponentTranslation,
        wasmtime_environ::PrimaryMap<StaticModuleIndex, wasmtime_environ::ModuleTranslation<'a>>,
        wasmtime_environ::component::ComponentTypes,
    )> {
        let tunables = wasmtime_environ::Tunables::default_u32();
        let mut types = ComponentTypesBuilder::default();
        let mut validator = Self::create_component_validator();

        let (translation, modules) = Translator::new(&tunables, &mut validator, &mut types, scope)
            .translate(bytes)
            .context("Could not translate input component to core WASM.")?;

        Ok((
            translation,
            modules,
            types.finish(&Default::default(), [], []).0,
        ))
    }

    /// Fills in all of the exports for a component.
    fn load_exports(
        mut inner: ComponentInner,
        types: &wasmtime_environ::component::ComponentTypes,
    ) -> Result<ComponentInner> {
        Self::export_names(&mut inner);

        for (export_name, export) in &inner.translation.component.exports {
            let world_key = &inner.export_names[export_name];
            let item = &inner.resolve.worlds[inner.world_id].exports[world_key];
            match export {
                wasmtime_environ::component::Export::LiftedFunction { ty, func, options } => {
                    let f = match item {
                        WorldItem::Function(f) => f,
                        WorldItem::Interface(_) | WorldItem::Type(_) => unreachable!(),
                    };

                    Self::update_resource_map(
                        &inner.resolve,
                        types,
                        f,
                        *ty,
                        &mut inner.resource_map,
                    );

                    let export_name = Arc::<str>::from(export_name.as_str());
                    let ty = crate::types::FuncType::from_component(f, &inner, None)?;

                    ensure!(
                        inner
                            .export_types
                            .root
                            .functions
                            .insert(export_name.clone(), ty.clone())
                            .is_none(),
                        "Duplicate function definition."
                    );

                    ensure!(
                        inner
                            .export_info
                            .root
                            .functions
                            .insert(
                                export_name,
                                ComponentExport {
                                    options: options.clone(),
                                    def: match func {
                                        CoreDef::Export(x) => x.clone(),
                                        _ => unreachable!(),
                                    },
                                    func: f.clone(),
                                    ty
                                }
                            )
                            .is_none(),
                        "Duplicate function definition."
                    );
                },
                wasmtime_environ::component::Export::Instance { exports, .. } => {
                    let id = match item {
                        WorldItem::Interface(id) => *id,
                        WorldItem::Function(_) | WorldItem::Type(_) => unreachable!(),
                    };
                    for (func_name, export) in exports {
                        let (func, options, ty) = match export {
                            wasmtime_environ::component::Export::LiftedFunction {
                                func,
                                options,
                                ty,
                            } => (func, options, ty),
                            wasmtime_environ::component::Export::Type(_) => continue, // ignored
                            _ => unreachable!(),
                        };

                        let f = &inner.resolve.interfaces[id].functions[func_name];

                        Self::update_resource_map(
                            &inner.resolve,
                            types,
                            f,
                            *ty,
                            &mut inner.resource_map,
                        );
                        let exp = ComponentExport {
                            options: options.clone(),
                            def: match func {
                                CoreDef::Export(x) => x.clone(),
                                _ => unreachable!(),
                            },
                            func: f.clone(),
                            ty: crate::types::FuncType::from_component(f, &inner, None)?,
                        };
                        let func_name = Arc::<str>::from(func_name.as_str());
                        ensure!(
                            inner
                                .export_types
                                .instances
                                .entry(inner.interface_identifiers[id.index()].clone())
                                .or_insert_with(ComponentTypesInstance::new)
                                .functions
                                .insert(func_name.clone(), exp.ty.clone())
                                .is_none(),
                            "Duplicate function definition."
                        );
                        ensure!(
                            inner
                                .export_info
                                .instances
                                .entry(inner.interface_identifiers[id.index()].clone())
                                .or_default()
                                .functions
                                .insert(func_name, exp)
                                .is_none(),
                            "Duplicate function definition."
                        );
                    }
                }

                // ignore type exports for now
                wasmtime_environ::component::Export::Type(_) => {}

                // This can't be tested at this time so leave it unimplemented
                wasmtime_environ::component::Export::ModuleStatic(_) => {
                    bail!("Not yet implemented.")
                },
                wasmtime_environ::component::Export::ModuleImport { .. } => {
                    bail!("Not yet implemented.")
                },
            }
        }

        Ok(inner)
    }

    /// Fills in the mapping of export names to the exports' respective worlds.
    fn export_names(inner: &mut ComponentInner) {
        let to_iter = &inner.resolve.worlds[inner.world_id].exports;
        let mut exports = FxHashMap::with_capacity_and_hasher(to_iter.len(), Default::default());
        for (key, _) in to_iter {
            let name = inner.resolve.name_world_key(key);
            exports.insert(name, key.clone());
        }
        inner.export_names = exports;
    }

    /// Updates the mapping from type IDs to table indices based upon the resources
    /// referenced by the provided function.
    fn update_resource_map(
        resolve: &Resolve,
        types: &wasmtime_environ::component::ComponentTypes,
        func: &Function,
        ty_func_idx: TypeFuncIndex,
        map: &mut Vec<TypeResourceTableIndex>,
    ) {
        let params_ty = &types[types[ty_func_idx].params];
        for ((_, ty), iface_ty) in func.params.iter().zip(params_ty.types.iter()) {
            Self::connect_resources(resolve, types, ty, iface_ty, map);
        }
        let results_ty = &types[types[ty_func_idx].results];
        for (ty, iface_ty) in func.results.iter_types().zip(results_ty.types.iter()) {
            Self::connect_resources(resolve, types, ty, iface_ty, map);
        }
    }

    /// Inspects the given type (and any referenced subtypes) for resources,
    /// and records the table indices of those resources in the map.
    fn connect_resources(
        resolve: &Resolve,
        types: &wasmtime_environ::component::ComponentTypes,
        ty: &Type,
        iface_ty: &InterfaceType,
        map: &mut Vec<TypeResourceTableIndex>,
    ) {
        let Type::Id(id) = ty else { return };
        match (&resolve.types[*id].kind, iface_ty) {
            (TypeDefKind::Flags(_), InterfaceType::Flags(_))
            | (TypeDefKind::Enum(_), InterfaceType::Enum(_)) => {}
            (TypeDefKind::Record(t1), InterfaceType::Record(t2)) => {
                let t2 = &types[*t2];
                for (f1, f2) in t1.fields.iter().zip(t2.fields.iter()) {
                    Self::connect_resources(resolve, types, &f1.ty, &f2.ty, map);
                }
            }
            (
                TypeDefKind::Handle(Handle::Own(t1) | Handle::Borrow(t1)),
                InterfaceType::Own(t2) | InterfaceType::Borrow(t2),
            ) => {
                map[t1.index()] = *t2;
            }
            (TypeDefKind::Tuple(t1), InterfaceType::Tuple(t2)) => {
                let t2 = &types[*t2];
                for (f1, f2) in t1.types.iter().zip(t2.types.iter()) {
                    Self::connect_resources(resolve, types, f1, f2, map);
                }
            }
            (TypeDefKind::Variant(t1), InterfaceType::Variant(t2)) => {
                let t2 = &types[*t2];
                for (f1, f2) in t1.cases.iter().zip(t2.cases.iter()) {
                    if let Some(t1) = &f1.ty {
                        Self::connect_resources(resolve, types, t1, f2.ty.as_ref().unwrap(), map);
                    }
                }
            }
            (TypeDefKind::Option(t1), InterfaceType::Option(t2)) => {
                let t2 = &types[*t2];
                Self::connect_resources(resolve, types, t1, &t2.ty, map);
            }
            (TypeDefKind::Result(t1), InterfaceType::Result(t2)) => {
                let t2 = &types[*t2];
                if let Some(t1) = &t1.ok {
                    Self::connect_resources(resolve, types, t1, &t2.ok.unwrap(), map);
                }
                if let Some(t1) = &t1.err {
                    Self::connect_resources(resolve, types, t1, &t2.err.unwrap(), map);
                }
            }
            (TypeDefKind::List(t1), InterfaceType::List(t2)) => {
                let t2 = &types[*t2];
                Self::connect_resources(resolve, types, t1, &t2.element, map);
            }
            (TypeDefKind::Type(ty), _) => {
                Self::connect_resources(resolve, types, ty, iface_ty, map);
            }
            (_, _) => unreachable!(),
        }
    }

    /// Creates a validator with the appropriate settings for component model resolution.
    fn create_component_validator() -> wasmtime_environ::wasmparser::Validator {
        wasmtime_environ::wasmparser::Validator::new_with_features(
            wasmtime_environ::wasmparser::WasmFeatures::all(),
        )
    }
}

/// Holds the inner, immutable state of an instantiated component.
struct ComponentInner {
    /// Maps from module indices to export indices for linking.
    pub export_mapping:
        FxHashMap<StaticModuleIndex, FxHashMap<wasmtime_environ::EntityIndex, String>>,
    /// Maps between export names and world keys.
    pub export_names: FxHashMap<String, WorldKey>,
    /// The exports of the component.
    pub export_types: ComponentTypes,
    /// Holds internal information for linking exports.
    pub export_info: ExportTypes,
    /// The memories that this component instantiates and references.
    pub extracted_memories: FxHashMap<RuntimeMemoryIndex, CoreExport<MemoryIndex>>,
    /// The reallocation functions that this component instantiates and references.
    pub extracted_reallocs:
        FxHashMap<RuntimeReallocIndex, CoreExport<wasmtime_environ::EntityIndex>>,
    /// The post-return functions that this component instantiates and references.
    pub extracted_post_returns:
        FxHashMap<RuntimePostReturnIndex, CoreExport<wasmtime_environ::EntityIndex>>,
    /// A mapping from type indices to resource table indices.
    pub resource_map: Vec<TypeResourceTableIndex>,
    /// The set of trampolines required to use this resource.
    pub generated_trampolines: FxHashMap<TrampolineIndex, GeneratedTrampoline>,
    /// The component's globally-unique ID.
    pub id: u64,
    /// The imports of the component.
    pub import_types: ComponentTypes,
    /// A mapping from runtime module indices to static indices.
    pub instance_modules: wasmtime_environ::PrimaryMap<RuntimeInstanceIndex, StaticModuleIndex>,
    /// A mapping from interface ID to parsed identifier.
    pub interface_identifiers: Vec<InterfaceIdentifier>,
    /// A mapping from type ID to parsed identifier.
    pub type_identifiers: Vec<Option<TypeIdentifier>>,
    /// The translated modules of this component.
    pub modules: FxHashMap<StaticModuleIndex, ModuleTranslation>,
    /// The resolved WIT of the component.
    pub resolve: Resolve,
    /// The size and alignment of component types.
    pub size_align: SizeAlign,
    /// The translated component data.
    pub translation: ComponentTranslation,
    /// The ID of the primary exported world.
    pub world_id: Id<World>,
    /// The package identifier for the component.
    pub package: PackageIdentifier,
}

impl std::fmt::Debug for ComponentInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComponentInner").finish()
    }
}

/// A translated module.
struct ModuleTranslation {
    /// The instantiated module that was translated.
    pub module: Module,
    /// The translation data for the module.
    pub translation: wasmtime_environ::Module,
}

/// Details the set of types and functions exported by a component.
#[derive(Debug, Default)]
struct ExportTypes {
    /// The root instance for component exports.
    root: ExportTypesInstance,
    /// The interfaces exported by the component.
    instances: FxHashMap<InterfaceIdentifier, ExportTypesInstance>,
}

/// Represents an interface that has been exported by a component.
#[derive(Debug, Default)]
struct ExportTypesInstance {
    /// The functions in the interface.
    functions: FxHashMap<Arc<str>, ComponentExport>,
}

/// Details a set of types within a component.
#[derive(Debug)]
pub struct ComponentTypes {
    /// The package root of the component.
    root: ComponentTypesInstance,
    /// All instances owned by the component.
    instances: FxHashMap<InterfaceIdentifier, ComponentTypesInstance>,
}

impl ComponentTypes {
    /// Creates a new, initially empty component type set.
    pub(crate) fn new() -> Self {
        Self {
            root: ComponentTypesInstance::new(),
            instances: FxHashMap::default(),
        }
    }

    /// Gets the root instance.
    pub fn root(&self) -> &ComponentTypesInstance {
        &self.root
    }

    /// Gets the instance with the specified name, if any.
    pub fn instance(&self, name: &InterfaceIdentifier) -> Option<&ComponentTypesInstance> {
        self.instances.get(name)
    }

    /// Gets an iterator over all instances by identifier.
    pub fn instances(
        &self,
    ) -> impl Iterator<Item = (&'_ InterfaceIdentifier, &'_ ComponentTypesInstance)> {
        self.instances.iter()
    }
}

/// Represents a specific interface from a component.
#[derive(Debug)]
pub struct ComponentTypesInstance {
    /// The functions of the interface.
    functions: FxHashMap<Arc<str>, crate::types::FuncType>,
    /// The resources of the interface.
    resources: FxHashMap<Arc<str>, ResourceType>,
}

impl ComponentTypesInstance {
    /// Creates a new, empty instance.
    pub(crate) fn new() -> Self {
        Self {
            functions: FxHashMap::default(),
            resources: FxHashMap::default(),
        }
    }

    /// Gets the associated function by name, if any.
    pub fn func(&self, name: impl AsRef<str>) -> Option<crate::types::FuncType> {
        self.functions.get(name.as_ref()).cloned()
    }

    /// Iterates over all associated functions by name.
    pub fn funcs(&self) -> impl Iterator<Item = (&'_ str, crate::types::FuncType)> {
        self.functions.iter().map(|(k, v)| (&**k, v.clone()))
    }

    /// Gets the associated abstract resource by name, if any.
    pub fn resource(&self, name: impl AsRef<str>) -> Option<ResourceType> {
        self.resources.get(name.as_ref()).cloned()
    }

    /// Iterates over all associated functions by name.
    pub fn resources(&self) -> impl Iterator<Item = (&'_ str, crate::types::ResourceType)> {
        self.resources.iter().map(|(k, v)| (&**k, v.clone()))
    }
}

/// Provides the ability to define imports for a component and create [`Instance`]s of it.
#[derive(Clone, Debug, Default)]
pub struct Linker {
    /// The root instance used for linking.
    root: LinkerInstance,
    /// The set of interfaces against which to link.
    instances: FxHashMap<InterfaceIdentifier, LinkerInstance>,
}

impl Linker {
    /// Immutably obtains the root interface for this linker.
    pub fn root(&self) -> &LinkerInstance {
        &self.root
    }

    /// Mutably obtains the root interface for this linker.
    pub fn root_mut(&mut self) -> &mut LinkerInstance {
        &mut self.root
    }

    /// Creates a new instance in the linker with the provided name. Returns an
    /// error if an instance with that name already exists.
    pub fn define_instance(&mut self, name: InterfaceIdentifier) -> Result<&mut LinkerInstance> {
        if self.instance(&name).is_none() {
            Ok(self.instances.entry(name).or_default())
        } else {
            bail!("Duplicate instance definition.");
        }
    }

    /// Immutably obtains the instance with the given name, if any.
    pub fn instance(&self, name: &InterfaceIdentifier) -> Option<&LinkerInstance> {
        self.instances.get(name)
    }

    /// Mutably obtains the instance with the given name, if any.
    pub fn instance_mut(&mut self, name: &InterfaceIdentifier) -> Option<&mut LinkerInstance> {
        self.instances.get_mut(name)
    }

    /// Gets an immutable iterator over all instances defined in this linker.
    pub fn instances(
        &self,
    ) -> impl ExactSizeIterator<Item = (&'_ InterfaceIdentifier, &'_ LinkerInstance)> {
        self.instances.iter()
    }

    /// Gets a mutable iterator over all instances defined in this linker.
    pub fn instances_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = (&'_ InterfaceIdentifier, &'_ mut LinkerInstance)> {
        self.instances.iter_mut()
    }

    /// Instantiates a component for the provided store, filling in its imports with externals
    /// defined in this linker. All imports must be defined for instantiation to succeed.
    pub fn instantiate(&self, ctx: impl AsContextMut, component: &Component) -> Result<Instance> {
        Instance::new(ctx, component, self)
    }
}

/// Describes a concrete interface which components may import.
#[derive(Clone, Debug, Default)]
pub struct LinkerInstance {
    /// The functions in the interface.
    functions: FxHashMap<Arc<str>, crate::func::Func>,
    /// The resource types in the interface.
    resources: FxHashMap<Arc<str>, ResourceType>,
}

impl LinkerInstance {
    /// Defines a new function for this interface with the provided name.
    /// Fails if the function already exists.
    pub fn define_func(
        &mut self,
        name: impl Into<Arc<str>>,
        func: crate::func::Func,
    ) -> Result<()> {
        let n = Into::<Arc<str>>::into(name);
        if self.functions.contains_key(&n) {
            bail!("Duplicate function definition.");
        }

        self.functions.insert(n, func);
        Ok(())
    }

    /// Gets the function in this interface with the given name, if any.
    pub fn func(&self, name: impl AsRef<str>) -> Option<crate::func::Func> {
        self.functions.get(name.as_ref()).cloned()
    }

    /// Defines a new resource type for this interface with the provided name.
    /// Fails if the resource type already exists, or if the resource is abstract.
    pub fn define_resource(
        &mut self,
        name: impl Into<Arc<str>>,
        resource_ty: ResourceType,
    ) -> Result<()> {
        ensure!(
            resource_ty.is_instantiated(),
            "Cannot link with abstract resource type."
        );

        let n = Into::<Arc<str>>::into(name);
        if self.resources.contains_key(&n) {
            bail!("Duplicate resource definition.");
        }

        self.resources.insert(n, resource_ty);
        Ok(())
    }

    /// Gets the resource in this interface with the given name, if any.
    pub fn resource(&self, name: impl AsRef<str>) -> Option<ResourceType> {
        self.resources.get(name.as_ref()).cloned()
    }

    /// Iterates over all associated functions by name.
    pub fn funcs(&self) -> impl Iterator<Item = (&'_ str, crate::func::Func)> {
        self.functions.iter().map(|(k, v)| (&**k, v.clone()))
    }

    /// Iterates over all associated functions by name.
    pub fn resources(&self) -> impl Iterator<Item = (&'_ str, ResourceType)> {
        self.resources.iter().map(|(k, v)| (&**k, v.clone()))
    }
}

/// An instantiated WebAssembly component.
#[derive(Clone, Debug)]
pub struct Instance(Arc<InstanceInner>);

impl Instance {
    /// Creates a new instance for the given component with the specified linker.
    pub(crate) fn new(
        mut ctx: impl AsContextMut,
        component: &Component,
        linker: &Linker,
    ) -> Result<Self> {
        /// A counter that uniquely identifies instances.
        static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

        let mut instance_flags = wasmtime_environ::PrimaryMap::default();
        // println!("{:?}", component.0.instance_modules);
        for _i in 0..component.0.instance_modules.len() + 20
        /* ??? */
        {
            instance_flags.push(Global::new(
                ctx.as_context_mut().inner,
                wasm_runtime_layer::Value::I32(
                    wasmtime_environ::component::FLAG_MAY_LEAVE
                        | wasmtime_environ::component::FLAG_MAY_ENTER,
                ),
                true,
            ));
        }
        // println!("{:?}", instance_flags);

        let id = ID_COUNTER.fetch_add(1, Ordering::AcqRel);
        let map = Self::create_resource_instantiation_map(id, component, linker)?;
        let types = Self::generate_types(component, &map)?;
        let resource_tables = Mutex::new(vec![
            HandleTable::default();
            component.0.translation.component.num_resource_tables
        ]);
        let resource_call_borrows = Mutex::new(Vec::new());

        let instance = InstanceInner {
            component: component.clone(),
            exports: Exports::new(),
            id,
            instances: Default::default(),
            instance_flags,
            state_table: Arc::new(StateTable {
                dropped: AtomicBool::new(false),
                resource_tables,
                resource_call_borrows,
            }),
            types,
            store_id: ctx.as_context().inner.data().id,
        };
        let initialized = Self::global_initialize(instance, &mut ctx, linker, &map)?;
        let exported = Self::load_exports(initialized, &ctx, &map)?;

        Ok(Self(Arc::new_cyclic(|w| {
            Self::fill_exports(exported, w.clone())
        })))
    }

    /// Gets the component associated with this instance.
    pub fn component(&self) -> &Component {
        &self.0.component
    }

    /// Gets the exports of this instance.
    pub fn exports(&self) -> &Exports {
        &self.0.exports
    }

    /// Drops the instance and all of its owned resources, removing its data from the given store.
    /// Returns the list of errors that occurred while dropping owned resources, but continues
    /// until all resources have been dropped.
    pub fn drop<T, E: backend::WasmEngine>(&self, ctx: &mut Store<T, E>) -> Result<Vec<Error>> {
        ensure!(self.0.store_id == ctx.inner.data().id, "Incorrect store.");
        self.0.state_table.dropped.store(true, Ordering::Release);

        let mut errors = Vec::new();

        let mut tables = self
            .0
            .state_table
            .resource_tables
            .try_lock()
            .expect("Could not lock resource tables.");
        for table in &mut *tables {
            if let Some(destructor) = table.destructor.as_ref() {
                for (_, val) in table.array.iter() {
                    if let Err(x) = destructor.call(
                        &mut ctx.inner,
                        &[wasm_runtime_layer::Value::I32(val.rep)],
                        &mut [],
                    ) {
                        errors.push(x);
                    }
                }
            }
        }

        Ok(errors)
    }

    /// Fills the export tables with pointers to the final instance.
    fn fill_exports(mut inner: InstanceInner, final_ptr: Weak<InstanceInner>) -> InstanceInner {
        for inst in inner.exports.instances.values_mut() {
            inst.instance = final_ptr.clone();
        }

        inner.exports.root.instance = final_ptr;
        inner
    }

    /// Generates the concrete list of types for this instance, after replacing abstract resources with instantiated ones.
    fn generate_types(
        component: &Component,
        map: &FxHashMap<ResourceType, ResourceType>,
    ) -> Result<Arc<[crate::types::ValueType]>> {
        let mut types = Vec::with_capacity(component.0.resolve.types.len());
        for (id, val) in &component.0.resolve.types {
            assert!(
                types.len() == id.index(),
                "Type definition IDs were not equal."
            );

            match val.kind {
                TypeDefKind::Resource => {
                    types.push(crate::types::ValueType::Bool);
                    continue;
                }
                TypeDefKind::Type(Type::Id(x)) => {
                    if component.0.resolve.types[x].kind == TypeDefKind::Resource {
                        types.push(crate::types::ValueType::Bool);
                        continue;
                    }
                }
                _ => {}
            };

            types.push(crate::types::ValueType::from_component_typedef(
                id,
                &component.0,
                Some(map),
            )?);
        }
        Ok(types.into())
    }

    /// Creates a mapping from component resources to instance resources,
    /// since resource types are unique per instantiation.
    fn create_resource_instantiation_map(
        instance_id: u64,
        component: &Component,
        linker: &Linker,
    ) -> Result<FxHashMap<ResourceType, ResourceType>> {
        let mut types = FxHashMap::default();

        for (name, resource) in component.imports().root().resources() {
            let instantiated = linker
                .root()
                .resource(name)
                .ok_or_else(|| Error::msg(format!("Could not find resource {name} in linker.")))?;
            types.insert(resource, instantiated);
        }

        for (id, interface) in component.imports().instances() {
            for (name, resource) in interface.resources() {
                let instantiated = linker
                    .instance(id)
                    .and_then(|x| x.resource(name))
                    .ok_or_else(|| {
                        Error::msg(format!(
                            "Could not find resource {name} from interface {id:?} in linker."
                        ))
                    })?;
                types.insert(resource, instantiated);
            }
        }

        for (_, resource) in component
            .exports()
            .instances()
            .flat_map(|(_, x)| x.resources())
            .chain(component.exports().root().resources())
        {
            let instantiated = resource.instantiate(instance_id)?;
            types.insert(resource, instantiated);
        }

        Ok(types)
    }

    /// Fills in the exports map for the instance.
    fn load_exports(
        mut inner: InstanceInner,
        ctx: impl AsContext,
        map: &FxHashMap<ResourceType, ResourceType>,
    ) -> Result<InstanceInner> {
        for (name, func) in &inner.component.0.export_info.root.functions {
            inner.exports.root.functions.insert(
                name.clone(),
                Self::export_function(
                    &inner,
                    &ctx,
                    &func.def,
                    &func.options,
                    &func.func,
                    map,
                    None,
                )?,
            );
        }
        for (name, res) in &inner.component.0.export_types.root.resources {
            inner
                .exports
                .root
                .resources
                .insert(name.clone(), res.instantiate(inner.id)?);
        }

        let mut generated_functions = Vec::new();
        for (inst_name, inst) in &inner.component.0.export_info.instances {
            for (name, func) in &inst.functions {
                let export = Self::export_function(
                    &inner,
                    &ctx,
                    &func.def,
                    &func.options,
                    &func.func,
                    map,
                    Some(inst_name.clone()),
                )?;
                generated_functions.push((inst_name.clone(), name.clone(), export));
            }
        }

        for (inst_name, inst) in &inner.component.0.export_types.instances {
            for (name, res) in &inst.resources {
                inner
                    .exports
                    .instances
                    .entry(inst_name.clone())
                    .or_insert_with(ExportInstance::new)
                    .resources
                    .insert(name.clone(), res.instantiate(inner.id)?);
            }
        }

        for (inst_name, name, func) in generated_functions {
            let interface = inner
                .exports
                .instances
                .entry(inst_name)
                .or_insert_with(ExportInstance::new);
            interface.functions.insert(name, func);
        }

        Ok(inner)
    }

    /// Creates the options used to invoke an imported function.
    fn import_function(
        inner: &InstanceInner,
        ctx: impl AsContext,
        options: &CanonicalOptions,
        func: &Function,
    ) -> GuestInvokeOptions {
        let memory = options.memory.map(|idx| {
            Self::core_export(inner, &ctx, &inner.component.0.extracted_memories[&idx])
                .expect("Could not get runtime memory export.")
                .into_memory()
                .expect("Export was not of memory type.")
        });
        let realloc = options.realloc.map(|idx| {
            Self::core_export(inner, &ctx, &inner.component.0.extracted_reallocs[&idx])
                .expect("Could not get runtime realloc export.")
                .into_func()
                .expect("Export was not of func type.")
        });
        let post_return = options.post_return.map(|idx| {
            Self::core_export(inner, &ctx, &inner.component.0.extracted_post_returns[&idx])
                .expect("Could not get runtime post return export.")
                .into_func()
                .expect("Export was not of func type.")
        });

        GuestInvokeOptions {
            component: inner.component.0.clone(),
            encoding: options.string_encoding,
            function: func.clone(),
            memory,
            realloc,
            state_table: inner.state_table.clone(),
            post_return,
            types: inner.types.clone(),
            instance_id: inner.id,
            store_id: ctx.as_context().inner.data().id,
        }
    }

    /// Creates an exported function from the provided definitions.
    fn export_function(
        inner: &InstanceInner,
        ctx: impl AsContext,
        def: &CoreExport<wasmtime_environ::EntityIndex>,
        options: &CanonicalOptions,
        func: &Function,
        mapping: &FxHashMap<ResourceType, ResourceType>,
        interface_id: Option<InterfaceIdentifier>,
    ) -> Result<crate::func::Func> {
        let callee = Self::core_export(inner, &ctx, def)
            .expect("Could not get callee export.")
            .into_func()
            .expect("Export was not of func type.");
        let memory = options.memory.map(|idx| {
            Self::core_export(inner, &ctx, &inner.component.0.extracted_memories[&idx])
                .expect("Could not get runtime memory export.")
                .into_memory()
                .expect("Export was not of memory type.")
        });
        let realloc = options.realloc.map(|idx| {
            Self::core_export(inner, &ctx, &inner.component.0.extracted_reallocs[&idx])
                .expect("Could not get runtime realloc export.")
                .into_func()
                .expect("Export was not of func type.")
        });
        let post_return = options.post_return.map(|idx| {
            Self::core_export(inner, &ctx, &inner.component.0.extracted_post_returns[&idx])
                .expect("Could not get runtime post return export.")
                .into_func()
                .expect("Export was not of func type.")
        });

        Ok(crate::func::Func {
            store_id: ctx.as_context().inner.data().id,
            ty: crate::types::FuncType::from_component(func, &inner.component.0, Some(mapping))?,
            backing: FuncImpl::GuestFunc(
                None,
                Arc::new(GuestFunc {
                    callee,
                    component: inner.component.0.clone(),
                    encoding: options.string_encoding,
                    function: func.clone(),
                    memory,
                    realloc,
                    state_table: inner.state_table.clone(),
                    post_return,
                    types: inner.types.clone(),
                    instance_id: inner.id,
                    interface_id,
                }),
            ),
        })
    }

    /// Gets the core WASM import associated with the provided definition.
    fn core_import(
        inner: &InstanceInner,
        mut ctx: impl AsContextMut,
        def: &CoreDef,
        linker: &Linker,
        ty: ExternType,
        destructors: &mut Vec<TrampolineIndex>,
        resource_map: &FxHashMap<ResourceType, ResourceType>,
    ) -> Result<Extern> {
        match def {
            CoreDef::Export(x) => {
                Self::core_export(inner, ctx, x).context("Could not find exported function.")
            }
            CoreDef::Trampoline(x) => {
                let ty = if let ExternType::Func(x) = ty {
                    x
                } else {
                    bail!("Incorrect extern type.")
                };
                match inner
                    .component
                    .0
                    .generated_trampolines
                    .get(x)
                    .context("Could not find exported trampoline.")?
                {
                    GeneratedTrampoline::ImportedFunction(component_import) => {
                        let expected = crate::types::FuncType::from_component(
                            &component_import.func,
                            &inner.component.0,
                            Some(resource_map),
                        )?;
                        let func = Self::get_component_import(component_import, linker)?;
                        ensure!(
                            func.ty() == expected,
                            "Function import {} had type {}, but expected {expected}",
                            component_import.name,
                            func.ty()
                        );
                        let guest_options = Self::import_function(
                            inner,
                            &ctx,
                            &component_import.options,
                            &component_import.func,
                        );

                        // Improve the name
                        // Due to indirect calls in wasm this function has the name `"0"`, `"1"`, etc, which is only their index in the indirect call table across modules
                        let ty = ty.with_name(component_import.name.clone());

                        Ok(Extern::Func(wasm_runtime_layer::Func::new(
                            ctx.as_context_mut().inner,
                            ty,
                            move |ctx, args, results| {
                                let ctx = StoreContextMut { inner: ctx };
                                func.call_from_guest(ctx, &guest_options, args, results)
                            },
                        )))
                    }
                    GeneratedTrampoline::ResourceNew(x) => {
                        let x = x.as_u32();
                        let tables = inner.state_table.clone();
                        let ty = ty.with_name(format!("resource-new-{}", x));
                        Ok(Extern::Func(wasm_runtime_layer::Func::new(
                            ctx.as_context_mut().inner,
                            ty,
                            move |_ctx, args, results| {
                                let rep =
                                    require_matches!(args[0], wasm_runtime_layer::Value::I32(x), x);
                                let mut table_array = tables
                                    .resource_tables
                                    .try_lock()
                                    .expect("Could not get mutual reference to table.");
                                results[0] = wasm_runtime_layer::Value::I32(
                                    table_array[x as usize].add(HandleElement {
                                        rep,
                                        own: true,
                                        lend_count: 0,
                                    }),
                                );
                                Ok(())
                            },
                        )))
                    }
                    GeneratedTrampoline::ResourceRep(x) => {
                        let x = x.as_u32();
                        let tables = inner.state_table.clone();
                        let ty = ty.with_name(format!("resource-rep-{}", x));
                        Ok(Extern::Func(wasm_runtime_layer::Func::new(
                            ctx.as_context_mut().inner,
                            ty,
                            move |_ctx, args, results| {
                                let idx =
                                    require_matches!(args[0], wasm_runtime_layer::Value::I32(x), x);
                                let table_array = tables
                                    .resource_tables
                                    .try_lock()
                                    .expect("Could not get mutual reference to table.");
                                results[0] = wasm_runtime_layer::Value::I32(
                                    table_array[x as usize].get(idx)?.rep,
                                );
                                Ok(())
                            },
                        )))
                    }
                    GeneratedTrampoline::ResourceDrop(y, _) => {
                        destructors.push(*x);
                        let x = y.as_u32();
                        let tables = inner.state_table.clone();
                        let ty = ty.with_name(format!("resource-drop-{}", x));
                        Ok(Extern::Func(wasm_runtime_layer::Func::new(
                            ctx.as_context_mut().inner,
                            ty,
                            move |ctx, args, _results| {
                                let idx =
                                    require_matches!(args[0], wasm_runtime_layer::Value::I32(x), x);
                                let mut table_array = tables
                                    .resource_tables
                                    .try_lock()
                                    .expect("Could not get mutual reference to table.");
                                let current_table = &mut table_array[x as usize];

                                let elem_borrow = current_table.get(idx)?;

                                if elem_borrow.own {
                                    ensure!(
                                        elem_borrow.lend_count == 0,
                                        "Attempted to drop loaned resource."
                                    );
                                    let elem = current_table.remove(idx)?;
                                    if let Some(destructor) =
                                        table_array[x as usize].destructor().cloned()
                                    {
                                        drop(table_array);
                                        destructor.call(
                                            ctx,
                                            &[wasm_runtime_layer::Value::I32(elem.rep)],
                                            &mut [],
                                        )?;
                                    }
                                }
                                Ok(())
                            },
                        )))
                    }
                    GeneratedTrampoline::Utf8CopyTranscoder { from, to } => {
                        let from_memory = Self::core_export(inner, &ctx, &inner.component.0.extracted_memories[&from])
                            .expect("Could not get runtime memory export.")
                            .into_memory()
                            .expect("Export was not of memory type.");
                        let to_memory = Self::core_export(inner, &ctx, &inner.component.0.extracted_memories[&to])
                            .expect("Could not get runtime memory export.")
                            .into_memory()
                            .expect("Export was not of memory type.");
                        let from = from.as_u32();
                        let to = to.as_u32();
                        let ty = ty.with_name(format!("transcode-copy-utf8-{}-{}", from, to));
                        Ok(Extern::Func(wasm_runtime_layer::Func::new(
                            ctx.as_context_mut().inner,
                            ty,
                            move |mut ctx, args, results| {
                                let (from_ptr, to_ptr, len) = match args {
                                    [
                                        wasm_runtime_layer::Value::I32(from_ptr),
                                        wasm_runtime_layer::Value::I32(to_ptr),
                                        wasm_runtime_layer::Value::I32(len),
                                    ] => (from_ptr, to_ptr, len),
                                    args => bail!(
                                        "transcode-copy-utf8-{}-{}(from-ptr: i32, to-ptr: i32, len: i32)\
                                         called with unexpected args {:?}", from, to, args
                                    ),
                                };
                                let from_ptr = usize::try_from(*from_ptr)?;
                                let to_ptr = usize::try_from(*to_ptr)?;
                                let len = usize::try_from(*len)?;
                                ensure!(
                                    results.is_empty(),
                                    "transcode-copy-utf8-{}-{}(from-ptr: i32, to-ptr: i32, len: i32) \
                                    call expects unexpected results {:?}", from, to, results
                                );
                                let mut buffer = vec![0_u8; len];
                                from_memory.read(&mut ctx, from_ptr, &mut buffer)?;
                                to_memory.write(ctx, to_ptr, &buffer)?;
                                Ok(())
                            },
                        )))
                    }
                    GeneratedTrampoline::AlwaysTrap => {
                        let ty = ty.with_name("always-trap");
                        Ok(Extern::Func(wasm_runtime_layer::Func::new(
                            ctx.as_context_mut().inner,
                            ty,
                            move |_ctx, args: &[wasm_runtime_layer::Value], results| {
                                ensure!(args.is_empty(), "always-trap() called with unexpected args {:?}", args);
                                ensure!(results.is_empty(), "always-trap() call expects unexpected results {:?}", results);
                                Err(wasmtime_environ::Trap::AlwaysTrapAdapter.into())
                            },
                        )))
                    }
                    GeneratedTrampoline::ResourceTransferOwn => {
                        let ty = ty.with_name("resource-transfer-own");
                        let tables = inner.state_table.clone();
                        Ok(Extern::Func(wasm_runtime_layer::Func::new(
                            ctx.as_context_mut().inner,
                            ty,
                            move |_ctx, args, results| {
                                let (handle, from_rid, to_rid) = match args {
                                    [
                                        wasm_runtime_layer::Value::I32(handle),
                                        wasm_runtime_layer::Value::I32(from_rid),
                                        wasm_runtime_layer::Value::I32(to_rid),
                                    ] => (handle, from_rid, to_rid),
                                    args => bail!(
                                        "resource-transfer-own(handle: i32, from-rid: i32, to-rid: i32)\
                                         called with unexpected args {:?}", args
                                    ),
                                };
                                let from_rid = usize::try_from(*from_rid)?;
                                let to_rid = usize::try_from(*to_rid)?;
                                let result = match results {
                                    [wasm_runtime_layer::Value::I32(result)] => result,
                                    results => bail!(
                                        "resource-transfer-own(handle: i32, from-rid: i32, to-rid: i32)\
                                         call expects unexpected results {:?}", results
                                    ),
                                };
                                let mut table_array = tables
                                    .resource_tables
                                    .try_lock()
                                    .expect("Could not get mutual reference to table.");
                                let from_table = &mut table_array[from_rid];
                                let handle_borrow = from_table.get(*handle)?;
                                ensure!(handle_borrow.own, "Attempted to owning-transfer a non-owned resource");
                                ensure!(
                                    handle_borrow.lend_count == 0,
                                    "Attempted to owning-transfer a loaned resource."
                                );
                                let from_handle = from_table.remove(*handle)?;
                                let to_handle = table_array[to_rid].add(HandleElement {
                                    rep: from_handle.rep,
                                    own: true,
                                    lend_count: 0,
                                });
                                *result = to_handle;
                                Ok(())
                            },
                        )))
                    }
                    GeneratedTrampoline::ResourceTransferBorrow => {
                        let ty = ty.with_name("resource-transfer-borrow");
                        let tables = inner.state_table.clone();
                        Ok(Extern::Func(wasm_runtime_layer::Func::new(
                            ctx.as_context_mut().inner,
                            ty,
                            move |_ctx, args, results| {
                                let (handle, from_rid, to_rid) = match args {
                                    [
                                        wasm_runtime_layer::Value::I32(handle),
                                        wasm_runtime_layer::Value::I32(from_rid),
                                        wasm_runtime_layer::Value::I32(to_rid),
                                    ] => (handle, from_rid, to_rid),
                                    args => bail!(
                                        "resource-transfer-borrow(handle: i32, from-rid: i32, to-rid: i32)\
                                         called with unexpected args {:?}", args
                                    ),
                                };
                                let from_rid = usize::try_from(*from_rid)?;
                                let to_rid = usize::try_from(*to_rid)?;
                                let result = match results {
                                    [wasm_runtime_layer::Value::I32(result)] => result,
                                    results => bail!(
                                        "resource-transfer-borrow(handle: i32, from-rid: i32, to-rid: i32)\
                                         call expects unexpected results {:?}", results
                                    ),
                                };
                                let mut table_array = tables
                                    .resource_tables
                                    .try_lock()
                                    .expect("Could not get mutual reference to table.");
                                let from_table = &mut table_array[from_rid];
                                let handle_borrow = from_table.get(*handle)?;
                                let handle_rep = handle_borrow.rep;
                                // FIXME: wrong condition, should be if from_rid is imported
                                if true {
                                    ensure!(
                                        handle_borrow.lend_count == 0,
                                        "Attempted to borrow-transfer a non-owned loaned resource."
                                    );
                                    from_table.remove(*handle)?;
                                }
                                let to_table = &mut table_array[to_rid];
                                // FIXME: wrong condition, should be if to_rid is local
                                let to_handle = if true {
                                    handle_rep
                                } else {
                                    let to_handle = to_table.add(HandleElement {
                                        rep: handle_rep,
                                        own: false,
                                        lend_count: 0,
                                    });
                                    tables
                                        .resource_call_borrows
                                        .try_lock()
                                        .expect("Could not get mutual reference to resource call borrows.")
                                        .push((to_rid, to_handle));
                                    to_handle
                                };
                                *result = to_handle;
                                Ok(())
                            },
                        )))
                    }
                    GeneratedTrampoline::ResourceEnterCall => {
                        let ty = ty.with_name("resource-enter-call");
                        let tables = inner.state_table.clone();
                        Ok(Extern::Func(wasm_runtime_layer::Func::new(
                            ctx.as_context_mut().inner,
                            ty,
                            move |_ctx, args, results| {
                                ensure!(args.is_empty(), "resource-enter-call() called with unexpected args {:?}", args);
                                ensure!(results.is_empty(), "resource-enter-call() call expects unexpected results {:?}", results);
                                // As in Jco, ResourceEnterCall is a no-op since all logic
                                //  is handled in ResourceEnterCall and resource_call_borrows
                                //  should be empty here
                                let resource_call_borrows = tables
                                    .resource_call_borrows
                                    .try_lock()
                                    .expect("Could not get mutual reference to resource call borrows.");
                                assert!(resource_call_borrows.is_empty());
                                Ok(())
                            },
                        )))
                    }
                    GeneratedTrampoline::ResourceExitCall => {
                        let ty = ty.with_name("resource-exit-call");
                        let tables = inner.state_table.clone();
                        Ok(Extern::Func(wasm_runtime_layer::Func::new(
                            ctx.as_context_mut().inner,
                            ty,
                            move |_ctx, args, results| {
                                ensure!(args.is_empty(), "resource-exit-call() called with unexpected args {:?}", args);
                                ensure!(results.is_empty(), "resource-exit-call() call expects unexpected results {:?}", results);
                                let table_array = tables
                                    .resource_tables
                                    .try_lock()
                                    .expect("Could not get mutual reference to table.");
                                let mut resource_call_borrows = tables
                                    .resource_call_borrows
                                    .try_lock()
                                    .expect("Could not get mutual reference to resource call borrows.");
                                for (rid, handle) in resource_call_borrows.iter().copied() {
                                    ensure!(!table_array[rid].contains(handle), "Borrow was not dropped for resource transfer call.");
                                }
                                resource_call_borrows.clear();
                                Ok(())
                            },
                        )))
                    }
                }
            }
            CoreDef::InstanceFlags(i) => Ok(Extern::Global(inner.instance_flags[*i].clone())),
        }
    }

    /// Gets the core WASM export associated with the provided definition.
    fn core_export<T: Copy + Into<wasmtime_environ::EntityIndex>>(
        inner: &InstanceInner,
        ctx: impl AsContext,
        export: &CoreExport<T>,
    ) -> Option<Extern> {
        let name = match &export.item {
            ExportItem::Index(idx) => {
                &inner.component.0.export_mapping
                    [&inner.component.0.instance_modules[export.instance]][&(*idx).into()]
            }
            ExportItem::Name(s) => s,
        };

        inner.instances[export.instance].get_export(&ctx.as_context().inner, name)
    }

    /// Handles all global initializers and instantiates the set of WASM modules for this component.
    fn global_initialize(
        mut inner: InstanceInner,
        mut ctx: impl AsContextMut,
        linker: &Linker,
        resource_map: &FxHashMap<ResourceType, ResourceType>,
    ) -> Result<InstanceInner> {
        let mut destructors = Vec::new();
        for initializer in &inner.component.0.translation.component.initializers {
            match initializer {
                GlobalInitializer::InstantiateModule(InstantiateModule::Static(idx, def)) => {
                    let module = &inner.component.0.modules[idx];
                    let imports = Self::generate_imports(
                        &inner,
                        &mut ctx,
                        linker,
                        module,
                        def,
                        &mut destructors,
                        resource_map,
                    )?;
                    let instance = wasm_runtime_layer::Instance::new(
                        &mut ctx.as_context_mut().inner,
                        &module.module,
                        &imports,
                    )?;
                    inner.instances.push(instance);
                }
                GlobalInitializer::ExtractMemory(_) => {}
                GlobalInitializer::ExtractRealloc(_) => {}
                GlobalInitializer::ExtractPostReturn(_) => {}
                GlobalInitializer::LowerImport { .. } => {}
                GlobalInitializer::Resource(_) => {}
                _ => bail!("Not yet implemented {initializer:?}."),
            }
        }

        Self::fill_destructors(inner, ctx, destructors, resource_map)
    }

    /// Generates the set of core WASM imports for this component.
    fn generate_imports(
        inner: &InstanceInner,
        mut store: impl AsContextMut,
        linker: &Linker,
        module: &ModuleTranslation,
        defs: &[CoreDef],
        destructors: &mut Vec<TrampolineIndex>,
        resource_map: &FxHashMap<ResourceType, ResourceType>,
    ) -> Result<Imports> {
        let mut import_ty_map = FxHashMap::default();

        let engine = store.as_context().engine().clone();
        for import in module.module.imports(&engine) {
            import_ty_map.insert((import.module, import.name), import.ty.clone());
        }

        let mut imports = Imports::default();

        for (host, name, def) in module
            .translation
            .imports()
            .zip(defs)
            .map(|((module, name, _), arg)| (module, name, arg))
        {
            let ty = import_ty_map
                .get(&(host, name))
                .context("Unrecognized import.")?
                .clone();
            imports.define(
                host,
                name,
                Self::core_import(
                    inner,
                    &mut store,
                    def,
                    linker,
                    ty,
                    destructors,
                    resource_map,
                )?,
            );
        }

        Ok(imports)
    }

    /// Gets an import from the linker for this component.
    fn get_component_import(
        import: &ComponentImport,
        linker: &Linker,
    ) -> Result<crate::func::Func> {
        let inst = if let Some(name) = &import.instance {
            linker
                .instance(name)
                .ok_or_else(|| Error::msg(format!("Could not find imported interface {name:?}")))?
        } else {
            linker.root()
        };

        inst.func(&import.name)
            .ok_or_else(|| Error::msg(format!("Could not find function import {}", import.name)))
    }

    /// Fills the resource tables with all resource destructors.
    fn fill_destructors(
        inner: InstanceInner,
        ctx: impl AsContext,
        destructors: Vec<TrampolineIndex>,
        resource_map: &FxHashMap<ResourceType, ResourceType>,
    ) -> Result<InstanceInner> {
        let mut tables = inner
            .state_table
            .resource_tables
            .try_lock()
            .expect("Could not get access to resource tables.");

        for index in destructors {
            let (x, def) = require_matches!(
                &inner.component.0.generated_trampolines[&index],
                GeneratedTrampoline::ResourceDrop(x, def),
                (x, def)
            );
            if let Some(def) = def {
                let export = require_matches!(def, CoreDef::Export(x), x);
                tables[x.as_u32() as usize].set_destructor(Some(require_matches!(
                    Self::core_export(&inner, &ctx, export),
                    Some(Extern::Func(x)),
                    x
                )));
            }
        }

        for (id, idx) in inner.component.0.resolve.types.iter().filter_map(|(i, _)| {
            let val = inner.component.0.resource_map[i.index()];
            (val.as_u32() < u32::MAX - 1).then_some((i, val))
        }) {
            let res = ResourceType::from_resolve(
                inner.component.0.type_identifiers[id.index()].clone(),
                id,
                &inner.component.0,
                Some(resource_map),
            )?;
            match res.host_destructor() {
                Some(Some(func)) => tables[idx.as_u32() as usize].set_destructor(Some(func)),
                Some(None) => tables[idx.as_u32() as usize]
                    .set_destructor(ctx.as_context().inner.data().drop_host_resource.clone()),
                _ => {}
            }
        }

        drop(tables);

        Ok(inner)
    }
}

/// Provides the exports for an instance.
#[derive(Debug)]
pub struct Exports {
    /// The root interface of this instance.
    root: ExportInstance,
    /// All of this instance's exported interfaces.
    instances: FxHashMap<InterfaceIdentifier, ExportInstance>,
}

impl Exports {
    /// Creates a new set of exports.
    pub(crate) fn new() -> Self {
        Self {
            root: ExportInstance::new(),
            instances: FxHashMap::default(),
        }
    }

    /// Gets the root instance.
    pub fn root(&self) -> &ExportInstance {
        &self.root
    }

    /// Gets the instance with the specified name, if any.
    pub fn instance(&self, name: &InterfaceIdentifier) -> Option<&ExportInstance> {
        self.instances.get(name)
    }

    /// Gets an iterator over all instances by identifier.
    pub fn instances(&self) -> impl Iterator<Item = (&'_ InterfaceIdentifier, &'_ ExportInstance)> {
        self.instances.iter()
    }
}

/// Represents a specific interface from a instance.
#[derive(Debug)]
pub struct ExportInstance {
    /// The functions of the interface.
    functions: FxHashMap<Arc<str>, crate::func::Func>,
    /// The resources of the interface.
    resources: FxHashMap<Arc<str>, ResourceType>,
    /// The instance that owns these exports.
    instance: Weak<InstanceInner>,
}

impl ExportInstance {
    /// Creates a new, empty instance.
    pub(crate) fn new() -> Self {
        Self {
            functions: FxHashMap::default(),
            resources: FxHashMap::default(),
            instance: Weak::new(),
        }
    }

    /// Gets the associated function by name, if any.
    pub fn func(&self, name: impl AsRef<str>) -> Option<crate::func::Func> {
        self.functions.get(name.as_ref()).map(|x| {
            x.instantiate(Instance(
                self.instance.upgrade().expect("Instance did not exist."),
            ))
        })
    }

    /// Iterates over all associated functions by name.
    pub fn funcs(&self) -> impl Iterator<Item = (&'_ str, crate::func::Func)> {
        let inst = self.instance.upgrade().expect("Instance did not exist.");
        self.functions
            .iter()
            .map(move |(k, v)| (&**k, v.instantiate(Instance(inst.clone()))))
    }

    /// Gets the associated abstract resource by name, if any.
    pub fn resource(&self, name: impl AsRef<str>) -> Option<ResourceType> {
        self.resources.get(name.as_ref()).cloned()
    }

    /// Iterates over all associated functions by name.
    pub fn resources(&self) -> impl Iterator<Item = (&'_ str, ResourceType)> {
        self.resources.iter().map(|(k, v)| (&**k, v.clone()))
    }
}

/// Stores the internal state for an instance.
#[derive(Debug)]
struct InstanceInner {
    /// The component from which this instance was created.
    pub component: Component,
    /// The exports of this instance.
    pub exports: Exports,
    /// The unique ID of this instance.
    pub id: u64,
    /// The flags associated with this instance.
    pub instance_flags: wasmtime_environ::PrimaryMap<RuntimeComponentInstanceIndex, Global>,
    /// The underlying instantiated WASM modules for this instance.
    pub instances: wasmtime_environ::PrimaryMap<RuntimeInstanceIndex, wasm_runtime_layer::Instance>,
    /// Stores the instance-specific state.
    pub state_table: Arc<StateTable>,
    /// The list of types for this instance.
    pub types: Arc<[crate::types::ValueType]>,
    /// The store ID associated with this instance.
    pub store_id: u64,
}

/// Stores the instance-specific state for a component.
#[derive(Debug)]
struct StateTable {
    /// Whether this instance has been dropped.
    pub dropped: AtomicBool,
    /// The set of resource tables and destructors.
    pub resource_tables: Mutex<Vec<HandleTable>>,
    /// The set of resource kind and handle ids that have been borrowed for a resource call.
    pub resource_call_borrows: Mutex<Vec<(usize, i32)>>,
}

/// Details an import for a component.
#[derive(Clone, Debug)]
struct ComponentImport {
    /// The interface from which this export originates.
    pub instance: Option<InterfaceIdentifier>,
    /// The name of the import.
    pub name: Arc<str>,
    /// The function associated with the import.
    pub func: Function,
    /// The canonical options with which the import will be called.
    pub options: CanonicalOptions,
}

/// Details an export from a component.
#[derive(Clone, Debug)]
struct ComponentExport {
    /// The canonical options with which the export will be called.
    pub options: CanonicalOptions,
    /// The function associated with the export.
    pub func: Function,
    /// The definition of the export.
    pub def: CoreExport<wasmtime_environ::EntityIndex>,
    /// The type of export.
    pub ty: crate::types::FuncType,
}

/// The store represents all global state that can be manipulated by
/// WebAssembly programs. It consists of the runtime representation
/// of all instances of functions, tables, memories, and globals that
/// have been allocated during the lifetime of the abstract machine.
///
/// The `Store` holds the engine (that is —amongst many things— used to compile
/// the Wasm bytes into a valid module artifact).
///
/// Spec: <https://webassembly.github.io/spec/core/exec/runtime.html#store>
pub struct Store<T, E: backend::WasmEngine> {
    /// The backing implementation.
    inner: wasm_runtime_layer::Store<StoreInner<T, E>, E>,
}

impl<T, E: backend::WasmEngine> Store<T, E> {
    /// Creates a new [`Store`] with a specific [`Engine`].
    pub fn new(engine: &Engine<E>, data: T) -> Self {
        /// A counter that uniquely identifies stores.
        static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

        let mut inner = wasm_runtime_layer::Store::new(
            engine,
            StoreInner {
                id: ID_COUNTER.fetch_add(1, Ordering::AcqRel),
                data,
                host_functions: FuncVec::default(),
                host_resources: Slab::default(),
                drop_host_resource: None,
            },
        );

        inner.data_mut().drop_host_resource = Some(wasm_runtime_layer::Func::new(
            &mut inner,
            wasm_runtime_layer::FuncType::new([wasm_runtime_layer::ValueType::I32], [])
                .with_name("drop-host-resources"),
            |mut ctx, args, _| {
                if let wasm_runtime_layer::Value::I32(index) = &args[0] {
                    ctx.data_mut().host_resources.remove(*index as usize);
                    Ok(())
                } else {
                    bail!("Could not drop resource.");
                }
            },
        ));

        Self { inner }
    }

    /// Returns the [`Engine`] that this store is associated with.
    pub fn engine(&self) -> &Engine<E> {
        self.inner.engine()
    }

    /// Returns a shared reference to the user provided data owned by this [`Store`].
    pub fn data(&self) -> &T {
        &self.inner.data().data
    }

    /// Returns an exclusive reference to the user provided data owned by this [`Store`].
    pub fn data_mut(&mut self) -> &mut T {
        &mut self.inner.data_mut().data
    }

    /// Consumes `self` and returns its user provided data.
    pub fn into_data(self) -> T {
        self.inner.into_data().data
    }
}

/// A temporary handle to a [`&Store<T>`][`Store`].
///
/// This type is suitable for [`AsContext`] trait bounds on methods if desired.
/// For more information, see [`Store`].
pub struct StoreContext<'a, T: 'a, E: backend::WasmEngine> {
    /// The backing implementation.
    inner: wasm_runtime_layer::StoreContext<'a, StoreInner<T, E>, E>,
}

impl<'a, T: 'a, E: backend::WasmEngine> StoreContext<'a, T, E> {
    /// Returns the underlying [`Engine`] this store is connected to.
    pub fn engine(&self) -> &Engine<E> {
        self.inner.engine()
    }

    /// Access the underlying data owned by this store.
    ///
    /// Same as [`Store::data`].
    pub fn data(&self) -> &T {
        &self.inner.data().data
    }
}

/// A temporary handle to a [`&mut Store<T>`][`Store`].
///
/// This type is suitable for [`AsContextMut`] or [`AsContext`] trait bounds on methods if desired.
/// For more information, see [`Store`].
pub struct StoreContextMut<'a, T: 'a, E: backend::WasmEngine> {
    /// The backing implementation.
    inner: wasm_runtime_layer::StoreContextMut<'a, StoreInner<T, E>, E>,
}

impl<'a, T: 'a, E: backend::WasmEngine> StoreContextMut<'a, T, E> {
    /// Returns the underlying [`Engine`] this store is connected to.
    pub fn engine(&self) -> &Engine<E> {
        self.inner.engine()
    }

    /// Access the underlying data owned by this store.
    ///
    /// Same as [`Store::data`].    
    pub fn data(&self) -> &T {
        &self.inner.data().data
    }

    /// Access the underlying data owned by this store.
    ///
    /// Same as [`Store::data_mut`].
    pub fn data_mut(&mut self) -> &mut T {
        &mut self.inner.data_mut().data
    }
}

/// A trait used to get shared access to a [`Store`].
pub trait AsContext {
    /// The engine type associated with the context.
    type Engine: backend::WasmEngine;

    /// The user state associated with the [`Store`], aka the `T` in `Store<T>`.
    type UserState;

    /// Returns the store context that this type provides access to.
    fn as_context(&self) -> StoreContext<Self::UserState, Self::Engine>;
}

/// A trait used to get exclusive access to a [`Store`].
pub trait AsContextMut: AsContext {
    /// Returns the store context that this type provides access to.
    fn as_context_mut(&mut self) -> StoreContextMut<Self::UserState, Self::Engine>;
}

impl<T, E: backend::WasmEngine> AsContext for Store<T, E> {
    type Engine = E;

    type UserState = T;

    fn as_context(&self) -> StoreContext<Self::UserState, Self::Engine> {
        StoreContext {
            inner: wasm_runtime_layer::AsContext::as_context(&self.inner),
        }
    }
}

impl<T, E: backend::WasmEngine> AsContextMut for Store<T, E> {
    fn as_context_mut(&mut self) -> StoreContextMut<Self::UserState, Self::Engine> {
        StoreContextMut {
            inner: wasm_runtime_layer::AsContextMut::as_context_mut(&mut self.inner),
        }
    }
}

impl<T: AsContext> AsContext for &T {
    type Engine = T::Engine;

    type UserState = T::UserState;

    fn as_context(&self) -> StoreContext<Self::UserState, Self::Engine> {
        (**self).as_context()
    }
}

impl<T: AsContext> AsContext for &mut T {
    type Engine = T::Engine;

    type UserState = T::UserState;

    fn as_context(&self) -> StoreContext<Self::UserState, Self::Engine> {
        (**self).as_context()
    }
}

impl<T: AsContextMut> AsContextMut for &mut T {
    fn as_context_mut(&mut self) -> StoreContextMut<Self::UserState, Self::Engine> {
        (**self).as_context_mut()
    }
}

impl<'a, T: 'a, E: backend::WasmEngine> AsContext for StoreContext<'a, T, E> {
    type Engine = E;

    type UserState = T;

    fn as_context(&self) -> StoreContext<Self::UserState, Self::Engine> {
        StoreContext {
            inner: wasm_runtime_layer::AsContext::as_context(&self.inner),
        }
    }
}

impl<'a, T: 'a, E: backend::WasmEngine> AsContext for StoreContextMut<'a, T, E> {
    type Engine = E;

    type UserState = T;

    fn as_context(&self) -> StoreContext<Self::UserState, Self::Engine> {
        StoreContext {
            inner: wasm_runtime_layer::AsContext::as_context(&self.inner),
        }
    }
}

impl<'a, T: 'a, E: backend::WasmEngine> AsContextMut for StoreContextMut<'a, T, E> {
    fn as_context_mut(&mut self) -> StoreContextMut<Self::UserState, Self::Engine> {
        StoreContextMut {
            inner: wasm_runtime_layer::AsContextMut::as_context_mut(&mut self.inner),
        }
    }
}

/// Holds the inner mutable state for a component model implementation.
struct StoreInner<T, E: backend::WasmEngine> {
    /// The unique ID of this store.
    pub id: u64,
    /// The consumer's custom data.
    pub data: T,
    /// The table of host functions.
    pub host_functions: FuncVec<T, E>,
    /// The table of host resources.
    pub host_resources: Slab<Box<dyn Any + Send + Sync>>,
    /// A function that drops a host resource from this store.
    pub drop_host_resource: Option<wasm_runtime_layer::Func>,
}

/// Denotes a trampoline used by components to interact with the host.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
enum GeneratedTrampoline {
    /// The guest would like to call an imported function.
    ImportedFunction(ComponentImport),
    /// The guest would like to create a new resource.
    ResourceNew(TypeResourceTableIndex),
    /// The guest would like to obtain the representation of a resource.
    ResourceRep(TypeResourceTableIndex),
    /// The guest would like to drop a resource.
    ResourceDrop(TypeResourceTableIndex, Option<CoreDef>),
    /// A Utf8 string is copied from one component's memory to the other's.
    Utf8CopyTranscoder {
        /// Index of the linear memory from which the string is copied.
        from: RuntimeMemoryIndex,
        /// Index of the linear memory to which the string is copied.
        to: RuntimeMemoryIndex,
    },
    /// A degenerate lift/lower combination forces a trap.
    AlwaysTrap,
    /// An owned resource is transferred from one table to another.
    ResourceTransferOwn,
    /// A borrowed resource is transferred from one table to another.
    ResourceTransferBorrow,
    /// A call is being entered, requiring bookkeeping for resource handles.
    ResourceEnterCall,
    /// A call is being exited, requiring bookkeeping for resource handles.
    ResourceExitCall,
}

/// Represents a resource handle owned by a guest instance.
#[derive(Copy, Clone, Debug, Default)]
struct HandleElement {
    /// The originating instance's representation of the handle.
    pub rep: i32,
    /// Whether this handle is owned by this instance.
    pub own: bool,
    /// The number of times that this handle has been lent, without any borrows being returned.
    pub lend_count: i32,
}

/// Stores a set of resource handles and associated type information.
#[derive(Clone, Debug, Default)]
struct HandleTable {
    /// The array of handles.
    array: Slab<HandleElement>,
    /// The destructor for this handle type.
    destructor: Option<wasm_runtime_layer::Func>,
}

impl HandleTable {
    /// Gets the destructor for this handle table.
    pub fn destructor(&self) -> Option<&wasm_runtime_layer::Func> {
        self.destructor.as_ref()
    }

    /// Sets the destructor for this handle table.
    pub fn set_destructor(&mut self, destructor: Option<wasm_runtime_layer::Func>) {
        self.destructor = destructor;
    }

    /// Gets the element at the specified slot, or fails if it is empty.
    pub fn get(&self, i: i32) -> Result<&HandleElement> {
        self.array.get(i as usize).context("Invalid handle index.")
    }

    /// Sets the element at the specified slot, panicking if an element was
    /// not already there.
    pub fn set(&mut self, i: i32, element: HandleElement) {
        *self
            .array
            .get_mut(i as usize)
            .expect("Invalid handle index.") = element;
    }

    /// Inserts a new handle into this table, returning its index.
    pub fn add(&mut self, handle: HandleElement) -> i32 {
        self.array.insert(handle) as i32
    }

    /// Removes the handle at the provided index from the table,
    /// or fails if there was no handle present.
    pub fn remove(&mut self, i: i32) -> Result<HandleElement> {
        Ok(self.array.remove(i as usize))
    }

    /// Check if a handle is present at the provided index.
    pub fn contains(&self, i: i32) -> bool {
        self.array.contains(i as usize)
    }
}
