#![allow(warnings)]

mod abi;
mod func;
mod types;
mod values;

use anyhow::*;
pub use crate::func::*;
pub use crate::types::*;
pub use crate::values::*;
use fxhash::*;
use id_arena::*;
use std::sync::*;
use wasm_runtime_layer::*;
use wasmtime_environ::component::*;
use wit_component::*;
use wit_parser::*;

pub use wasm_runtime_layer::Engine as Engine;
pub use wasm_runtime_layer::Store as Store;
pub use wasm_runtime_layer::AsContext;
pub use wasm_runtime_layer::AsContextMut;

pub struct Component(Arc<ComponentInner>);

impl Component {
    pub fn new<E: backend::WasmEngine>(engine: &Engine<E>, bytes: &[u8]) -> Result<Self> {
        let inner = Self::generate_component(engine, bytes)?;
        Ok(Self(Arc::new(Self::extract_initializers(inner)?)))
    }

    fn generate_component<E: backend::WasmEngine>(engine: &Engine<E>, bytes: &[u8]) -> Result<ComponentInner> {
        let decoded = wit_component::decode(bytes)
        .context("Could not decode component information from bytes.")?;

        let (resolve, world_id) = match decoded {
            DecodedWasm::WitPackage(..) => bail!("Cannot instantiate WIT package as module."),
            DecodedWasm::Component(resolve, id) => (resolve, id)
        };

        let adapter_vec = wasmtime_environ::ScopeVec::new();
        let (translation, types, module_data) = Self::translate_modules(bytes, &adapter_vec)?;

        let export_mapping = Self::generate_export_mapping(&module_data);
        let mut modules = FxHashMap::with_capacity_and_hasher(module_data.len(), Default::default());

        for (id, module) in module_data {
            modules.insert(id, Module::new(engine, std::io::Cursor::new(module.wasm))?);
        }
        
        let mut size_align = SizeAlign::default();
        size_align.fill(&resolve);

        Ok(ComponentInner {
            export_mapping,
            extracted_memories: FxHashMap::default(),
            extracted_reallocs: FxHashMap::default(),
            extracted_post_returns: FxHashMap::default(),
            instance_modules: wasmtime_environ::PrimaryMap::default(),
            modules,
            resolve,
            size_align,
            translation,
            types,
            world_id
        })
    }

    fn generate_export_mapping(module_data: &wasmtime_environ::PrimaryMap<StaticModuleIndex, wasmtime_environ::ModuleTranslation>) -> FxHashMap<StaticModuleIndex, FxHashMap<wasmtime_environ::EntityIndex, String>> {
        let mut export_mapping = FxHashMap::with_capacity_and_hasher(module_data.len(), Default::default());

        for (idx, module) in module_data {
            let entry: &mut FxHashMap<wasmtime_environ::EntityIndex, String> = export_mapping.entry(idx).or_default();
            for (name, index) in module.module.exports.clone() {
                entry.insert(index, name);
            }
        }

        export_mapping
    }

    fn extract_initializers(mut inner: ComponentInner) -> Result<ComponentInner> {
        for initializer in &inner.translation.component.initializers {
            match initializer {
                GlobalInitializer::InstantiateModule(InstantiateModule::Static(idx, def)) => {
                    inner.instance_modules.push(*idx);
                },
                GlobalInitializer::ExtractMemory(ExtractMemory { index, export }) => {
                    ensure!(inner.extracted_memories.insert(*index, export.clone()).is_none(), "Extracted the same memory more than once.");
                },
                GlobalInitializer::ExtractRealloc(ExtractRealloc { index, def }) => {
                    if let CoreDef::Export(export) = def {
                        ensure!(inner.extracted_reallocs.insert(*index, export.clone()).is_none(), "Extracted the same memory more than once.");
                    }
                    else {
                        bail!("Unexpected post return definition type.");
                    }
                },
                GlobalInitializer::ExtractPostReturn(ExtractPostReturn { index, def }) => {
                    if let CoreDef::Export(export) = def {
                        ensure!(inner.extracted_post_returns.insert(*index, export.clone()).is_none(), "Extracted the same memory more than once.");
                    }
                    else {
                        bail!("Unexpected post return definition type.");
                    }
                },
                _ => bail!("Not yet implemented {initializer:?}.")
            }
        }

        Ok(inner)
    }

    fn translate_modules<'a>(bytes: &'a [u8], scope: &'a wasmtime_environ::ScopeVec<u8>) -> Result<(ComponentTranslation, ComponentTypes, wasmtime_environ::PrimaryMap<StaticModuleIndex, wasmtime_environ::ModuleTranslation<'a>>)> {
        let tunables = wasmtime_environ::Tunables::default();
        let mut types = ComponentTypesBuilder::default();
        let mut validator = Self::create_component_validator();

        let (translation, modules) = Translator::new(&tunables, &mut validator, &mut types, &scope)
            .translate(bytes)
            .context("Could not translate input component to core WASM.")?;

        Ok((translation, types.finish(), modules))
    }

    fn create_component_validator() -> wasmtime_environ::wasmparser::Validator {
        wasmtime_environ::wasmparser::Validator::new_with_features(wasmtime_environ::wasmparser::WasmFeatures {
            relaxed_simd: true,
            threads: true,
            multi_memory: true,
            exceptions: true,
            memory64: true,
            extended_const: true,
            component_model: true,
            function_references: true,
            memory_control: true,
            gc: true,
            component_model_values: true,
            mutable_global: true,
            saturating_float_to_int: true,
            sign_extension: true,
            bulk_memory: true,
            multi_value: true,
            reference_types: true,
            tail_call: true,
            simd: true,
            floats: true,
        })
    }
}

struct ComponentInner {
    pub export_mapping: FxHashMap<StaticModuleIndex, FxHashMap<wasmtime_environ::EntityIndex, String>>,
    pub extracted_memories: FxHashMap<RuntimeMemoryIndex, CoreExport<MemoryIndex>>,
    pub extracted_reallocs: FxHashMap<RuntimeReallocIndex, CoreExport<wasmtime_environ::EntityIndex>>,
    pub extracted_post_returns: FxHashMap<RuntimePostReturnIndex, CoreExport<wasmtime_environ::EntityIndex>>,
    pub instance_modules: wasmtime_environ::PrimaryMap<RuntimeInstanceIndex, StaticModuleIndex>,
    pub modules: FxHashMap<StaticModuleIndex, Module>,
    pub resolve: Resolve,
    pub size_align: SizeAlign,
    pub translation: ComponentTranslation,
    pub types: ComponentTypes,
    pub world_id: Id<World>
}

impl std::fmt::Debug for ComponentInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComponentInner").finish()
    }
}

#[derive(Clone, Debug)]
pub struct Instance(Arc<InstanceInner>);

impl Instance {
    pub fn new(mut ctx: impl AsContextMut, component: &Component) -> Result<Self> {
        let instance = InstanceInner { component: component.0.clone(), instances: Default::default(), funcs: FxHashMap::default() };
        let initialized = Self::global_initialize(instance, &mut ctx)?;
        let exported = Self::load_exports(initialized, &ctx)?;
        Ok(Self(Arc::new(exported)))
    }

    fn load_exports(mut inner: InstanceInner, ctx: impl AsContext) -> Result<InstanceInner> {
        let names = Self::export_names(&inner);

        for (export_name, export) in &inner.component.translation.component.exports {
            let world_key = &names[export_name];
            let item = &inner.component.resolve.worlds[inner.component.world_id].exports[world_key];
            match export {
                wasmtime_environ::component::Export::LiftedFunction {
                    ty,
                    func,
                    options,
                } => {
                    let f = match item {
                        WorldItem::Function(f) => f,
                        WorldItem::Interface(_) | WorldItem::Type(_) => unreachable!(),
                    };

                    inner.funcs.insert(("".to_string(), export_name.clone()), Self::export_function(&inner, &ctx, match func { CoreDef::Export(x) => x, _ => unreachable!() }, *ty, options, f));
                }
                wasmtime_environ::component::Export::Instance(iface) => {
                    let id = match item {
                        WorldItem::Interface(id) => *id,
                        WorldItem::Function(_) | WorldItem::Type(_) => unreachable!(),
                    };
                    for (func_name, export) in iface {
                        let (func, options, ty) = match export {
                            wasmtime_environ::component::Export::LiftedFunction { func, options, ty } => (func, options, ty),
                            wasmtime_environ::component::Export::Type(_) => continue, // ignored
                            _ => unreachable!(),
                        };

                        let f = &inner.component.resolve.interfaces[id].functions[func_name];
    
                        inner.funcs.insert((export_name.clone(), func_name.clone()), Self::export_function(&inner, &ctx, match func { CoreDef::Export(x) => x, _ => unreachable!() }, *ty, options, f));
                    }
                }

                // ignore type exports for now
                wasmtime_environ::component::Export::Type(_) => {}

                // This can't be tested at this time so leave it unimplemented
                wasmtime_environ::component::Export::ModuleStatic(_) => bail!("Not yet implemented."),
                wasmtime_environ::component::Export::ModuleImport(_) => bail!("Not yet implemented."),
            }
        }

        Ok(inner)
    }

    fn export_function(inner: &InstanceInner, ctx: impl AsContext, def: &CoreExport<wasmtime_environ::EntityIndex>, ty: TypeFuncIndex, options: &CanonicalOptions, func: &Function) -> crate::func::Func {
        let callee = Self::core_export(inner, &ctx, def).expect("Could not get callee export.").into_func().expect("Export was not of func type.");
        let memory = options.memory.map(|idx| Self::core_export(inner, &ctx, &inner.component.extracted_memories[&idx]).expect("Could not get runtime memory export.").into_memory().expect("Export was not of memory type."));
        let realloc = options.realloc.map(|idx| Self::core_export(inner, &ctx, &inner.component.extracted_reallocs[&idx]).expect("Could not get runtime realloc export.").into_func().expect("Export was not of func type."));
        let post_return = options.post_return.map(|idx| Self::core_export(inner, &ctx, &inner.component.extracted_post_returns[&idx]).expect("Could not get runtime post return export.").into_func().expect("Export was not of func type."));

        crate::func::Func(Arc::new(FuncInner {
            callee,
            component: inner.component.clone(),
            encoding: options.string_encoding,
            function: func.clone(),
            memory,
            realloc,
            post_return,
            ty
        }))
    }

    fn export_names(inner: &InstanceInner) -> FxHashMap<String, WorldKey> {
        let to_iter = &inner.component.resolve.worlds[inner.component.world_id].exports;
        let mut exports = FxHashMap::with_capacity_and_hasher(to_iter.len(), Default::default());
        for (key, _) in to_iter {
            let name = inner.component.resolve.name_world_key(key);
            exports.insert(name, key.clone());
        }
        exports
    }

    fn core_export<T: Copy + Into<wasmtime_environ::EntityIndex>>(inner: &InstanceInner, ctx: impl AsContext, export: &CoreExport<T>) -> Option<Extern> {
        let name = match &export.item {
            ExportItem::Index(idx) => &inner.component.export_mapping[&inner.component.instance_modules[export.instance]][&(*idx).into()],
            ExportItem::Name(s) => s,
        };

        inner.instances[export.instance].get_export(ctx, &name)
    }

    fn global_initialize(mut inner: InstanceInner, mut ctx: impl AsContextMut) -> Result<InstanceInner> {
        for initializer in &inner.component.translation.component.initializers {
            match initializer {
                GlobalInitializer::InstantiateModule(InstantiateModule::Static(idx, def)) => {
                    let instance = wasm_runtime_layer::Instance::new(&mut ctx, &inner.component.modules[idx], &Imports::default())?;
                    inner.instances.push(instance);
                },
                GlobalInitializer::ExtractMemory(_) => {},
                GlobalInitializer::ExtractPostReturn(_) => {},
                _ => bail!("Not yet implemented {initializer:?}.")
            }
        }

        Ok(inner)
    }
}

#[derive(Debug)]
struct InstanceInner {
    pub component: Arc<ComponentInner>,
    pub instances: wasmtime_environ::PrimaryMap<RuntimeInstanceIndex, wasm_runtime_layer::Instance>,
    pub funcs: FxHashMap<(String, String), crate::func::Func>
}

#[cfg(test)]
mod tests {
    use super::*;
    
    const WASM: &[u8] = include_bytes!("test_guest_component.wasm");

    #[test]
    fn run() {
        let engine = Engine::new(wasmi::Engine::default());
        let mut store = Store::new(&engine, ());
        let comp = Component::new(&engine, WASM).unwrap();
        let inst = crate::Instance::new(&mut store, &comp).unwrap();

        let test_simple = inst.0.funcs.get(&("test:guest/tester".to_owned(), "test-simple-a".to_owned())).unwrap();
        let mut res = [crate::values::Value::U32(0)];
        test_simple.call(&mut store, &[crate::values::Value::U32(23)], &mut res).unwrap();
        println!("omg we got {res:?}");

        let get_string = inst.0.funcs.get(&("test:guest/tester".to_owned(), "get-a-string".to_owned())).unwrap();
        let mut res = [crate::values::Value::U32(0)];
        get_string.call(&mut store, &[], &mut res).unwrap();

        println!("omg it returned {res:?}");
    }
}