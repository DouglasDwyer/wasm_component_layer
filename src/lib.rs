#![allow(warnings)]

mod types;
mod values;

use anyhow::*;
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

        let mut modules = FxHashMap::with_capacity_and_hasher(module_data.len(), Default::default());

        for (id, module) in module_data {
            modules.insert(id, Module::new(engine, std::io::Cursor::new(module.wasm))?);
        }

        Ok(Self::create_component_inner(translation, types, modules, resolve, world_id))
    }
    
    fn create_component_inner(translation: ComponentTranslation, types: ComponentTypes, modules: FxHashMap<StaticModuleIndex, Module>, resolve: Resolve, world_id: Id<World>) -> ComponentInner {
        ComponentInner {
            extracted_memories: FxHashMap::default(),
            extracted_post_returns: FxHashMap::default(),
            instance_modules: wasmtime_environ::PrimaryMap::default(),
            modules,
            resolve,
            translation,
            types,
            world_id
        }
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
    pub extracted_memories: FxHashMap<RuntimeMemoryIndex, CoreExport<MemoryIndex>>,
    pub extracted_post_returns: FxHashMap<RuntimePostReturnIndex, CoreExport<wasmtime_environ::EntityIndex>>,
    pub instance_modules: wasmtime_environ::PrimaryMap<RuntimeInstanceIndex, StaticModuleIndex>,
    pub modules: FxHashMap<StaticModuleIndex, Module>,
    pub resolve: Resolve,
    pub translation: ComponentTranslation,
    pub types: ComponentTypes,
    pub world_id: Id<World>
}

pub struct Instance(Arc<InstanceInner>);

impl Instance {
    pub fn new(mut ctx: impl AsContextMut, component: &Component) -> Result<Self> {
        let instance = InstanceInner { component: component.0.clone(), instances: Default::default() };
        let initialized = Self::global_initialize(instance, ctx)?;
        Ok(Self(Arc::new(initialized)))
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

struct InstanceInner {
    pub component: Arc<ComponentInner>,
    pub instances: wasmtime_environ::PrimaryMap<RuntimeInstanceIndex, wasm_runtime_layer::Instance>
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
        let inst = Instance::new(&mut store, &comp).unwrap();
        //let inst = Linker::new(&engine).instantiate(&engine, &comp).unwrap();
    }
}