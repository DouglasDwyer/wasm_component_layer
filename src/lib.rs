#![allow(warnings)]

mod abi;
mod func;
mod types;
mod values;

use anyhow::*;
use once_cell::sync::*;
pub use crate::func::*;
pub use crate::types::*;
pub use crate::values::*;
use fxhash::*;
use id_arena::*;
use ref_cast::*;
use std::sync::*;
use std::sync::atomic::*;
use wasm_runtime_layer::*;
use wasmtime_environ::component::*;
use wit_component::*;
use wit_parser::*;

pub use wasm_runtime_layer::Engine as Engine;

pub struct Component(Arc<ComponentInner>);

impl Component {
    pub fn new<E: backend::WasmEngine>(engine: &Engine<E>, bytes: &[u8]) -> Result<Self> {
        let inner = Self::generate_component(engine, bytes)?;
        Ok(Self(Arc::new(Self::load_exports(Self::extract_initializers(Self::generate_types(inner)?)?)?)))
    }

    pub fn exports(&self) -> &ExportTypes {
        &self.0.export_types
    }

    pub fn imports(&self) -> &ImportTypes {
        &self.0.import_types
    }

    fn generate_component<E: backend::WasmEngine>(engine: &Engine<E>, bytes: &[u8]) -> Result<ComponentInner> {
        let decoded = wit_component::decode(bytes)
        .context("Could not decode component information from bytes.")?;

        let (resolve, world_id) = match decoded {
            DecodedWasm::WitPackage(..) => bail!("Cannot instantiate WIT package as module."),
            DecodedWasm::Component(resolve, id) => (resolve, id)
        };

        let adapter_vec = wasmtime_environ::ScopeVec::new();
        let (translation, module_data) = Self::translate_modules(bytes, &adapter_vec)?;

        let export_mapping = Self::generate_export_mapping(&module_data);
        let mut modules = FxHashMap::with_capacity_and_hasher(module_data.len(), Default::default());

        for (id, module) in module_data {
            modules.insert(id, Module::new(engine, std::io::Cursor::new(module.wasm))?);
        }
        
        let mut size_align = SizeAlign::default();
        size_align.fill(&resolve);

        Ok(ComponentInner {
            export_mapping,
            export_names: FxHashMap::default(),
            export_types: ExportTypes::new(),
            extracted_memories: FxHashMap::default(),
            extracted_reallocs: FxHashMap::default(),
            extracted_post_returns: FxHashMap::default(),
            imported_functions: FxHashMap::default(),
            import_types: ImportTypes::new(),
            instance_modules: wasmtime_environ::PrimaryMap::default(),
            modules,
            resolve,
            size_align,
            translation,
            types: Vec::default(),
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

    fn generate_types(mut inner: ComponentInner) -> Result<ComponentInner> {
        for (id, val) in &inner.resolve.types {
            assert!(inner.types.len() == id.index(), "Type definition IDs were not equal.");
            inner.types.push(crate::types::ValueType::from_typedef(val, &inner.resolve)?);
        }
        Ok(inner)
    } 

    fn extract_initializers(mut inner: ComponentInner) -> Result<ComponentInner> {
        let lowering_options = Self::get_lowering_options(&inner.translation.trampolines);
        let mut imports = FxHashMap::default();
        for (key, _) in &inner.resolve.worlds[inner.world_id].imports {
            let name = inner.resolve.name_world_key(key);
            imports.insert(name, key.clone());
        }

        let root_name = Arc::<str>::from("$root");

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
                GlobalInitializer::LowerImport { index, import } => {
                    let (import_index, path) = &inner.translation.component.imports[*import];
                    let (import_name, _) = &inner.translation.component.import_types[*import_index];
                    let world_key = &imports[import_name];

                    let (core, imp) = match &inner.resolve.worlds[inner.world_id].imports[world_key] {
                        WorldItem::Function(func) => {
                            assert_eq!(path.len(), 0);
                            (CoreImport { module: root_name.clone(), name: import_name.as_str().into() }, ComponentImport {
                                instance: None,
                                func: func.clone(),
                                options: lowering_options[*index].clone()
                            })
                        }
                        WorldItem::Interface(i) => {
                            assert_eq!(path.len(), 1);
                            let iface = &inner.resolve.interfaces[*i];
                            let func = &iface.functions[&path[0]];
                            let imp_name = Arc::<str>::from(import_name.as_str());
                            (CoreImport { module: imp_name.clone(), name: path[0].as_str().into() }, ComponentImport {
                                instance: Some(imp_name),
                                func: func.clone(),
                                options: lowering_options[*index].clone()
                            })
                        }
                        WorldItem::Type(_) => unreachable!(),
                    };

                    //
                    let inst = if let Some(inst) = &imp.instance {
                        inner.import_types.instances.entry(inst.clone()).or_insert_with(ImportTypesInstance::new)
                    }
                    else {
                        &mut inner.import_types.root
                    };

                    ensure!(inst.functions.insert(core.name.clone(), crate::types::FuncType::from_resolve(&imp.func, &inner.resolve)?).is_none(), "Attempted to insert duplicate import.");

                    ensure!(inner.imported_functions.insert(core, imp).is_none(), "Attempted to insert duplicate import.");
                },
                _ => bail!("Not yet implemented {initializer:?}.")
            }
        }

        Ok(inner)
    }

    fn get_lowering_options<'a>(trampolines: &'a wasmtime_environ::PrimaryMap<TrampolineIndex, Trampoline>) -> wasmtime_environ::PrimaryMap<LoweredIndex, &'a CanonicalOptions> {
        let mut lowers = wasmtime_environ::PrimaryMap::default();
        for (_, trampoline) in trampolines {
            if let Trampoline::LowerImport { index, lower_ty, options } = trampoline {
                assert!(lowers.push(options) == *index, "Indices did not match.");
            }
        }
        lowers
    }

    fn translate_modules<'a>(bytes: &'a [u8], scope: &'a wasmtime_environ::ScopeVec<u8>) -> Result<(ComponentTranslation, wasmtime_environ::PrimaryMap<StaticModuleIndex, wasmtime_environ::ModuleTranslation<'a>>)> {
        let tunables = wasmtime_environ::Tunables::default();
        let mut types = ComponentTypesBuilder::default();
        let mut validator = Self::create_component_validator();

        let (translation, modules) = Translator::new(&tunables, &mut validator, &mut types, &scope)
            .translate(bytes)
            .context("Could not translate input component to core WASM.")?;

        Ok((translation, modules))
    }

    fn load_exports(mut inner: ComponentInner) -> Result<ComponentInner> {
        Self::export_names(&mut inner);

        for (export_name, export) in &inner.translation.component.exports {
            let world_key = &inner.export_names[export_name];
            let item = &inner.resolve.worlds[inner.world_id].exports[world_key];
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

                    ensure!(inner.export_types.root.functions.insert(export_name.as_str().into(), ComponentExport {
                        options: options.clone(),
                        def: match func { CoreDef::Export(x) => x.clone(), _ => unreachable!() },
                        func: f.clone(),
                        ty: crate::types::FuncType::from_resolve(f, &inner.resolve)?
                    }).is_none(), "Duplicate function definition.");
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

                        let f = &inner.resolve.interfaces[id].functions[func_name];
    
                        let exp = ComponentExport { options: options.clone(), def: match func { CoreDef::Export(x) => x.clone(), _ => unreachable!() }, func: f.clone(), ty: crate::types::FuncType::from_resolve(f, &inner.resolve)? };
                        if let Some(inst) = inner.export_types.instances.get_mut(export_name.as_str()) {
                            ensure!(inst.functions.insert(func_name.as_str().into(), exp).is_none(), "Duplicate function definition.");
                        }
                        else {
                            ensure!(inner.export_types.instances.entry(export_name.as_str().into()).or_insert(ExportTypesInstance::new()).functions.insert(func_name.as_str().into(), exp).is_none(), "Duplicate function definition.");
                        }
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

    fn export_names(inner: &mut ComponentInner) {
        let to_iter = &inner.resolve.worlds[inner.world_id].exports;
        let mut exports = FxHashMap::with_capacity_and_hasher(to_iter.len(), Default::default());
        for (key, _) in to_iter {
            let name = inner.resolve.name_world_key(key);
            exports.insert(name, key.clone());
        }
        inner.export_names = exports;
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
    pub export_names: FxHashMap<String, WorldKey>,
    pub export_types: ExportTypes,
    pub extracted_memories: FxHashMap<RuntimeMemoryIndex, CoreExport<MemoryIndex>>,
    pub extracted_reallocs: FxHashMap<RuntimeReallocIndex, CoreExport<wasmtime_environ::EntityIndex>>,
    pub extracted_post_returns: FxHashMap<RuntimePostReturnIndex, CoreExport<wasmtime_environ::EntityIndex>>,
    pub imported_functions: FxHashMap<CoreImport, ComponentImport>,
    pub import_types: ImportTypes,
    pub instance_modules: wasmtime_environ::PrimaryMap<RuntimeInstanceIndex, StaticModuleIndex>,
    pub modules: FxHashMap<StaticModuleIndex, Module>,
    pub resolve: Resolve,
    pub size_align: SizeAlign,
    pub translation: ComponentTranslation,
    pub types: Vec<crate::types::ValueType>,
    pub world_id: Id<World>
}

impl std::fmt::Debug for ComponentInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComponentInner").finish()
    }
}

#[derive(Debug)]
pub struct ExportTypes {
    root: ExportTypesInstance,
    instances: FxHashMap<Arc<str>, ExportTypesInstance>
}

impl ExportTypes {
    pub(crate) fn new() -> Self {
        Self { root: ExportTypesInstance::new(), instances: FxHashMap::default() }
    }

    pub fn root(&self) -> &ExportTypesInstance {
        &self.root
    }

    pub fn instance(&self, name: impl AsRef<str>) -> Option<&ExportTypesInstance> {
        self.instances.get(name.as_ref())
    }

    pub fn instances<'a>(&'a self) -> impl Iterator<Item = (&'a str, &'a ExportTypesInstance)> {
        self.instances.iter().map(|(k, v)| (&**k, v))
    }
}

#[derive(Debug)]
pub struct ExportTypesInstance {
    functions: FxHashMap<Arc<str>, ComponentExport>
}

impl ExportTypesInstance {
    pub(crate) fn new() -> Self {
        Self { functions: FxHashMap::default() }
    }

    pub fn func(&self, name: impl AsRef<str>) -> Option<crate::types::FuncType> {
        self.functions.get(name.as_ref()).map(|x| x.ty.clone())
    }

    pub fn funcs<'a>(&'a self) -> impl Iterator<Item = (&'a str, crate::types::FuncType)> {
        self.functions.iter().map(|(k, v)| (&**k, v.ty.clone()))
    }
}

#[derive(Debug)]
pub struct ImportTypes {
    root: ImportTypesInstance,
    instances: FxHashMap<Arc<str>, ImportTypesInstance>
}

impl ImportTypes {
    pub(crate) fn new() -> Self {
        Self { root: ImportTypesInstance::new(), instances: FxHashMap::default() }
    }

    pub fn root(&self) -> &ImportTypesInstance {
        &self.root
    }

    pub fn instance(&self, name: impl AsRef<str>) -> Option<&ImportTypesInstance> {
        self.instances.get(name.as_ref())
    }

    pub fn instances<'a>(&'a self) -> impl Iterator<Item = (&'a str, &'a ImportTypesInstance)> {
        self.instances.iter().map(|(k, v)| (&**k, v))
    }
}

#[derive(Debug)]
pub struct ImportTypesInstance {
    functions: FxHashMap<Arc<str>, crate::types::FuncType>
}

impl ImportTypesInstance {
    pub(crate) fn new() -> Self {
        Self { functions: FxHashMap::default() }
    }

    pub fn func(&self, name: impl AsRef<str>) -> Option<crate::types::FuncType> {
        self.functions.get(name.as_ref()).cloned()
    }

    pub fn funcs<'a>(&'a self) -> impl Iterator<Item = (&'a str, crate::types::FuncType)> {
        self.functions.iter().map(|(k, v)| (&**k, v.clone()))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Linker {
    root: LinkerInstance,
    instances: FxHashMap<Arc<str>, LinkerInstance>
}

impl Linker {
    pub fn root(&self) -> &LinkerInstance {
        &self.root
    }

    pub fn root_mut(&mut self) -> &mut LinkerInstance {
        &mut self.root
    }

    pub fn define_instance(&mut self, name: impl Into<Arc<str>>) -> Result<&mut LinkerInstance> {
        let n = Into::<Arc<str>>::into(name);
        if self.instance(&n).is_none() {
            Ok(self.instances.entry(n).or_default())
        }
        else {
            bail!("Duplicate instance definition.");
        }
    }

    pub fn instance(&self, name: impl AsRef<str>) -> Option<&LinkerInstance> {
        self.instances.get(name.as_ref())
    }

    pub fn instance_mut(&mut self, name: impl AsRef<str>) -> Option<&mut LinkerInstance> {
        self.instances.get_mut(name.as_ref())
    }

    pub fn instantiate(&self, ctx: impl AsContextMut, component: &Component) -> Result<Instance> {
        Instance::new(ctx, component, self)
    }
}

#[derive(Clone, Debug, Default)]
pub struct LinkerInstance {
    functions: FxHashMap<Arc<str>, crate::func::Func>
}

impl LinkerInstance {
    pub fn define_func(&mut self, name: impl Into<Arc<str>>, func: crate::func::Func) -> Result<()> {
        let n = Into::<Arc<str>>::into(name);
        if self.functions.contains_key(&n) {
            bail!("Duplicate function definition.");
        }

        self.functions.insert(n, func);
        Ok(())
    }

    pub fn func(&self, name: impl AsRef<str>) -> Option<crate::func::Func> {
        self.functions.get(name.as_ref()).cloned()
    }
}

#[derive(Clone, Debug)]
pub struct Instance(Arc<InstanceInner>);

impl Instance {
    pub(crate) fn new(mut ctx: impl AsContextMut, component: &Component, linker: &Linker) -> Result<Self> {
        let instance = InstanceInner { component: component.0.clone(), instances: Default::default(), funcs: FxHashMap::default() };
        let initialized = Self::global_initialize(instance, &mut ctx, linker)?;
        let exported = Self::load_exports(initialized, &ctx)?;
        Ok(Self(Arc::new(exported)))
    }

    fn load_exports(mut inner: InstanceInner, ctx: impl AsContext) -> Result<InstanceInner> {
        for (name, func) in &inner.component.export_types.root.functions {
            inner.funcs.insert(("".to_string(), name.to_string()), Self::export_function(&inner, &ctx, &func.def, &func.options, &func.func, func.ty.clone())?);
        }
        
        for (inst_name, inst) in &inner.component.export_types.instances {
            for (name, func) in &inst.functions {
                inner.funcs.insert((inst_name.to_string(), name.to_string()), Self::export_function(&inner, &ctx, &func.def, &func.options, &func.func, func.ty.clone())?);
            }
        }

        Ok(inner)
    }

    fn import_function(inner: &InstanceInner, ctx: impl AsContext, options: CanonicalOptions, func: Function) -> GuestInvokeOptions {
        let memory = options.memory.map(|idx| Self::core_export(inner, &ctx, &inner.component.extracted_memories[&idx]).expect("Could not get runtime memory export.").into_memory().expect("Export was not of memory type."));
        let realloc = options.realloc.map(|idx| Self::core_export(inner, &ctx, &inner.component.extracted_reallocs[&idx]).expect("Could not get runtime realloc export.").into_func().expect("Export was not of func type."));
        let post_return = options.post_return.map(|idx| Self::core_export(inner, &ctx, &inner.component.extracted_post_returns[&idx]).expect("Could not get runtime post return export.").into_func().expect("Export was not of func type."));

        GuestInvokeOptions {
            component: inner.component.clone(),
            encoding: options.string_encoding,
            function: func,
            memory,
            realloc,
            post_return
        }
    }

    fn export_function(inner: &InstanceInner, ctx: impl AsContext, def: &CoreExport<wasmtime_environ::EntityIndex>, options: &CanonicalOptions, func: &Function, ty: crate::types::FuncType) -> Result<crate::func::Func> {
        let callee = Self::core_export(inner, &ctx, def).expect("Could not get callee export.").into_func().expect("Export was not of func type.");
        let memory = options.memory.map(|idx| Self::core_export(inner, &ctx, &inner.component.extracted_memories[&idx]).expect("Could not get runtime memory export.").into_memory().expect("Export was not of memory type."));
        let realloc = options.realloc.map(|idx| Self::core_export(inner, &ctx, &inner.component.extracted_reallocs[&idx]).expect("Could not get runtime realloc export.").into_func().expect("Export was not of func type."));
        let post_return = options.post_return.map(|idx| Self::core_export(inner, &ctx, &inner.component.extracted_post_returns[&idx]).expect("Could not get runtime post return export.").into_func().expect("Export was not of func type."));

        Ok(crate::func::Func {
            store_id: ctx.as_context().inner.data().id,
            ty,
            backing: FuncImpl::GuestFunc(Arc::new(GuestFunc {
                callee,
                component: inner.component.clone(),
                encoding: options.string_encoding,
                function: func.clone(),
                memory,
                realloc,
                post_return,
            }))
        })
    }

    fn core_export<T: Copy + Into<wasmtime_environ::EntityIndex>>(inner: &InstanceInner, ctx: impl AsContext, export: &CoreExport<T>) -> Option<Extern> {
        let name = match &export.item {
            ExportItem::Index(idx) => &inner.component.export_mapping[&inner.component.instance_modules[export.instance]][&(*idx).into()],
            ExportItem::Name(s) => s,
        };

        println!("need instancee {:?}", export.instance);
        inner.instances[export.instance].get_export(&ctx.as_context().inner, &name)
    }

    fn global_initialize(mut inner: InstanceInner, mut ctx: impl AsContextMut, linker: &Linker) -> Result<InstanceInner> {
        println!("Initializers include {:?}", inner.component.translation.component.initializers);
        for initializer in &inner.component.translation.component.initializers {
            match initializer {
                GlobalInitializer::InstantiateModule(InstantiateModule::Static(idx, def)) => {
                    println!("Instantiate module {idx:?} (needs defs {def:?})");
                    let module = &inner.component.modules[idx];
                    let (imports, to_fill) = Self::generate_imports(&inner, &mut ctx, linker, module)?;
                    let instance = wasm_runtime_layer::Instance::new(&mut ctx.as_context_mut().inner, module, &imports)?;
                    inner.instances.push(instance);

                    for (val, options, func) in to_fill {
                        drop(val.set(Self::import_function(&inner, &ctx, options, func)));
                    }
                },
                GlobalInitializer::ExtractMemory(_) => {},
                GlobalInitializer::ExtractRealloc(_) => {},
                GlobalInitializer::ExtractPostReturn(_) => {},
                GlobalInitializer::LowerImport { .. } => {}
                _ => bail!("Not yet implemented {initializer:?}.")
            }
        }

        Ok(inner)
    }

    fn generate_imports(inner: &InstanceInner, mut store: impl AsContextMut, linker: &Linker, module: &Module) -> Result<(Imports, Vec<(Arc<OnceCell<GuestInvokeOptions>>, CanonicalOptions, Function)>)> {
        let mut imports = Imports::default();
        let mut options_to_fill = Vec::default();
        
        let core_imports = module.imports(store.as_context().engine()).filter_map(|x| x.ty.func().map(|y| (Arc::<str>::from(x.module), Arc::<str>::from(x.name), y.clone()))).collect::<Vec<_>>();
        for (module, name, ty) in core_imports {
            if let Some(component_import) = inner.component.imported_functions.get(&CoreImport { module: module.clone(), name: name.clone() }) {
                let func = Self::get_component_import(&name, component_import, linker)?;

                let guest_options = Arc::new(OnceCell::new());
                options_to_fill.push((guest_options.clone(), component_import.options.clone(), component_import.func.clone()));

                imports.define(&module, &name, Extern::Func(Func::new(store.as_context_mut().inner, ty, move |ctx, args, results| {
                    let ctx = StoreContextMut { inner: ctx };
                    func.call_from_guest(ctx, &guest_options.get().expect("Function was not initialized."), args, results)
                })));
            }
            else {
                bail!("Undefined import {module:?} {name:?}")
            }
        }

        Ok((imports, options_to_fill))
    }

    fn get_component_import(name: &str, import: &ComponentImport, linker: &Linker) -> Result<crate::func::Func> {
        let inst = if let Some(name) = &import.instance {
            linker.instance(name).ok_or_else(|| Error::msg(format!("Could not find imported interface {name}")))?
        }
        else {
            linker.root()
        };

        inst.func(name).ok_or_else(|| Error::msg(format!("Could not find function import {name}")))
    }
}

#[derive(Debug)]
struct InstanceInner {
    pub component: Arc<ComponentInner>,
    pub instances: wasmtime_environ::PrimaryMap<RuntimeInstanceIndex, wasm_runtime_layer::Instance>,
    pub funcs: FxHashMap<(String, String), crate::func::Func>
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct CoreImport {
    pub module: Arc<str>,
    pub name: Arc<str>
}

#[derive(Clone, Debug)]
struct ComponentImport {
    pub instance: Option<Arc<str>>,
    pub func: Function,
    pub options: CanonicalOptions
}

#[derive(Clone, Debug)]
struct ComponentExport {
    pub options: CanonicalOptions,
    pub func: Function,
    pub def: CoreExport<wasmtime_environ::EntityIndex>,
    pub ty: crate::types::FuncType
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
        static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

        Self {
            inner: wasm_runtime_layer::Store::new(&engine, StoreInner { id: ID_COUNTER.fetch_add(1, Ordering::AcqRel), data, host_functions: FuncVec::default() }),
        }
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

struct StoreInner<T, E: backend::WasmEngine> {
    pub id: u64,
    pub data: T,
    pub host_functions: FuncVec<T, E>
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imports() {
        const WASM: &[u8] = include_bytes!("test_guest_component2.wasm");

        let engine = Engine::new(wasmi::Engine::default());
        let mut store = Store::new(&engine, ());
        let comp = Component::new(&engine, WASM).unwrap();

        let mut linker = Linker::default();
        
        let func_ty = comp.imports().instance("test:guest/tester").unwrap().func("get-a-string").unwrap();
        linker.define_instance("test:guest/tester").unwrap().define_func("get-a-string", crate::func::Func::new(&mut store, func_ty, |_, _, results| {
            results[0] = crate::values::Value::String("borger".into());
            Ok(())
        }));

        let inst = linker.instantiate(&mut store, &comp).unwrap();
    }
}