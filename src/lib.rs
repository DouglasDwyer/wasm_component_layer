#![allow(warnings)]

mod abi;
mod func;
mod identifier;
#[macro_use]
mod require_matches;
mod types;
mod values;

use std::collections::hash_map::*;
use std::sync::atomic::*;
use std::sync::*;

use anyhow::*;
use fxhash::*;
use id_arena::*;
use ref_cast::*;
use vec_option::*;
pub use wasm_runtime_layer::Engine;
use wasm_runtime_layer::*;
use wasmtime_environ::component::*;
use wit_component::*;
use wit_parser::*;

pub use crate::func::*;
pub use crate::identifier::*;
use crate::require_matches::*;
pub use crate::types::*;
pub use crate::values::*;

pub struct Component(Arc<ComponentInner>);

impl Component {
    pub fn new<E: backend::WasmEngine>(engine: &Engine<E>, bytes: &[u8]) -> Result<Self> {
        let (inner, types) = Self::generate_component(engine, bytes)?;
        Ok(Self(Arc::new(Self::generate_resources(
            Self::load_exports(Self::extract_initializers(inner, &types)?, &types)?,
        )?)))
    }

    pub fn exports(&self) -> &ExportTypes {
        &self.0.export_types
    }

    pub fn imports(&self) -> &ImportTypes {
        &self.0.import_types
    }

    fn generate_component<E: backend::WasmEngine>(
        engine: &Engine<E>,
        bytes: &[u8],
    ) -> Result<(ComponentInner, ComponentTypes)> {
        static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

        let decoded = wit_component::decode(bytes)
            .context("Could not decode component information from bytes.")?;

        let (resolve, world_id) = match decoded {
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

        let export_types = ExportTypes::new(
            (&resolve.packages[resolve.worlds[world_id]
                .package
                .context("No package associated with world.")?]
            .name)
                .into(),
        );

        let package_identifiers = Self::generate_package_identifiers(&resolve)?;
        let interface_identifiers =
            Self::generate_interface_identifiers(&resolve, &package_identifiers)?;

        Ok((
            ComponentInner {
                export_mapping,
                export_names: FxHashMap::default(),
                import_types: ImportTypes::new(),
                export_types,
                extracted_memories: FxHashMap::default(),
                extracted_reallocs: FxHashMap::default(),
                extracted_post_returns: FxHashMap::default(),
                id: ID_COUNTER.fetch_add(1, Ordering::AcqRel),
                generated_trampolines: FxHashMap::default(),
                instance_modules: wasmtime_environ::PrimaryMap::default(),
                interface_identifiers,
                modules,
                package_identifiers,
                resource_map: vec![
                    TypeResourceTableIndex::from_u32(u32::MAX - 1);
                    resolve.types.len()
                ],
                resolve,
                size_align,
                translation,
                world_id,
            },
            component_types,
        ))
    }

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

    fn generate_resources(mut inner: ComponentInner) -> Result<ComponentInner> {
        for (key, item) in &inner.resolve.worlds[inner.world_id].imports {
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
                                        ResourceType::from_resolve(*x, &inner, None)?
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
                            let ty = ResourceType::from_resolve(*ty, &inner, None)?;
                            let entry = inner
                                .import_types
                                .instances
                                .entry(inner.interface_identifiers[x.index()].clone())
                                .or_insert_with(ImportTypesInstance::new);
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

        for (key, item) in &inner.resolve.worlds[inner.world_id].exports {
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
                                        ResourceType::from_resolve(*x, &inner, None)?
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
                            let ty = ResourceType::from_resolve(*ty, &inner, None)?;
                            let entry = inner
                                .export_types
                                .instances
                                .entry(inner.interface_identifiers[x.index()].clone())
                                .or_insert_with(ExportTypesInstance::new);
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

    fn generate_package_identifiers(resolve: &Resolve) -> Result<Vec<PackageIdentifier>> {
        let mut res = Vec::with_capacity(resolve.packages.len());

        for (_, pkg) in &resolve.packages {
            res.push(PackageIdentifier::try_from(&pkg.name)?);
        }

        Ok(res)
    }

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

    fn extract_initializers(
        mut inner: ComponentInner,
        types: &ComponentTypes,
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

        let root_name = Arc::<str>::from("$root");

        let mut destructors = FxHashMap::default();

        for initializer in &inner.translation.component.initializers {
            match initializer {
                GlobalInitializer::InstantiateModule(InstantiateModule::Static(idx, def)) => {
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
                            .or_insert_with(ImportTypesInstance::new)
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

        for (_, trampoline) in &mut inner.generated_trampolines {
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
                Trampoline::ResourceNew(x) => {
                    output_trampolines.insert(idx, GeneratedTrampoline::ResourceNew(*x));
                }
                Trampoline::ResourceRep(x) => {
                    output_trampolines.insert(idx, GeneratedTrampoline::ResourceRep(*x));
                }
                Trampoline::ResourceDrop(x) => {
                    output_trampolines.insert(idx, GeneratedTrampoline::ResourceDrop(*x, None));
                }
                _ => bail!("Trampoline not implemented."),
            }
        }
        Ok(lowers)
    }

    fn translate_modules<'a>(
        bytes: &'a [u8],
        scope: &'a wasmtime_environ::ScopeVec<u8>,
    ) -> Result<(
        ComponentTranslation,
        wasmtime_environ::PrimaryMap<StaticModuleIndex, wasmtime_environ::ModuleTranslation<'a>>,
        ComponentTypes,
    )> {
        let tunables = wasmtime_environ::Tunables::default();
        let mut types = ComponentTypesBuilder::default();
        let mut validator = Self::create_component_validator();

        let (translation, modules) = Translator::new(&tunables, &mut validator, &mut types, &scope)
            .translate(bytes)
            .context("Could not translate input component to core WASM.")?;

        Ok((translation, modules, types.finish()))
    }

    fn load_exports(mut inner: ComponentInner, types: &ComponentTypes) -> Result<ComponentInner> {
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

                    ensure!(
                        inner
                            .export_types
                            .root
                            .functions
                            .insert(
                                export_name.as_str().into(),
                                ComponentExport {
                                    options: options.clone(),
                                    def: match func {
                                        CoreDef::Export(x) => x.clone(),
                                        _ => unreachable!(),
                                    },
                                    func: f.clone(),
                                    ty: crate::types::FuncType::from_component(f, &inner, None)?
                                }
                            )
                            .is_none(),
                        "Duplicate function definition."
                    );
                }
                wasmtime_environ::component::Export::Instance(iface) => {
                    let id = match item {
                        WorldItem::Interface(id) => *id,
                        WorldItem::Function(_) | WorldItem::Type(_) => unreachable!(),
                    };
                    for (func_name, export) in iface {
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
                        ensure!(
                            inner
                                .export_types
                                .instances
                                .entry(inner.interface_identifiers[id.index()].clone())
                                .or_insert_with(ExportTypesInstance::new)
                                .functions
                                .insert(func_name.as_str().into(), exp)
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
                }
                wasmtime_environ::component::Export::ModuleImport(_) => {
                    bail!("Not yet implemented.")
                }
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
            (TypeDefKind::Union(t1), InterfaceType::Union(t2)) => {
                let t2 = &types[*t2];
                for (f1, f2) in t1.cases.iter().zip(t2.types.iter()) {
                    Self::connect_resources(resolve, types, &f1.ty, f2, map);
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

    fn create_component_validator() -> wasmtime_environ::wasmparser::Validator {
        wasmtime_environ::wasmparser::Validator::new_with_features(
            wasmtime_environ::wasmparser::WasmFeatures {
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
            },
        )
    }
}

struct ComponentInner {
    pub export_mapping:
        FxHashMap<StaticModuleIndex, FxHashMap<wasmtime_environ::EntityIndex, String>>,
    pub export_names: FxHashMap<String, WorldKey>,
    pub export_types: ExportTypes,
    pub extracted_memories: FxHashMap<RuntimeMemoryIndex, CoreExport<MemoryIndex>>,
    pub extracted_reallocs:
        FxHashMap<RuntimeReallocIndex, CoreExport<wasmtime_environ::EntityIndex>>,
    pub extracted_post_returns:
        FxHashMap<RuntimePostReturnIndex, CoreExport<wasmtime_environ::EntityIndex>>,
    pub resource_map: Vec<TypeResourceTableIndex>,
    pub generated_trampolines: FxHashMap<TrampolineIndex, GeneratedTrampoline>,
    pub id: u64,
    pub import_types: ImportTypes,
    pub instance_modules: wasmtime_environ::PrimaryMap<RuntimeInstanceIndex, StaticModuleIndex>,
    pub interface_identifiers: Vec<InterfaceIdentifier>,
    pub modules: FxHashMap<StaticModuleIndex, ModuleTranslation>,
    pub package_identifiers: Vec<PackageIdentifier>,
    pub resolve: Resolve,
    pub size_align: SizeAlign,
    pub translation: ComponentTranslation,
    pub world_id: Id<World>,
}

impl std::fmt::Debug for ComponentInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComponentInner").finish()
    }
}

struct ModuleTranslation {
    pub module: Module,
    pub translation: wasmtime_environ::Module,
}

#[derive(Debug)]
pub struct ExportTypes {
    root: ExportTypesInstance,
    instances: FxHashMap<InterfaceIdentifier, ExportTypesInstance>,
    package: PackageIdentifier,
}

impl ExportTypes {
    pub(crate) fn new(package: PackageIdentifier) -> Self {
        Self {
            root: ExportTypesInstance::new(),
            instances: FxHashMap::default(),
            package,
        }
    }

    pub fn root(&self) -> &ExportTypesInstance {
        &self.root
    }

    pub fn package(&self) -> &PackageIdentifier {
        &self.package
    }

    pub fn instance(&self, name: &InterfaceIdentifier) -> Option<&ExportTypesInstance> {
        self.instances.get(name)
    }

    pub fn instances<'a>(
        &'a self,
    ) -> impl Iterator<Item = (&'a InterfaceIdentifier, &'a ExportTypesInstance)> {
        self.instances.iter()
    }
}

#[derive(Debug)]
pub struct ExportTypesInstance {
    functions: FxHashMap<Arc<str>, ComponentExport>,
    resources: FxHashMap<Arc<str>, ResourceType>,
}

impl ExportTypesInstance {
    pub(crate) fn new() -> Self {
        Self {
            functions: FxHashMap::default(),
            resources: FxHashMap::default(),
        }
    }

    pub fn func(&self, name: impl AsRef<str>) -> Option<crate::types::FuncType> {
        self.functions.get(name.as_ref()).map(|x| x.ty.clone())
    }

    pub fn funcs<'a>(&'a self) -> impl Iterator<Item = (&'a str, crate::types::FuncType)> {
        self.functions.iter().map(|(k, v)| (&**k, v.ty.clone()))
    }

    pub fn resource(&self, name: impl AsRef<str>) -> Option<ResourceType> {
        self.resources.get(name.as_ref()).cloned()
    }

    pub fn resources<'a>(&'a self) -> impl Iterator<Item = (&'a str, crate::types::ResourceType)> {
        self.resources.iter().map(|(k, v)| (&**k, v.clone()))
    }
}

#[derive(Debug)]
pub struct ImportTypes {
    root: ImportTypesInstance,
    instances: FxHashMap<InterfaceIdentifier, ImportTypesInstance>,
}

impl ImportTypes {
    pub(crate) fn new() -> Self {
        Self {
            root: ImportTypesInstance::new(),
            instances: FxHashMap::default(),
        }
    }

    pub fn root(&self) -> &ImportTypesInstance {
        &self.root
    }

    pub fn instance(&self, name: &InterfaceIdentifier) -> Option<&ImportTypesInstance> {
        self.instances.get(name)
    }

    pub fn instances<'a>(
        &'a self,
    ) -> impl Iterator<Item = (&'a InterfaceIdentifier, &'a ImportTypesInstance)> {
        self.instances.iter()
    }
}

#[derive(Debug)]
pub struct ImportTypesInstance {
    functions: FxHashMap<Arc<str>, crate::types::FuncType>,
    resources: FxHashMap<Arc<str>, ResourceType>,
}

impl ImportTypesInstance {
    pub(crate) fn new() -> Self {
        Self {
            functions: FxHashMap::default(),
            resources: FxHashMap::default(),
        }
    }

    pub fn func(&self, name: impl AsRef<str>) -> Option<crate::types::FuncType> {
        self.functions.get(name.as_ref()).cloned()
    }

    pub fn funcs<'a>(&'a self) -> impl Iterator<Item = (&'a str, crate::types::FuncType)> {
        self.functions.iter().map(|(k, v)| (&**k, v.clone()))
    }

    pub fn resource(&self, name: impl AsRef<str>) -> Option<ResourceType> {
        self.resources.get(name.as_ref()).cloned()
    }

    pub fn resources<'a>(&'a self) -> impl Iterator<Item = (&'a str, crate::types::ResourceType)> {
        self.resources.iter().map(|(k, v)| (&**k, v.clone()))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Linker {
    root: LinkerInstance,
    instances: FxHashMap<InterfaceIdentifier, LinkerInstance>,
}

impl Linker {
    pub fn root(&self) -> &LinkerInstance {
        &self.root
    }

    pub fn root_mut(&mut self) -> &mut LinkerInstance {
        &mut self.root
    }

    pub fn define_instance(&mut self, name: InterfaceIdentifier) -> Result<&mut LinkerInstance> {
        if self.instance(&name).is_none() {
            Ok(self.instances.entry(name).or_default())
        } else {
            bail!("Duplicate instance definition.");
        }
    }

    pub fn instance(&self, name: &InterfaceIdentifier) -> Option<&LinkerInstance> {
        self.instances.get(name)
    }

    pub fn instance_mut(&mut self, name: &InterfaceIdentifier) -> Option<&mut LinkerInstance> {
        self.instances.get_mut(name)
    }

    pub fn instantiate(&self, ctx: impl AsContextMut, component: &Component) -> Result<Instance> {
        Instance::new(ctx, component, self)
    }
}

#[derive(Clone, Debug, Default)]
pub struct LinkerInstance {
    functions: FxHashMap<Arc<str>, crate::func::Func>,
    resources: FxHashMap<Arc<str>, ResourceType>,
}

impl LinkerInstance {
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

    pub fn func(&self, name: impl AsRef<str>) -> Option<crate::func::Func> {
        self.functions.get(name.as_ref()).cloned()
    }

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

    pub fn resource(&self, name: impl AsRef<str>) -> Option<ResourceType> {
        self.resources.get(name.as_ref()).cloned()
    }
}

#[derive(Clone, Debug)]
pub struct Instance(Arc<InstanceInner>);

impl Instance {
    pub(crate) fn new(
        mut ctx: impl AsContextMut,
        component: &Component,
        linker: &Linker,
    ) -> Result<Self> {
        static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

        let mut instance_flags = wasmtime_environ::PrimaryMap::default();
        for i in 0..component.0.instance_modules.len() {
            instance_flags.push(Global::new(
                ctx.as_context_mut().inner,
                wasm_runtime_layer::Value::I32(
                    wasmtime_environ::component::FLAG_MAY_LEAVE
                        | wasmtime_environ::component::FLAG_MAY_ENTER,
                ),
                true,
            ));
        }

        let id = ID_COUNTER.fetch_add(1, Ordering::AcqRel);
        let map = Self::create_resource_instantiation_map(&ctx, id, component, linker)?;
        let types = Self::generate_types(&ctx, id, component, linker, &map)?;
        let resource_tables = Arc::new(Mutex::new(vec![
            HandleTable::default();
            component
                .0
                .translation
                .component
                .num_resource_tables
        ]));
        let instance = InstanceInner {
            component: component.0.clone(),
            exports: Exports::new(component.exports().package().clone()),
            id,
            instances: Default::default(),
            instance_flags,
            resource_tables,
            types,
        };
        let initialized = Self::global_initialize(instance, &mut ctx, linker, &map)?;
        let exported = Self::load_exports(initialized, &ctx, &map)?;
        Ok(Self(Arc::new(exported)))
    }

    pub fn component(&self) -> Component {
        Component(self.0.component.clone())
    }

    pub fn exports(&self) -> &Exports {
        &self.0.exports
    }

    fn generate_types(
        ctx: impl AsContext,
        instance_id: u64,
        component: &Component,
        linker: &Linker,
        map: &FxHashMap<ResourceType, ResourceType>,
    ) -> Result<Arc<[crate::types::ValueType]>> {
        let mut types = Vec::with_capacity(component.0.resolve.types.len());
        for (id, val) in &component.0.resolve.types {
            assert!(
                types.len() == id.index(),
                "Type definition IDs were not equal."
            );
            if val.kind == TypeDefKind::Resource {
                types.push(crate::types::ValueType::Bool);
            } else {
                types.push(crate::types::ValueType::from_component_typedef(
                    id,
                    &component.0,
                    Some(&map),
                )?);
            }
        }
        Ok(types.into())
    }

    fn create_resource_instantiation_map(
        ctx: impl AsContext,
        instance_id: u64,
        component: &Component,
        linker: &Linker,
    ) -> Result<FxHashMap<ResourceType, ResourceType>> {
        let store_id = ctx.as_context().inner.data().id;
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
            let instantiated = resource.instantiate(store_id, instance_id)?;
            types.insert(resource, instantiated);
        }

        Ok(types)
    }

    fn load_exports(
        mut inner: InstanceInner,
        ctx: impl AsContext,
        map: &FxHashMap<ResourceType, ResourceType>,
    ) -> Result<InstanceInner> {
        let store_id = ctx.as_context().inner.data().id;
        for (name, func) in &inner.component.export_types.root.functions {
            inner.exports.root.functions.insert(
                name.clone(),
                Self::export_function(&inner, &ctx, &func.def, &func.options, &func.func, map)?,
            );
        }
        for (name, res) in &inner.component.export_types.root.resources {
            inner
                .exports
                .root
                .resources
                .insert(name.clone(), res.instantiate(store_id, inner.id)?);
        }

        let mut generated_functions = Vec::new();
        for (inst_name, inst) in &inner.component.export_types.instances {
            for (name, func) in &inst.functions {
                let export =
                    Self::export_function(&inner, &ctx, &func.def, &func.options, &func.func, map)?;
                generated_functions.push((inst_name.clone(), name.clone(), export));
            }
            for (name, res) in &inst.resources {
                inner
                    .exports
                    .instances
                    .entry(inst_name.clone())
                    .or_insert_with(ExportInstance::new)
                    .resources
                    .insert(name.clone(), res.instantiate(store_id, inner.id)?);
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

    fn import_function(
        inner: &InstanceInner,
        ctx: impl AsContext,
        options: &CanonicalOptions,
        func: &Function,
    ) -> GuestInvokeOptions {
        let memory = options.memory.map(|idx| {
            Self::core_export(inner, &ctx, &inner.component.extracted_memories[&idx])
                .expect("Could not get runtime memory export.")
                .into_memory()
                .expect("Export was not of memory type.")
        });
        let realloc = options.realloc.map(|idx| {
            Self::core_export(inner, &ctx, &inner.component.extracted_reallocs[&idx])
                .expect("Could not get runtime realloc export.")
                .into_func()
                .expect("Export was not of func type.")
        });
        let post_return = options.post_return.map(|idx| {
            Self::core_export(inner, &ctx, &inner.component.extracted_post_returns[&idx])
                .expect("Could not get runtime post return export.")
                .into_func()
                .expect("Export was not of func type.")
        });

        GuestInvokeOptions {
            component: inner.component.clone(),
            encoding: options.string_encoding,
            function: func.clone(),
            memory,
            realloc,
            resource_tables: inner.resource_tables.clone(),
            post_return,
            types: inner.types.clone(),
            instance_id: inner.id,
        }
    }

    fn export_function(
        inner: &InstanceInner,
        ctx: impl AsContext,
        def: &CoreExport<wasmtime_environ::EntityIndex>,
        options: &CanonicalOptions,
        func: &Function,
        mapping: &FxHashMap<ResourceType, ResourceType>,
    ) -> Result<crate::func::Func> {
        let callee = Self::core_export(inner, &ctx, def)
            .expect("Could not get callee export.")
            .into_func()
            .expect("Export was not of func type.");
        let memory = options.memory.map(|idx| {
            Self::core_export(inner, &ctx, &inner.component.extracted_memories[&idx])
                .expect("Could not get runtime memory export.")
                .into_memory()
                .expect("Export was not of memory type.")
        });
        let realloc = options.realloc.map(|idx| {
            Self::core_export(inner, &ctx, &inner.component.extracted_reallocs[&idx])
                .expect("Could not get runtime realloc export.")
                .into_func()
                .expect("Export was not of func type.")
        });
        let post_return = options.post_return.map(|idx| {
            Self::core_export(inner, &ctx, &inner.component.extracted_post_returns[&idx])
                .expect("Could not get runtime post return export.")
                .into_func()
                .expect("Export was not of func type.")
        });

        Ok(crate::func::Func {
            store_id: ctx.as_context().inner.data().id,
            ty: crate::types::FuncType::from_component(func, &inner.component, Some(mapping))?,
            backing: FuncImpl::GuestFunc(Arc::new(GuestFunc {
                callee,
                component: inner.component.clone(),
                encoding: options.string_encoding,
                function: func.clone(),
                memory,
                realloc,
                resource_tables: inner.resource_tables.clone(),
                post_return,
                types: inner.types.clone(),
                instance_id: inner.id,
            })),
        })
    }

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
                    .generated_trampolines
                    .get(x)
                    .context("Could not find exported trampoline.")?
                {
                    GeneratedTrampoline::ImportedFunction(component_import) => {
                        let expected = crate::types::FuncType::from_component(
                            &component_import.func,
                            &inner.component,
                            Some(resource_map),
                        )?;
                        let func = Self::get_component_import(inner, component_import, linker)?;
                        ensure!(
                            func.ty() == expected,
                            "Function import {} had type {:?}, but expected {expected:?}",
                            component_import.name,
                            func.ty()
                        );
                        let guest_options = Self::import_function(
                            inner,
                            &ctx,
                            &component_import.options,
                            &component_import.func,
                        );

                        Ok(Extern::Func(Func::new(
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
                        let tables = inner.resource_tables.clone();
                        Ok(Extern::Func(Func::new(
                            ctx.as_context_mut().inner,
                            ty,
                            move |ctx, args, results| {
                                let rep =
                                    require_matches!(args[0], wasm_runtime_layer::Value::I32(x), x);
                                let mut table_array = tables
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
                        let tables = inner.resource_tables.clone();
                        Ok(Extern::Func(Func::new(
                            ctx.as_context_mut().inner,
                            ty,
                            move |ctx, args, results| {
                                let idx =
                                    require_matches!(args[0], wasm_runtime_layer::Value::I32(x), x);
                                let mut table_array = tables
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
                        let tables = inner.resource_tables.clone();
                        Ok(Extern::Func(Func::new(
                            ctx.as_context_mut().inner,
                            ty,
                            move |mut ctx, args, results| {
                                let idx =
                                    require_matches!(args[0], wasm_runtime_layer::Value::I32(x), x);
                                let mut table_array = tables
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
                }
            }
            CoreDef::InstanceFlags(i) => Ok(Extern::Global(inner.instance_flags[*i].clone())),
        }
    }

    fn core_export<T: Copy + Into<wasmtime_environ::EntityIndex>>(
        inner: &InstanceInner,
        ctx: impl AsContext,
        export: &CoreExport<T>,
    ) -> Option<Extern> {
        let name = match &export.item {
            ExportItem::Index(idx) => {
                &inner.component.export_mapping[&inner.component.instance_modules[export.instance]]
                    [&(*idx).into()]
            }
            ExportItem::Name(s) => s,
        };

        inner.instances[export.instance].get_export(&ctx.as_context().inner, &name)
    }

    fn global_initialize(
        mut inner: InstanceInner,
        mut ctx: impl AsContextMut,
        linker: &Linker,
        resource_map: &FxHashMap<ResourceType, ResourceType>,
    ) -> Result<InstanceInner> {
        let mut destructors = Vec::new();
        for initializer in &inner.component.translation.component.initializers {
            match initializer {
                GlobalInitializer::InstantiateModule(InstantiateModule::Static(idx, def)) => {
                    let module = &inner.component.modules[idx];
                    let imports = Self::generate_imports(
                        &inner,
                        &mut ctx,
                        linker,
                        module,
                        &def,
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

    fn get_component_import(
        inner: &InstanceInner,
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

    fn fill_destructors(
        mut inner: InstanceInner,
        ctx: impl AsContext,
        destructors: Vec<TrampolineIndex>,
        resource_map: &FxHashMap<ResourceType, ResourceType>,
    ) -> Result<InstanceInner> {
        let mut tables = inner
            .resource_tables
            .try_lock()
            .expect("Could not get access to resource tables.");

        for index in destructors {
            let (x, def) = require_matches!(
                &inner.component.generated_trampolines[&index],
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

        for (id, idx) in inner.component.resolve.types.iter().filter_map(|(i, _)| {
            let val = inner.component.resource_map[i.index()];
            (val.as_u32() < u32::MAX - 1).then_some((i, val))
        }) {
            let res = ResourceType::from_resolve(id, &inner.component, Some(resource_map))?;
            if let Some(Some(func)) = res.host_destructor() {
                tables[idx.as_u32() as usize].set_destructor(Some(func));
            }
        }

        drop(tables);

        Ok(inner)
    }
}

#[derive(Debug)]
pub struct Exports {
    root: ExportInstance,
    instances: FxHashMap<InterfaceIdentifier, ExportInstance>,
    package: PackageIdentifier,
}

impl Exports {
    pub(crate) fn new(package: PackageIdentifier) -> Self {
        Self {
            root: ExportInstance::new(),
            instances: FxHashMap::default(),
            package,
        }
    }

    pub fn root(&self) -> &ExportInstance {
        &self.root
    }

    pub fn package(&self) -> &PackageIdentifier {
        &self.package
    }

    pub fn instance(&self, name: &InterfaceIdentifier) -> Option<&ExportInstance> {
        self.instances.get(name)
    }

    pub fn instances<'a>(
        &'a self,
    ) -> impl Iterator<Item = (&'a InterfaceIdentifier, &'a ExportInstance)> {
        self.instances.iter()
    }
}

#[derive(Debug)]
pub struct ExportInstance {
    functions: FxHashMap<Arc<str>, crate::func::Func>,
    resources: FxHashMap<Arc<str>, ResourceType>,
}

impl ExportInstance {
    pub(crate) fn new() -> Self {
        Self {
            functions: FxHashMap::default(),
            resources: FxHashMap::default(),
        }
    }

    pub fn func(&self, name: impl AsRef<str>) -> Option<crate::func::Func> {
        self.functions.get(name.as_ref()).cloned()
    }

    pub fn funcs<'a>(&'a self) -> impl Iterator<Item = (&'a str, crate::func::Func)> {
        self.functions.iter().map(|(k, v)| (&**k, v.clone()))
    }

    pub fn resource(&self, name: impl AsRef<str>) -> Option<ResourceType> {
        self.resources.get(name.as_ref()).cloned()
    }

    pub fn resources<'a>(&'a self) -> impl Iterator<Item = (&'a str, ResourceType)> {
        self.resources.iter().map(|(k, v)| (&**k, v.clone()))
    }
}

#[derive(Debug)]
struct InstanceInner {
    pub component: Arc<ComponentInner>,
    pub exports: Exports,
    pub id: u64,
    pub instance_flags: wasmtime_environ::PrimaryMap<RuntimeComponentInstanceIndex, Global>,
    pub instances: wasmtime_environ::PrimaryMap<RuntimeInstanceIndex, wasm_runtime_layer::Instance>,
    pub resource_tables: Arc<Mutex<Vec<HandleTable>>>,
    pub types: Arc<[crate::types::ValueType]>,
}

#[derive(Clone, Debug)]
struct ComponentImport {
    pub instance: Option<InterfaceIdentifier>,
    pub name: Arc<str>,
    pub func: Function,
    pub options: CanonicalOptions,
}

#[derive(Clone, Debug)]
struct ComponentExport {
    pub options: CanonicalOptions,
    pub func: Function,
    pub def: CoreExport<wasmtime_environ::EntityIndex>,
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
        static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

        Self {
            inner: wasm_runtime_layer::Store::new(
                &engine,
                StoreInner {
                    id: ID_COUNTER.fetch_add(1, Ordering::AcqRel),
                    data,
                    host_functions: FuncVec::default(),
                },
            ),
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
    pub host_functions: FuncVec<T, E>,
}

#[derive(Clone, Debug)]
enum GeneratedTrampoline {
    ImportedFunction(ComponentImport),
    ResourceNew(TypeResourceTableIndex),
    ResourceRep(TypeResourceTableIndex),
    ResourceDrop(TypeResourceTableIndex, Option<CoreDef>),
}

#[derive(Copy, Clone, Debug, Default)]
struct HandleElement {
    pub rep: i32,
    pub own: bool,
    pub lend_count: i32,
}

#[derive(Clone, Debug, Default)]
struct HandleTable {
    array: VecOption<HandleElement>,
    destructor: Option<wasm_runtime_layer::Func>,
    free: Vec<i32>,
}

impl HandleTable {
    pub fn destructor(&self) -> Option<&wasm_runtime_layer::Func> {
        self.destructor.as_ref()
    }

    pub fn set_destructor(&mut self, destructor: Option<wasm_runtime_layer::Func>) {
        self.destructor = destructor;
    }

    pub fn get(&self, i: i32) -> Result<&HandleElement> {
        self.array
            .get(i as usize)
            .and_then(std::convert::identity)
            .context("Invalid handle index.")
    }

    pub fn set(&mut self, i: i32, element: HandleElement) {
        *self
            .array
            .get_mut(i as usize)
            .expect("Invalid handle index.") = Some(element);
    }

    pub fn add(&mut self, handle: HandleElement) -> i32 {
        if let Some(i) = self.free.pop() {
            *self
                .array
                .get_mut(i as usize)
                .expect("Could not get free index from list.") = Some(handle);
            i
        } else {
            let i = self.array.len();
            self.array.push(handle);
            i as i32
        }
    }

    pub fn remove(&mut self, i: i32) -> Result<HandleElement> {
        let res = self
            .array
            .get_mut(i as usize)
            .context("Invalid handle index.")?
            .take()
            .context("Invalid handle index.")?;
        self.free.push(i);
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imports() {
        const WASM_0: &[u8] = include_bytes!("test_guest_component.wasm");
        const WASM: &[u8] = include_bytes!("test_guest_component2.wasm");

        let engine = Engine::new(wasmi::Engine::default());
        let mut store = Store::new(&engine, ());
        let comp_0 = Component::new(&engine, WASM_0).unwrap();
        let comp = Component::new(&engine, WASM).unwrap();

        let resource = ResourceType::new(&mut store, None).unwrap();

        let f_func = TypedFunc::new(&mut store, |ctx, ()| Ok(("aa".to_owned(),)));
        println!("Calling got {}", f_func.call(&mut store, (())).unwrap().0);

        let mut linker = Linker::default();

        let inst_0 = linker.instantiate(&mut store, &comp_0).unwrap();

        let real_inst = inst_0
            .exports()
            .instance(&"test:guest/tester".try_into().unwrap())
            .unwrap();

        let link_int = linker
            .define_instance("test:guest/tester".try_into().unwrap())
            .unwrap();
        //link_int.define_resource("exp", real_inst.resource("exp").unwrap()).unwrap();

        //link_int.define_func("[constructor]exp", real_inst.func("[constructor]exp").unwrap()).unwrap();
        //link_int.define_func("[method]exp.test", real_inst.func("[method]exp.test").unwrap()).unwrap();

        let defun = crate::func::Func::new(
            &mut store,
            crate::types::FuncType::new([crate::types::ValueType::S32], []),
            |_, _, _| {
                println!("im ded an gone");
                Ok(())
            },
        );
        let my_resource = ResourceType::new(&mut store, Some(defun)).unwrap();
        let cpy = my_resource.clone();
        link_int.define_resource("exp", my_resource.clone());

        link_int
            .define_func(
                "[constructor]exp",
                crate::func::Func::new(
                    &mut store,
                    crate::types::FuncType::new(
                        [],
                        [crate::types::ValueType::Own(my_resource.clone())],
                    ),
                    move |_, _, res| {
                        res[0] = crate::values::Value::Own(
                            crate::values::ResourceOwn::new(29, cpy.clone()).unwrap(),
                        );
                        Ok(())
                    },
                ),
            )
            .unwrap();
        link_int
            .define_func(
                "[method]exp.test",
                crate::func::Func::new(
                    &mut store,
                    crate::types::FuncType::new(
                        [crate::types::ValueType::Borrow(my_resource.clone())],
                        [crate::types::ValueType::String],
                    ),
                    |_, _, res| {
                        res[0] = crate::values::Value::String("yay".into());
                        Ok(())
                    },
                ),
            )
            .unwrap();

        let inst_1 = linker.instantiate(&mut store, &comp).unwrap();

        let func = inst_1
            .exports()
            .instance(&"test-two:guest/bester".try_into().unwrap())
            .unwrap()
            .func("the-string")
            .unwrap();
        let mut res = [crate::values::Value::Bool(false)];
        func.call(&mut store, &[], &mut res).unwrap();
        println!("OMG WE GOT {res:?}");
        //println!("COmp 0 had version {:?} and comp wanted {:?}", comp_0.exports().instances().map(|(a, _)| a).collect::<Vec<_>>(), comp.imports().instances().map(|(a, _)| a).collect::<Vec<_>>());
        /*let func_ty = comp.imports().instance(&"test:guest/tester@0.1.1".try_into().unwrap()).unwrap().func("get-a-string").unwrap();

        linker.define_instance("test:guest/tester@0.1.1".try_into().unwrap()).unwrap().define_func("get-a-string", inst_0.exports().instance(&"test:guest/tester@0.1.1".try_into().unwrap()).unwrap().func("get-a-string").unwrap()).unwrap();

        let inst = linker.instantiate(&mut store, &comp).unwrap();
        let double = inst.exports().root().func("doubled-string").unwrap();

        let mut res = [crate::values::Value::Bool(false)];
        double.call(&mut store, &[], &mut res).unwrap();
        println!("AND HIS NAMES {res:?}");*/
    }
}
