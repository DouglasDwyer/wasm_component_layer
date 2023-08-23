use std::any::*;
use std::hash::*;
use std::sync::atomic::*;
use std::sync::*;

use anyhow::*;
use fxhash::*;
use id_arena::*;

use crate::require_matches;
use crate::values::ComponentType;
use crate::{AsContextMut, ComponentInner, StoreContextMut};

/// Represents a component model interface type
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ValueType {
    Bool,
    S8,
    U8,
    S16,
    U16,
    S32,
    U32,
    S64,
    U64,
    F32,
    F64,
    Char,
    String,
    List(ListType),
    Record(RecordType),
    Tuple(TupleType),
    Variant(VariantType),
    Enum(EnumType),
    Union(UnionType),
    Option(OptionType),
    Result(ResultType),
    Flags(FlagsType),
    Own(ResourceType),
    Borrow(ResourceType),
}

impl ValueType {
    pub(crate) fn from_component(
        ty: &wit_parser::Type,
        component: &ComponentInner,
        resource_map: Option<&FxHashMap<ResourceType, ResourceType>>,
    ) -> Result<Self> {
        Ok(match ty {
            wit_parser::Type::Bool => Self::Bool,
            wit_parser::Type::U8 => Self::U8,
            wit_parser::Type::U16 => Self::U16,
            wit_parser::Type::U32 => Self::U32,
            wit_parser::Type::U64 => Self::U64,
            wit_parser::Type::S8 => Self::S8,
            wit_parser::Type::S16 => Self::S16,
            wit_parser::Type::S32 => Self::S32,
            wit_parser::Type::S64 => Self::S64,
            wit_parser::Type::Float32 => Self::F32,
            wit_parser::Type::Float64 => Self::F64,
            wit_parser::Type::Char => Self::Char,
            wit_parser::Type::String => Self::String,
            wit_parser::Type::Id(x) => Self::from_component_typedef(*x, component, resource_map)?,
        })
    }

    pub(crate) fn from_component_typedef(
        def: Id<wit_parser::TypeDef>,
        component: &ComponentInner,
        resource_map: Option<&FxHashMap<ResourceType, ResourceType>>,
    ) -> Result<Self> {
        Ok(match &component.resolve.types[def].kind {
            wit_parser::TypeDefKind::Record(x) => {
                Self::Record(RecordType::from_component(x, component, resource_map)?)
            }
            wit_parser::TypeDefKind::Resource => bail!("Cannot instantiate resource as type."),
            wit_parser::TypeDefKind::Handle(x) => match x {
                wit_parser::Handle::Own(t) => {
                    Self::Own(ResourceType::from_resolve(*t, component, resource_map)?)
                }
                wit_parser::Handle::Borrow(t) => {
                    Self::Borrow(ResourceType::from_resolve(*t, component, resource_map)?)
                }
            },
            wit_parser::TypeDefKind::Flags(x) => {
                Self::Flags(FlagsType::from_component(x, component)?)
            }
            wit_parser::TypeDefKind::Tuple(x) => {
                Self::Tuple(TupleType::from_component(x, component, resource_map)?)
            }
            wit_parser::TypeDefKind::Variant(x) => {
                Self::Variant(VariantType::from_component(x, component, resource_map)?)
            }
            wit_parser::TypeDefKind::Enum(x) => Self::Enum(EnumType::from_component(x, component)),
            wit_parser::TypeDefKind::Option(x) => Self::Option(OptionType::new(
                Self::from_component(x, component, resource_map)?,
            )),
            wit_parser::TypeDefKind::Result(x) => Self::Result(ResultType::new(
                match &x.ok {
                    Some(t) => Some(Self::from_component(t, component, resource_map)?),
                    None => None,
                },
                match &x.err {
                    Some(t) => Some(Self::from_component(t, component, resource_map)?),
                    None => None,
                },
            )),
            wit_parser::TypeDefKind::Union(x) => {
                Self::Union(UnionType::from_component(x, component, resource_map)?)
            }
            wit_parser::TypeDefKind::List(x) => Self::List(ListType::new(Self::from_component(
                x,
                component,
                resource_map,
            )?)),
            wit_parser::TypeDefKind::Future(_) => bail!("Unimplemented."),
            wit_parser::TypeDefKind::Stream(_) => bail!("Unimplemented."),
            wit_parser::TypeDefKind::Type(x) => Self::from_component(x, component, resource_map)?,
            wit_parser::TypeDefKind::Unknown => unreachable!(),
        })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ListType {
    element: Arc<ValueType>,
}

impl ListType {
    pub fn new(element_ty: ValueType) -> Self {
        Self {
            element: Arc::new(element_ty),
        }
    }

    pub fn element_ty(&self) -> ValueType {
        (*self.element).clone()
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RecordType {
    pub(crate) fields: Arc<[(usize, Arc<str>, ValueType)]>,
}

impl RecordType {
    pub fn new<S: Into<Arc<str>>>(
        fields: impl IntoIterator<Item = (S, ValueType)>,
    ) -> Result<Self> {
        let mut to_sort = fields
            .into_iter()
            .enumerate()
            .map(|(i, (name, val))| (i, Into::<Arc<str>>::into(name), val))
            .collect::<Arc<_>>();
        Arc::get_mut(&mut to_sort)
            .expect("Could not get exclusive reference.")
            .sort_by(|a, b| a.1.cmp(&b.1));

        for pair in to_sort.windows(2) {
            ensure!(pair[0].1 != pair[1].1, "Duplicate field names");
        }

        Ok(Self { fields: to_sort })
    }

    pub fn fields(&self) -> impl ExactSizeIterator<Item = (&str, ValueType)> {
        self.fields
            .iter()
            .map(|(_, name, val)| (&**name, val.clone()))
    }

    pub(crate) fn new_sorted(
        fields: impl IntoIterator<Item = (Arc<str>, ValueType)>,
    ) -> Result<Self> {
        let result = Self {
            fields: fields
                .into_iter()
                .enumerate()
                .map(|(i, (a, b))| (i, a, b))
                .collect(),
        };

        for pair in result.fields.windows(2) {
            ensure!(pair[0].0 != pair[1].0, "Duplicate field names");
        }

        Ok(result)
    }

    fn from_component(
        ty: &wit_parser::Record,
        component: &ComponentInner,
        resource_map: Option<&FxHashMap<ResourceType, ResourceType>>,
    ) -> Result<Self> {
        let mut to_sort = ty
            .fields
            .iter()
            .enumerate()
            .map(|(i, x)| {
                Ok((
                    i,
                    Into::<Arc<str>>::into(x.name.as_str()),
                    ValueType::from_component(&x.ty, component, resource_map)?,
                ))
            })
            .collect::<Result<Arc<_>>>()?;
        Arc::get_mut(&mut to_sort)
            .expect("Could not get exclusive reference.")
            .sort_by(|a, b| a.1.cmp(&b.1));

        for pair in to_sort.windows(2) {
            ensure!(pair[0].0 != pair[1].0, "Duplicate field names");
        }

        Ok(Self { fields: to_sort })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct TupleType {
    fields: Arc<[ValueType]>,
}

impl TupleType {
    pub fn new(fields: impl IntoIterator<Item = ValueType>) -> Self {
        Self {
            fields: fields.into_iter().collect(),
        }
    }

    pub fn fields(&self) -> &[ValueType] {
        &self.fields
    }

    fn from_component(
        ty: &wit_parser::Tuple,
        component: &ComponentInner,
        resource_map: Option<&FxHashMap<ResourceType, ResourceType>>,
    ) -> Result<Self> {
        let fields = ty
            .types
            .iter()
            .map(|x| ValueType::from_component(x, component, resource_map))
            .collect::<Result<_>>()?;
        Ok(Self { fields })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct VariantCase {
    name: Arc<str>,
    ty: Option<ValueType>,
}

impl VariantCase {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn ty(&self) -> Option<ValueType> {
        self.ty.clone()
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct VariantType {
    cases: Arc<[VariantCase]>,
}

impl VariantType {
    pub fn new(cases: impl IntoIterator<Item = VariantCase>) -> Result<Self> {
        let cases: Arc<_> = cases.into_iter().collect();

        for i in 0..cases.len() {
            for j in 0..i {
                ensure!(cases[i].name() != cases[j].name(), "Duplicate case names.");
            }
        }

        Ok(Self { cases })
    }

    pub fn cases(&self) -> &[VariantCase] {
        &self.cases
    }

    fn from_component(
        ty: &wit_parser::Variant,
        component: &ComponentInner,
        resource_map: Option<&FxHashMap<ResourceType, ResourceType>>,
    ) -> Result<Self> {
        let cases: Arc<_> = ty
            .cases
            .iter()
            .map(|x| {
                Ok(VariantCase {
                    name: x.name.as_str().into(),
                    ty: match &x.ty {
                        Some(t) => Some(ValueType::from_component(t, component, resource_map)?),
                        None => None,
                    },
                })
            })
            .collect::<Result<_>>()?;

        for i in 0..cases.len() {
            for j in 0..i {
                ensure!(cases[i].name() != cases[j].name(), "Duplicate case names.");
            }
        }

        Ok(Self { cases })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct EnumType {
    cases: Arc<[Arc<str>]>,
}

impl EnumType {
    pub fn new<S: Into<Arc<str>>>(cases: impl IntoIterator<Item = S>) -> Self {
        Self {
            cases: cases
                .into_iter()
                .map(|x| Into::<Arc<str>>::into(x))
                .collect(),
        }
    }

    pub fn cases(&self) -> impl ExactSizeIterator<Item = &str> {
        self.cases.iter().map(|x| &**x)
    }

    fn from_component(ty: &wit_parser::Enum, _component: &ComponentInner) -> Self {
        Self::new(ty.cases.iter().map(|x| x.name.as_str()))
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnionType {
    cases: Arc<[ValueType]>,
}

impl UnionType {
    pub fn new(cases: impl IntoIterator<Item = ValueType>) -> Self {
        Self {
            cases: cases.into_iter().collect(),
        }
    }

    pub fn cases(&self) -> &[ValueType] {
        &self.cases
    }

    fn from_component(
        ty: &wit_parser::Union,
        component: &ComponentInner,
        resource_map: Option<&FxHashMap<ResourceType, ResourceType>>,
    ) -> Result<Self> {
        let cases = ty
            .cases
            .iter()
            .map(|x| ValueType::from_component(&x.ty, component, resource_map))
            .collect::<Result<_>>()?;

        Ok(Self { cases })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct OptionType {
    ty: Arc<ValueType>,
}

impl OptionType {
    pub fn new(ty: ValueType) -> Self {
        Self { ty: Arc::new(ty) }
    }

    pub fn some_ty(&self) -> ValueType {
        (*self.ty).clone()
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ResultType {
    ok_err: Arc<(Option<ValueType>, Option<ValueType>)>,
}

impl ResultType {
    pub fn new(ok: Option<ValueType>, err: Option<ValueType>) -> Self {
        Self {
            ok_err: Arc::new((ok, err)),
        }
    }

    pub fn ok_ty(&self) -> Option<ValueType> {
        self.ok_err.0.clone()
    }

    pub fn err_ty(&self) -> Option<ValueType> {
        self.ok_err.1.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlagsType {
    names: Arc<[Arc<str>]>,
    pub(crate) indices: Arc<FxHashMap<Arc<str>, usize>>,
}

impl FlagsType {
    pub fn new<S: Into<Arc<str>>>(names: impl IntoIterator<Item = S>) -> Result<Self> {
        let names: Arc<_> = names
            .into_iter()
            .map(|x| Into::<Arc<str>>::into(x))
            .collect();

        for i in 0..names.len() {
            for j in 0..i {
                ensure!(names[i] != names[j], "Duplicate case names.");
            }
        }

        let indices = Arc::new(
            names
                .iter()
                .enumerate()
                .map(|(i, val)| (val.clone(), i))
                .collect(),
        );

        Ok(Self { names, indices })
    }

    pub fn names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.names.iter().map(|x| &**x)
    }

    fn from_component(ty: &wit_parser::Flags, _component: &ComponentInner) -> Result<Self> {
        Self::new(ty.flags.iter().map(|x| x.name.as_ref()))
    }
}

impl Hash for FlagsType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.names.hash(state)
    }
}

static RESOURCE_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ResourceType {
    kind: ResourceKindValue,
}

impl ResourceType {
    pub fn new<T: 'static + Send + Sync + Sized>() -> Self {
        Self {
            kind: ResourceKindValue::Host {
                resource_id: RESOURCE_ID_COUNTER.fetch_add(1, Ordering::AcqRel),
                type_id: TypeId::of::<T>(),
                associated_store: None
            },
        }
    }

    pub fn with_destructor<T: 'static + Send + Sync + Sized, C: AsContextMut>(mut ctx: C, destructor: impl 'static + Send + Sync + Fn(StoreContextMut<'_, C::UserState, C::Engine>, T) -> Result<()>) -> Result<Self> {
        let store_id = ctx.as_context().inner.data().id;
        let destructor = wasm_runtime_layer::Func::new(
            ctx.as_context_mut().inner,
            wasm_runtime_layer::FuncType::new([wasm_runtime_layer::ValueType::I32], []),
            move |mut ctx, val, _res| {
                let resource = wasm_runtime_layer::AsContextMut::as_context_mut(&mut ctx).data_mut().host_resources.remove(require_matches!(val[0], wasm_runtime_layer::Value::I32(x), x) as usize);
                destructor(StoreContextMut { inner: ctx }, *resource.downcast().expect("Resource was of incorrect type."))
            },
        );

        Ok(Self {
            kind: ResourceKindValue::Host {
                resource_id: RESOURCE_ID_COUNTER.fetch_add(1, Ordering::AcqRel),
                type_id: TypeId::of::<T>(),
                associated_store: Some((store_id, destructor))
            },
        })
    }

    pub(crate) fn valid_for<T: 'static + Send + Sync>(&self, store_id: u64) -> bool {
        if let ResourceKindValue::Host { type_id, associated_store, .. } = &self.kind {
            *type_id == TypeId::of::<T>() && associated_store.as_ref().map(|(id, _)| *id == store_id).unwrap_or(true)
        }
        else {
            false
        }
    }

    pub(crate) fn is_owned_by_instance(&self, instance: u64) -> bool {
        if let ResourceKindValue::Instantiated { instance: a, .. } = &self.kind {
            instance == *a
        } else {
            false
        }
    }

    pub(crate) fn from_resolve(
        id: Id<wit_parser::TypeDef>,
        component: &ComponentInner,
        resource_map: Option<&FxHashMap<ResourceType, ResourceType>>,
    ) -> Result<Self> {
        let ab = Self {
            kind: ResourceKindValue::Abstract {
                id: id.index(),
                component: component.id,
            },
        };
        if let Some(map) = resource_map {
            Ok(map[&ab].clone())
        } else {
            Ok(ab)
        }
    }

    pub(crate) fn host_destructor(&self) -> Option<Option<wasm_runtime_layer::Func>> {
        if let ResourceKindValue::Host { associated_store, .. } = &self.kind {
            Some(associated_store.as_ref().map(|(_, x)| x.clone()))
        } else {
            None
        }
    }

    pub(crate) fn instantiate(&self, instance: u64) -> Result<Self> {
        if let ResourceKindValue::Abstract { id, component: _ } = &self.kind {
            Ok(Self {
                kind: ResourceKindValue::Instantiated {
                    id: *id,
                    instance
                },
            })
        } else {
            bail!("Resource was not abstract.");
        }
    }

    pub(crate) fn is_instantiated(&self) -> bool {
        match &self.kind {
            ResourceKindValue::Abstract { .. } => false,
            _ => true,
        }
    }
}

#[derive(Clone, Debug)]
enum ResourceKindValue {
    Abstract {
        id: usize,
        component: u64,
    },
    Instantiated {
        id: usize,
        instance: u64,
    },
    Host {
        resource_id: u64,
        type_id: TypeId,
        associated_store: Option<(u64, wasm_runtime_layer::Func)>,
    },
}

impl PartialEq for ResourceKindValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                ResourceKindValue::Abstract {
                    id: a,
                    component: b,
                },
                ResourceKindValue::Abstract {
                    id: x,
                    component: y,
                },
            ) => a == x && b == y,
            (
                ResourceKindValue::Instantiated {
                    id: a, instance: b
                },
                ResourceKindValue::Instantiated {
                    id: x, instance: y
                },
            ) => a == x && b == y,
            (
                ResourceKindValue::Host { resource_id: a, .. },
                ResourceKindValue::Host { resource_id: x, .. },
            ) => a == x,
            _ => false,
        }
    }
}

impl Eq for ResourceKindValue {}

impl Hash for ResourceKindValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            ResourceKindValue::Abstract { id, component } => {
                id.hash(state);
                component.hash(state);
            }
            ResourceKindValue::Instantiated {
                id,
                instance
            } => {
                id.hash(state);
                instance.hash(state);
            }
            ResourceKindValue::Host {
                resource_id,
                ..
            } => resource_id.hash(state),
        }
    }
}

/// A function type representing a function's parameter and result types.
///
/// # Note
///
/// Can be cloned cheaply.
#[derive(Clone, PartialEq, Eq)]
pub struct FuncType {
    /// The number of function parameters.
    len_params: usize,
    /// The ordered and merged parameter and result types of the function type.
    ///
    /// # Note
    ///
    /// The parameters and results are ordered and merged in a single
    /// vector starting with parameters in their order and followed
    /// by results in their order.
    /// The `len_params` field denotes how many parameters there are in
    /// the head of the vector before the results.
    params_results: Arc<[ValueType]>,
}

impl std::fmt::Debug for FuncType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("FuncType")
            .field("params", &self.params())
            .field("results", &self.results())
            .finish()
    }
}

impl FuncType {
    pub(crate) fn from_component(
        func: &wit_parser::Function,
        component: &ComponentInner,
        resource_map: Option<&FxHashMap<ResourceType, ResourceType>>,
    ) -> Result<Self> {
        let mut params_results = func
            .params
            .iter()
            .map(|(_, ty)| ValueType::from_component(ty, component, resource_map))
            .collect::<Result<Vec<_>>>()?;
        let len_params = params_results.len();

        for result in func
            .results
            .iter_types()
            .map(|ty| ValueType::from_component(ty, component, resource_map))
        {
            params_results.push(result?);
        }

        Ok(Self {
            params_results: params_results.into(),
            len_params,
        })
    }

    /// Creates a new [`FuncType`].
    pub fn new<P, R>(params: P, results: R) -> Self
    where
        P: IntoIterator<Item = ValueType>,
        R: IntoIterator<Item = ValueType>,
    {
        let mut params_results = params.into_iter().collect::<Vec<_>>();
        let len_params = params_results.len();
        params_results.extend(results);
        Self {
            params_results: params_results.into(),
            len_params,
        }
    }

    /// Returns the parameter types of the function type.
    pub fn params(&self) -> &[ValueType] {
        &self.params_results[..self.len_params]
    }

    /// Returns the result types of the function type.
    pub fn results(&self) -> &[ValueType] {
        &self.params_results[self.len_params..]
    }

    /// Returns `Ok` if the number and types of items in `params` matches as expected by the [`FuncType`].
    pub(crate) fn match_params(&self, params: &[crate::values::Value]) -> Result<()> {
        if self.params().len() != params.len() {
            bail!("Incorrect parameter length.");
        }
        if self
            .params()
            .iter()
            .cloned()
            .ne(params.iter().map(crate::values::Value::ty))
        {
            bail!("Incorrect parameter types.");
        }
        Ok(())
    }

    /// Returns `Ok` if the number and types of items in `results` matches as expected by the [`FuncType`].
    pub(crate) fn match_results(&self, results: &[crate::values::Value]) -> Result<()> {
        if self.results().len() != results.len() {
            bail!("Incorrect result length.");
        }
        if self
            .results()
            .iter()
            .cloned()
            .ne(results.iter().map(crate::values::Value::ty))
        {
            bail!("Incorrect result types.");
        }
        Ok(())
    }
}

pub trait ComponentList: Sized {
    const LEN: usize;

    fn into_tys(types: &mut [ValueType]);
    fn into_values(self, values: &mut [crate::values::Value]) -> Result<()>;
    fn from_values(values: &[crate::values::Value]) -> Result<Self>;
}

impl ComponentList for () {
    const LEN: usize = 0;

    fn into_tys(_types: &mut [ValueType]) {}

    fn from_values(_values: &[crate::values::Value]) -> Result<Self> {
        Ok(())
    }

    fn into_values(self, _values: &mut [crate::values::Value]) -> Result<()> {
        Ok(())
    }
}

const fn one<T>() -> usize { 1 }

macro_rules! impl_component_list {
    ( $( ($name:ident, $extra:ident) )+ ) => {
        impl<$($name: ComponentType),+> ComponentList for ($($name,)+)
        {
            const LEN: usize = { $(one::<$name>() + )+ 0 };

            #[allow(warnings)]
            fn into_tys(types: &mut [ValueType]) {
                let mut counter = 0;
                $(types[{ let res = counter; counter += 1; res }] = $name::ty();)+
            }

            #[allow(warnings)]
            fn into_values(self, values: &mut [crate::values::Value]) -> Result<()> {
                let ($($extra,)+) = self;
                let mut counter = 0;
                $(values[{ let res = counter; counter += 1; res }] = $extra.into_value()?;)+
                Ok(())
            }

            #[allow(warnings)]
            fn from_values(values: &[crate::values::Value]) -> Result<Self> {
                let mut counter = 0;
                Ok(($($name::from_value(&values[{ let res = counter; counter += 1; res }])?, )+))
            }
        }
    };
}

impl_component_list!((A, a));
impl_component_list!((A, a)(B, b));
impl_component_list!((A, a)(B, b)(C, c));
impl_component_list!((A, a)(B, b)(C, c)(D, d));
impl_component_list!((A, a)(B, b)(C, c)(D, d)(E, e));
impl_component_list!((A, a)(B, b)(C, c)(D, d)(E, e)(F, f));
impl_component_list!((A, a)(B, b)(C, c)(D, d)(E, e)(F, f)(G, g));
impl_component_list!((A, a)(B, b)(C, c)(D, d)(E, e)(F, f)(G, g)(H, h));
impl_component_list!((A, a)(B, b)(C, c)(D, d)(E, e)(F, f)(G, g)(H, h)(I, i));
impl_component_list!((A, a)(B, b)(C, c)(D, d)(E, e)(F, f)(G, g)(H, h)(I, i)(J, j));
impl_component_list!((A, a)(B, b)(C, c)(D, d)(E, e)(F, f)(G, g)(H, h)(I, i)(J, j)(K, k));
impl_component_list!((A, a)(B, b)(C, c)(D, d)(E, e)(F, f)(G, g)(H, h)(I, i)(J, j)(K, k)(L, l));
