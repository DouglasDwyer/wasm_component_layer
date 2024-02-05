use std::any::*;
use std::fmt::Display;
use std::hash::*;
use std::sync::atomic::*;
use std::sync::*;

use anyhow::*;
use fxhash::*;
use id_arena::*;
#[cfg(feature = "serde")]
use serde::*;

use crate::values::ComponentType;
use crate::TypeIdentifier;
use crate::{require_matches, UnaryComponentType};
use crate::{AsContextMut, ComponentInner, StoreContextMut};

/// Represents a component model interface type.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ValueType {
    /// The boolean type.
    Bool,
    /// The eight-bit signed integer type.
    S8,
    /// The eight-bit unsigned integer type.
    U8,
    ///The 16-bit signed integer type.
    S16,
    /// The 16-bit unsigned integer type.
    U16,
    /// The 32-bit signed integer type.
    S32,
    /// The 32-bit unsigned integer type.
    U32,
    /// The 64-bit signed integer type.
    S64,
    /// The 64-bit unsigned integer type.
    U64,
    /// The 32-bit floating point number type.
    F32,
    /// The 64-bit floating point number type.
    F64,
    /// The UTF-8 character type.
    Char,
    /// The string type.
    String,
    /// The homogenous list of elements type.
    List(ListType),
    /// The record with heterogenous fields type.
    Record(RecordType),
    /// The tuple with heterogenous fields type.
    Tuple(TupleType),
    /// The variant which may be one of multiple types or cases type.
    Variant(VariantType),
    /// The enum which may be one of multiple cases type.
    Enum(EnumType),
    /// The type which may or may not have an underlying value type.
    Option(OptionType),
    /// The type that indicates success or failure type.
    Result(ResultType),
    /// The set of boolean values type.
    Flags(FlagsType),
    /// The owned resource handle type.
    Own(ResourceType),
    /// The borrowed resource handle type.
    Borrow(ResourceType),
}

impl Display for ValueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueType::Bool => write!(f, "bool"),
            ValueType::S8 => write!(f, "s8"),
            ValueType::U8 => write!(f, "u8"),
            ValueType::S16 => write!(f, "s16"),
            ValueType::U16 => write!(f, "u16"),
            ValueType::S32 => write!(f, "s32"),
            ValueType::U32 => write!(f, "u32"),
            ValueType::S64 => write!(f, "s64"),
            ValueType::U64 => write!(f, "u64"),
            ValueType::F32 => write!(f, "f32"),
            ValueType::F64 => write!(f, "f64"),
            ValueType::Char => write!(f, "char"),
            ValueType::String => write!(f, "string"),
            ValueType::List(x) => write!(f, "list<{}>", x.element_ty()),
            // record<field1: type1, field2: type2, ...>
            ValueType::Record(x) => {
                write!(f, "record<")?;
                let mut first = true;
                for (name, ty) in x.fields() {
                    if !first {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", name, ty)?;
                    first = false;
                }
                write!(f, ">")
            }
            // tuple<type1, type2, ...>
            ValueType::Tuple(x) => {
                write!(f, "tuple<")?;
                let mut first = true;
                for ty in x.fields() {
                    if !first {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                    first = false;
                }
                write!(f, ">")
            }
            // variant<case1: type1?, case2: type2?, ...>
            ValueType::Variant(x) => {
                write!(f, "variant<")?;
                let mut first = true;
                for case in x.cases() {
                    if !first {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", case.name())?;
                    if let Some(ty) = case.ty() {
                        write!(f, ": {}", ty)?;
                    }
                    first = false;
                }
                write!(f, ">")
            }
            // enum<case1, case2, ...>
            ValueType::Enum(x) => {
                write!(f, "enum<")?;
                let mut first = true;
                for case in x.cases() {
                    if !first {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", case)?;
                    first = false;
                }
                write!(f, ">")
            }
            ValueType::Option(x) => write!(f, "option<{}>", x.some_ty()),
            ValueType::Result(x) => {
                write!(f, "result<")?;
                if let Some(ty) = x.ok_ty() {
                    write!(f, "{}", ty)?;
                } else {
                    write!(f, "unit")?;
                }
                write!(f, ", ")?;
                if let Some(ty) = x.err_ty() {
                    write!(f, "{}", ty)?;
                } else {
                    write!(f, "unit")?;
                }
                write!(f, ">")
            }
            // flags<flag1, flag2, ...>
            ValueType::Flags(x) => {
                write!(f, "flags<")?;
                let mut first = true;
                for name in x.names() {
                    if !first {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", name)?;
                    first = false;
                }
                write!(f, ">")
            }
            // own<type>
            ValueType::Own(x) => write!(
                f,
                "own<{}>",
                x.name().map(ToString::to_string).unwrap_or("".to_string())
            ),
            // borrow<type>
            ValueType::Borrow(x) => write!(
                f,
                "borrow<{}>",
                x.name().map(ToString::to_string).unwrap_or("".to_string())
            ),
        }
    }
}

impl ValueType {
    /// Creates a value type from a component.
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

    /// Creates a value type from a component type definition.
    pub(crate) fn from_component_typedef(
        def: Id<wit_parser::TypeDef>,
        component: &ComponentInner,
        resource_map: Option<&FxHashMap<ResourceType, ResourceType>>,
    ) -> Result<Self> {
        let name = component.type_identifiers[def.index()].clone();
        Ok(match &component.resolve.types[def].kind {
            wit_parser::TypeDefKind::Record(x) => Self::Record(RecordType::from_component(
                name,
                x,
                component,
                resource_map,
            )?),
            wit_parser::TypeDefKind::Resource => bail!("Cannot instantiate resource as type."),
            wit_parser::TypeDefKind::Handle(x) => match x {
                wit_parser::Handle::Own(t) => Self::Own(ResourceType::from_resolve(
                    name,
                    *t,
                    component,
                    resource_map,
                )?),
                wit_parser::Handle::Borrow(t) => Self::Borrow(ResourceType::from_resolve(
                    name,
                    *t,
                    component,
                    resource_map,
                )?),
            },
            wit_parser::TypeDefKind::Flags(x) => {
                Self::Flags(FlagsType::from_component(name, x, component)?)
            }
            wit_parser::TypeDefKind::Tuple(x) => {
                Self::Tuple(TupleType::from_component(name, x, component, resource_map)?)
            }
            wit_parser::TypeDefKind::Variant(x) => Self::Variant(VariantType::from_component(
                name,
                x,
                component,
                resource_map,
            )?),
            wit_parser::TypeDefKind::Enum(x) => {
                Self::Enum(EnumType::from_component(name, x, component)?)
            }
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

/// Describes the type of a list of values, all of the same type.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ListType {
    /// The element of the list.
    element: Arc<ValueType>,
}

impl ListType {
    /// Creates a new list type for the given element type.
    pub fn new(element_ty: ValueType) -> Self {
        Self {
            element: Arc::new(element_ty),
        }
    }

    /// Gets the element type for this list.
    pub fn element_ty(&self) -> ValueType {
        (*self.element).clone()
    }
}

/// Describes the type of an unordered collection of named fields, each associated with the values.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RecordType {
    /// The fields of the record.
    pub(crate) fields: Arc<[(usize, Arc<str>, ValueType)]>,
    /// The identifier associated with this type.
    name: Option<TypeIdentifier>,
}

impl RecordType {
    /// Creates a new record type from the given set of fields. The field names must be unique.
    pub fn new<S: Into<Arc<str>>>(
        name: Option<TypeIdentifier>,
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

        Ok(Self {
            fields: to_sort,
            name,
        })
    }

    /// Gets the type of the provided field, if any.
    pub fn field_ty(&self, name: impl AsRef<str>) -> Option<ValueType> {
        self.fields
            .iter()
            .filter(|&(_, x, _val)| (&**x == name.as_ref()))
            .map(|(_, _x, val)| val.clone())
            .next()
    }

    /// Gets the name of this type, if any.
    pub fn name(&self) -> Option<&TypeIdentifier> {
        self.name.as_ref()
    }

    /// Gets an iterator over all field names and values in this record.
    pub fn fields(&self) -> impl ExactSizeIterator<Item = (&str, ValueType)> {
        self.fields
            .iter()
            .map(|(_, name, val)| (&**name, val.clone()))
    }

    /// Creates a new record type, assuming that the fields are already sorted.
    pub(crate) fn new_sorted(
        name: Option<TypeIdentifier>,
        fields: impl IntoIterator<Item = (Arc<str>, ValueType)>,
    ) -> Result<Self> {
        let result = Self {
            fields: fields
                .into_iter()
                .enumerate()
                .map(|(i, (a, b))| (i, a, b))
                .collect(),
            name,
        };

        for pair in result.fields.windows(2) {
            ensure!(pair[0].0 != pair[1].0, "Duplicate field names");
        }

        Ok(result)
    }

    /// Creates the record type from the given component.
    fn from_component(
        name: Option<TypeIdentifier>,
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

        Ok(Self {
            fields: to_sort,
            name,
        })
    }
}

impl PartialEq for RecordType {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields
    }
}

impl Eq for RecordType {}

impl Hash for RecordType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fields.hash(state);
    }
}

/// Describes the type of an ordered, unnamed sequence of heterogenously-typed values.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TupleType {
    /// The types of the tuple fields.
    fields: Arc<[ValueType]>,
    /// The name of the type.
    name: Option<TypeIdentifier>,
}

impl TupleType {
    /// Creates a tuple for the specified list of fields.
    pub fn new(name: Option<TypeIdentifier>, fields: impl IntoIterator<Item = ValueType>) -> Self {
        Self {
            name,
            fields: fields.into_iter().collect(),
        }
    }

    /// The set of field types for this tuple.
    pub fn fields(&self) -> &[ValueType] {
        &self.fields
    }

    /// Gets the name of this type, if any.
    pub fn name(&self) -> Option<&TypeIdentifier> {
        self.name.as_ref()
    }

    /// Creates this type from the given component.
    fn from_component(
        name: Option<TypeIdentifier>,
        ty: &wit_parser::Tuple,
        component: &ComponentInner,
        resource_map: Option<&FxHashMap<ResourceType, ResourceType>>,
    ) -> Result<Self> {
        let fields = ty
            .types
            .iter()
            .map(|x| ValueType::from_component(x, component, resource_map))
            .collect::<Result<_>>()?;
        Ok(Self { name, fields })
    }
}

impl PartialEq for TupleType {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields
    }
}

impl Eq for TupleType {}

impl Hash for TupleType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fields.hash(state);
    }
}

/// Describes a single branch of a variant.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VariantCase {
    /// The name of this case.
    name: Arc<str>,
    /// The type of this case's values, if any.
    ty: Option<ValueType>,
}

impl VariantCase {
    /// Creates a new variant case with the specified name and optional associated type.
    pub fn new(name: impl Into<Arc<str>>, ty: Option<ValueType>) -> Self {
        Self {
            name: name.into(),
            ty,
        }
    }

    /// The name of this case.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The type of values associated with this case, if any.
    pub fn ty(&self) -> Option<ValueType> {
        self.ty.clone()
    }
}

/// Describes a type has multiple possible states. Each state may optionally
/// have a type associated with it.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VariantType {
    /// The cases of this variant.
    cases: Arc<[VariantCase]>,
    /// The name of the type.
    name: Option<TypeIdentifier>,
}

impl VariantType {
    /// Creates a new type for the given set of variant cases. Each case must have a unique name.
    pub fn new(
        name: Option<TypeIdentifier>,
        cases: impl IntoIterator<Item = VariantCase>,
    ) -> Result<Self> {
        let cases: Arc<_> = cases.into_iter().collect();

        for i in 0..cases.len() {
            for j in 0..i {
                ensure!(cases[i].name() != cases[j].name(), "Duplicate case names.");
            }
        }

        Ok(Self { name, cases })
    }

    /// Gets all of the cases for this variant.
    pub fn cases(&self) -> &[VariantCase] {
        &self.cases
    }

    /// Gets the name of this type, if any.
    pub fn name(&self) -> Option<&TypeIdentifier> {
        self.name.as_ref()
    }

    /// Creates this type from the given component.
    fn from_component(
        name: Option<TypeIdentifier>,
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

        Ok(Self { name, cases })
    }
}

impl PartialEq for VariantType {
    fn eq(&self, other: &Self) -> bool {
        self.cases == other.cases
    }
}

impl Eq for VariantType {}

impl Hash for VariantType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.cases.hash(state);
    }
}

/// A type that has multiple possible states.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EnumType {
    /// The cases of the enum.
    cases: Arc<[Arc<str>]>,
    /// The name of the type.
    name: Option<TypeIdentifier>,
}

impl EnumType {
    /// Creates a new enumeration from the list of case names. The case names must be unique.
    pub fn new<S: Into<Arc<str>>>(
        name: Option<TypeIdentifier>,
        cases: impl IntoIterator<Item = S>,
    ) -> Result<Self> {
        let res = Self {
            name,
            cases: cases
                .into_iter()
                .map(|x| Into::<Arc<str>>::into(x))
                .collect(),
        };

        for i in 0..res.cases.len() {
            for j in 0..i {
                ensure!(res.cases[i] != res.cases[j], "Duplicate case names.");
            }
        }

        Ok(res)
    }

    /// Gets the name of this type, if any.
    pub fn name(&self) -> Option<&TypeIdentifier> {
        self.name.as_ref()
    }

    /// Gets a list of all cases in this enum.
    pub fn cases(&self) -> impl ExactSizeIterator<Item = &str> {
        self.cases.iter().map(|x| &**x)
    }

    /// Creates this type from the given component.
    fn from_component(
        name: Option<TypeIdentifier>,
        ty: &wit_parser::Enum,
        _component: &ComponentInner,
    ) -> Result<Self> {
        Self::new(name, ty.cases.iter().map(|x| x.name.as_str()))
    }
}

impl PartialEq for EnumType {
    fn eq(&self, other: &Self) -> bool {
        self.cases == other.cases
    }
}

impl Eq for EnumType {}

impl Hash for EnumType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.cases.hash(state)
    }
}

/// A type that may also be the absence of anything.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OptionType {
    /// The type of this option when something exists.
    ty: Arc<ValueType>,
}

impl OptionType {
    /// Creates a new option type that holds the given value.
    pub fn new(ty: ValueType) -> Self {
        Self { ty: Arc::new(ty) }
    }

    /// Gets the type associated with the `Some` variant of this type.
    pub fn some_ty(&self) -> ValueType {
        (*self.ty).clone()
    }
}

/// A type that denotes successful or unsuccessful operation.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ResultType {
    /// The types associated with the result variant.
    ok_err: Arc<(Option<ValueType>, Option<ValueType>)>,
}

impl ResultType {
    /// Creates a new result type for the given `Ok` and `Err` variant types.
    pub fn new(ok: Option<ValueType>, err: Option<ValueType>) -> Self {
        Self {
            ok_err: Arc::new((ok, err)),
        }
    }

    /// Gets the type of value returned for the `Ok` variant, if any.
    pub fn ok_ty(&self) -> Option<ValueType> {
        self.ok_err.0.clone()
    }

    /// Gets the type of value returned for the `Err` variant, if any.
    pub fn err_ty(&self) -> Option<ValueType> {
        self.ok_err.1.clone()
    }
}

/// A type that denotes a set of named bitflags.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FlagsType {
    /// The names of each flags.
    names: Arc<[Arc<str>]>,
    /// A mapping from flag name to index.
    pub(crate) indices: Arc<FxHashMap<Arc<str>, usize>>,
    /// The name of the type.
    name: Option<TypeIdentifier>,
}

impl FlagsType {
    /// Creates a new flags type with the specified list of names. The names must be unique.
    pub fn new<S: Into<Arc<str>>>(
        name: Option<TypeIdentifier>,
        names: impl IntoIterator<Item = S>,
    ) -> Result<Self> {
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

        Ok(Self {
            name,
            names,
            indices,
        })
    }

    /// Gets the name of this type, if any.
    pub fn name(&self) -> Option<&TypeIdentifier> {
        self.name.as_ref()
    }

    /// Gets an iterator over the names of the flags in this collection.
    pub fn names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.names.iter().map(|x| &**x)
    }

    /// Creates this type from the given component.
    fn from_component(
        name: Option<TypeIdentifier>,
        ty: &wit_parser::Flags,
        _component: &ComponentInner,
    ) -> Result<Self> {
        Self::new(name, ty.flags.iter().map(|x| x.name.as_ref()))
    }
}

impl PartialEq for FlagsType {
    fn eq(&self, other: &Self) -> bool {
        self.names == other.names
    }
}

impl Eq for FlagsType {}

impl Hash for FlagsType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.names.hash(state)
    }
}

/// A counter that uniquely identifies resources.
static RESOURCE_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Describes the type of a resource. This may either be:
///
/// - An abstract guest resource, associated with a component. Abstract resources
/// cannot be used to instantiate any values.
/// - An instantiated guest resource, associated with an instance. Instantiated guest
/// resources identify resources created by WASM.
/// - A host resource, which is associated with a native value.
#[derive(Clone, Debug)]
pub struct ResourceType {
    /// The kind of resource that this is.
    kind: ResourceKindValue,
    /// The name of the type.
    name: Option<TypeIdentifier>,
}

impl ResourceType {
    /// Creates a new host resource for storing values of the given type. Note that multiple
    /// resource types may be created for the same `T`, and they will be distinct.
    pub fn new<T: 'static + Send + Sync + Sized>(name: Option<TypeIdentifier>) -> Self {
        Self {
            name,
            kind: ResourceKindValue::Host {
                resource_id: RESOURCE_ID_COUNTER.fetch_add(1, Ordering::AcqRel),
                type_id: TypeId::of::<T>(),
                associated_store: None,
            },
        }
    }

    /// Creates a new host resource for storing values of the given type. Additionally,
    /// adds a destructor that is called when the resource is dropped. Note that multiple
    /// resource types may be created for the same `T`, and they will be distinct.
    pub fn with_destructor<T: 'static + Send + Sync + Sized, C: AsContextMut>(
        mut ctx: C,
        name: Option<TypeIdentifier>,
        destructor: impl 'static
            + Send
            + Sync
            + Fn(StoreContextMut<'_, C::UserState, C::Engine>, T) -> Result<()>,
    ) -> Result<Self> {
        let store_id = ctx.as_context().inner.data().id;
        let destructor = wasm_runtime_layer::Func::new(
            ctx.as_context_mut().inner,
            wasm_runtime_layer::FuncType::new([wasm_runtime_layer::ValueType::I32], [])
                .with_name("destructor"),
            move |mut ctx, val, _res| {
                let resource = wasm_runtime_layer::AsContextMut::as_context_mut(&mut ctx)
                    .data_mut()
                    .host_resources
                    .remove(
                        require_matches!(val[0], wasm_runtime_layer::Value::I32(x), x) as usize,
                    );
                destructor(
                    StoreContextMut { inner: ctx },
                    *resource
                        .downcast()
                        .expect("Resource was of incorrect type."),
                )
            },
        );

        Ok(Self {
            name,
            kind: ResourceKindValue::Host {
                resource_id: RESOURCE_ID_COUNTER.fetch_add(1, Ordering::AcqRel),
                type_id: TypeId::of::<T>(),
                associated_store: Some((store_id, destructor)),
            },
        })
    }

    /// Gets the name of this type, if any.
    pub fn name(&self) -> Option<&TypeIdentifier> {
        self.name.as_ref()
    }

    /// Determines whether this resource type can be used to extract host values of `T` from the provided store.
    pub(crate) fn valid_for<T: 'static + Send + Sync>(&self, store_id: u64) -> bool {
        if let ResourceKindValue::Host {
            type_id,
            associated_store,
            ..
        } = &self.kind
        {
            *type_id == TypeId::of::<T>()
                && associated_store
                    .as_ref()
                    .map(|(id, _)| *id == store_id)
                    .unwrap_or(true)
        } else {
            false
        }
    }

    /// Determines whether this resource type is owned by the specified instance.
    pub(crate) fn is_owned_by_instance(&self, instance: u64) -> bool {
        if let ResourceKindValue::Instantiated { instance: a, .. } = &self.kind {
            instance == *a
        } else {
            false
        }
    }

    /// Creates this type from the given component.
    pub(crate) fn from_resolve(
        name: Option<TypeIdentifier>,
        id: Id<wit_parser::TypeDef>,
        component: &ComponentInner,
        resource_map: Option<&FxHashMap<ResourceType, ResourceType>>,
    ) -> Result<Self> {
        let resolve_type = &component.resolve.types[id];
        if resolve_type.kind == wit_parser::TypeDefKind::Resource {
            let ab = Self {
                name,
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
        } else if let wit_parser::TypeDefKind::Type(wit_parser::Type::Id(x)) = resolve_type.kind {
            Self::from_resolve(name, x, component, resource_map)
        } else {
            bail!("Unrecognized resource type.")
        }
    }

    /// Determines whether this is a host resource, and if so, returns the optional destructor.
    pub(crate) fn host_destructor(&self) -> Option<Option<wasm_runtime_layer::Func>> {
        if let ResourceKindValue::Host {
            associated_store, ..
        } = &self.kind
        {
            Some(associated_store.as_ref().map(|(_, x)| x.clone()))
        } else {
            None
        }
    }

    /// Converts this type from an abstract guest resource to an instantiated guest resource.
    pub(crate) fn instantiate(&self, instance: u64) -> Result<Self> {
        if let ResourceKindValue::Abstract { id, component: _ } = &self.kind {
            Ok(Self {
                name: self.name.clone(),
                kind: ResourceKindValue::Instantiated { id: *id, instance },
            })
        } else {
            bail!("Resource was not abstract.");
        }
    }

    /// Determines whether this is an instantiated or host resource.
    pub(crate) fn is_instantiated(&self) -> bool {
        !matches!(&self.kind, ResourceKindValue::Abstract { .. })
    }
}

impl Hash for ResourceType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
    }
}

impl PartialEq for ResourceType {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

impl Eq for ResourceType {}

#[cfg(feature = "serde")]
impl Serialize for ResourceType {
    fn serialize<S: Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
        use serde::ser::*;
        std::result::Result::Err(S::Error::custom("Cannot serialize resources."))
    }
}

#[cfg(feature = "serde")]
impl<'a> Deserialize<'a> for ResourceType {
    fn deserialize<D: Deserializer<'a>>(_: D) -> Result<Self, D::Error> {
        use serde::de::*;
        std::result::Result::Err(D::Error::custom("Cannot deserialize resources."))
    }
}

/// Marks the backing for a resource type.
#[derive(Clone, Debug)]
enum ResourceKindValue {
    /// This is an abstract, uninstantiated resource.
    Abstract {
        /// The ID of the resource.
        id: usize,
        /// The ID of the component.
        component: u64,
    },
    /// A resource associated with an instance.
    Instantiated {
        /// The ID of the resource.
        id: usize,
        /// The ID of the instance.
        instance: u64,
    },
    /// A resource associated with the host.
    Host {
        /// The ID of the resource.
        resource_id: u64,
        /// The type ID of the representation.
        type_id: TypeId,
        /// The associated store and destructor, if any.
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
                ResourceKindValue::Instantiated { id: a, instance: b },
                ResourceKindValue::Instantiated { id: x, instance: y },
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
            ResourceKindValue::Instantiated { id, instance } => {
                id.hash(state);
                instance.hash(state);
            }
            ResourceKindValue::Host { resource_id, .. } => resource_id.hash(state),
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

impl Display for FuncType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params = self.params();
        let results = self.results();

        let mut first = true;

        write!(f, "func(")?;
        for param in params {
            if !first {
                write!(f, ", ")?;
            } else {
                first = false;
            }
            write!(f, "{param}")?;
        }

        write!(f, ")")?;

        let mut first = true;

        for result in results {
            if !first {
                write!(f, ", ")?;
            } else {
                first = false;
                write!(f, " -> ")?;
            }

            write!(f, "{result}")?;
        }

        std::fmt::Result::Ok(())
    }
}

impl FuncType {
    /// Creates this type from the given component.
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

/// Marks a list of components that can be passed as parameters and results.
pub trait ComponentList: 'static + Sized {
    /// The length of the component list.
    const LEN: usize;

    /// Stores the types of this list into the given slice. Panics
    /// if the slice is not big enough.
    fn into_tys(types: &mut [ValueType]);

    /// Attempts to convert this component list into values, storing them
    /// in the provided slice.
    fn into_values(self, values: &mut [crate::values::Value]) -> Result<()>;

    /// Attempts to convert a list of values into a component list of this type.
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

impl<T: UnaryComponentType> ComponentList for T {
    const LEN: usize = 1;

    fn into_tys(types: &mut [ValueType]) {
        assert!(types.len() == 1);
        types[0] = T::ty();
    }

    fn into_values(self, values: &mut [crate::values::Value]) -> Result<()> {
        assert!(values.len() == 1);
        values[0] = T::into_value(self)?;
        Ok(())
    }

    fn from_values(values: &[crate::values::Value]) -> Result<Self> {
        assert!(values.len() == 1);
        T::from_value(&values[0])
    }
}

/// A function that returns a single result, and eats a macro parameter in the process.
/// Used to count the number of parameters in the macro.
#[allow(clippy::extra_unused_type_parameters)]
const fn one<T>() -> usize {
    1
}

/// Implements the component list for a tuple with the provided set of parameters.
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
