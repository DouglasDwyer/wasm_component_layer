use std::any::*;
use std::marker::*;
#[cfg(feature = "serde")]
use std::mem::*;
use std::ops::*;
use std::sync::atomic::*;
use std::sync::*;

use anyhow::*;
#[cfg(feature = "serde")]
use bytemuck::*;
use private::*;
#[cfg(feature = "serde")]
use serde::*;

use crate::require_matches::require_matches;
use crate::types::*;
use crate::AsContext;
use crate::AsContextMut;
use crate::TypeIdentifier;

/// Represents a component model type.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Value {
    /// A boolean value.
    Bool(bool),
    /// An eight-bit signed integer.
    S8(i8),
    /// An eight-bit unsigned integer.
    U8(u8),
    /// A 16-bit signed integer.
    S16(i16),
    /// A 16-bit unsigned integer.
    U16(u16),
    /// A 32-bit signed integer.
    S32(i32),
    /// A 32-bit unsigned integer.
    U32(u32),
    /// A 64-bit signed integer.
    S64(i64),
    /// A 64-bit unsigned integer.
    U64(u64),
    /// A 32-bit floating point number.
    F32(f32),
    /// A 64-bit floating point number.
    F64(f64),
    /// A UTF-8 character.
    Char(char),
    /// A string.
    String(Arc<str>),
    /// A homogenous list of elements.
    List(List),
    /// A record with heterogenous fields.
    Record(Record),
    /// A tuple with heterogenous fields.
    Tuple(Tuple),
    /// A variant which may be one of multiple types or cases.
    Variant(Variant),
    /// An enum which may be one of multiple cases.
    Enum(Enum),
    /// A type which may or may not have an underlying value.
    Option(OptionValue),
    /// A type that indicates success or failure.
    Result(ResultValue),
    /// A set of boolean values.
    Flags(Flags),
    /// An owned resource handle.
    Own(ResourceOwn),
    /// A borrowed resource handle.
    Borrow(ResourceBorrow),
}

impl Value {
    /// Gets the type of this value.
    pub fn ty(&self) -> ValueType {
        match self {
            Value::Bool(_) => ValueType::Bool,
            Value::S8(_) => ValueType::S8,
            Value::U8(_) => ValueType::U8,
            Value::S16(_) => ValueType::S16,
            Value::U16(_) => ValueType::U16,
            Value::S32(_) => ValueType::S32,
            Value::U32(_) => ValueType::U32,
            Value::S64(_) => ValueType::S64,
            Value::U64(_) => ValueType::U64,
            Value::F32(_) => ValueType::F32,
            Value::F64(_) => ValueType::F64,
            Value::Char(_) => ValueType::Char,
            Value::String(_) => ValueType::String,
            Value::List(x) => ValueType::List(x.ty()),
            Value::Record(x) => ValueType::Record(x.ty()),
            Value::Tuple(x) => ValueType::Tuple(x.ty()),
            Value::Variant(x) => ValueType::Variant(x.ty()),
            Value::Enum(x) => ValueType::Enum(x.ty()),
            Value::Option(x) => ValueType::Option(x.ty()),
            Value::Result(x) => ValueType::Result(x.ty()),
            Value::Flags(x) => ValueType::Flags(x.ty()),
            Value::Own(x) => ValueType::Own(x.ty()),
            Value::Borrow(x) => ValueType::Borrow(x.ty()),
        }
    }
}

impl TryFrom<&Value> for wasm_runtime_layer::Value {
    type Error = Error;

    fn try_from(value: &Value) -> Result<Self> {
        match value {
            Value::S32(x) => Ok(Self::I32(*x)),
            Value::S64(x) => Ok(Self::I64(*x)),
            Value::F32(x) => Ok(Self::F32(*x)),
            Value::F64(x) => Ok(Self::F64(*x)),
            _ => bail!("Unable to convert {value:?} to core type."),
        }
    }
}

impl TryFrom<&wasm_runtime_layer::Value> for Value {
    type Error = Error;

    fn try_from(value: &wasm_runtime_layer::Value) -> Result<Self> {
        match value {
            wasm_runtime_layer::Value::I32(x) => Ok(Self::S32(*x)),
            wasm_runtime_layer::Value::I64(x) => Ok(Self::S64(*x)),
            wasm_runtime_layer::Value::F32(x) => Ok(Self::F32(*x)),
            wasm_runtime_layer::Value::F64(x) => Ok(Self::F64(*x)),
            _ => bail!("Unable to convert {value:?} to component type."),
        }
    }
}

/// Implements the `From` trait for primitive values.
macro_rules! impl_primitive_from {
    ($(($type_name: ident, $enum_name: ident))*) => {
        $(
            impl From<&$type_name> for Value {
                fn from(value: &$type_name) -> Value {
                    Value::$enum_name(*value)
                }
            }

            impl TryFrom<$type_name> for Value {
                type Error = Error;

                fn try_from(value: $type_name) -> Result<Self> {
                    Ok(Value::$enum_name(value))
                }
            }

            impl TryFrom<&Value> for $type_name {
                type Error = Error;

                fn try_from(value: &Value) -> Result<Self> {
                    Ok(require_matches!(value, Value::$enum_name(x), *x))
                }
            }
        )*
    };
}

impl_primitive_from!((bool, Bool)(i8, S8)(u8, U8)(i16, S16)(u16, U16)(i32, S32)(
    u32, U32
)(i64, S64)(u64, U64)(f32, F32)(f64, F64)(char, Char));

/// Represents a list of values, all of the same type.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct List {
    /// The underlying representation of the list.
    values: ListSpecialization,
    /// The type of the list.
    ty: ListType,
}

impl List {
    /// Creates a new list with the provided values. Every value must match
    /// the element in the given list type.
    pub fn new(ty: ListType, values: impl IntoIterator<Item = Value>) -> Result<Self> {
        let values = match ty.element_ty() {
            ValueType::Bool => bool::from_value_iter(values)?,
            ValueType::S8 => i8::from_value_iter(values)?,
            ValueType::U8 => u8::from_value_iter(values)?,
            ValueType::S16 => i16::from_value_iter(values)?,
            ValueType::U16 => u16::from_value_iter(values)?,
            ValueType::S32 => i32::from_value_iter(values)?,
            ValueType::U32 => u32::from_value_iter(values)?,
            ValueType::S64 => i64::from_value_iter(values)?,
            ValueType::U64 => u64::from_value_iter(values)?,
            ValueType::F32 => f32::from_value_iter(values)?,
            ValueType::F64 => f64::from_value_iter(values)?,
            ValueType::Char => char::from_value_iter(values)?,
            _ => ListSpecialization::Other(
                values
                    .into_iter()
                    .map(|x| {
                        (x.ty() == ty.element_ty()).then_some(x).ok_or_else(|| {
                            Error::msg("List elements were not all of the same type.")
                        })
                    })
                    .collect::<Result<_>>()?,
            ),
        };

        Ok(Self { values, ty })
    }

    /// Gets the type of this value.
    pub fn ty(&self) -> ListType {
        self.ty.clone()
    }

    /// Casts this list to a strongly-typed slice, if possible. For performance
    /// reasons, lists are specialized to store primitive types without any
    /// wrappers or indirection. This function allows one to access that representation.
    pub fn typed<T: ListPrimitive>(&self) -> Result<&[T]> {
        if self.ty.element_ty() == T::ty() {
            Ok(T::from_specialization(&self.values))
        } else {
            bail!(
                "List type mismatch: expected {:?} but got {:?}",
                T::ty(),
                self.ty()
            );
        }
    }

    /// Gets an iterator over the values in the list.
    pub fn iter(&self) -> impl '_ + Iterator<Item = Value> {
        self.into_iter()
    }

    /// Whether this list is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets the length of the list.
    pub fn len(&self) -> usize {
        match &self.values {
            ListSpecialization::Bool(x) => x.len(),
            ListSpecialization::S8(x) => x.len(),
            ListSpecialization::U8(x) => x.len(),
            ListSpecialization::S16(x) => x.len(),
            ListSpecialization::U16(x) => x.len(),
            ListSpecialization::S32(x) => x.len(),
            ListSpecialization::U32(x) => x.len(),
            ListSpecialization::S64(x) => x.len(),
            ListSpecialization::U64(x) => x.len(),
            ListSpecialization::F32(x) => x.len(),
            ListSpecialization::F64(x) => x.len(),
            ListSpecialization::Char(x) => x.len(),
            ListSpecialization::Other(x) => x.len(),
        }
    }
}

impl PartialEq for List {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl<'a> IntoIterator for &'a List {
    type IntoIter = ListSpecializationIter<'a>;

    type Item = Value;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl<T: ListPrimitive> From<&[T]> for List {
    fn from(value: &[T]) -> Self {
        Self {
            values: T::from_arc(value.into()),
            ty: ListType::new(T::ty()),
        }
    }
}

impl<T: ListPrimitive> From<Arc<[T]>> for List {
    fn from(value: Arc<[T]>) -> Self {
        Self {
            values: T::from_arc(value),
            ty: ListType::new(T::ty()),
        }
    }
}

/// An unordered collection of named fields, each associated with the values.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Record {
    /// The internal set of keys and values, ordered lexicographically.
    fields: Arc<[(Arc<str>, Value)]>,
    /// The type of this record.
    ty: RecordType,
}

impl Record {
    /// Creates a new record out of the given fields. Each field must match with
    /// the type given in the `RecordType`.
    pub fn new<S: Into<Arc<str>>>(
        ty: RecordType,
        values: impl IntoIterator<Item = (S, Value)>,
    ) -> Result<Self> {
        let mut to_sort = values
            .into_iter()
            .map(|(name, val)| (Into::<Arc<str>>::into(name), val))
            .collect::<Arc<_>>();
        Arc::get_mut(&mut to_sort)
            .expect("Could not get exclusive reference.")
            .sort_by(|a, b| a.0.cmp(&b.0));

        ensure!(
            to_sort.len() == ty.fields().len(),
            "Record fields did not match type."
        );

        for ((name, val), (ty_name, ty_val)) in to_sort.iter().zip(ty.fields()) {
            ensure!(
                **name == *ty_name && val.ty() == ty_val,
                "Record fields did not match type."
            );
        }

        Ok(Self {
            fields: to_sort,
            ty,
        })
    }

    /// Constructs a record from the provided fields, dynamically determining the type.
    pub fn from_fields<S: Into<Arc<str>>>(
        name: Option<TypeIdentifier>,
        values: impl IntoIterator<Item = (S, Value)>,
    ) -> Result<Self> {
        let mut fields = values
            .into_iter()
            .map(|(name, val)| (Into::<Arc<str>>::into(name), val))
            .collect::<Arc<_>>();
        Arc::get_mut(&mut fields)
            .expect("Could not get exclusive reference.")
            .sort_by(|a, b| a.0.cmp(&b.0));
        let ty = RecordType::new_sorted(
            name,
            fields.iter().map(|(name, val)| (name.clone(), val.ty())),
        )?;
        Ok(Self { fields, ty })
    }

    /// Gets the field with the provided name, if any.
    pub fn field(&self, field: impl AsRef<str>) -> Option<Value> {
        self.fields
            .iter()
            .filter(|&(name, _val)| (&**name == field.as_ref()))
            .map(|(_name, val)| val.clone())
            .next()
    }

    /// Gets an iterator over the fields of this record.
    pub fn fields(&self) -> impl ExactSizeIterator<Item = (&str, Value)> {
        self.fields.iter().map(|(name, val)| (&**name, val.clone()))
    }

    /// Gets the type of this value.
    pub fn ty(&self) -> RecordType {
        self.ty.clone()
    }

    /// Creates a new record from the already-sorted list of values.
    pub(crate) fn from_sorted(
        ty: RecordType,
        values: impl IntoIterator<Item = (Arc<str>, Value)>,
    ) -> Self {
        Self {
            fields: values.into_iter().collect(),
            ty,
        }
    }
}

impl PartialEq for Record {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields
    }
}

/// An ordered, unnamed sequence of heterogenously-typed values.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Tuple {
    /// The fields of this tuple.
    fields: Arc<[Value]>,
    /// The type of the tuple.
    ty: TupleType,
}

impl Tuple {
    /// Creates a new tuple of the given type, from the provided fields. Fails if the provided
    /// fields do not match the specified type.
    pub fn new(ty: TupleType, fields: impl IntoIterator<Item = Value>) -> Result<Self> {
        Ok(Self {
            fields: fields
                .into_iter()
                .enumerate()
                .map(|(i, val)| {
                    ensure!(i < ty.fields().len(), "Field count was out-of-range.");
                    (val.ty() == ty.fields()[i])
                        .then_some(val)
                        .ok_or_else(|| Error::msg("Value was not of correct type."))
                })
                .collect::<Result<_>>()?,
            ty,
        })
    }

    /// Creates a new tuple from the provided fields, inferring the type.
    pub fn from_fields(
        name: Option<TypeIdentifier>,
        fields: impl IntoIterator<Item = Value>,
    ) -> Self {
        let fields: Arc<_> = fields.into_iter().collect();
        let ty = TupleType::new(name, fields.iter().map(|x| x.ty()));
        Self { fields, ty }
    }

    /// Gets the type of this value.
    pub fn ty(&self) -> TupleType {
        self.ty.clone()
    }

    /// Creates a new tuple of the given type without any typechecking.
    pub(crate) fn new_unchecked(ty: TupleType, fields: impl IntoIterator<Item = Value>) -> Self {
        Self {
            fields: fields.into_iter().collect(),
            ty,
        }
    }
}

impl PartialEq for Tuple {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields
    }
}

impl Deref for Tuple {
    type Target = [Value];

    fn deref(&self) -> &Self::Target {
        &self.fields
    }
}

impl IntoIterator for Tuple {
    type IntoIter = std::vec::IntoIter<Value>;
    type Item = Value;

    fn into_iter(self) -> Self::IntoIter {
        self.fields.iter().cloned().collect::<Vec<_>>().into_iter()
    }
}

impl<'a> IntoIterator for &'a Tuple {
    type IntoIter = std::slice::Iter<'a, Value>;
    type Item = &'a Value;

    fn into_iter(self) -> Self::IntoIter {
        self.fields.iter()
    }
}

/// A value that exists in one of multiple possible states. Each state may optionally
/// have a type associated with it.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Variant {
    /// Determines in which state this value exists.
    discriminant: u32,
    /// The value of this variant.
    value: Option<Arc<Value>>,
    /// The type of the variant.
    ty: VariantType,
}

impl Variant {
    /// Creates a new variant for the given discriminant and optional value. The value's type
    /// must match the variant's type for the selected state.
    pub fn new(ty: VariantType, discriminant: usize, value: Option<Value>) -> Result<Self> {
        ensure!(
            discriminant < ty.cases().len(),
            "Discriminant out-of-range."
        );
        ensure!(
            ty.cases()[discriminant].ty() == value.as_ref().map(|x| x.ty()),
            "Provided value was of incorrect type for case."
        );
        Ok(Self {
            discriminant: discriminant as u32,
            value: value.map(Arc::new),
            ty,
        })
    }

    /// Gets the index that describes in which state this value exists.
    pub fn discriminant(&self) -> usize {
        self.discriminant as usize
    }

    /// Gets the typed value associated with the current state, if any.
    pub fn value(&self) -> Option<Value> {
        self.value.as_ref().map(|x| (**x).clone())
    }

    /// Gets the type of this value.
    pub fn ty(&self) -> VariantType {
        self.ty.clone()
    }
}

/// A value that may exist in one of multiple possible states.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Enum {
    /// Determines in which state this value exists.
    discriminant: u32,
    /// The type of the enum.
    ty: EnumType,
}

impl Enum {
    /// Creates a new enum value with the given discriminant. The discriminant must be
    /// in range with respect to the enum type.
    pub fn new(ty: EnumType, discriminant: usize) -> Result<Self> {
        ensure!(
            discriminant < ty.cases().len(),
            "Discriminant out-of-range."
        );
        Ok(Self {
            discriminant: discriminant as u32,
            ty,
        })
    }

    /// Gets the index that describes in which state this value exists.
    pub fn discriminant(&self) -> usize {
        self.discriminant as usize
    }

    /// Gets the type of this value.
    pub fn ty(&self) -> EnumType {
        self.ty.clone()
    }
}

/// Represents a value or lack thereof.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OptionValue {
    /// The type of this option.
    ty: OptionType,
    /// The value, if any.
    value: Arc<Option<Value>>,
}

impl OptionValue {
    /// Creates a new option with the given type and value.
    pub fn new(ty: OptionType, value: Option<Value>) -> Result<Self> {
        ensure!(
            value
                .as_ref()
                .map(|x| x.ty() == ty.some_ty())
                .unwrap_or(true),
            "Provided option value was of incorrect type."
        );
        Ok(Self {
            ty,
            value: Arc::new(value),
        })
    }

    /// Gets the type of this value.
    pub fn ty(&self) -> OptionType {
        self.ty.clone()
    }
}

impl Deref for OptionValue {
    type Target = Option<Value>;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

/// Denotes a successful or unsuccessful operation, associated optionally with types.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ResultValue {
    /// The type of this result.
    ty: ResultType,
    /// The value of this result.
    value: Arc<Result<Option<Value>, Option<Value>>>,
}

impl ResultValue {
    /// Creates a new result from the provided type and value. The value must match that which
    /// is described in the type.
    pub fn new(ty: ResultType, value: Result<Option<Value>, Option<Value>>) -> Result<Self> {
        ensure!(
            match &value {
                std::result::Result::Ok(x) => x.as_ref().map(|y| y.ty()) == ty.ok_ty(),
                std::result::Result::Err(x) => x.as_ref().map(|y| y.ty()) == ty.err_ty(),
            },
            "Provided result value was of incorrect type. (expected {ty:?}, had {value:?})"
        );
        Ok(Self {
            ty,
            value: Arc::new(value),
        })
    }

    /// The type of this result.
    pub fn ty(&self) -> ResultType {
        self.ty.clone()
    }
}

impl Deref for ResultValue {
    type Target = Result<Option<Value>, Option<Value>>;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

/// A finite set of boolean flags that may be `false` or `true`.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Flags {
    /// The type of this flags list.
    ty: FlagsType,
    /// The internal representation of the flags.
    flags: FlagsList,
}

impl Flags {
    /// Creates a new, zeroed set of flags.
    pub fn new(ty: FlagsType) -> Self {
        let names = ty.names().len() as u32;
        Self {
            flags: if names > usize::BITS {
                FlagsList::Multiple(Arc::new(vec![0; (((names - 1) / u32::BITS) + 1) as usize]))
            } else {
                FlagsList::Single(0)
            },
            ty,
        }
    }

    /// Gets the value of the flag with the given name.
    pub fn get(&self, name: impl AsRef<str>) -> bool {
        self.get_index(self.index_of(name))
    }

    /// Gets the value of the flag with the given index.
    pub fn get_index(&self, index: usize) -> bool {
        let index = index as u32;
        match &self.flags {
            FlagsList::Single(x) => (*x >> index) == 1,
            FlagsList::Multiple(x) => {
                let arr_index = index / u32::BITS;
                let sub_index = index % u32::BITS;
                (x[arr_index as usize] >> sub_index) == 1
            },
        }
    }

    /// Sets the value of the flag with the given name.
    pub fn set(&mut self, name: impl AsRef<str>, value: bool) {
        self.set_index(self.index_of(name), value)
    }

    /// Sets the value of the flag with the given index.
    pub fn set_index(&mut self, index: usize, value: bool) {
        let index = index as u32;
        match &mut self.flags {
            FlagsList::Single(x) => {
                if value {
                    *x |= 1 << index;
                } else {
                    *x &= !(1 << index);
                }
            },
            FlagsList::Multiple(x) => {
                let list = Arc::make_mut(x);
                let arr_index = index / u32::BITS;
                let sub_index = index % u32::BITS;
                let x = &mut list[arr_index as usize];
                if value {
                    *x |= 1 << sub_index;
                } else {
                    *x &= !(1 << sub_index);
                }
            },
        }
    }

    /// Gets the type of this value.
    pub fn ty(&self) -> FlagsType {
        self.ty.clone()
    }

    /// Creates a new flags list from the provided raw parts.
    pub(crate) fn new_unchecked(ty: FlagsType, flags: FlagsList) -> Self {
        Self { ty, flags }
    }

    /// Gets the list of flags represented as a slice of `u32` values.
    pub(crate) fn as_u32_list(&self) -> &[u32] {
        match &self.flags {
            FlagsList::Single(x) => std::slice::from_ref(x),
            FlagsList::Multiple(x) => x,
        }
    }

    /// Gets the flag index associated with the provided name.
    fn index_of(&self, name: impl AsRef<str>) -> usize {
        *self
            .ty
            .indices
            .get(name.as_ref())
            .expect("Unknown flag name")
    }
}

/// Internally represents a set of bitflags.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(crate) enum FlagsList {
    /// A group of bitflags less than or equal to one `u32` in length.
    Single(u32),
    /// A group of bitflags bigger than a `u32`.
    Multiple(Arc<Vec<u32>>),
}

/// Represents a resource that is owned by the host.
#[derive(Clone, Debug)]
pub struct ResourceOwn {
    /// A tracker that determines when this resource is borrowed or dropped.
    tracker: Arc<AtomicUsize>,
    /// The representation for the resource.
    rep: i32,
    /// The destructor for the resource, if any.
    destructor: Option<wasm_runtime_layer::Func>,
    /// The store with which the resource was created.
    store_id: u64,
    /// The type of the resource.
    ty: ResourceType,
}

impl ResourceOwn {
    /// Creates a new resource for the given value. The value must match the resource type, which must be a host resource type.
    pub fn new<T: 'static + Send + Sync + Sized>(
        mut ctx: impl AsContextMut,
        value: T,
        ty: ResourceType,
    ) -> Result<Self> {
        let mut store_ctx = ctx.as_context_mut();
        let store_id = store_ctx.inner.data().id;
        ensure!(
            ty.valid_for::<T>(store_id),
            "Resource value was of incorrect type."
        );
        let rep = store_ctx
            .inner
            .data_mut()
            .host_resources
            .insert(Box::new(value)) as i32;

        Ok(Self {
            tracker: Arc::default(),
            rep,
            destructor: match ty
                .host_destructor()
                .expect("Could not get host destructor value.")
            {
                Some(x) => Some(x),
                None => store_ctx.inner.data().drop_host_resource.clone(),
            },
            store_id,
            ty,
        })
    }

    /// Creates a new owned resource that is received from a guest.
    pub(crate) fn new_guest(
        rep: i32,
        ty: ResourceType,
        store_id: u64,
        destructor: Option<wasm_runtime_layer::Func>,
    ) -> Self {
        Self {
            tracker: Arc::default(),
            rep,
            destructor,
            ty,
            store_id,
        }
    }

    /// Creates a borrow of this owned resource. The resulting borrow must be manually released via [`ResourceBorrow::drop`] afterward.
    pub fn borrow(&self, ctx: impl crate::AsContextMut) -> Result<ResourceBorrow> {
        ensure!(
            self.store_id == ctx.as_context().inner.data().id,
            "Incorrect store."
        );
        ensure!(
            self.tracker.load(Ordering::Acquire) < usize::MAX,
            "Resource was already destroyed."
        );
        Ok(ResourceBorrow {
            dead: Arc::default(),
            host_tracker: Some(self.tracker.clone()),
            rep: self.rep,
            store_id: self.store_id,
            ty: self.ty.clone(),
        })
    }

    /// Gets the internal representation of this resource. Fails if this is not a host resource, or if the resource was already dropped.
    pub fn rep<'a, T: 'static + Send + Sync, S, E: wasm_runtime_layer::backend::WasmEngine>(
        &self,
        ctx: &'a crate::StoreContext<S, E>,
    ) -> Result<&'a T> {
        ensure!(
            self.store_id == ctx.as_context().inner.data().id,
            "Incorrect store."
        );
        ensure!(
            self.tracker.load(Ordering::Acquire) < usize::MAX,
            "Resource was already destroyed."
        );

        if self.ty.host_destructor().is_some() {
            ctx.inner
                .data()
                .host_resources
                .get(self.rep as usize)
                .expect("Resource was not present.")
                .downcast_ref()
                .context("Resource was not of requested type.")
        } else {
            bail!("Cannot get the representation for a guest-owned resource.");
        }
    }

    /// Gets the type of this value.
    pub fn ty(&self) -> ResourceType {
        self.ty.clone()
    }

    /// Removes this resource from the context without invoking the destructor, and returns the value.
    /// Fails if this is not a host resource, or if the resource is borrowed.
    pub fn take<T: 'static + Send + Sync>(&self, mut ctx: impl crate::AsContextMut) -> Result<()> {
        ensure!(
            self.store_id == ctx.as_context().inner.data().id,
            "Incorrect store."
        );
        ensure!(
            self.tracker.load(Ordering::Acquire) == 0,
            "Resource had remaining borrows or was already dropped."
        );

        ensure!(
            self.ty.host_destructor().is_some(),
            "Resource did not originate from host."
        );

        ensure!(
            ctx.as_context_mut()
                .inner
                .data_mut()
                .host_resources
                .get(self.rep as usize)
                .expect("Resource was not present.")
                .is::<T>(),
            "Resource was of incorrect type."
        );

        *ctx.as_context_mut()
            .inner
            .data_mut()
            .host_resources
            .remove(self.rep as usize)
            .downcast()
            .expect("Could not downcast resource.")
    }

    /// Drops this resource and invokes the destructor, removing it from the context.
    /// Fails if the resource is borrowed or already destroyed.
    pub fn drop(&self, mut ctx: impl crate::AsContextMut) -> Result<()> {
        ensure!(
            self.store_id == ctx.as_context().inner.data().id,
            "Incorrect store."
        );
        ensure!(
            self.tracker.load(Ordering::Acquire) == 0,
            "Resource had remaining borrows or was already dropped."
        );

        if let Some(destructor) = &self.destructor {
            destructor.call(
                ctx.as_context_mut().inner,
                &[wasm_runtime_layer::Value::I32(self.rep)],
                &mut [],
            )?;
        }

        self.tracker.store(usize::MAX, Ordering::Release);
        Ok(())
    }

    /// Lowers this owned resource into a guest context.
    pub(crate) fn lower(&self, ctx: impl crate::AsContextMut) -> Result<i32> {
        ensure!(
            self.store_id == ctx.as_context().inner.data().id,
            "Incorrect store."
        );
        ensure!(
            self.tracker.load(Ordering::Acquire) < usize::MAX,
            "Resource was already destroyed."
        );
        self.tracker.store(usize::MAX, Ordering::Release);
        Ok(self.rep)
    }
}

impl PartialEq for ResourceOwn {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.tracker, &other.tracker)
    }
}

#[cfg(feature = "serde")]
impl Serialize for ResourceOwn {
    fn serialize<S: Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
        use serde::ser::*;
        std::result::Result::Err(S::Error::custom("Cannot serialize resources."))
    }
}

#[cfg(feature = "serde")]
impl<'a> Deserialize<'a> for ResourceOwn {
    fn deserialize<D: Deserializer<'a>>(_: D) -> Result<Self, D::Error> {
        use serde::de::*;
        std::result::Result::Err(D::Error::custom("Cannot deserialize resources."))
    }
}

/// Represents a resource that is borrowed by the host. If this borrow originated from a host-owned resource,
/// then it must be manually released via [`ResourceBorrow::drop`], or the owned resource will be considered
/// borrowed indefinitely.
#[derive(Clone, Debug)]
pub struct ResourceBorrow {
    /// Whether this resource borrow has been destroyed.
    dead: Arc<AtomicBool>,
    /// The original host resource handle, if any.
    host_tracker: Option<Arc<AtomicUsize>>,
    /// The representation of this resource.
    rep: i32,
    /// The store ID of this resource.
    store_id: u64,
    /// The type of this resource.
    ty: ResourceType,
}

impl ResourceBorrow {
    /// Creates a new borrowed resource.
    pub(crate) fn new(rep: i32, store_id: u64, ty: ResourceType) -> Self {
        Self {
            dead: Arc::default(),
            host_tracker: None,
            rep,
            ty,
            store_id,
        }
    }

    /// Gets the internal representation of this resource. Fails if this is not a host resource, or if the resource was already dropped.
    pub fn rep<'a, T: 'static + Send + Sync, S, E: wasm_runtime_layer::backend::WasmEngine>(
        &self,
        ctx: &'a crate::StoreContext<S, E>,
    ) -> Result<&'a T> {
        ensure!(
            self.store_id == ctx.as_context().inner.data().id,
            "Incorrect store."
        );
        ensure!(
            !self.dead.load(Ordering::Acquire),
            "Borrow was already dropped."
        );

        if self.ty.host_destructor().is_some() {
            ctx.inner
                .data()
                .host_resources
                .get(self.rep as usize)
                .expect("Resource was not present.")
                .downcast_ref()
                .context("Resource was not of requested type.")
        } else {
            bail!("Cannot get the representation for a guest-owned resource.");
        }
    }

    /// Gets the type of this value.
    pub fn ty(&self) -> ResourceType {
        self.ty.clone()
    }

    /// Drops this borrow. Fails if this was not a manual borrow of a host resource.
    pub fn drop(&self, ctx: impl crate::AsContextMut) -> Result<()> {
        ensure!(
            self.store_id == ctx.as_context().inner.data().id,
            "Incorrect store."
        );
        ensure!(
            !self.dead.load(Ordering::Acquire),
            "Borrow was already dropped."
        );
        let tracker = self
            .host_tracker
            .as_ref()
            .context("Only host borrows require dropping.")?;
        tracker.fetch_sub(1, Ordering::AcqRel);
        Ok(())
    }

    /// Lowers this borrow into its representation.
    pub(crate) fn lower(&self, ctx: impl crate::AsContextMut) -> Result<i32> {
        ensure!(
            self.store_id == ctx.as_context().inner.data().id,
            "Incorrect store."
        );
        ensure!(
            !self.dead.load(Ordering::Acquire),
            "Borrow was already dropped."
        );
        Ok(self.rep)
    }

    /// Gets a reference to the tracker that determines if this resource is dead.
    pub(crate) fn dead_ref(&self) -> Arc<AtomicBool> {
        self.dead.clone()
    }
}

impl PartialEq for ResourceBorrow {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.dead, &other.dead)
    }
}

#[cfg(feature = "serde")]
impl Serialize for ResourceBorrow {
    fn serialize<S: Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
        use serde::ser::*;
        std::result::Result::Err(S::Error::custom("Cannot serialize resources."))
    }
}

#[cfg(feature = "serde")]
impl<'a> Deserialize<'a> for ResourceBorrow {
    fn deserialize<D: Deserializer<'a>>(_: D) -> Result<Self, D::Error> {
        use serde::de::*;
        std::result::Result::Err(D::Error::custom("Cannot deserialize resources."))
    }
}

/// A type which can convert itself to and from component model values.
pub trait ComponentType: 'static + Sized {
    /// Gets the component model type for instances of `Self`.
    fn ty() -> ValueType;

    /// Attempts to create an instance of `Self` from a component model value.
    fn from_value(value: &Value) -> Result<Self>;

    /// Attempts to convert `Self` into a component model value.
    fn into_value(self) -> Result<Value>;
}

/// Implements the `ComponentType` trait for primitive values.
macro_rules! impl_primitive_component_type {
    ($(($type_name: ident, $enum_name: ident))*) => {
        $(
            impl ComponentType for $type_name {
                fn ty() -> ValueType {
                    ValueType::$enum_name
                }

                fn from_value(value: &Value) -> Result<Self> {
                    Ok(require_matches!(value, Value::$enum_name(x), *x))
                }

                fn into_value(self) -> Result<Value> {
                    Ok(Value::$enum_name(self))
                }
            }
        )*
    };
}

impl_primitive_component_type!((bool, Bool)(i8, S8)(u8, U8)(i16, S16)(u16, U16)(i32, S32)(
    u32, U32
)(i64, S64)(u64, U64)(f32, F32)(f64, F64)(char, Char));

impl ComponentType for String {
    fn ty() -> ValueType {
        ValueType::String
    }

    fn from_value(value: &Value) -> Result<Self> {
        Ok(require_matches!(value, Value::String(x), (**x).into()))
    }

    fn into_value(self) -> Result<Value> {
        Ok(Value::String(self.into()))
    }
}

impl ComponentType for Box<str> {
    fn ty() -> ValueType {
        ValueType::String
    }

    fn from_value(value: &Value) -> Result<Self> {
        Ok(require_matches!(value, Value::String(x), x)
            .to_string()
            .into())
    }

    fn into_value(self) -> Result<Value> {
        Ok(Value::String(self.into()))
    }
}

impl ComponentType for Arc<str> {
    fn ty() -> ValueType {
        ValueType::String
    }

    fn from_value(value: &Value) -> Result<Self> {
        Ok(require_matches!(value, Value::String(x), x).clone())
    }

    fn into_value(self) -> Result<Value> {
        Ok(Value::String(self))
    }
}

impl<T: ComponentType> ComponentType for Option<T> {
    fn ty() -> ValueType {
        ValueType::Option(OptionType::new(T::ty()))
    }

    fn from_value(value: &Value) -> Result<Self> {
        let inner = require_matches!(value, Value::Option(x), x);
        if let Some(val) = &**inner {
            Ok(Some(T::from_value(val)?))
        } else {
            Ok(None)
        }
    }

    fn into_value(self) -> Result<Value> {
        if let Some(val) = self {
            Ok(Value::Option(OptionValue::new(
                OptionType::new(T::ty()),
                Some(T::into_value(val)?),
            )?))
        } else {
            Ok(Value::Option(OptionValue::new(
                OptionType::new(T::ty()),
                None,
            )?))
        }
    }
}

impl<T: ComponentType> ComponentType for Box<T> {
    fn ty() -> ValueType {
        T::ty()
    }

    fn from_value(value: &Value) -> Result<Self> {
        Ok(Box::new(T::from_value(value)?))
    }

    fn into_value(self) -> Result<Value> {
        Ok(T::into_value(*self)?)
    }
}

impl ComponentType for Result<(), ()> {
    fn ty() -> ValueType {
        ValueType::Result(ResultType::new(None, None))
    }

    fn from_value(value: &Value) -> Result<Self> {
        match &**require_matches!(value, Value::Result(x), x) {
            std::result::Result::Ok(None) => Ok(std::result::Result::Ok(())),
            std::result::Result::Err(None) => Ok(std::result::Result::Err(())),
            _ => bail!("Incorrect result type."),
        }
    }

    fn into_value(self) -> Result<Value> {
        match self {
            std::result::Result::Ok(()) => Ok(Value::Result(ResultValue::new(
                ResultType::new(None, None),
                std::result::Result::Ok(None),
            )?)),
            std::result::Result::Err(()) => Ok(Value::Result(ResultValue::new(
                ResultType::new(None, None),
                std::result::Result::Err(None),
            )?)),
        }
    }
}

impl<T: ComponentType> ComponentType for Result<T, ()> {
    fn ty() -> ValueType {
        ValueType::Result(ResultType::new(Some(T::ty()), None))
    }

    fn from_value(value: &Value) -> Result<Self> {
        match &**require_matches!(value, Value::Result(x), x) {
            std::result::Result::Ok(Some(x)) => Ok(std::result::Result::Ok(T::from_value(x)?)),
            std::result::Result::Err(None) => Ok(std::result::Result::Err(())),
            _ => bail!("Incorrect result type."),
        }
    }

    fn into_value(self) -> Result<Value> {
        match self {
            std::result::Result::Ok(x) => Ok(Value::Result(ResultValue::new(
                ResultType::new(Some(T::ty()), None),
                std::result::Result::Ok(Some(T::into_value(x)?)),
            )?)),
            std::result::Result::Err(()) => Ok(Value::Result(ResultValue::new(
                ResultType::new(Some(T::ty()), None),
                std::result::Result::Err(None),
            )?)),
        }
    }
}

impl<T: ComponentType> ComponentType for Result<(), T> {
    fn ty() -> ValueType {
        ValueType::Result(ResultType::new(None, Some(T::ty())))
    }

    fn from_value(value: &Value) -> Result<Self> {
        match &**require_matches!(value, Value::Result(x), x) {
            std::result::Result::Ok(None) => Ok(std::result::Result::Ok(())),
            std::result::Result::Err(Some(v)) => Ok(std::result::Result::Err(T::from_value(v)?)),
            _ => bail!("Incorrect result type."),
        }
    }

    fn into_value(self) -> Result<Value> {
        match self {
            std::result::Result::Ok(()) => Ok(Value::Result(ResultValue::new(
                ResultType::new(None, Some(T::ty())),
                std::result::Result::Ok(None),
            )?)),
            std::result::Result::Err(v) => Ok(Value::Result(ResultValue::new(
                ResultType::new(None, Some(T::ty())),
                std::result::Result::Err(Some(T::into_value(v)?)),
            )?)),
        }
    }
}

impl<U: ComponentType, V: ComponentType> ComponentType for Result<U, V> {
    fn ty() -> ValueType {
        ValueType::Result(ResultType::new(Some(U::ty()), Some(V::ty())))
    }

    fn from_value(value: &Value) -> Result<Self> {
        match &**require_matches!(value, Value::Result(x), x) {
            std::result::Result::Ok(Some(u)) => Ok(std::result::Result::Ok(U::from_value(u)?)),
            std::result::Result::Err(Some(v)) => Ok(std::result::Result::Err(V::from_value(v)?)),
            _ => bail!("Incorrect result type."),
        }
    }

    fn into_value(self) -> Result<Value> {
        match self {
            std::result::Result::Ok(u) => Ok(Value::Result(ResultValue::new(
                ResultType::new(Some(U::ty()), Some(V::ty())),
                std::result::Result::Ok(Some(U::into_value(u)?)),
            )?)),
            std::result::Result::Err(v) => Ok(Value::Result(ResultValue::new(
                ResultType::new(Some(U::ty()), Some(V::ty())),
                std::result::Result::Err(Some(V::into_value(v)?)),
            )?)),
        }
    }
}

impl<T: ComponentType> ComponentType for Vec<T> {
    fn ty() -> ValueType {
        ValueType::List(ListType::new(T::ty()))
    }

    fn from_value(value: &Value) -> Result<Self> {
        let list = require_matches!(value, Value::List(x), x);

        let id = TypeId::of::<T>();
        Ok(if id == TypeId::of::<bool>() {
            *(Box::new(require_matches!(&list.values, ListSpecialization::Bool(x), x).to_vec())
                as Box<dyn Any>)
                .downcast()
                .expect("Could not downcast vector.")
        } else if id == TypeId::of::<i8>() {
            *(Box::new(require_matches!(&list.values, ListSpecialization::S8(x), x).to_vec())
                as Box<dyn Any>)
                .downcast()
                .expect("Could not downcast vector.")
        } else if id == TypeId::of::<u8>() {
            *(Box::new(require_matches!(&list.values, ListSpecialization::U8(x), x).to_vec())
                as Box<dyn Any>)
                .downcast()
                .expect("Could not downcast vector.")
        } else if id == TypeId::of::<i16>() {
            *(Box::new(require_matches!(&list.values, ListSpecialization::S16(x), x).to_vec())
                as Box<dyn Any>)
                .downcast()
                .expect("Could not downcast vector.")
        } else if id == TypeId::of::<u16>() {
            *(Box::new(require_matches!(&list.values, ListSpecialization::U16(x), x).to_vec())
                as Box<dyn Any>)
                .downcast()
                .expect("Could not downcast vector.")
        } else if id == TypeId::of::<i32>() {
            *(Box::new(require_matches!(&list.values, ListSpecialization::S32(x), x).to_vec())
                as Box<dyn Any>)
                .downcast()
                .expect("Could not downcast vector.")
        } else if id == TypeId::of::<u32>() {
            *(Box::new(require_matches!(&list.values, ListSpecialization::U32(x), x).to_vec())
                as Box<dyn Any>)
                .downcast()
                .expect("Could not downcast vector.")
        } else if id == TypeId::of::<i64>() {
            *(Box::new(require_matches!(&list.values, ListSpecialization::S64(x), x).to_vec())
                as Box<dyn Any>)
                .downcast()
                .expect("Could not downcast vector.")
        } else if id == TypeId::of::<u64>() {
            *(Box::new(require_matches!(&list.values, ListSpecialization::U64(x), x).to_vec())
                as Box<dyn Any>)
                .downcast()
                .expect("Could not downcast vector.")
        } else if id == TypeId::of::<f32>() {
            *(Box::new(require_matches!(&list.values, ListSpecialization::F32(x), x).to_vec())
                as Box<dyn Any>)
                .downcast()
                .expect("Could not downcast vector.")
        } else if id == TypeId::of::<f64>() {
            *(Box::new(require_matches!(&list.values, ListSpecialization::F64(x), x).to_vec())
                as Box<dyn Any>)
                .downcast()
                .expect("Could not downcast vector.")
        } else if id == TypeId::of::<char>() {
            *(Box::new(require_matches!(&list.values, ListSpecialization::Char(x), x).to_vec())
                as Box<dyn Any>)
                .downcast()
                .expect("Could not downcast vector.")
        } else {
            require_matches!(&list.values, ListSpecialization::Other(x), x)
                .iter()
                .map(|x| T::from_value(x))
                .collect::<Result<_>>()?
        })
    }

    fn into_value(self) -> Result<Value> {
        let holder = Box::new(self) as Box<dyn Any>;
        let id = TypeId::of::<T>();
        Ok(Value::List(if id == TypeId::of::<bool>() {
            List {
                ty: ListType::new(ValueType::Bool),
                values: ListSpecialization::Bool(Arc::from(
                    *(holder)
                        .downcast::<Vec<bool>>()
                        .expect("Could not downcast vector."),
                )),
            }
        } else if id == TypeId::of::<i8>() {
            List {
                ty: ListType::new(ValueType::S8),
                values: ListSpecialization::S8(Arc::from(
                    *(holder)
                        .downcast::<Vec<i8>>()
                        .expect("Could not downcast vector."),
                )),
            }
        } else if id == TypeId::of::<u8>() {
            List {
                ty: ListType::new(ValueType::U8),
                values: ListSpecialization::U8(Arc::from(
                    *(holder)
                        .downcast::<Vec<u8>>()
                        .expect("Could not downcast vector."),
                )),
            }
        } else if id == TypeId::of::<i16>() {
            List {
                ty: ListType::new(ValueType::S16),
                values: ListSpecialization::S16(Arc::from(
                    *(holder)
                        .downcast::<Vec<i16>>()
                        .expect("Could not downcast vector."),
                )),
            }
        } else if id == TypeId::of::<u16>() {
            List {
                ty: ListType::new(ValueType::U16),
                values: ListSpecialization::U16(Arc::from(
                    *(holder)
                        .downcast::<Vec<u16>>()
                        .expect("Could not downcast vector."),
                )),
            }
        } else if id == TypeId::of::<i32>() {
            List {
                ty: ListType::new(ValueType::S32),
                values: ListSpecialization::S32(Arc::from(
                    *(holder)
                        .downcast::<Vec<i32>>()
                        .expect("Could not downcast vector."),
                )),
            }
        } else if id == TypeId::of::<u32>() {
            List {
                ty: ListType::new(ValueType::U32),
                values: ListSpecialization::U32(Arc::from(
                    *(holder)
                        .downcast::<Vec<u32>>()
                        .expect("Could not downcast vector."),
                )),
            }
        } else if id == TypeId::of::<i64>() {
            List {
                ty: ListType::new(ValueType::S64),
                values: ListSpecialization::S64(Arc::from(
                    *(holder)
                        .downcast::<Vec<i64>>()
                        .expect("Could not downcast vector."),
                )),
            }
        } else if id == TypeId::of::<u64>() {
            List {
                ty: ListType::new(ValueType::U64),
                values: ListSpecialization::U64(Arc::from(
                    *(holder)
                        .downcast::<Vec<u64>>()
                        .expect("Could not downcast vector."),
                )),
            }
        } else if id == TypeId::of::<f32>() {
            List {
                ty: ListType::new(ValueType::F32),
                values: ListSpecialization::F32(Arc::from(
                    *(holder)
                        .downcast::<Vec<f32>>()
                        .expect("Could not downcast vector."),
                )),
            }
        } else if id == TypeId::of::<f64>() {
            List {
                ty: ListType::new(ValueType::F64),
                values: ListSpecialization::F64(Arc::from(
                    *(holder)
                        .downcast::<Vec<f64>>()
                        .expect("Could not downcast vector."),
                )),
            }
        } else if id == TypeId::of::<char>() {
            List {
                ty: ListType::new(ValueType::Char),
                values: ListSpecialization::Char(Arc::from(
                    *(holder)
                        .downcast::<Vec<char>>()
                        .expect("Could not downcast vector."),
                )),
            }
        } else {
            List {
                ty: ListType::new(T::ty()),
                values: ListSpecialization::Other(
                    holder
                        .downcast::<Vec<T>>()
                        .expect("Could not downcast vector.")
                        .into_iter()
                        .map(|x| x.into_value())
                        .collect::<Result<_>>()?,
                ),
            }
        }))
    }
}

/// Implements `ComponentType` for tuples
macro_rules! tuple_impl {
    ($($idx: tt $ty: ident), *) => {
        impl<$($ty: ComponentType),*> ComponentType for ($($ty,)*) {
            fn ty() -> ValueType {
                ValueType::Tuple(TupleType::new(None, [$(<$ty as ComponentType>::ty(),)*]))
            }

            fn from_value(value: &Value) -> Result<Self> {
                Ok(require_matches!(
                    value,
                    Value::Tuple(x),
                    ($(<$ty as ComponentType>::from_value(&x.fields[$idx])?,)*)
                ))
            }

            fn into_value(self) -> Result<Value> {
                Ok(Value::Tuple(Tuple::new(
                    TupleType::new(None, [$(<$ty as ComponentType>::ty(),)*]),
                    [$(<$ty as ComponentType>::into_value(self.$idx)?,)*]
                )?))
            }
        }
    };
}

tuple_impl! { 0 A }
tuple_impl! { 0 A, 1 B }
tuple_impl! { 0 A, 1 B, 2 C }
tuple_impl! { 0 A, 1 B, 2 C, 3 D }
tuple_impl! { 0 A, 1 B, 2 C, 3 D, 4 E }
tuple_impl! { 0 A, 1 B, 2 C, 3 D, 4 E, 5 F }
tuple_impl! { 0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G }
tuple_impl! { 0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H }

/// Specialization of a non-tuple component type for disambiguation between multi-value and tuple.
pub trait UnaryComponentType: ComponentType {}

/// Implements `UnaryComponentType` for a group of types
macro_rules! impl_unary {
    ($([$($param: ident: $bound: ident),*] $ty: ty,)*) => {
        $( impl<$($param: $bound),*> UnaryComponentType for $ty {} )*
    };
}

impl_unary!(
    [] bool,
    [] i8,
    [] u8,
    [] i16,
    [] u16,
    [] i32,
    [] u32,
    [] i64,
    [] u64,
    [] f32,
    [] f64,
    [] char,
    [] String,
    [] Box<str>,
    [] Arc<str>,
    [T: ComponentType] Option<T>,
    [T: ComponentType] Box<T>,
    [T: ComponentType] Vec<T>,
    [T: ComponentType, U: ComponentType] Result<T, U>,
    [U: ComponentType] Result<(), U>,
    [T: ComponentType] Result<T, ()>,
);

/// A module used to hide traits that are implementation details.
mod private {
    use super::*;

    /// The inner backing for a list, specialized over primitive types for efficient access.
    #[derive(Clone, Debug, PartialEq)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub enum ListSpecialization {
        /// A list of booleans.
        Bool(Arc<[bool]>),
        /// A list of eight-bit signed integers.
        S8(#[cfg_attr(feature = "serde", serde(with = "serialize_specialized"))] Arc<[i8]>),
        /// A list of eight-bit unsigned integers.
        U8(#[cfg_attr(feature = "serde", serde(with = "serialize_specialized"))] Arc<[u8]>),
        /// A list of 16-bit signed integers.
        S16(#[cfg_attr(feature = "serde", serde(with = "serialize_specialized"))] Arc<[i16]>),
        /// A list of 16-bit unsigned integers.
        U16(#[cfg_attr(feature = "serde", serde(with = "serialize_specialized"))] Arc<[u16]>),
        /// A list of 32-bit signed integers.
        S32(#[cfg_attr(feature = "serde", serde(with = "serialize_specialized"))] Arc<[i32]>),
        /// A list of 32-bit unsigned integers.
        U32(#[cfg_attr(feature = "serde", serde(with = "serialize_specialized"))] Arc<[u32]>),
        /// A list of 64-bit signed integers.
        S64(#[cfg_attr(feature = "serde", serde(with = "serialize_specialized"))] Arc<[i64]>),
        /// A list of 64-bit unsigned integers.
        U64(#[cfg_attr(feature = "serde", serde(with = "serialize_specialized"))] Arc<[u64]>),
        /// A list of 32-bit floating point numbers.
        F32(#[cfg_attr(feature = "serde", serde(with = "serialize_specialized"))] Arc<[f32]>),
        /// A list of 64-bit floating point numbers.
        F64(#[cfg_attr(feature = "serde", serde(with = "serialize_specialized"))] Arc<[f64]>),
        /// A list of characters.
        Char(Arc<[char]>),
        /// A list of other, non-specialized values.
        Other(Arc<[Value]>),
    }

    #[cfg(feature = "serde")]
    /// Allows a list specialization to be serialized in the most efficient way possible.
    mod serialize_specialized {
        use super::*;

        /// Serializes a list specialization in the most efficient way possible.
        pub fn serialize<S: Serializer, A: Pod>(
            value: &Arc<[A]>,
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            if cfg!(target_endian = "little") || size_of::<A>() == 1 {
                serializer.serialize_bytes(cast_slice(value))
            } else {
                let mut bytes = cast_slice::<_, u8>(value).to_vec();

                for chunk in bytes.chunks_exact_mut(size_of::<A>()) {
                    chunk.reverse();
                }

                serializer.serialize_bytes(&bytes)
            }
        }

        /// Deserializes a list specialization in the most efficient way possible.
        pub fn deserialize<'a, D: Deserializer<'a>, A: Pod>(
            deserializer: D,
        ) -> Result<Arc<[A]>, D::Error> {
            use serde::de::*;

            let mut byte_data = Arc::<[u8]>::deserialize(deserializer)?;

            if !(cfg!(target_endian = "little") || size_of::<A>() == 1) {
                for chunk in Arc::get_mut(&mut byte_data)
                    .expect("Could not get exclusive reference.")
                    .chunks_exact_mut(size_of::<A>())
                {
                    chunk.reverse();
                }
            }

            try_cast_slice_arc(byte_data).map_err(|(x, _)| D::Error::custom(x))
        }
    }

    impl<'a> IntoIterator for &'a ListSpecialization {
        type Item = Value;

        type IntoIter = ListSpecializationIter<'a>;

        fn into_iter(self) -> Self::IntoIter {
            match self {
                ListSpecialization::Bool(x) => ListSpecializationIter::Bool(x.iter()),
                ListSpecialization::S8(x) => ListSpecializationIter::S8(x.iter()),
                ListSpecialization::U8(x) => ListSpecializationIter::U8(x.iter()),
                ListSpecialization::S16(x) => ListSpecializationIter::S16(x.iter()),
                ListSpecialization::U16(x) => ListSpecializationIter::U16(x.iter()),
                ListSpecialization::S32(x) => ListSpecializationIter::S32(x.iter()),
                ListSpecialization::U32(x) => ListSpecializationIter::U32(x.iter()),
                ListSpecialization::S64(x) => ListSpecializationIter::S64(x.iter()),
                ListSpecialization::U64(x) => ListSpecializationIter::U64(x.iter()),
                ListSpecialization::F32(x) => ListSpecializationIter::F32(x.iter()),
                ListSpecialization::F64(x) => ListSpecializationIter::F64(x.iter()),
                ListSpecialization::Char(x) => ListSpecializationIter::Char(x.iter()),
                ListSpecialization::Other(x) => ListSpecializationIter::Other(x.iter()),
            }
        }
    }

    /// An iterator over specialized list values that yields `Value`s.
    pub enum ListSpecializationIter<'a> {
        /// An iterator over booleans.
        Bool(std::slice::Iter<'a, bool>),
        /// An iterator over eight-bit signed integers.
        S8(std::slice::Iter<'a, i8>),
        /// An iterator over eight-bit unsigned integers.
        U8(std::slice::Iter<'a, u8>),
        /// An iterator over 16-bit signed integers.
        S16(std::slice::Iter<'a, i16>),
        /// An iterator over 16-bit unsigned integers.
        U16(std::slice::Iter<'a, u16>),
        /// An iterator over 32-bit signed integers.
        S32(std::slice::Iter<'a, i32>),
        /// An iterator over 32-bit unsigned integers.
        U32(std::slice::Iter<'a, u32>),
        /// An iterator over 64-bit signed integers.
        S64(std::slice::Iter<'a, i64>),
        /// An iterator over 64-bit unsigned integers.
        U64(std::slice::Iter<'a, u64>),
        /// An iterator over 32-bit floating point numbers.
        F32(std::slice::Iter<'a, f32>),
        /// An iterator over 64-bit floating point numbers.
        F64(std::slice::Iter<'a, f64>),
        /// An iterator over characters.
        Char(std::slice::Iter<'a, char>),
        /// An iterator over unspecialized values.
        Other(std::slice::Iter<'a, Value>),
    }

    impl<'a> Iterator for ListSpecializationIter<'a> {
        type Item = Value;

        fn next(&mut self) -> Option<Self::Item> {
            Some(match self {
                ListSpecializationIter::Bool(x) => Value::from(x.next()?),
                ListSpecializationIter::S8(x) => Value::from(x.next()?),
                ListSpecializationIter::U8(x) => Value::from(x.next()?),
                ListSpecializationIter::S16(x) => Value::from(x.next()?),
                ListSpecializationIter::U16(x) => Value::from(x.next()?),
                ListSpecializationIter::S32(x) => Value::from(x.next()?),
                ListSpecializationIter::U32(x) => Value::from(x.next()?),
                ListSpecializationIter::S64(x) => Value::from(x.next()?),
                ListSpecializationIter::U64(x) => Value::from(x.next()?),
                ListSpecializationIter::F32(x) => Value::from(x.next()?),
                ListSpecializationIter::F64(x) => Value::from(x.next()?),
                ListSpecializationIter::Char(x) => Value::from(x.next()?),
                ListSpecializationIter::Other(x) => x.next()?.clone(),
            })
        }
    }

    /// Denotes a type that can be stored in a specialized list contiguously.
    pub trait ListPrimitive: Copy + Sized {
        /// Creates a list specialization from a reference to a slice of this kind of value.
        fn from_arc(arc: Arc<[Self]>) -> ListSpecialization;
        /// Attempts to create a list specialization from an iterator over this kind of value.
        fn from_value_iter(iter: impl IntoIterator<Item = Value>) -> Result<ListSpecialization>;
        /// Gets the slice of primitive values of this type from the given list, or panics.
        fn from_specialization(specialization: &ListSpecialization) -> &[Self];
        /// Gets the type of this value.
        fn ty() -> ValueType;
    }

    /// Implements the `ListPrimitive` trait for a primitive type.
    macro_rules! impl_list_primitive {
        ($(($type_name: ident, $enum_name: ident))*) => {
            $(
                impl ListPrimitive for $type_name {
                    fn from_arc(arc: Arc<[Self]>) -> ListSpecialization {
                        ListSpecialization::$enum_name(arc)
                    }

                    fn from_value_iter(iter: impl IntoIterator<Item = Value>) -> Result<ListSpecialization> {
                        let values: Arc<[Self]> = iter.into_iter().map(|x| TryInto::try_into(&x)).collect::<Result<_>>()?;
                        Ok(ListSpecialization::$enum_name(values))
                    }

                    fn from_specialization(specialization: &ListSpecialization) -> &[Self] {
                        if let ListSpecialization::$enum_name(vals) = specialization {
                            &vals
                        }
                        else {
                            panic!("Incorrect specialization type.");
                        }
                    }

                    fn ty() -> ValueType {
                        ValueType::$enum_name
                    }
                }
            )*
        };
    }

    impl_list_primitive!((bool, Bool)(i8, S8)(u8, U8)(i16, S16)(u16, U16)(i32, S32)(
        u32, U32
    )(i64, S64)(u64, U64)(f32, F32)(f64, F64)(char, Char));
}
