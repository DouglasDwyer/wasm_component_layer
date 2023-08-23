use std::any::*;
use std::marker::*;

use std::ops::*;
use std::sync::atomic::*;
use std::sync::*;

use anyhow::*;
use once_cell::sync::*;
use private::*;

use crate::AsContextMut;
use crate::require_matches::require_matches;
use crate::types::*;
use crate::AsContext;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    S8(i8),
    U8(u8),
    S16(i16),
    U16(u16),
    S32(i32),
    U32(u32),
    S64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
    Char(char),
    String(Arc<str>),
    List(List),
    Record(Record),
    Tuple(Tuple),
    Variant(Variant),
    Enum(Enum),
    Union(Union),
    Option(OptionValue),
    Result(ResultValue),
    Flags(Flags),
    Own(ResourceOwn),
    Borrow(ResourceBorrow),
}

impl Value {
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
            Value::Union(x) => ValueType::Union(x.ty()),
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

#[derive(Clone, Debug)]
pub struct List {
    values: ListSpecialization,
    ty: ListType,
}

impl List {
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

    pub fn ty(&self) -> ListType {
        self.ty.clone()
    }

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

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

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

#[derive(Clone, Debug)]
pub struct Record {
    fields: Arc<[(Arc<str>, Value)]>,
    ty: RecordType,
}

impl Record {
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

    pub fn from_fields<S: Into<Arc<str>>>(
        values: impl IntoIterator<Item = (S, Value)>,
    ) -> Result<Self> {
        let mut fields = values
            .into_iter()
            .map(|(name, val)| (Into::<Arc<str>>::into(name), val))
            .collect::<Arc<_>>();
        Arc::get_mut(&mut fields)
            .expect("Could not get exclusive reference.")
            .sort_by(|a, b| a.0.cmp(&b.0));
        let ty = RecordType::new_sorted(fields.iter().map(|(name, val)| (name.clone(), val.ty())))?;
        Ok(Self { fields, ty })
    }

    pub fn fields(&self) -> impl ExactSizeIterator<Item = (&str, Value)> {
        self.fields.iter().map(|(name, val)| (&**name, val.clone()))
    }

    pub fn ty(&self) -> RecordType {
        self.ty.clone()
    }

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

#[derive(Clone, Debug)]
pub struct Tuple {
    fields: Arc<[Value]>,
    ty: TupleType,
}

impl Tuple {
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

    pub fn from_fields(fields: impl IntoIterator<Item = Value>) -> Self {
        let fields: Arc<_> = fields.into_iter().collect();
        let ty = TupleType::new(fields.iter().map(|x| x.ty()));
        Self { fields, ty }
    }

    pub fn ty(&self) -> TupleType {
        self.ty.clone()
    }

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

#[derive(Clone, Debug, PartialEq)]
pub struct Variant {
    discriminant: u32,
    value: Option<Arc<Value>>,
    ty: VariantType,
}

impl Variant {
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
            value: value.map(|x| Arc::new(x)),
            ty,
        })
    }

    pub fn discriminant(&self) -> usize {
        self.discriminant as usize
    }

    pub fn value(&self) -> Option<Value> {
        self.value.as_ref().map(|x| (**x).clone())
    }

    pub fn ty(&self) -> VariantType {
        self.ty.clone()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Enum {
    discriminant: u32,
    ty: EnumType,
}

impl Enum {
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

    pub fn discriminant(&self) -> usize {
        self.discriminant as usize
    }

    pub fn ty(&self) -> EnumType {
        self.ty.clone()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Union {
    discriminant: u32,
    value: Arc<Value>,
    ty: UnionType,
}

impl Union {
    pub fn new(ty: UnionType, discriminant: usize, value: Value) -> Result<Self> {
        ensure!(
            discriminant < ty.cases().len(),
            "Discriminant out-of-range."
        );
        ensure!(
            ty.cases()[discriminant] == value.ty(),
            "Provided value was of incorrect type."
        );

        Ok(Self {
            discriminant: discriminant as u32,
            value: Arc::new(value),
            ty,
        })
    }

    pub fn discriminant(&self) -> usize {
        self.discriminant as usize
    }

    pub fn value(&self) -> Value {
        (*self.value).clone()
    }

    pub fn ty(&self) -> UnionType {
        self.ty.clone()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OptionValue {
    ty: OptionType,
    value: Arc<Option<Value>>,
}

impl OptionValue {
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

    pub fn ty(&self) -> OptionType {
        self.ty.clone()
    }
}

impl Deref for OptionValue {
    type Target = Option<Value>;

    fn deref(&self) -> &Self::Target {
        &*self.value
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResultValue {
    ty: ResultType,
    value: Arc<Result<Option<Value>, Option<Value>>>,
}

impl ResultValue {
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

    pub fn ty(&self) -> ResultType {
        self.ty.clone()
    }
}

impl Deref for ResultValue {
    type Target = Result<Option<Value>, Option<Value>>;

    fn deref(&self) -> &Self::Target {
        &*self.value
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Flags {
    ty: FlagsType,
    flags: FlagsList,
}

impl Flags {
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

    pub fn get(&self, name: impl AsRef<str>) -> bool {
        self.get_index(self.index_of(name))
    }

    pub fn get_index(&self, index: usize) -> bool {
        let index = index as u32;
        match &self.flags {
            FlagsList::Single(x) => {
                if (*x >> index) == 1 {
                    true
                } else {
                    false
                }
            }
            FlagsList::Multiple(x) => {
                let arr_index = index / u32::BITS;
                let sub_index = index % u32::BITS;
                if (x[arr_index as usize] >> sub_index) == 1 {
                    true
                } else {
                    false
                }
            }
        }
    }

    pub fn set(&mut self, name: impl AsRef<str>, value: bool) {
        self.set_index(self.index_of(name), value)
    }

    pub fn set_index(&mut self, index: usize, value: bool) {
        let index = index as u32;
        match &mut self.flags {
            FlagsList::Single(x) => {
                if value {
                    *x |= 1 << index;
                } else {
                    *x &= !(1 << index);
                }
            }
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
            }
        }
    }

    pub fn ty(&self) -> FlagsType {
        self.ty.clone()
    }

    pub(crate) fn new_unchecked(ty: FlagsType, flags: FlagsList) -> Self {
        Self { ty, flags }
    }

    pub(crate) fn as_u32_list(&self) -> &[u32] {
        match &self.flags {
            FlagsList::Single(x) => std::slice::from_ref(x),
            FlagsList::Multiple(x) => &**x,
        }
    }

    fn index_of(&self, name: impl AsRef<str>) -> usize {
        *self
            .ty
            .indices
            .get(name.as_ref())
            .expect("Unknown flag name")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FlagsList {
    Single(u32),
    Multiple(Arc<Vec<u32>>),
}

#[derive(Clone, Debug)]
pub struct ResourceOwn {
    tracker: Arc<AtomicUsize>,
    rep: i32,
    destructor: Option<wasm_runtime_layer::Func>,
    store_id: u64,
    ty: ResourceType,
}

impl ResourceOwn {
    pub fn new<T: 'static + Send + Sync + Sized>(mut ctx: impl AsContextMut, value: T, ty: ResourceType) -> Result<Self> {
        let mut store_ctx = ctx.as_context_mut();
        let store_id = store_ctx.inner.data().id;
        ensure!(ty.valid_for::<T>(store_id), "Resource value was of incorrect type.");
        let rep = store_ctx.inner.data_mut().host_resources.insert(Box::new(value)) as i32;

        Ok(Self {
            tracker: Arc::default(),
            rep,
            destructor: None,
            store_id,
            ty,
        })
    }

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
            store_id
        }
    }

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

    pub fn rep(&self, ctx: impl AsContext) -> Result<i32> {
        ensure!(
            self.store_id == ctx.as_context().inner.data().id,
            "Incorrect store."
        );
        Ok(self.rep)
    }

    pub fn ty(&self) -> ResourceType {
        self.ty.clone()
    }

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
        } else if let Some(Some(destructor)) = self.ty.host_destructor() {
            destructor.call(
                ctx.as_context_mut().inner,
                &[wasm_runtime_layer::Value::I32(self.rep)],
                &mut [],
            )?;
        }

        self.tracker.store(usize::MAX, Ordering::Release);
        Ok(())
    }

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

#[derive(Clone, Debug)]
pub struct ResourceBorrow {
    dead: Arc<AtomicBool>,
    host_tracker: Option<Arc<AtomicUsize>>,
    rep: i32,
    store_id: u64,
    ty: ResourceType,
}

impl ResourceBorrow {
    pub(crate) fn new(rep: i32, store_id: u64, ty: ResourceType) -> Self {
        Self {
            dead: Arc::default(),
            host_tracker: None,
            rep,
            ty,
            store_id
        }
    }

    pub fn rep(&self, ctx: impl AsContext) -> Result<i32> {
        ensure!(
            self.store_id == ctx.as_context().inner.data().id,
            "Incorrect store."
        );
        Ok(self.rep)
    }

    pub fn ty(&self) -> ResourceType {
        self.ty.clone()
    }

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

    pub(crate) fn dead_ref(&self) -> Arc<AtomicBool> {
        self.dead.clone()
    }
}

impl PartialEq for ResourceBorrow {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.dead, &other.dead)
    }
}

pub trait ComponentType: 'static + Sized {
    fn ty() -> ValueType;

    fn from_value(value: &Value) -> Result<Self>;

    fn into_value(self) -> Result<Value>;
}

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
        Ok(require_matches!(value, Value::String(x), x).to_string().into())
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

struct OptionTypeVal<T: ComponentType>(T);

impl<T: ComponentType> OptionTypeVal<T> {
    fn ty() -> OptionType {
        static TY: OnceCell<OptionType> = OnceCell::new();
        TY.get_or_init(|| OptionType::new(T::ty())).clone()
    }
}

impl<T: ComponentType> ComponentType for Option<T> {
    fn ty() -> ValueType {
        ValueType::Option(OptionTypeVal::<T>::ty())
    }

    fn from_value(value: &Value) -> Result<Self> {
        let inner = require_matches!(value, Value::Option(x), x);
        if let Some(val) = &**inner {
            Ok(Some(T::from_value(val)?))
        }
        else {
            Ok(None)
        }
    }

    fn into_value(self) -> Result<Value> {
        if let Some(val) = self {
            Ok(Value::Option(OptionValue::new(OptionTypeVal::<T>::ty(), Some(T::into_value(val)?))?))
        }
        else {
            Ok(Value::Option(OptionValue::new(OptionTypeVal::<T>::ty(), None)?))
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

struct ResultTypeValOk<T: ComponentType>(T);

impl<T: ComponentType> ResultTypeValOk<T> {
    fn ty() -> ResultType {
        static TY: OnceCell<ResultType> = OnceCell::new();
        TY.get_or_init(|| ResultType::new(Some(T::ty()), None)).clone()
    }
}

impl<T: ComponentType> ComponentType for Result<T, ()> {
    fn ty() -> ValueType {
        ValueType::Result(ResultTypeValOk::<T>::ty())
    }

    fn from_value(value: &Value) -> Result<Self> {
        match &**require_matches!(value, Value::Result(x), x) {
            std::result::Result::Ok(Some(x)) => Ok(std::result::Result::Ok(T::from_value(x)?)),
            std::result::Result::Err(None) => Ok(std::result::Result::Err(())),
            _ => bail!("Incorrect result type.")
        }
    }

    fn into_value(self) -> Result<Value> {
        match self {
            std::result::Result::Ok(x) => Ok(Value::Result(ResultValue::new(ResultTypeValOk::<T>::ty(), std::result::Result::Ok(Some(T::into_value(x)?)))?)),
            std::result::Result::Err(()) => Ok(Value::Result(ResultValue::new(ResultTypeValOk::<T>::ty(), std::result::Result::Err(None))?))
        }
    }
}

struct ResultTypeValErr<T: ComponentType>(T);

impl<T: ComponentType> ResultTypeValErr<T> {
    fn ty() -> ResultType {
        static TY: OnceCell<ResultType> = OnceCell::new();
        TY.get_or_init(|| ResultType::new(None, Some(T::ty()))).clone()
    }
}

impl<T: ComponentType> ComponentType for Result<(), T> {
    fn ty() -> ValueType {
        ValueType::Result(ResultTypeValErr::<T>::ty())
    }

    fn from_value(value: &Value) -> Result<Self> {
        match &**require_matches!(value, Value::Result(x), x) {
            std::result::Result::Ok(None) => Ok(std::result::Result::Ok(())),
            std::result::Result::Err(Some(v)) => Ok(std::result::Result::Err(T::from_value(v)?)),
            _ => bail!("Incorrect result type.")
        }
    }

    fn into_value(self) -> Result<Value> {
        match self {
            std::result::Result::Ok(()) => Ok(Value::Result(ResultValue::new(ResultTypeValErr::<T>::ty(), std::result::Result::Ok(None))?)),
            std::result::Result::Err(v) => Ok(Value::Result(ResultValue::new(ResultTypeValErr::<T>::ty(), std::result::Result::Err(Some(T::into_value(v)?)))?))
        }
    }
}

struct ResultTypeValTwo<U: ComponentType, V: ComponentType>(U, V);

impl<U: ComponentType, V: ComponentType> ResultTypeValTwo<U, V> {
    fn ty() -> ResultType {
        static TY: OnceCell<ResultType> = OnceCell::new();
        TY.get_or_init(|| ResultType::new(Some(U::ty()), Some(V::ty()))).clone()
    }
}

impl<U: ComponentType, V: ComponentType> ComponentType for Result<U, V> {
    fn ty() -> ValueType {
        ValueType::Result(ResultTypeValTwo::<U, V>::ty())
    }

    fn from_value(value: &Value) -> Result<Self> {
        match &**require_matches!(value, Value::Result(x), x) {
            std::result::Result::Ok(Some(u)) => Ok(std::result::Result::Ok(U::from_value(u)?)),
            std::result::Result::Err(Some(v)) => Ok(std::result::Result::Err(V::from_value(v)?)),
            _ => bail!("Incorrect result type.")
        }
    }

    fn into_value(self) -> Result<Value> {
        match self {
            std::result::Result::Ok(u) => Ok(Value::Result(ResultValue::new(ResultTypeValTwo::<U, V>::ty(), std::result::Result::Ok(Some(U::into_value(u)?)))?)),
            std::result::Result::Err(v) => Ok(Value::Result(ResultValue::new(ResultTypeValTwo::<U, V>::ty(), std::result::Result::Err(Some(V::into_value(v)?)))?))
        }
    }
}

impl<T: ComponentType> ComponentType for Vec<T> {
    fn ty() -> ValueType {
        static TY: OnceCell<ValueType> = OnceCell::new();
        TY.get_or_init(|| ValueType::List(ListType::new(T::ty()))).clone()
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
                .into_iter()
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

mod private {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    pub enum ListSpecialization {
        Bool(Arc<[bool]>),
        S8(Arc<[i8]>),
        U8(Arc<[u8]>),
        S16(Arc<[i16]>),
        U16(Arc<[u16]>),
        S32(Arc<[i32]>),
        U32(Arc<[u32]>),
        S64(Arc<[i64]>),
        U64(Arc<[u64]>),
        F32(Arc<[f32]>),
        F64(Arc<[f64]>),
        Char(Arc<[char]>),
        Other(Arc<[Value]>),
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

    pub enum ListSpecializationIter<'a> {
        Bool(std::slice::Iter<'a, bool>),
        S8(std::slice::Iter<'a, i8>),
        U8(std::slice::Iter<'a, u8>),
        S16(std::slice::Iter<'a, i16>),
        U16(std::slice::Iter<'a, u16>),
        S32(std::slice::Iter<'a, i32>),
        U32(std::slice::Iter<'a, u32>),
        S64(std::slice::Iter<'a, i64>),
        U64(std::slice::Iter<'a, u64>),
        F32(std::slice::Iter<'a, f32>),
        F64(std::slice::Iter<'a, f64>),
        Char(std::slice::Iter<'a, char>),
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

    pub trait ListPrimitive: Copy + Sized {
        fn from_arc(arc: Arc<[Self]>) -> ListSpecialization;
        fn from_value_iter(iter: impl IntoIterator<Item = Value>) -> Result<ListSpecialization>;

        fn from_specialization(specialization: &ListSpecialization) -> &[Self];

        fn ty() -> ValueType;
    }

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
