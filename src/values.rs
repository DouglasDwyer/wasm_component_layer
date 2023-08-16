use anyhow::*;
use fxhash::*;
use std::iter;
use std::mem::*;
use std::ops::*;
use wasmtime_environ::component::*;

/// Represents runtime list values
#[derive(PartialEq, Eq, Clone)]
pub struct List {
    ty: crate::types::List,
    values: Box<[Value]>,
}

impl List {
    /// Instantiate the specified type with the specified `values`.
    pub fn new(ty: &crate::types::List, values: Box<[Value]>) -> Result<Self> {
        let element_type = ty.ty();
        for (index, value) in values.iter().enumerate() {
            element_type
                .check(value)
                .with_context(|| format!("type mismatch for element {index} of list"))?;
        }

        Ok(Self {
            ty: ty.clone(),
            values,
        })
    }

    /// Returns the corresponding type of this list
    pub fn ty(&self) -> &crate::types::List {
        &self.ty
    }
}

impl Deref for List {
    type Target = [Value];

    fn deref(&self) -> &[Value] {
        &self.values
    }
}

impl std::fmt::Debug for List {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_list();
        for val in self.iter() {
            f.entry(val);
        }
        f.finish()
    }
}

/// Represents runtime record values
#[derive(PartialEq, Eq, Clone)]
pub struct Record {
    ty: crate::types::Record,
    values: Box<[Value]>,
}

impl Record {
    /// Instantiate the specified type with the specified `values`.
    pub fn new<'a>(
        ty: &crate::types::Record,
        values: impl IntoIterator<Item = (&'a str, Value)>,
    ) -> Result<Self> {
        let mut fields = ty.fields();
        let expected_len = fields.len();
        let mut iter = values.into_iter();
        let mut values = Vec::with_capacity(expected_len);
        loop {
            match (fields.next(), iter.next()) {
                (Some(field), Some((name, value))) => {
                    if name == field.name {
                        field
                            .ty
                            .check(&value)
                            .with_context(|| format!("type mismatch for field {name} of record"))?;

                        values.push(value);
                    } else {
                        bail!("field name mismatch: expected {}; got {name}", field.name)
                    }
                }
                (None, Some((_, value))) => values.push(value),
                _ => break,
            }
        }

        if values.len() != expected_len {
            bail!("expected {} value(s); got {}", expected_len, values.len());
        }

        Ok(Self {
            ty: ty.clone(),
            values: values.into(),
        })
    }

    /// Returns the corresponding type of this record.
    pub fn ty(&self) -> &crate::types::Record {
        &self.ty
    }

    /// Gets the value of the specified field `name` from this record.
    pub fn fields(&self) -> impl Iterator<Item = (&str, &Value)> {
        assert_eq!(self.values.len(), self.ty.fields().len());
        self.ty
            .fields()
            .zip(self.values.iter())
            .map(|(ty, val)| (ty.name, val))
    }
}

impl std::fmt::Debug for Record {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("Record");
        for (name, val) in self.fields() {
            f.field(name, val);
        }
        f.finish()
    }
}

/// Represents runtime tuple values
#[derive(PartialEq, Eq, Clone)]
pub struct Tuple {
    ty: crate::types::Tuple,
    values: Box<[Value]>,
}

impl Tuple {
    /// Instantiate the specified type ith the specified `values`.
    pub fn new(ty: &crate::types::Tuple, values: Box<[Value]>) -> Result<Self> {
        if values.len() != ty.types().len() {
            bail!(
                "expected {} value(s); got {}",
                ty.types().len(),
                values.len()
            );
        }

        for (index, (value, ty)) in values.iter().zip(ty.types()).enumerate() {
            ty.check(value)
                .with_context(|| format!("type mismatch for field {index} of tuple"))?;
        }

        Ok(Self {
            ty: ty.clone(),
            values,
        })
    }

    /// Returns the type of this tuple.
    pub fn ty(&self) -> &crate::types::Tuple {
        &self.ty
    }

    /// Returns the list of values that this tuple contains.
    pub fn values(&self) -> &[Value] {
        &self.values
    }
}

impl std::fmt::Debug for Tuple {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut tuple = f.debug_tuple("");
        for val in self.values() {
            tuple.field(val);
        }
        tuple.finish()
    }
}

/// Represents runtime variant values
#[derive(PartialEq, Eq, Clone)]
pub struct Variant {
    ty: crate::types::Variant,
    discriminant: u32,
    value: Option<Box<Value>>,
}

impl Variant {
    /// Instantiate the specified type with the specified case `name` and `value`.
    pub fn new(ty: &crate::types::Variant, name: &str, value: Option<Value>) -> Result<Self> {
        let (discriminant, case_type) = ty
            .cases()
            .enumerate()
            .find_map(|(index, case)| {
                if case.name == name {
                    Some((index, case.ty))
                } else {
                    None
                }
            })
            .ok_or_else(|| anyhow!("unknown variant case: {name}"))?;

        typecheck_payload(name, case_type.as_ref(), value.as_ref())?;

        Ok(Self {
            ty: ty.clone(),
            discriminant: u32::try_from(discriminant)?,
            value: value.map(Box::new),
        })
    }

    /// Returns the type of this variant.
    pub fn ty(&self) -> &crate::types::Variant {
        &self.ty
    }

    /// Returns name of the discriminant of this value within the variant type.
    pub fn discriminant(&self) -> &str {
        self.ty
            .cases()
            .nth(self.discriminant as usize)
            .unwrap()
            .name
    }

    /// Returns the payload value for this variant.
    pub fn payload(&self) -> Option<&Value> {
        self.value.as_deref()
    }
}

fn typecheck_payload(name: &str, case_type: Option<&crate::types::Type>, value: Option<&Value>) -> Result<()> {
    match (case_type, value) {
        (Some(expected), Some(actual)) => expected
            .check(&actual)
            .with_context(|| format!("type mismatch for case {name} of variant")),
        (None, None) => Ok(()),
        (Some(_), None) => bail!("expected a payload for case `{name}`"),
        (None, Some(_)) => bail!("did not expect payload for case `{name}`"),
    }
}

impl std::fmt::Debug for Variant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple(self.discriminant())
            .field(&self.payload())
            .finish()
    }
}

/// Represents runtime enum values
#[derive(PartialEq, Eq, Clone)]
pub struct Enum {
    ty: crate::types::Enum,
    discriminant: u32,
}

impl Enum {
    /// Instantiate the specified type with the specified case `name`.
    pub fn new(ty: &crate::types::Enum, name: &str) -> Result<Self> {
        let discriminant = u32::try_from(
            ty.names()
                .position(|n| n == name)
                .ok_or_else(|| anyhow!("unknown enum case: {name}"))?,
        )?;

        Ok(Self {
            ty: ty.clone(),
            discriminant,
        })
    }

    /// Returns the type of this value.
    pub fn ty(&self) -> &crate::types::Enum {
        &self.ty
    }

    /// Returns name of this enum value.
    pub fn discriminant(&self) -> &str {
        self.ty.names().nth(self.discriminant as usize).unwrap()
    }
}

impl std::fmt::Debug for Enum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.discriminant(), f)
    }
}

/// Represents runtime union values
#[derive(PartialEq, Eq, Clone)]
pub struct Union {
    ty: crate::types::Union,
    discriminant: u32,
    value: Option<Box<Value>>,
}

impl Union {
    /// Instantiate the specified type with the specified `discriminant` and `value`.
    pub fn new(ty: &crate::types::Union, discriminant: u32, value: Value) -> Result<Self> {
        if let Some(case_ty) = ty.types().nth(usize::try_from(discriminant)?) {
            case_ty
                .check(&value)
                .with_context(|| format!("type mismatch for case {discriminant} of union"))?;

            Ok(Self {
                ty: ty.clone(),
                discriminant,
                value: Some(Box::new(value)),
            })
        } else {
            Err(anyhow!(
                "discriminant {discriminant} out of range: [0,{})",
                ty.types().len()
            ))
        }
    }

    /// Returns the type of this value.
    pub fn ty(&self) -> &crate::types::Union {
        &self.ty
    }

    /// Returns name of the discriminant of this value within the union type.
    pub fn discriminant(&self) -> u32 {
        self.discriminant
    }

    /// Returns the payload value for this union.
    pub fn payload(&self) -> &Value {
        self.value.as_ref().unwrap()
    }
}

impl std::fmt::Debug for Union {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple(&format!("U{}", self.discriminant()))
            .field(self.payload())
            .finish()
    }
}

/// Represents runtime option values
#[derive(PartialEq, Eq, Clone)]
pub struct OptionVal {
    ty: crate::types::OptionType,
    discriminant: u32,
    value: Option<Box<Value>>,
}

impl OptionVal {
    /// Instantiate the specified type with the specified `value`.
    pub fn new(ty: &crate::types::OptionType, value: Option<Value>) -> Result<Self> {
        let value = value
            .map(|value| {
                ty.ty().check(&value).context("type mismatch for option")?;

                std::result::Result::Ok::<_, Error>(value)
            })
            .transpose()?;

        Ok(Self {
            ty: ty.clone(),
            discriminant: if value.is_none() { 0 } else { 1 },
            value: value.map(Box::new),
        })
    }

    /// Returns the type of this value.
    pub fn ty(&self) -> &crate::types::OptionType {
        &self.ty
    }

    /// Returns the optional value contained within.
    pub fn value(&self) -> Option<&Value> {
        self.value.as_deref()
    }
}

impl std::fmt::Debug for OptionVal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.value().fmt(f)
    }
}

/// Represents runtime result values
#[derive(PartialEq, Eq, Clone)]
pub struct ResultVal {
    ty: crate::types::ResultType,
    discriminant: u32,
    value: Option<Box<Value>>,
}

impl ResultVal {
    /// Instantiate the specified type with the specified `value`.
    pub fn new(ty: &crate::types::ResultType, value: Result<Option<Value>, Option<Value>>) -> Result<Self> {
        Ok(Self {
            ty: ty.clone(),
            discriminant: if value.is_ok() { 0 } else { 1 },
            value: match value {
                std::result::Result::Ok(value) => {
                    typecheck_payload("ok", ty.ok().as_ref(), value.as_ref())?;
                    value.map(Box::new)
                }
                std::result::Result::Err(value) => {
                    typecheck_payload("err", ty.err().as_ref(), value.as_ref())?;
                    value.map(Box::new)
                }
            },
        })
    }

    /// Returns the type of this value.
    pub fn ty(&self) -> &crate::types::ResultType {
        &self.ty
    }

    /// Returns the result value contained within.
    pub fn value(&self) -> Result<Option<&Value>, Option<&Value>> {
        if self.discriminant == 0 {
            std::result::Result::Ok(self.value.as_deref())
        } else {
            std::result::Result::Err(self.value.as_deref())
        }
    }
}

impl std::fmt::Debug for ResultVal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.value().fmt(f)
    }
}

/// Represents runtime flag values
#[derive(PartialEq, Eq, Clone)]
pub struct Flags {
    ty: crate::types::Flags,
    count: u32,
    value: Box<[u32]>,
}

impl Flags {
    /// Instantiate the specified type with the specified flag `names`.
    pub fn new(ty: &crate::types::Flags, names: &[&str]) -> Result<Self> {
        let map = ty
            .names()
            .enumerate()
            .map(|(index, name)| (name, index))
            .collect::<FxHashMap<_, _>>();

        let count = usize::from(ty.canonical_abi().flat_count.unwrap());
        let mut values = vec![0_u32; count];

        for name in names {
            let index = map
                .get(name)
                .ok_or_else(|| anyhow!("unknown flag: {name}"))?;
            values[index / 32] |= 1 << (index % 32);
        }

        Ok(Self {
            ty: ty.clone(),
            count: u32::try_from(map.len())?,
            value: values.into(),
        })
    }

    /// Returns the type of this value.
    pub fn ty(&self) -> &crate::types::Flags {
        &self.ty
    }

    /// Returns an iterator over the set of names that this flags set contains.
    pub fn flags(&self) -> impl Iterator<Item = &str> {
        (0..self.count).filter_map(|i| {
            let (idx, bit) = ((i / 32) as usize, i % 32);
            if self.value[idx] & (1 << bit) != 0 {
                Some(self.ty.names().nth(i as usize).unwrap())
            } else {
                None
            }
        })
    }
}

impl std::fmt::Debug for Flags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut set = f.debug_set();
        for flag in self.flags() {
            set.entry(&flag);
        }
        set.finish()
    }
}

/// Represents possible runtime values which a component function can either
/// consume or produce
///
/// This is a dynamic representation of possible values in the component model.
/// Note that this is not an efficient representation but is instead intended to
/// be a flexible and somewhat convenient representation. The most efficient
/// representation of component model types is to use the `bindgen!` macro to
/// generate native Rust types with specialized liftings and lowerings.
///
/// This type is used in conjunction with [`Func::call`] for example if the
/// signature of a component is not statically known ahead of time.
///
/// # Notes on Equality
///
/// This type implements both the Rust `PartialEq` and `Eq` traits. This type
/// additionally contains values which are not necessarily easily equated,
/// however, such as floats (`Float32` and `Float64`) and resources. Equality
/// does require that two values have the same type, and then these cases are
/// handled as:
///
/// * Floats are tested if they are "semantically the same" meaning all NaN
///   values are equal to all other NaN values. Additionally zero values must be
///   exactly the same, so positive zero is not equal to negative zero. The
///   primary use case at this time is fuzzing-related equality which this is
///   sufficient for.
///
/// * Resources are tested if their types and indices into the host table are
///   equal. This does not compare the underlying representation so borrows of
///   the same guest resource are not considered equal. This additionally
///   doesn't go further and test for equality in the guest itself (for example
///   two different heap allocations of `Box<u32>` can be equal in normal Rust
///   if they contain the same value, but will never be considered equal when
///   compared as `Val::Resource`s).
///
/// In general if a strict guarantee about equality is required here it's
/// recommended to "build your own" as this equality intended for fuzzing
/// Wasmtime may not be suitable for you.
///
/// [`Func::call`]: crate::component::Func::call
#[derive(Debug, Clone)]
#[allow(missing_docs)]
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
    String(String),
    List(List),
    Record(Record),
    Tuple(Tuple),
    Variant(Variant),
    Enum(Enum),
    Union(Union),
    Option(OptionVal),
    Result(ResultVal),
    Flags(Flags),
    Resource(()),
}

impl Value {
    /// Retrieve the [`Type`] of this value.
    pub fn ty(&self) -> crate::types::Type {
        match self {
            Value::Bool(_) => crate::types::Type::Bool,
            Value::S8(_) => crate::types::Type::S8,
            Value::U8(_) => crate::types::Type::U8,
            Value::S16(_) => crate::types::Type::S16,
            Value::U16(_) => crate::types::Type::U16,
            Value::S32(_) => crate::types::Type::S32,
            Value::U32(_) => crate::types::Type::U32,
            Value::S64(_) => crate::types::Type::S64,
            Value::U64(_) => crate::types::Type::U64,
            Value::F32(_) => crate::types::Type::Float32,
            Value::F64(_) => crate::types::Type::Float64,
            Value::Char(_) => crate::types::Type::Char,
            Value::String(_) => crate::types::Type::String,
            Value::List(List { ty, .. }) => crate::types::Type::List(ty.clone()),
            Value::Record(Record { ty, .. }) => crate::types::Type::Record(ty.clone()),
            Value::Tuple(Tuple { ty, .. }) => crate::types::Type::Tuple(ty.clone()),
            Value::Variant(Variant { ty, .. }) => crate::types::Type::Variant(ty.clone()),
            Value::Enum(Enum { ty, .. }) => crate::types::Type::Enum(ty.clone()),
            Value::Union(Union { ty, .. }) => crate::types::Type::Union(ty.clone()),
            Value::Option(OptionVal { ty, .. }) => crate::types::Type::Option(ty.clone()),
            Value::Result(ResultVal { ty, .. }) => crate::types::Type::Result(ty.clone()),
            Value::Flags(Flags { ty, .. }) => crate::types::Type::Flags(ty.clone()),
            Value::Resource(_) => crate::types::Type::Bool //todo
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            // IEEE 754 equality considers NaN inequal to NaN and negative zero
            // equal to positive zero, however we do the opposite here, because
            // this logic is used by testing and fuzzing, which want to know
            // whether two values are semantically the same, rather than
            // numerically equal.
            (Self::F32(l), Self::F32(r)) => {
                (*l != 0.0 && l == r)
                    || (*l == 0.0 && l.to_bits() == r.to_bits())
                    || (l.is_nan() && r.is_nan())
            }
            (Self::F32(_), _) => false,
            (Self::F64(l), Self::F64(r)) => {
                (*l != 0.0 && l == r)
                    || (*l == 0.0 && l.to_bits() == r.to_bits())
                    || (l.is_nan() && r.is_nan())
            }
            (Self::F64(_), _) => false,

            (Self::Bool(l), Self::Bool(r)) => l == r,
            (Self::Bool(_), _) => false,
            (Self::S8(l), Self::S8(r)) => l == r,
            (Self::S8(_), _) => false,
            (Self::U8(l), Self::U8(r)) => l == r,
            (Self::U8(_), _) => false,
            (Self::S16(l), Self::S16(r)) => l == r,
            (Self::S16(_), _) => false,
            (Self::U16(l), Self::U16(r)) => l == r,
            (Self::U16(_), _) => false,
            (Self::S32(l), Self::S32(r)) => l == r,
            (Self::S32(_), _) => false,
            (Self::U32(l), Self::U32(r)) => l == r,
            (Self::U32(_), _) => false,
            (Self::S64(l), Self::S64(r)) => l == r,
            (Self::S64(_), _) => false,
            (Self::U64(l), Self::U64(r)) => l == r,
            (Self::U64(_), _) => false,
            (Self::Char(l), Self::Char(r)) => l == r,
            (Self::Char(_), _) => false,
            (Self::String(l), Self::String(r)) => l == r,
            (Self::String(_), _) => false,
            (Self::List(l), Self::List(r)) => l == r,
            (Self::List(_), _) => false,
            (Self::Record(l), Self::Record(r)) => l == r,
            (Self::Record(_), _) => false,
            (Self::Tuple(l), Self::Tuple(r)) => l == r,
            (Self::Tuple(_), _) => false,
            (Self::Variant(l), Self::Variant(r)) => l == r,
            (Self::Variant(_), _) => false,
            (Self::Enum(l), Self::Enum(r)) => l == r,
            (Self::Enum(_), _) => false,
            (Self::Union(l), Self::Union(r)) => l == r,
            (Self::Union(_), _) => false,
            (Self::Option(l), Self::Option(r)) => l == r,
            (Self::Option(_), _) => false,
            (Self::Result(l), Self::Result(r)) => l == r,
            (Self::Result(_), _) => false,
            (Self::Flags(l), Self::Flags(r)) => l == r,
            (Self::Flags(_), _) => false,
            (Self::Resource(l), Self::Resource(r)) => l == r,
            (Self::Resource(_), _) => false,
        }
    }
}

impl Eq for Value {}

impl TryFrom<&Value> for wasm_runtime_layer::Value {
    type Error = Error;

    fn try_from(value: &Value) -> Result<Self> {
        match value {
            Value::S32(x) => Ok(Self::I32(*x)),
            Value::S64(x) => Ok(Self::I64(*x)),
            Value::F32(x) => Ok(Self::F32(*x)),
            Value::F64(x) => Ok(Self::F64(*x)),
            _ => bail!("Unable to convert {value:?} to core type.")
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
            _ => bail!("Unable to convert {value:?} to component type.")
        }
    }
}