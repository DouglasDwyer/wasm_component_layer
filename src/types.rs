use anyhow::*;
use fxhash::*;
use std::hash::*;
use std::sync::*;

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
    /*Own(()),
    Borrow(()),*/
}

impl ValueType {
    pub(crate) fn from_resolve(ty: &wit_parser::Type, resolve: &wit_parser::Resolve) -> Result<Self> {
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
            wit_parser::Type::Id(x) => Self::from_typedef(&resolve.types[*x], resolve)?,
        })
    }

    pub(crate) fn from_typedef(def: &wit_parser::TypeDef, resolve: &wit_parser::Resolve) -> Result<Self> {
        Ok(match &def.kind {
            wit_parser::TypeDefKind::Record(x) => Self::Record(RecordType::from_resolve(x, resolve)?),
            wit_parser::TypeDefKind::Resource => bail!("Unimplemented"),
            wit_parser::TypeDefKind::Handle(_) => bail!("Unimplemented"),
            wit_parser::TypeDefKind::Flags(x) => Self::Flags(FlagsType::from_resolve(x, resolve)?),
            wit_parser::TypeDefKind::Tuple(x) => Self::Tuple(TupleType::from_resolve(x, resolve)?),
            wit_parser::TypeDefKind::Variant(x) => Self::Variant(VariantType::from_resolve(x, resolve)?),
            wit_parser::TypeDefKind::Enum(x) => Self::Enum(EnumType::from_resolve(x, resolve)),
            wit_parser::TypeDefKind::Option(x) => Self::Option(OptionType::new(Self::from_resolve(x, resolve)?)),
            wit_parser::TypeDefKind::Result(x) => Self::Result(ResultType::new(match &x.ok { Some(t) => Some(Self::from_resolve(t, resolve)?), None => None }, match &x.err { Some(t) => Some(Self::from_resolve(t, resolve)?), None => None })),
            wit_parser::TypeDefKind::Union(x) => Self::Union(UnionType::from_resolve(x, resolve)?),
            wit_parser::TypeDefKind::List(x) => Self::List(ListType::new(Self::from_resolve(x, resolve)?)),
            wit_parser::TypeDefKind::Future(_) => bail!("Unimplemented."),
            wit_parser::TypeDefKind::Stream(_) => bail!("Unimplemented."),
            wit_parser::TypeDefKind::Type(x) => Self::from_resolve(x, resolve)?,
            wit_parser::TypeDefKind::Unknown => unreachable!(),
        })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ListType {
    element: Arc<ValueType>
}

impl ListType {
    pub fn new(element_ty: ValueType) -> Self {
        Self { element: Arc::new(element_ty) }
    }

    pub fn element_ty(&self) -> ValueType {
        (*self.element).clone()
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RecordType {
    pub(crate) fields: Arc<[(usize, Arc<str>, ValueType)]>
}

impl RecordType {
    pub fn new<S: Into<Arc<str>>>(fields: impl IntoIterator<Item = (S, ValueType)>) -> Result<Self> {
        let mut to_sort = fields.into_iter().enumerate().map(|(i, (name, val))| (i, Into::<Arc<str>>::into(name), val)).collect::<Arc<_>>();
        Arc::get_mut(&mut to_sort).expect("Could not get exclusive reference.").sort_by(|a, b| a.1.cmp(&b.1));

        for pair in to_sort.windows(2) {
            ensure!(pair[0].1 != pair[1].1, "Duplicate field names");
        }

        Ok(Self { fields: to_sort })
    }

    pub fn fields(&self) -> impl ExactSizeIterator<Item = (&str, ValueType)> {
        self.fields.iter().map(|(_, name, val)| (&**name, val.clone()))
    }

    pub(crate) fn new_sorted(fields: impl IntoIterator<Item = (Arc<str>, ValueType)>) -> Result<Self> {
        let result = Self { fields: fields.into_iter().enumerate().map(|(i, (a, b))| (i, a, b)).collect() };

        for pair in result.fields.windows(2) {
            ensure!(pair[0].0 != pair[1].0, "Duplicate field names");
        }

        Ok(result)
    }

    fn from_resolve(ty: &wit_parser::Record, resolve: &wit_parser::Resolve) -> Result<Self> {
        let mut to_sort = ty.fields.iter().enumerate().map(|(i, x)| Ok((i, Into::<Arc<str>>::into(x.name.as_str()), ValueType::from_resolve(&x.ty, resolve)?))).collect::<Result<Arc<_>>>()?;
        Arc::get_mut(&mut to_sort).expect("Could not get exclusive reference.").sort_by(|a, b| a.1.cmp(&b.1));

        for pair in to_sort.windows(2) {
            ensure!(pair[0].0 != pair[1].0, "Duplicate field names");
        }

        Ok(Self { fields: to_sort })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct TupleType {
    fields: Arc<[ValueType]>
}

impl TupleType {
    pub fn new(fields: impl IntoIterator<Item = ValueType>) -> Self {
        Self { fields: fields.into_iter().collect() }
    }

    pub fn fields(&self) -> &[ValueType] {
        &self.fields
    }

    fn from_resolve(ty: &wit_parser::Tuple, resolve: &wit_parser::Resolve) -> Result<Self> {
        let fields = ty.types.iter().map(|x| ValueType::from_resolve(x, resolve)).collect::<Result<_>>()?;
        Ok(Self { fields })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct VariantCase {
    name: Arc<str>,
    ty: Option<ValueType>
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
    cases: Arc<[VariantCase]>
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

    fn from_resolve(ty: &wit_parser::Variant, resolve: &wit_parser::Resolve) -> Result<Self> {
        let cases: Arc<_> = ty.cases.iter().map(|x| Ok(VariantCase { name: x.name.as_str().into(), ty: match &x.ty { Some(t) => Some(ValueType::from_resolve(t, resolve)?), None => None } })).collect::<Result<_>>()?;

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
    cases: Arc<[Arc<str>]>
}

impl EnumType {
    pub fn new<S: Into<Arc<str>>>(cases: impl IntoIterator<Item = S>) -> Self {
        Self { cases: cases.into_iter().map(|x| Into::<Arc<str>>::into(x)).collect() }
    }

    pub fn cases(&self) -> impl ExactSizeIterator<Item = &str> {
        self.cases.iter().map(|x| &**x)
    }

    fn from_resolve(ty: &wit_parser::Enum, resolve: &wit_parser::Resolve) -> Self {
        Self::new(ty.cases.iter().map(|x| x.name.as_str()))
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnionType {
    cases: Arc<[ValueType]>
}

impl UnionType {
    pub fn new(cases: impl IntoIterator<Item = ValueType>) -> Self {
        Self { cases: cases.into_iter().collect() }
    }

    pub fn cases(&self) -> &[ValueType] {
        &self.cases
    }

    fn from_resolve(ty: &wit_parser::Union, resolve: &wit_parser::Resolve) -> Result<Self> {
        let cases = ty.cases.iter().map(|x| ValueType::from_resolve(&x.ty, resolve)).collect::<Result<_>>()?;

        Ok(Self { cases })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct OptionType {
    ty: Arc<ValueType>
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
    ok_err: Arc<(Option<ValueType>, Option<ValueType>)>
}

impl ResultType {
    pub fn new(ok: Option<ValueType>, err: Option<ValueType>) -> Self {
        Self { ok_err: Arc::new((ok, err)) }
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
    pub(crate) indices: Arc<FxHashMap<Arc<str>, usize>>
}

impl FlagsType {
    pub fn new<S: Into<Arc<str>>>(names: impl IntoIterator<Item = S>) -> Result<Self> {
        let names: Arc<_> = names.into_iter().map(|x| Into::<Arc<str>>::into(x)).collect();

        for i in 0..names.len() {
            for j in 0..i {
                ensure!(names[i] != names[j], "Duplicate case names.");
            }
        }

        let indices = Arc::new(names.iter().enumerate().map(|(i, val)| (val.clone(), i)).collect());
        
        Ok(Self { names, indices })
    }

    pub fn names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.names.iter().map(|x| &**x)
    }

    fn from_resolve(ty: &wit_parser::Flags, resolve: &wit_parser::Resolve) -> Result<Self> {
        Self::new(ty.flags.iter().map(|x| x.name.as_ref()))
    }
}

impl Hash for FlagsType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.names.hash(state)
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
    pub(crate) fn from_resolve(func: &wit_parser::Function, resolve: &wit_parser::Resolve) -> Result<Self> {
        let mut params_results = func.params.iter().map(|(_, ty)| ValueType::from_resolve(ty, resolve)).collect::<Result<Vec<_>>>()?;
        let len_params = params_results.len();
        
        for result in func.results.iter_types().map(|ty| ValueType::from_resolve(ty, resolve)) {
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

    /// Returns the pair of parameter and result types of the function type.
    pub(crate) fn params_results(&self) -> (&[ValueType], &[ValueType]) {
        self.params_results.split_at(self.len_params)
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
        if self.params().len() != results.len() {
            bail!("Incorrect parameter length.");
        }
        if self
            .results()
            .iter()
            .cloned()
            .ne(results.iter().map(crate::values::Value::ty))
        {
            bail!("Incorrect parameter types.");
        }
        Ok(())
    }
}