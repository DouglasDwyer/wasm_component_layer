use std::marker::*;
use std::mem::*;
use std::sync::atomic::*;
use std::sync::*;
use std::usize;

use bytemuck::*;
use wasm_runtime_layer::*;
#[allow(unused_imports)]
use wasmtime_environ::component::StringEncoding;

#[allow(unused_imports)]
use crate::abi::{Generator, *};
use crate::types::{FuncType, ValueType};
use crate::values::Value;
use crate::{AsContext, AsContextMut, StoreContextMut, *};

/// Stores the backing implementation for a function.
#[derive(Clone, Debug)]
pub(crate) enum FuncImpl {
    /// A function backed by a guest implementation.
    GuestFunc(Option<crate::Instance>, Arc<GuestFunc>),
    /// A host-provided function.
    HostFunc(Arc<AtomicUsize>),
}

/// Stores the data necessary to call a guest function.
#[derive(Debug)]
pub(crate) struct GuestFunc {
    /// The core function to call.
    pub callee: wasm_runtime_layer::Func,
    /// The component for this function.
    pub component: Arc<ComponentInner>,
    /// The string encoding to use.
    pub encoding: StringEncoding,
    /// The function definition to use.
    pub function: Function,
    /// The memory to use.
    pub memory: Option<Memory>,
    /// The reallocation function to use.
    pub realloc: Option<wasm_runtime_layer::Func>,
    /// The post-return function to use.
    pub post_return: Option<wasm_runtime_layer::Func>,
    /// The state table to use.
    pub state_table: Arc<StateTable>,
    /// The types to use.
    pub types: Arc<[crate::types::ValueType]>,
    /// The instance ID to use.
    pub instance_id: u64,
    /// The ID of the interface associated with this function.
    pub interface_id: Option<InterfaceIdentifier>,
}

/// A component model function that may be invoked to interact with an `Instance`.
#[derive(Clone, Debug)]
pub struct Func {
    /// The store ID associated with this function.
    pub(crate) store_id: u64,
    /// The type of this function.
    pub(crate) ty: FuncType,
    /// The backing implementation for this function.
    pub(crate) backing: FuncImpl,
}

impl Func {
    /// Creates a new function with the provided type and arguments.
    pub fn new<C: AsContextMut>(
        mut ctx: C,
        ty: FuncType,
        f: impl 'static
            + Send
            + Sync
            + Fn(StoreContextMut<C::UserState, C::Engine>, &[Value], &mut [Value]) -> Result<()>,
    ) -> Self {
        let mut ctx_mut = ctx.as_context_mut();
        let data = ctx_mut.inner.data_mut();
        let idx = data.host_functions.push(f);

        Self {
            store_id: data.id,
            ty,
            backing: FuncImpl::HostFunc(idx),
        }
    }

    /// Calls this function, returning an error if:
    ///
    /// - The store did not match the original.
    /// - The arguments or results did not match the signature.
    /// - A trap occurred.
    pub fn call<C: AsContextMut>(
        &self,
        mut ctx: C,
        arguments: &[Value],
        results: &mut [Value],
    ) -> Result<()> {
        if ctx.as_context().inner.data().id != self.store_id {
            panic!("Attempted to call function with incorrect store.");
        }

        self.ty.match_params(arguments)?;

        if self.ty.results().len() != results.len() {
            bail!("Incorrect result length.");
        }

        match &self.backing {
            FuncImpl::GuestFunc(i, x) => {
                let GuestFunc {
                    callee,
                    component,
                    encoding,
                    function,
                    memory,
                    realloc,
                    state_table,
                    post_return,
                    types,
                    instance_id,
                    interface_id,
                } = &**x;

                ensure!(
                    !state_table.dropped.load(Ordering::Acquire),
                    "Instance had been dropped."
                );

                let mut bindgen = FuncBindgen {
                    ctx,
                    flat_results: Vec::default(),
                    arguments,
                    results,
                    callee_interface: None,
                    callee_wasm: Some(callee),
                    component,
                    encoding,
                    memory,
                    realloc,
                    resource_tables: &state_table.resource_tables,
                    post_return,
                    types,
                    handles_to_drop: Vec::new(),
                    required_dropped: Vec::new(),
                    instance_id: *instance_id,
                    store_id: self.store_id,
                };

                Ok(Generator::new(
                    &component.resolve,
                    AbiVariant::GuestExport,
                    LiftLower::LowerArgsLiftResults,
                    &mut bindgen,
                )
                .call(function)
                .map_err(|error| FuncError {
                    name: function.name.clone(),
                    interface: interface_id.clone(),
                    instance: i.as_ref().expect("No instance available.").clone(),
                    error,
                })?)
            },
            FuncImpl::HostFunc(idx) => {
                let callee = ctx.as_context().inner.data().host_functions.get(idx);
                (callee)(ctx.as_context_mut(), arguments, results)?;
                self.ty.match_results(results)
            },
        }
    }

    /// Gets the type of this value.
    pub fn ty(&self) -> FuncType {
        self.ty.clone()
    }

    /// Converts this function to a [`TypedFunc`], failing if the signatures do not match.
    pub fn typed<P: ComponentList, R: ComponentList>(&self) -> Result<TypedFunc<P, R>> {
        let mut params_results = vec![ValueType::Bool; P::LEN + R::LEN];
        P::into_tys(&mut params_results[..P::LEN]);
        R::into_tys(&mut params_results[P::LEN..]);
        ensure!(
            &params_results[..P::LEN] == self.ty.params(),
            "Parameters did not match function signature. Expected {:?} but got {:?}",
            self.ty.params(),
            &params_results[..P::LEN]
        );
        ensure!(
            &params_results[P::LEN..] == self.ty.results(),
            "Results did not match function signature. Expected {:?} but got {:?}",
            self.ty.results(),
            &params_results[P::LEN..]
        );
        Ok(TypedFunc {
            inner: self.clone(),
            data: PhantomData,
        })
    }

    /// Ties the given instance to this function.
    pub(crate) fn instantiate(&self, inst: crate::Instance) -> Self {
        if let FuncImpl::GuestFunc(None, y) = &self.backing {
            Self {
                store_id: self.store_id,
                backing: FuncImpl::GuestFunc(Some(inst), y.clone()),
                ty: self.ty.clone(),
            }
        } else {
            panic!("Function was not an uninitialized guest function.");
        }
    }

    /// Calls this function from a guest context.
    pub(crate) fn call_from_guest<C: AsContextMut>(
        &self,
        ctx: C,
        options: &GuestInvokeOptions,
        arguments: &[wasm_runtime_layer::Value],
        results: &mut [wasm_runtime_layer::Value],
    ) -> Result<()> {
        ensure!(
            self.store_id == options.store_id,
            "Function stores did not match."
        );

        let args = arguments
            .iter()
            .map(TryFrom::try_from)
            .collect::<Result<Vec<_>>>()?;
        let mut res = vec![Value::Bool(false); results.len()];

        let mut bindgen = FuncBindgen {
            ctx,
            flat_results: Vec::default(),
            arguments: &args,
            results: &mut res,
            callee_interface: Some(self),
            callee_wasm: None,
            component: &options.component,
            encoding: &options.encoding,
            memory: &options.memory,
            realloc: &options.realloc,
            resource_tables: &options.state_table.resource_tables,
            post_return: &options.post_return,
            types: &options.types,
            handles_to_drop: Vec::new(),
            required_dropped: Vec::new(),
            instance_id: options.instance_id,
            store_id: self.store_id,
        };

        Generator::new(
            &options.component.resolve,
            AbiVariant::GuestImport,
            LiftLower::LiftArgsLowerResults,
            &mut bindgen,
        )
        .call(&options.function)?;

        for (idx, val) in res.into_iter().enumerate() {
            results[idx] = (&val).try_into()?;
        }

        Ok(())
    }
}

/// Describes options to invoke an imported function from a guest.
pub(crate) struct GuestInvokeOptions {
    /// The component to use.
    pub component: Arc<ComponentInner>,
    /// The string encoding to use.
    pub encoding: StringEncoding,
    /// The function definition to use.
    pub function: Function,
    /// The memory to use.
    pub memory: Option<Memory>,
    /// The reallocation function to use.
    pub realloc: Option<wasm_runtime_layer::Func>,
    /// The post-return function to use.
    pub post_return: Option<wasm_runtime_layer::Func>,
    /// The resource tables to use.
    pub state_table: Arc<StateTable>,
    /// The types to use.
    pub types: Arc<[crate::types::ValueType]>,
    /// The instance ID to use.
    pub instance_id: u64,
    /// The store ID to use.
    pub store_id: u64,
}

/// Manages the invocation of a component model function with the canonical ABI.
struct FuncBindgen<'a, C: AsContextMut> {
    /// The interface function to call.
    pub callee_interface: Option<&'a Func>,
    /// The core WASM function to call.
    pub callee_wasm: Option<&'a wasm_runtime_layer::Func>,
    /// The component to use.
    pub component: &'a ComponentInner,
    /// The context.
    pub ctx: C,
    /// The encoding to use.
    pub encoding: &'a StringEncoding,
    /// The list of flat results.
    pub flat_results: Vec<wasm_runtime_layer::Value>,
    /// The memory to use.
    pub memory: &'a Option<Memory>,
    /// The reallocation function to use.
    pub realloc: &'a Option<wasm_runtime_layer::Func>,
    /// The post-return function to use.
    pub post_return: &'a Option<wasm_runtime_layer::Func>,
    /// The arguments to use.
    pub arguments: &'a [Value],
    /// The results to use.
    pub results: &'a mut [Value],
    /// The resource tables to use.
    pub resource_tables: &'a Mutex<Vec<HandleTable>>,
    /// The types to use.
    pub types: &'a [crate::types::ValueType],
    /// The handles to drop at the call's end.
    pub handles_to_drop: Vec<(u32, i32)>,
    /// The handles to require dropped at the call's end.
    pub required_dropped: Vec<(bool, u32, i32, Arc<AtomicBool>)>,
    /// The instance ID to use.
    pub instance_id: u64,
    /// The store ID to use.
    pub store_id: u64,
}

impl<'a, C: AsContextMut> FuncBindgen<'a, C> {
    /// Loads a type from the given offset in guest memory.
    fn load<B: Blittable>(&self, offset: usize) -> Result<B> {
        Ok(B::from_bytes(<B::Array as ByteArray>::load(
            &self.ctx,
            self.memory.as_ref().expect("No memory."),
            offset,
        )?))
    }

    /// Stores a type to the given offset in guest memory.
    fn store<B: Blittable>(&mut self, offset: usize, value: B) -> Result<()> {
        value.to_bytes().store(
            &mut self.ctx,
            self.memory.as_ref().expect("No memory."),
            offset,
        )
    }

    /// Loads a list of types from the given offset in guest memory.
    fn load_array<B: Blittable>(&self, offset: usize, len: usize) -> Result<Arc<[B]>> {
        let mut raw_memory = B::zeroed_array(len);
        self.memory.as_ref().expect("No memory").read(
            self.ctx.as_context().inner,
            offset,
            Arc::get_mut(&mut raw_memory).expect("Could not get exclusive reference."),
        )?;
        Ok(B::from_le_array(raw_memory))
    }

    /// Stores a list of types to the given offset in guest memory.
    fn store_array<B: Blittable>(&mut self, offset: usize, value: &[B]) -> Result<()> {
        self.memory.as_ref().expect("No memory.").write(
            self.ctx.as_context_mut().inner,
            offset,
            B::to_le_slice(value),
        )
    }
}

impl<'a, C: AsContextMut> Bindgen for FuncBindgen<'a, C> {
    type Operand = Value;

    fn emit(
        &mut self,
        _resolve: &Resolve,
        inst: &Instruction<'_>,
        operands: &mut Vec<Self::Operand>,
        results: &mut Vec<Self::Operand>,
    ) -> Result<()> {
        match inst {
            Instruction::GetArg { nth } => results.push(
                self.arguments
                    .get(*nth)
                    .cloned()
                    .ok_or_else(|| Error::msg("Invalid argument count."))?,
            ),
            Instruction::I32Const { val } => results.push(Value::S32(*val)),
            Instruction::Bitcasts { casts } => {
                for (cast, op) in casts.iter().zip(operands) {
                    match cast {
                        Bitcast::I32ToF32 => require_matches!(
                            op,
                            Value::S32(x),
                            results.push(Value::F32(f32::from_bits(*x as u32)))
                        ),
                        Bitcast::F32ToI32 => require_matches!(
                            op,
                            Value::F32(x),
                            results.push(Value::S32(x.to_bits() as i32))
                        ),
                        Bitcast::I64ToF64 => require_matches!(
                            op,
                            Value::S64(x),
                            results.push(Value::F64(f64::from_bits(*x as u64)))
                        ),
                        Bitcast::F64ToI64 => require_matches!(
                            op,
                            Value::F64(x),
                            results.push(Value::S64(x.to_bits() as i64))
                        ),
                        Bitcast::I32ToI64 => {
                            require_matches!(op, Value::S32(x), results.push(Value::S64(*x as i64)))
                        },
                        Bitcast::I64ToI32 => {
                            require_matches!(op, Value::S64(x), results.push(Value::S32(*x as i32)))
                        },
                        Bitcast::I64ToF32 => {
                            require_matches!(op, Value::S64(x), results.push(Value::F32(*x as f32)))
                        },
                        Bitcast::F32ToI64 => {
                            require_matches!(op, Value::F32(x), results.push(Value::S64(*x as i64)))
                        },
                        Bitcast::None => results.push(op.clone()),
                    }
                }
            },
            Instruction::ConstZero { tys } => {
                for t in tys.iter() {
                    match t {
                        WasmType::I32 => results.push(Value::S32(0)),
                        WasmType::I64 => results.push(Value::S64(0)),
                        WasmType::F32 => results.push(Value::F32(0.0)),
                        WasmType::F64 => results.push(Value::F64(0.0)),
                    }
                }
            },
            Instruction::I32Load { offset } => require_matches!(
                operands.pop(),
                Some(Value::S32(x)),
                results.push(Value::S32(self.load((x as usize) + (*offset as usize))?))
            ),
            Instruction::I32Load8U { offset } => require_matches!(
                operands.pop(),
                Some(Value::S32(x)),
                results.push(Value::S32(
                    self.load::<u8>((x as usize) + (*offset as usize))? as i32
                ))
            ),
            Instruction::I32Load8S { offset } => require_matches!(
                operands.pop(),
                Some(Value::S32(x)),
                results.push(Value::S32(
                    self.load::<i8>((x as usize) + (*offset as usize))? as i32
                ))
            ),
            Instruction::I32Load16U { offset } => require_matches!(
                operands.pop(),
                Some(Value::S32(x)),
                results.push(Value::S32(
                    self.load::<u16>((x as usize) + (*offset as usize))? as i32
                ))
            ),
            Instruction::I32Load16S { offset } => require_matches!(
                operands.pop(),
                Some(Value::S32(x)),
                results.push(Value::S32(
                    self.load::<i16>((x as usize) + (*offset as usize))? as i32
                ))
            ),
            Instruction::I64Load { offset } => require_matches!(
                operands.pop(),
                Some(Value::S32(x)),
                results.push(Value::S64(self.load((x as usize) + (*offset as usize))?))
            ),
            Instruction::F32Load { offset } => require_matches!(
                operands.pop(),
                Some(Value::S32(x)),
                results.push(Value::F32(self.load((x as usize) + (*offset as usize))?))
            ),
            Instruction::F64Load { offset } => require_matches!(
                operands.pop(),
                Some(Value::S32(x)),
                results.push(Value::F64(self.load((x as usize) + (*offset as usize))?))
            ),
            Instruction::I32Store { offset } => require_matches!(
                operands.pop(),
                Some(Value::S32(address)),
                require_matches!(
                    operands.pop(),
                    Some(Value::S32(x)),
                    self.store((address as usize) + (*offset as usize), x)?
                )
            ),
            Instruction::I32Store8 { offset } => require_matches!(
                operands.pop(),
                Some(Value::S32(address)),
                require_matches!(
                    operands.pop(),
                    Some(Value::S32(x)),
                    self.store((address as usize) + (*offset as usize), x as u8)?
                )
            ),
            Instruction::I32Store16 { offset } => require_matches!(
                operands.pop(),
                Some(Value::S32(address)),
                require_matches!(
                    operands.pop(),
                    Some(Value::S32(x)),
                    self.store((address as usize) + (*offset as usize), x as u16)?
                )
            ),
            Instruction::I64Store { offset } => require_matches!(
                operands.pop(),
                Some(Value::S32(address)),
                require_matches!(
                    operands.pop(),
                    Some(Value::S64(x)),
                    self.store((address as usize) + (*offset as usize), x)?
                )
            ),
            Instruction::F32Store { offset } => require_matches!(
                operands.pop(),
                Some(Value::S32(address)),
                require_matches!(
                    operands.pop(),
                    Some(Value::F32(x)),
                    self.store((address as usize) + (*offset as usize), x)?
                )
            ),
            Instruction::F64Store { offset } => require_matches!(
                operands.pop(),
                Some(Value::S32(address)),
                require_matches!(
                    operands.pop(),
                    Some(Value::F64(x)),
                    self.store((address as usize) + (*offset as usize), x)?
                )
            ),
            Instruction::I32FromChar => require_matches!(
                operands.pop(),
                Some(Value::Char(x)),
                results.push(Value::S32(x as i32))
            ),
            Instruction::I64FromU64 => require_matches!(
                operands.pop(),
                Some(Value::U64(x)),
                results.push(Value::S64(x as i64))
            ),
            Instruction::I64FromS64 => require_matches!(
                operands.pop(),
                Some(Value::S64(x)),
                results.push(Value::S64(x))
            ),
            Instruction::I32FromU32 => require_matches!(
                operands.pop(),
                Some(Value::U32(x)),
                results.push(Value::S32(x as i32))
            ),
            Instruction::I32FromS32 => require_matches!(
                operands.pop(),
                Some(Value::S32(x)),
                results.push(Value::S32(x))
            ),
            Instruction::I32FromU16 => require_matches!(
                operands.pop(),
                Some(Value::U16(x)),
                results.push(Value::S32(x as i32))
            ),
            Instruction::I32FromS16 => require_matches!(
                operands.pop(),
                Some(Value::S16(x)),
                results.push(Value::S32(x as i32))
            ),
            Instruction::I32FromU8 => require_matches!(
                operands.pop(),
                Some(Value::U8(x)),
                results.push(Value::S32(x as i32))
            ),
            Instruction::I32FromS8 => require_matches!(
                operands.pop(),
                Some(Value::S8(x)),
                results.push(Value::S32(x as i32))
            ),
            Instruction::F32FromFloat32 => require_matches!(
                operands.pop(),
                Some(Value::F32(x)),
                results.push(Value::F32(x))
            ),
            Instruction::F64FromFloat64 => require_matches!(
                operands.pop(),
                Some(Value::F64(x)),
                results.push(Value::F64(x))
            ),
            Instruction::S8FromI32 => require_matches!(
                operands.pop(),
                Some(Value::S32(x)),
                results.push(Value::S8(x as i8))
            ),
            Instruction::U8FromI32 => require_matches!(
                operands.pop(),
                Some(Value::S32(x)),
                results.push(Value::U8(x as u8))
            ),
            Instruction::S16FromI32 => require_matches!(
                operands.pop(),
                Some(Value::S32(x)),
                results.push(Value::S16(x as i16))
            ),
            Instruction::U16FromI32 => require_matches!(
                operands.pop(),
                Some(Value::S32(x)),
                results.push(Value::U16(x as u16))
            ),
            Instruction::S32FromI32 => require_matches!(
                operands.pop(),
                Some(Value::S32(x)),
                results.push(Value::S32(x))
            ),
            Instruction::U32FromI32 => require_matches!(
                operands.pop(),
                Some(Value::S32(x)),
                results.push(Value::U32(x as u32))
            ),
            Instruction::S64FromI64 => require_matches!(
                operands.pop(),
                Some(Value::S64(x)),
                results.push(Value::S64(x))
            ),
            Instruction::U64FromI64 => require_matches!(
                operands.pop(),
                Some(Value::S64(x)),
                results.push(Value::U64(x as u64))
            ),
            Instruction::CharFromI32 => require_matches!(
                operands.pop(),
                Some(Value::S32(x)),
                results.push(Value::Char(char::from_u32(x as u32).ok_or_else(|| {
                    Error::msg("Could not convert integer to char.")
                })?))
            ),
            Instruction::Float32FromF32 => require_matches!(
                operands.pop(),
                Some(Value::F32(x)),
                results.push(Value::F32(x))
            ),
            Instruction::Float64FromF64 => require_matches!(
                operands.pop(),
                Some(Value::F64(x)),
                results.push(Value::F64(x))
            ),
            Instruction::BoolFromI32 => require_matches!(
                operands.pop(),
                Some(Value::S32(x)),
                results.push(Value::Bool(x > 0))
            ),
            Instruction::I32FromBool => require_matches!(
                operands.pop(),
                Some(Value::Bool(x)),
                results.push(Value::S32(x as i32))
            ),
            Instruction::StringLower { realloc: _ } => {
                let encoded = require_matches!(
                    operands.pop(),
                    Some(Value::String(x)),
                    match self.encoding {
                        StringEncoding::Utf8 => Vec::from_iter(x.bytes()),
                        StringEncoding::Utf16 | StringEncoding::CompactUtf16 =>
                            x.encode_utf16().flat_map(|a| a.to_le_bytes()).collect(),
                    }
                );

                let realloc = self.realloc.as_ref().expect("No realloc.");
                let args = [
                    wasm_runtime_layer::Value::I32(0),
                    wasm_runtime_layer::Value::I32(0),
                    wasm_runtime_layer::Value::I32(1),
                    wasm_runtime_layer::Value::I32(encoded.len() as i32),
                ];
                let mut res = [wasm_runtime_layer::Value::I32(0)];
                realloc.call(&mut self.ctx.as_context_mut().inner, &args, &mut res)?;
                let ptr = require_matches!(&res[0], wasm_runtime_layer::Value::I32(x), *x);

                let memory = self.memory.as_ref().expect("No memory.");
                memory.write(&mut self.ctx.as_context_mut().inner, ptr as usize, &encoded)?;

                results.push(Value::S32(ptr));
                results.push(Value::S32(encoded.len() as i32));
            },
            Instruction::ListCanonLower {
                element,
                realloc: _,
            } => {
                let list = require_matches!(operands.pop(), Some(Value::List(x)), x);
                let align = self.component.size_align.align(element);
                let size = self.component.size_align.size(element);

                let realloc = self.realloc.as_ref().expect("No realloc.");
                let args = [
                    wasm_runtime_layer::Value::I32(0),
                    wasm_runtime_layer::Value::I32(0),
                    wasm_runtime_layer::Value::I32(align as i32),
                    wasm_runtime_layer::Value::I32((list.len() * size) as i32),
                ];
                let mut res = [wasm_runtime_layer::Value::I32(0)];
                realloc.call(&mut self.ctx.as_context_mut().inner, &args, &mut res)?;
                let ptr = require_matches!(res[0], wasm_runtime_layer::Value::I32(x), x);

                match element {
                    Type::U8 => self.store_array(ptr as usize, list.typed::<u8>()?)?,
                    Type::U16 => self.store_array(ptr as usize, list.typed::<u16>()?)?,
                    Type::U32 => self.store_array(ptr as usize, list.typed::<u32>()?)?,
                    Type::U64 => self.store_array(ptr as usize, list.typed::<u64>()?)?,
                    Type::S8 => self.store_array(ptr as usize, list.typed::<i8>()?)?,
                    Type::S16 => self.store_array(ptr as usize, list.typed::<i16>()?)?,
                    Type::S32 => self.store_array(ptr as usize, list.typed::<i32>()?)?,
                    Type::S64 => self.store_array(ptr as usize, list.typed::<i64>()?)?,
                    Type::Float32 => self.store_array(ptr as usize, list.typed::<f32>()?)?,
                    Type::Float64 => self.store_array(ptr as usize, list.typed::<f64>()?)?,
                    _ => unreachable!(),
                }

                results.push(Value::S32(ptr));
                results.push(Value::S32(list.len() as i32));
            },
            Instruction::ListLower {
                element,
                realloc: _,
                len,
            } => {
                let list = require_matches!(operands.pop(), Some(Value::List(x)), x);
                let align = self.component.size_align.align(element);
                let size = self.component.size_align.size(element);

                let realloc = self.realloc.as_ref().expect("No realloc.");
                let args = [
                    wasm_runtime_layer::Value::I32(0),
                    wasm_runtime_layer::Value::I32(0),
                    wasm_runtime_layer::Value::I32(align as i32),
                    wasm_runtime_layer::Value::I32((list.len() * size) as i32),
                ];
                let mut res = [wasm_runtime_layer::Value::I32(0)];
                realloc.call(&mut self.ctx.as_context_mut().inner, &args, &mut res)?;
                let ptr = require_matches!(res[0], wasm_runtime_layer::Value::I32(x), x);

                len.set(list.len() as i32);

                results.push(Value::S32(ptr));
                results.push(Value::S32(list.len() as i32));

                for item in &list {
                    results.push(item.clone());
                }

                results.push(Value::S32(ptr));
            },
            Instruction::StringLift => {
                let memory = self.memory.as_ref().expect("No memory.");
                let len = require_matches!(operands.pop(), Some(Value::S32(len)), len) as usize;
                let mut result = vec![0; len];
                require_matches!(
                    operands.pop(),
                    Some(Value::S32(ptr)),
                    memory.read(&self.ctx.as_context().inner, ptr as usize, &mut result)?
                );

                match self.encoding {
                    StringEncoding::Utf8 => {
                        results.push(Value::String(String::from_utf8(result)?.into()))
                    },
                    StringEncoding::Utf16 | StringEncoding::CompactUtf16 => {
                        ensure!(result.len() & 0b1 == 0, "Invalid string length");
                        results.push(Value::String(
                            String::from_utf16(
                                &result
                                    .chunks_exact(2)
                                    .map(|e| {
                                        u16::from_be_bytes(
                                            e.try_into().expect("All chunks must have size 2."),
                                        )
                                    })
                                    .collect::<Vec<_>>(),
                            )?
                            .into(),
                        ));
                    },
                }
            },
            Instruction::ListCanonLift { element, ty: _ } => {
                let len = require_matches!(operands.pop(), Some(Value::S32(x)), x);
                let ptr = require_matches!(operands.pop(), Some(Value::S32(x)), x);

                results.push(Value::List(match element {
                    Type::U8 => self.load_array::<u8>(ptr as usize, len as usize)?.into(),
                    Type::U16 => self.load_array::<u16>(ptr as usize, len as usize)?.into(),
                    Type::U32 => self.load_array::<u32>(ptr as usize, len as usize)?.into(),
                    Type::U64 => self.load_array::<u64>(ptr as usize, len as usize)?.into(),
                    Type::S8 => self.load_array::<i8>(ptr as usize, len as usize)?.into(),
                    Type::S16 => self.load_array::<i16>(ptr as usize, len as usize)?.into(),
                    Type::S32 => self.load_array::<i32>(ptr as usize, len as usize)?.into(),
                    Type::S64 => self.load_array::<i64>(ptr as usize, len as usize)?.into(),
                    Type::Float32 => self.load_array::<f32>(ptr as usize, len as usize)?.into(),
                    Type::Float64 => self.load_array::<f64>(ptr as usize, len as usize)?.into(),
                    _ => unreachable!(),
                }));
            },
            Instruction::ListLift {
                element: _,
                ty,
                len: _,
            } => {
                let ty = self.types[ty.index()].clone();
                results.push(Value::List(List::new(
                    require_matches!(ty, crate::types::ValueType::List(x), x),
                    operands.drain(..),
                )?));
            },
            Instruction::ReadI32 { value } => {
                value.set(require_matches!(operands.pop(), Some(Value::S32(x)), x))
            },
            Instruction::RecordLower { record: _, ty } => {
                let official_ty =
                    require_matches!(&self.types[ty.index()], ValueType::Record(x), x);
                let record = require_matches!(operands.pop(), Some(Value::Record(x)), x);
                ensure!(&record.ty() == official_ty, "Record types did not match.");

                for _i in 0..record.fields().len() {
                    results.push(Value::Bool(false));
                }

                for (index, value) in official_ty
                    .fields
                    .iter()
                    .map(|x| x.0)
                    .zip(record.fields().map(|x| x.1))
                {
                    results[index] = value;
                }
            },
            Instruction::RecordLift { record: _, ty } => {
                let official_ty =
                    require_matches!(&self.types[ty.index()], ValueType::Record(x), x);
                ensure!(
                    operands.len() == official_ty.fields().len(),
                    "Record types did not match."
                );

                results.push(Value::Record(crate::values::Record::from_sorted(
                    official_ty.clone(),
                    official_ty.fields.iter().map(|(i, name, _)| {
                        (name.clone(), replace(&mut operands[*i], Value::Bool(false)))
                    }),
                )));
                operands.clear();
            },
            Instruction::HandleLower { handle, ty } => match &self.types[ty.index()] {
                ValueType::Own(_ty) => {
                    let def = match handle {
                        Handle::Own(x) => x,
                        Handle::Borrow(x) => x,
                    };
                    let val = require_matches!(operands.pop(), Some(Value::Own(x)), x);
                    let rep = val.lower(&mut self.ctx)?;

                    let mut tables = self
                        .resource_tables
                        .try_lock()
                        .expect("Could not acquire table access.");
                    results.push(Value::S32(
                        tables[self.component.resource_map[def.index()].as_u32() as usize].add(
                            HandleElement {
                                rep,
                                own: true,
                                lend_count: 0,
                            },
                        ),
                    ));
                },
                ValueType::Borrow(_ty) => {
                    let def = match handle {
                        Handle::Own(x) => x,
                        Handle::Borrow(x) => x,
                    };
                    let val = require_matches!(operands.pop(), Some(Value::Borrow(x)), x);
                    let rep = val.lower(&mut self.ctx)?;

                    if val.ty().is_owned_by_instance(self.instance_id) {
                        results.push(Value::S32(rep));
                    } else {
                        let mut tables = self
                            .resource_tables
                            .try_lock()
                            .expect("Could not acquire table access.");
                        let res = self.component.resource_map[def.index()].as_u32();
                        let idx = tables[res as usize].add(HandleElement {
                            rep,
                            own: false,
                            lend_count: 0,
                        });
                        results.push(Value::S32(idx));
                        self.handles_to_drop.push((res, idx));
                    }
                },
                _ => unreachable!(),
            },
            Instruction::HandleLift { handle, ty } => match &self.types[ty.index()] {
                ValueType::Own(ty) => {
                    let def = match handle {
                        Handle::Own(x) => x,
                        Handle::Borrow(x) => x,
                    };
                    let val = require_matches!(operands.pop(), Some(Value::S32(x)), x);

                    let mut tables = self
                        .resource_tables
                        .try_lock()
                        .expect("Could not acquire table access.");
                    let table =
                        &mut tables[self.component.resource_map[def.index()].as_u32() as usize];
                    let elem = table.remove(val)?;
                    ensure!(
                        elem.lend_count == 0,
                        "Attempted to transfer ownership while handle was lent."
                    );
                    ensure!(
                        elem.own,
                        "Attempted to transfer ownership of non-owned handle."
                    );

                    results.push(Value::Own(ResourceOwn::new_guest(
                        elem.rep,
                        ty.clone(),
                        self.store_id,
                        table.destructor().cloned(),
                    )));
                },
                ValueType::Borrow(ty) => {
                    let def = match handle {
                        Handle::Own(x) => x,
                        Handle::Borrow(x) => x,
                    };
                    let val = require_matches!(operands.pop(), Some(Value::S32(x)), x);

                    let mut tables = self
                        .resource_tables
                        .try_lock()
                        .expect("Could not acquire table access.");
                    let res = self.component.resource_map[def.index()].as_u32();
                    let table = &mut tables[res as usize];
                    let mut elem = *table.get(val)?;

                    if elem.own {
                        elem.lend_count += 1;
                        table.set(val, elem);
                    }

                    let borrow = ResourceBorrow::new(elem.rep, self.store_id, ty.clone());
                    self.required_dropped
                        .push((elem.own, res, val, borrow.dead_ref()));
                    results.push(Value::Borrow(borrow));
                },
                _ => unreachable!(),
            },
            Instruction::TupleLower { tuple: _, ty: _ } => {
                let tuple = require_matches!(operands.pop(), Some(Value::Tuple(x)), x);
                results.extend(tuple.iter().cloned());
            },
            Instruction::TupleLift { tuple: _, ty } => {
                results.push(Value::Tuple(crate::values::Tuple::new_unchecked(
                    require_matches!(&self.types[ty.index()], ValueType::Tuple(x), x.clone()),
                    operands.drain(..),
                )));
            },
            Instruction::FlagsLower { flags: _, ty: _ } => {
                let flags = require_matches!(operands.pop(), Some(Value::Flags(x)), x);
                if flags.ty().names().len() > 0 {
                    results.extend(flags.as_u32_list().iter().map(|x| Value::S32(*x as i32)));
                }
            },
            Instruction::FlagsLift { flags: _, ty } => {
                let flags = require_matches!(&self.types[ty.index()], ValueType::Flags(x), x);

                let list = match operands.len() {
                    0 => FlagsList::Single(0),
                    1 => FlagsList::Single(require_matches!(
                        operands.pop(),
                        Some(Value::S32(x)),
                        x as u32
                    )),
                    _ => FlagsList::Multiple(Arc::new(
                        operands
                            .drain(..)
                            .map(|x| Ok(require_matches!(x, Value::S32(y), y) as u32))
                            .collect::<Result<_>>()?,
                    )),
                };

                results.push(Value::Flags(crate::values::Flags::new_unchecked(
                    flags.clone(),
                    list,
                )));
            },
            Instruction::ExtractVariantDiscriminant { discriminant_value } => {
                let (discriminant, val) = match operands
                    .pop()
                    .expect("No operand on stack for which to extract discriminant.")
                {
                    Value::Variant(x) => (x.discriminant(), x.value()),
                    Value::Enum(x) => (x.discriminant(), None),
                    Value::Option(x) => {
                        (x.is_some().then_some(1).unwrap_or_default(), (*x).clone())
                    },
                    Value::Result(x) => (
                        x.is_err().then_some(1).unwrap_or_default(),
                        match &*x {
                            std::result::Result::Ok(y) => y,
                            std::result::Result::Err(y) => y,
                        }
                        .clone(),
                    ),
                    _ => bail!("Invalid type for which to extract variant."),
                };

                if let Some(value) = val {
                    results.push(value);
                    discriminant_value.set((discriminant as i32, true));
                } else {
                    discriminant_value.set((discriminant as i32, false));
                }
            },
            Instruction::VariantLift {
                ty, discriminant, ..
            } => {
                let variant_ty =
                    require_matches!(&self.types[ty.index()], ValueType::Variant(x), x);
                results.push(Value::Variant(crate::values::Variant::new(
                    variant_ty.clone(),
                    *discriminant as usize,
                    operands.pop(),
                )?));
            },
            Instruction::EnumLower { enum_: _, ty: _ } => {
                let en = require_matches!(operands.pop(), Some(Value::Enum(x)), x);
                results.push(Value::S32(en.discriminant() as i32));
            },
            Instruction::EnumLift {
                enum_: _,
                ty,
                discriminant,
            } => {
                let enum_ty = require_matches!(&self.types[ty.index()], ValueType::Enum(x), x);
                results.push(Value::Enum(crate::values::Enum::new(
                    enum_ty.clone(),
                    *discriminant as usize,
                )?));
            },
            Instruction::OptionLift {
                ty, discriminant, ..
            } => {
                let option_ty = require_matches!(&self.types[ty.index()], ValueType::Option(x), x);
                results.push(Value::Option(OptionValue::new(
                    option_ty.clone(),
                    if *discriminant == 0 {
                        None
                    } else {
                        Some(require_matches!(operands.pop(), Some(x), x))
                    },
                )?));
            },
            Instruction::ResultLift {
                discriminant, ty, ..
            } => {
                let result_ty = require_matches!(&self.types[ty.index()], ValueType::Result(x), x);
                results.push(Value::Result(ResultValue::new(
                    result_ty.clone(),
                    if *discriminant == 0 {
                        std::result::Result::Ok(operands.pop())
                    } else {
                        std::result::Result::Err(operands.pop())
                    },
                )?));
            },
            Instruction::CallWasm { name: _, sig } => {
                let args = operands
                    .iter()
                    .map(TryFrom::try_from)
                    .collect::<Result<Vec<_>>>()?;
                self.flat_results = vec![wasm_runtime_layer::Value::I32(0); sig.results.len()];
                self.callee_wasm.expect("No available WASM callee.").call(
                    &mut self.ctx.as_context_mut().inner,
                    &args,
                    &mut self.flat_results,
                )?;
                results.extend(
                    self.flat_results
                        .iter()
                        .map(TryFrom::try_from)
                        .collect::<Result<Vec<_>>>()?,
                );
            },
            Instruction::CallInterface { func } => {
                for _i in 0..func.results.len() {
                    results.push(Value::Bool(false));
                }

                self.callee_interface
                    .expect("No available interface callee.")
                    .call(self.ctx.as_context_mut(), operands, &mut results[..])?;
            },
            Instruction::Return { amt: _, func: _ } => {
                if let Some(post) = &self.post_return {
                    post.call(
                        &mut self.ctx.as_context_mut().inner,
                        &self.flat_results,
                        &mut [],
                    )?;
                }

                let mut tables = self
                    .resource_tables
                    .try_lock()
                    .expect("Could not lock resource table.");
                for (res, idx) in &self.handles_to_drop {
                    tables[*res as usize]
                        .remove(*idx)
                        .expect("Could not find handle to drop.");
                }

                for (own, res, idx, ptr) in &self.required_dropped {
                    ensure!(
                        Arc::strong_count(ptr) == 1,
                        "Borrow was not dropped at the end of method."
                    );

                    if *own {
                        let table = &mut tables[*res as usize];
                        let mut elem = *table.get(*idx)?;

                        elem.lend_count -= 1;
                        table.set(*idx, elem);
                    }
                }

                for (i, val) in operands.drain(..).enumerate() {
                    *self
                        .results
                        .get_mut(i)
                        .ok_or_else(|| Error::msg("Unexpected number of output arguments."))? = val;
                }
            },
            Instruction::Malloc {
                realloc: _,
                size,
                align,
            } => {
                let realloc = self.realloc.as_ref().expect("No realloc.");
                let args = [
                    wasm_runtime_layer::Value::I32(0),
                    wasm_runtime_layer::Value::I32(0),
                    wasm_runtime_layer::Value::I32(*align as i32),
                    wasm_runtime_layer::Value::I32(*size as i32),
                ];
                let mut res = [wasm_runtime_layer::Value::I32(0)];
                realloc.call(&mut self.ctx.as_context_mut().inner, &args, &mut res)?;
                require_matches!(
                    &res[0],
                    wasm_runtime_layer::Value::I32(x),
                    results.push(Value::S32(*x))
                );
            },
        }

        Ok(())
    }

    fn sizes(&self) -> &SizeAlign {
        &self.component.size_align
    }

    fn is_list_canonical(&self, element: &Type) -> bool {
        /// Whether this is a little-endian machine.
        const LITTLE_ENDIAN: bool = cfg!(target_endian = "little");

        match element {
            Type::Bool => false,
            Type::U8 => true,
            Type::U16 => LITTLE_ENDIAN,
            Type::U32 => LITTLE_ENDIAN,
            Type::U64 => LITTLE_ENDIAN,
            Type::S8 => true,
            Type::S16 => LITTLE_ENDIAN,
            Type::S32 => LITTLE_ENDIAN,
            Type::S64 => LITTLE_ENDIAN,
            Type::Float32 => LITTLE_ENDIAN,
            Type::Float64 => LITTLE_ENDIAN,
            Type::Char => false,
            Type::String => false,
            Type::Id(_) => false,
        }
    }
}

/// A strongly-typed component model function that can be called to interact with [`Instance`]s.
#[derive(Clone, Debug)]
pub struct TypedFunc<P: ComponentList, R: ComponentList> {
    /// The inner function to call.
    inner: Func,
    /// A marker to prevent compiler errors.
    data: PhantomData<fn(P) -> R>,
}

impl<P: ComponentList, R: ComponentList> TypedFunc<P, R> {
    /// Creates a new function, wrapping the given closure.
    pub fn new<C: AsContextMut>(
        ctx: C,
        f: impl 'static + Send + Sync + Fn(StoreContextMut<C::UserState, C::Engine>, P) -> Result<R>,
    ) -> Self {
        let mut params_results = vec![ValueType::Bool; P::LEN + R::LEN];
        P::into_tys(&mut params_results[..P::LEN]);
        R::into_tys(&mut params_results[P::LEN..]);

        Self {
            inner: Func::new(
                ctx,
                FuncType::new(
                    params_results[..P::LEN].iter().cloned(),
                    params_results[P::LEN..].iter().cloned(),
                ),
                move |ctx, args, res| {
                    let p = P::from_values(args)?;
                    let r = f(ctx, p)?;
                    r.into_values(res)
                },
            ),
            data: PhantomData,
        }
    }

    /// Calls this function, returning an error if:
    ///
    /// - The store did not match the original.
    /// - A trap occurred.
    pub fn call(&self, ctx: impl AsContextMut, params: P) -> Result<R> {
        let mut params_results = vec![Value::Bool(false); P::LEN + R::LEN];
        params.into_values(&mut params_results[0..P::LEN])?;
        let (params, results) = params_results.split_at_mut(P::LEN);
        self.inner.call(ctx, params, results)?;
        R::from_values(results)
    }

    /// Gets the underlying, untyped function.
    pub fn func(&self) -> Func {
        self.inner.clone()
    }

    /// Gets the component model type of this function.
    pub fn ty(&self) -> FuncType {
        self.inner.ty.clone()
    }
}

/// Details the function name and instance in which an error occurred.
pub struct FuncError {
    /// The name of the function.
    name: String,
    /// The ID of the interface associated with the function.
    interface: Option<InterfaceIdentifier>,
    /// The instance.
    instance: crate::Instance,
    /// The error.
    error: Error,
}

impl FuncError {
    /// Gets the name of the function for which the error was thrown.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Gets the instance for which this error occurred.
    pub fn instance(&self) -> &crate::Instance {
        &self.instance
    }
}

impl std::fmt::Debug for FuncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(inter) = &self.interface {
            f.write_fmt(format_args!("in {}.{}: {:?}", inter, self.name, self.error))
        } else {
            f.write_fmt(format_args!("in {}: {:?}", self.name, self.error))
        }
    }
}

impl std::fmt::Display for FuncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(inter) = &self.interface {
            f.write_fmt(format_args!("in {}.{}: {}", inter, self.name, self.error))
        } else {
            f.write_fmt(format_args!("in {}: {}", self.name, self.error))
        }
    }
}

impl std::error::Error for FuncError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error.as_ref())
    }
}

/// Marks a type that may be blitted directly to and from guest memory.
trait Blittable: Sized {
    /// The type of byte array that matches the layout of this type.
    type Array: ByteArray;

    /// Creates this type from a byte array.
    fn from_bytes(array: Self::Array) -> Self;
    /// Converts this type to a byte array.
    fn to_bytes(self) -> Self::Array;

    /// Creates a new, zeroed byte array for an array of `Self` of the given size.
    fn zeroed_array(len: usize) -> Arc<[u8]>;
    /// Converts an array of bytes to an array of `Self`.
    fn from_le_array(array: Arc<[u8]>) -> Arc<[Self]>;
    /// Converts a slice of `Self` to a slice of bytes.
    fn to_le_slice(data: &[Self]) -> &[u8];
}

/// Implements the `Blittable` interface for primitive types.
macro_rules! impl_blittable {
    ($($type_impl: ident)*) => {
        $(
            impl Blittable for $type_impl {
                type Array = [u8; std::mem::size_of::<$type_impl>()];

                fn from_bytes(array: Self::Array) -> Self {
                    Self::from_le_bytes(array)
                }

                fn to_bytes(self) -> Self::Array {
                    Self::to_le_bytes(self)
                }

                fn zeroed_array(len: usize) -> Arc<[u8]> {
                    Arc::from(cast_slice_box(zeroed_slice_box::<Self>(len)))
                }

                fn from_le_array(array: Arc<[u8]>) -> Arc<[Self]> {
                    assert!(cfg!(target_endian = "little"), "Attempted to bitcast to little-endian bytes on a big endian platform.");
                    cast_slice_arc(array)
                }

                fn to_le_slice(data: &[Self]) -> &[u8] {
                    assert!(cfg!(target_endian = "little"), "Attempted to bitcast to little-endian bytes on a big endian platform.");
                    cast_slice(data)
                }
            }
        )*
    };
}

impl_blittable!(u8 u16 u32 u64 i8 i16 i32 i64 f32 f64);

/// Denotes a byte array of any size.
trait ByteArray: Sized {
    /// Loads this byte array from a WASM memory.
    fn load(ctx: impl AsContext, memory: &Memory, offset: usize) -> Result<Self>;

    /// Stores the contents of this byte array into a WASM memory.
    fn store(self, ctx: impl AsContextMut, memory: &Memory, offset: usize) -> Result<()>;
}

impl<const N: usize> ByteArray for [u8; N] {
    fn load(ctx: impl AsContext, memory: &Memory, offset: usize) -> Result<Self> {
        let mut res = [0; N];
        memory.read(ctx.as_context().inner, offset, &mut res)?;
        Ok(res)
    }

    fn store(self, mut ctx: impl AsContextMut, memory: &Memory, offset: usize) -> Result<()> {
        memory.write(ctx.as_context_mut().inner, offset, &self)?;
        Ok(())
    }
}

/// The type of a dynamic host function.
type FunctionBacking<T, E> =
    dyn 'static + Send + Sync + Fn(StoreContextMut<T, E>, &[Value], &mut [Value]) -> Result<()>;

/// The type of the key used in the vector of host functions.
type FunctionBackingKeyPair<T, E> = (Arc<AtomicUsize>, Arc<FunctionBacking<T, E>>);

/// A vector for functions that automatically drops items when the references are dropped.
pub(crate) struct FuncVec<T, E: backend::WasmEngine> {
    /// The functions stored in the vector.
    functions: Vec<FunctionBackingKeyPair<T, E>>,
}

impl<T, E: backend::WasmEngine> FuncVec<T, E> {
    /// Pushes a new function into the vector.
    pub fn push(
        &mut self,
        f: impl 'static + Send + Sync + Fn(StoreContextMut<T, E>, &[Value], &mut [Value]) -> Result<()>,
    ) -> Arc<AtomicUsize> {
        if self.functions.capacity() == self.functions.len() {
            self.clear_dead_functions();
        }
        let idx = Arc::new(AtomicUsize::new(self.functions.len()));
        self.functions.push((idx.clone(), Arc::new(f)));
        idx
    }

    /// Gets a function from the vector.
    pub fn get(&self, value: &AtomicUsize) -> Arc<FunctionBacking<T, E>> {
        self.functions[value.load(Ordering::Acquire)].1.clone()
    }

    /// Clears all dead functions from the vector, and doubles its capacity.
    fn clear_dead_functions(&mut self) {
        let new_len = 2 * self.functions.len();
        let old = replace(&mut self.functions, Vec::with_capacity(new_len));
        for (idx, val) in old {
            if Arc::strong_count(&idx) > 1 {
                idx.store(self.functions.len(), Ordering::Release);
                self.functions.push((idx, val));
            }
        }
    }
}

impl<T, E: backend::WasmEngine> Default for FuncVec<T, E> {
    fn default() -> Self {
        Self {
            functions: Vec::new(),
        }
    }
}
