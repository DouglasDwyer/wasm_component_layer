
use std::marker::*;
use std::mem::*;
use std::sync::atomic::*;
use std::sync::*;
use std::usize;

use bytemuck::*;
use wasm_runtime_layer::*;
use wasmtime_environ::component::*;

use crate::abi::{Generator, *};
use crate::types::{FuncType, ValueType};
use crate::values::Value;
use crate::{AsContext, AsContextMut, StoreContextMut, *};

#[derive(Clone, Debug)]
pub(crate) enum FuncImpl {
    GuestFunc(Arc<GuestFunc>),
    HostFunc(Arc<AtomicUsize>),
}

#[derive(Debug)]
pub(crate) struct GuestFunc {
    pub callee: wasm_runtime_layer::Func,
    pub component: Arc<ComponentInner>,
    pub encoding: StringEncoding,
    pub function: Function,
    pub memory: Option<Memory>,
    pub realloc: Option<wasm_runtime_layer::Func>,
    pub post_return: Option<wasm_runtime_layer::Func>,
    pub resource_tables: Arc<Mutex<Vec<HandleTable>>>,
    pub types: Arc<[crate::types::ValueType]>,
    pub instance_id: u64
}

#[derive(Clone, Debug)]
pub struct Func {
    pub(crate) store_id: u64,
    pub(crate) ty: FuncType,
    pub(crate) backing: FuncImpl,
}

impl Func {
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
            FuncImpl::GuestFunc(x) => {
                let GuestFunc {
                    callee,
                    component,
                    encoding,
                    function,
                    memory,
                    realloc,
                    resource_tables,
                    post_return,
                    types,
                    instance_id,
                } = &**x;

                let mut bindgen = FuncBindgen {
                    ctx,
                    flat_results: Vec::default(),
                    arguments,
                    results,
                    callee_interface: None,
                    callee_wasm: Some(callee),
                    component: &**component,
                    encoding,
                    memory,
                    realloc,
                    resource_tables: &resource_tables,
                    post_return,
                    types: &types,
                    handles_to_drop: Vec::new(),
                    required_dropped: Vec::new(),
                    instance_id: *instance_id,
                    store_id: self.store_id
                };

                Generator::new(
                    &component.resolve,
                    AbiVariant::GuestExport,
                    LiftLower::LowerArgsLiftResults,
                    &mut bindgen,
                )
                .call(function)
            }
            FuncImpl::HostFunc(idx) => {
                let callee = ctx.as_context().inner.data().host_functions.get(&idx);
                (callee)(ctx.as_context_mut(), arguments, results)?;
                self.ty.match_results(results)
            }
        }
    }

    pub fn ty(&self) -> FuncType {
        self.ty.clone()
    }

    pub fn typed<P: ComponentList, R: ComponentList>(&self) -> Result<TypedFunc<P, R>> {
        let mut params_results = vec![ValueType::Bool; P::LEN + R::LEN];
        P::into_tys(&mut params_results[..P::LEN]);
        R::into_tys(&mut params_results[P::LEN..]);
        ensure!(
            &params_results[..P::LEN] == self.ty.params(),
            "Parameters did not match function signature."
        );
        ensure!(
            &params_results[P::LEN..] == self.ty.results(),
            "Results did not match function signature."
        );
        Ok(TypedFunc {
            inner: self.clone(),
            data: PhantomData,
        })
    }

    pub(crate) fn call_from_guest<C: AsContextMut>(
        &self,
        ctx: C,
        options: &GuestInvokeOptions,
        arguments: &[wasm_runtime_layer::Value],
        results: &mut [wasm_runtime_layer::Value],
    ) -> Result<()> {
        ensure!(self.store_id == options.store_id, "Function stores did not match.");

        let args = arguments
            .iter()
            .map(TryFrom::try_from)
            .collect::<Result<Vec<_>>>()?;
        let mut res = results
            .iter()
            .map(TryFrom::try_from)
            .collect::<Result<Vec<_>>>()?;

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
            resource_tables: &options.resource_tables,
            post_return: &options.post_return,
            types: &options.types,
            handles_to_drop: Vec::new(),
            required_dropped: Vec::new(),
            instance_id: options.instance_id,
            store_id: self.store_id
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

pub(crate) struct GuestInvokeOptions {
    pub component: Arc<ComponentInner>,
    pub encoding: StringEncoding,
    pub function: Function,
    pub memory: Option<Memory>,
    pub realloc: Option<wasm_runtime_layer::Func>,
    pub post_return: Option<wasm_runtime_layer::Func>,
    pub resource_tables: Arc<Mutex<Vec<HandleTable>>>,
    pub types: Arc<[crate::types::ValueType]>,
    pub instance_id: u64,
    pub store_id: u64
}
struct FuncBindgen<'a, C: AsContextMut> {
    pub callee_interface: Option<&'a Func>,
    pub callee_wasm: Option<&'a wasm_runtime_layer::Func>,
    pub component: &'a ComponentInner,
    pub ctx: C,
    pub encoding: &'a StringEncoding,
    pub flat_results: Vec<wasm_runtime_layer::Value>,
    pub memory: &'a Option<Memory>,
    pub realloc: &'a Option<wasm_runtime_layer::Func>,
    pub post_return: &'a Option<wasm_runtime_layer::Func>,
    pub arguments: &'a [Value],
    pub results: &'a mut [Value],
    pub resource_tables: &'a Mutex<Vec<HandleTable>>,
    pub types: &'a [crate::types::ValueType],
    pub handles_to_drop: Vec<(u32, i32)>,
    pub required_dropped: Vec<(bool, u32, i32, Arc<AtomicBool>)>,
    pub instance_id: u64,
    pub store_id: u64
}

impl<'a, C: AsContextMut> FuncBindgen<'a, C> {
    fn load<B: Blittable>(&self, offset: usize) -> Result<B> {
        Ok(B::from_bytes(<B::Array as ByteArray>::load(
            &self.ctx,
            self.memory.as_ref().expect("No memory."),
            offset,
        )?))
    }

    fn store<B: Blittable>(&mut self, offset: usize, value: B) -> Result<()> {
        value.to_bytes().store(
            &mut self.ctx,
            self.memory.as_ref().expect("No memory."),
            offset,
        )
    }

    fn load_array<B: Blittable>(&self, offset: usize, len: usize) -> Result<Arc<[B]>> {
        let mut raw_memory = B::zeroed_array(len);
        self.memory.as_ref().expect("No memory").read(
            self.ctx.as_context().inner,
            offset,
            &mut Arc::get_mut(&mut raw_memory).expect("Could not get exclusive reference."),
        )?;
        Ok(B::from_le_array(raw_memory))
    }

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
                        }
                        Bitcast::I64ToI32 => {
                            require_matches!(op, Value::S64(x), results.push(Value::S32(*x as i32)))
                        }
                        Bitcast::I64ToF32 => {
                            require_matches!(op, Value::S64(x), results.push(Value::F32(*x as f32)))
                        }
                        Bitcast::F32ToI64 => {
                            require_matches!(op, Value::F32(x), results.push(Value::S64(*x as i64)))
                        }
                        Bitcast::None => results.push(op.clone()),
                    }
                }
            }
            Instruction::ConstZero { tys } => {
                for t in tys.iter() {
                    match t {
                        WasmType::I32 => results.push(Value::S32(0)),
                        WasmType::I64 => results.push(Value::S64(0)),
                        WasmType::F32 => results.push(Value::F32(0.0)),
                        WasmType::F64 => results.push(Value::F64(0.0)),
                    }
                }
            }
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
                results.push(Value::Bool(if x > 0 { true } else { false }))
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
            }
            Instruction::ListCanonLower { element, realloc: _ } => {
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
            }
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
            }
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
                    }
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
                    }
                }
            }
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
            }
            Instruction::ListLift { element: _, ty, len: _ } => {
                let ty = self.types[ty.index()].clone();
                results.push(Value::List(List::new(
                    require_matches!(ty, crate::types::ValueType::List(x), x),
                    operands.drain(..),
                )?));
            }
            Instruction::ReadI32 { value } => {
                value.set(require_matches!(operands.pop(), Some(Value::S32(x)), x))
            }
            Instruction::RecordLower { record: _, name: _, ty } => {
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
            }
            Instruction::RecordLift { record: _, name: _, ty } => {
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
            }
            Instruction::HandleLower { handle, name: _, ty } => match &self.types[ty.index()] {
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
                }
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
                }
                _ => unreachable!(),
            },
            Instruction::HandleLift { handle, name: _, ty } => match &self.types[ty.index()] {
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

                    if ty.host_destructor().is_some() {
                        results.push(Value::Own(ResourceOwn::new_guest(elem.rep, ty.clone(), self.store_id, None)));
                    } else {
                        results.push(Value::Own(ResourceOwn::new_guest(
                            elem.rep,
                            ty.clone(),
                            self.store_id,
                            table.destructor().cloned(),
                        )));
                    }
                }
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
                }
                _ => unreachable!(),
            },
            Instruction::TupleLower { tuple: _, ty: _ } => {
                let tuple = require_matches!(operands.pop(), Some(Value::Tuple(x)), x);
                results.extend(tuple.iter().cloned());
            }
            Instruction::TupleLift { tuple: _, ty } => {
                results.push(Value::Tuple(crate::values::Tuple::new_unchecked(
                    require_matches!(&self.types[ty.index()], ValueType::Tuple(x), x.clone()),
                    operands.drain(..),
                )));
            }
            Instruction::FlagsLower { flags: _, name: _, ty: _ } => {
                let flags = require_matches!(operands.pop(), Some(Value::Flags(x)), x);
                if flags.ty().names().len() > 0 {
                    results.extend(flags.as_u32_list().iter().map(|x| Value::S32(*x as i32)));
                }
            }
            Instruction::FlagsLift { flags: _, name: _, ty } => {
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
            }
            Instruction::ExtractVariantDiscriminant { discriminant_value } => {
                let (discriminant, val) = match operands
                    .pop()
                    .expect("No operand on stack for which to extract discriminant.")
                {
                    Value::Variant(x) => (x.discriminant(), x.value()),
                    Value::Enum(x) => (x.discriminant(), None),
                    Value::Union(x) => (x.discriminant(), Some(x.value())),
                    Value::Option(x) => {
                        (x.is_some().then_some(1).unwrap_or_default(), (*x).clone())
                    }
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
            }
            Instruction::VariantLift {
                
                
                ty,
                discriminant,
                ..
            } => {
                let variant_ty =
                    require_matches!(&self.types[ty.index()], ValueType::Variant(x), x);
                results.push(Value::Variant(crate::values::Variant::new(
                    variant_ty.clone(),
                    *discriminant as usize,
                    operands.pop(),
                )?));
            }
            Instruction::UnionLift {
                union: _,
                name: _,
                ty,
                discriminant,
            } => {
                let union_ty = require_matches!(&self.types[ty.index()], ValueType::Union(x), x);
                let value = require_matches!(operands.pop(), Some(x), x);
                results.push(Value::Union(crate::values::Union::new(
                    union_ty.clone(),
                    *discriminant as usize,
                    value,
                )?));
            }
            Instruction::EnumLower { enum_: _, name: _, ty: _ } => {
                let en = require_matches!(operands.pop(), Some(Value::Enum(x)), x);
                results.push(Value::S32(en.discriminant() as i32));
            }
            Instruction::EnumLift {
                enum_: _,
                name: _,
                ty,
                discriminant,
            } => {
                let enum_ty = require_matches!(&self.types[ty.index()], ValueType::Enum(x), x);
                results.push(Value::Enum(crate::values::Enum::new(
                    enum_ty.clone(),
                    *discriminant as usize,
                )?));
            }
            Instruction::OptionLift {
                
                ty,
                discriminant,
                ..
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
            }
            Instruction::ResultLift {
                
                discriminant,
                ty,
                ..
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
            }
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
            }
            Instruction::CallInterface { func } => {
                for _i in 0..func.results.len() {
                    results.push(Value::Bool(false));
                }

                self.callee_interface
                    .expect("No available interface callee.")
                    .call(self.ctx.as_context_mut(), &operands, &mut results[..])?;
            }
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
                    tables[*res as usize].remove(*idx).expect("Could not find handle to drop.");
                }

                for (own, res, idx, ptr) in &self.required_dropped {
                    if *own {
                        let table = &mut tables[*res as usize];
                        let mut elem = *table.get(*idx)?;

                        elem.lend_count -= 1;
                        table.set(*idx, elem);
                    }

                    ensure!(
                        Arc::strong_count(ptr) == 1,
                        "Borrow was not dropped at the end of method."
                    );
                }

                for (i, val) in operands.drain(..).enumerate() {
                    *self
                        .results
                        .get_mut(i)
                        .ok_or_else(|| Error::msg("Unexpected number of output arguments."))? = val;
                }
            }
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
            }
        }

        Ok(())
    }

    fn sizes(&self) -> &SizeAlign {
        &self.component.size_align
    }

    fn is_list_canonical(&self, element: &Type) -> bool {
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

#[derive(Clone, Debug)]
pub struct TypedFunc<P: ComponentList, R: ComponentList> {
    inner: Func,
    data: PhantomData<fn(P) -> R>,
}

impl<P: ComponentList, R: ComponentList> TypedFunc<P, R> {
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

    pub fn call(&self, ctx: impl AsContextMut, params: P) -> Result<R> {
        let mut params_results = vec![Value::Bool(false); P::LEN + R::LEN];
        params.into_values(&mut params_results[0..P::LEN])?;
        let (params, results) = params_results.split_at_mut(P::LEN);
        self.inner.call(ctx, params, results)?;
        R::from_values(results)
    }

    pub fn func(&self) -> Func {
        self.inner.clone()
    }

    pub fn ty(&self) -> FuncType {
        self.inner.ty.clone()
    }
}

trait Blittable: Sized {
    type Array: ByteArray;

    fn from_bytes(array: Self::Array) -> Self;
    fn to_bytes(self) -> Self::Array;

    fn zeroed_array(len: usize) -> Arc<[u8]>;
    fn from_le_array(array: Arc<[u8]>) -> Arc<[Self]>;
    fn to_le_slice(data: &[Self]) -> &[u8];
}

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
trait ByteArray: Sized {
    fn load(ctx: impl AsContext, memory: &Memory, offset: usize) -> Result<Self>;
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

pub(crate) struct FuncVec<T, E: backend::WasmEngine> {
    functions: Vec<(
        Arc<AtomicUsize>,
        Arc<
            dyn 'static
                + Send
                + Sync
                + Fn(StoreContextMut<T, E>, &[Value], &mut [Value]) -> Result<()>,
        >,
    )>,
}

impl<T, E: backend::WasmEngine> FuncVec<T, E> {
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

    pub fn get(
        &self,
        value: &AtomicUsize,
    ) -> Arc<
        dyn 'static + Send + Sync + Fn(StoreContextMut<T, E>, &[Value], &mut [Value]) -> Result<()>,
    > {
        self.functions[value.load(Ordering::Acquire)].1.clone()
    }

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
