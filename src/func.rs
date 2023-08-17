use crate::*;
use crate::abi::*;
use crate::values::Value;
use crate::types::ValueType;
use std::mem::*;
use std::sync::*;
use std::usize;
use wasmtime_environ::component::*;
use wasm_runtime_layer::*;

#[derive(Clone, Debug)]
pub struct Func(pub(crate) Arc<FuncInner>);

impl Func {
    pub fn call(&self, ctx: impl AsContextMut, arguments: &[Value], results: &mut [Value]) -> Result<()> {
        let mut bindgen = FuncBindgen {
            ctx,
            func: &self.0,
            error: None,
            flat_results: Vec::default(),
            arguments,
            results
        };

        Generator::new(&self.0.component.resolve, AbiVariant::GuestExport, LiftLower::LowerArgsLiftResults, &mut bindgen).call(&self.0.function)?;

        if let Some(err) = bindgen.error {
            Err(err)
        }
        else {
            Ok(())
        }
    }

    /*pub fn params(&self) -> Box<[crate::types::Type]> {
        self.0.component.types[self.0.component.types[self.0.ty].params]
            .types
            .iter()
            .map(|ty| crate::types::Type::from(ty, self.0.component.clone()))
            .collect()
    }

    pub fn results(&self) -> Box<[crate::types::Type]> {
        self.0.component.types[self.0.component.types[self.0.ty].results]
            .types
            .iter()
            .map(|ty| crate::types::Type::from(ty, self.0.component.clone()))
            .collect()
    }*/
}

macro_rules! require_matches {
    ($expression:expr, $pattern:pat $(if $guard:expr)?, $then: expr) => {
        match $expression {
            $pattern $(if $guard)? => $then,
            _ => bail!("Incorrect type.")
        }
    };
}

#[derive(Clone, Debug)]
pub(crate) struct FuncInner {
    pub callee: wasm_runtime_layer::Func,
    pub component: Arc<ComponentInner>,
    pub encoding: StringEncoding,
    pub function: Function,
    pub memory: Option<Memory>,
    pub realloc: Option<wasm_runtime_layer::Func>,
    pub post_return: Option<wasm_runtime_layer::Func>,
    pub ty: TypeFuncIndex,
}

//, F: Fn(StoreContextMut<'_, C::UserState, C::Engine>, &[Value], &mut [Value]) -> Result<()>

struct FuncBindgen<'a, C: AsContextMut> {
    pub ctx: C,
    pub func: &'a FuncInner,
    pub error: Option<Error>,
    pub flat_results: Vec<wasm_runtime_layer::Value>,
    pub arguments: &'a [Value],
    pub results: &'a mut [Value]
}

impl<'a, C: AsContextMut> FuncBindgen<'a, C> {
    fn load<B: Blittable>(&self, offset: usize) -> Result<B> {
        Ok(B::from_bytes(<B::Array as ByteArray>::load(&self.ctx, self.func.memory.as_ref().expect("No memory."), offset)?))
    }

    fn store<B: Blittable>(&mut self, offset: usize, value: B) -> Result<()> {
        value.to_bytes().store(&mut self.ctx, self.func.memory.as_ref().expect("No memory."), offset)
    }
}

impl<'a, C: AsContextMut> Bindgen for FuncBindgen<'a, C> {
    type Operand = Value;

    fn emit(
        &mut self,
        resolve: &Resolve,
        inst: &Instruction<'_>,
        operands: &mut Vec<Self::Operand>,
        results: &mut Vec<Self::Operand>,
    ) -> Result<()> {
        match inst {
            Instruction::GetArg { nth } => results.push(self.arguments.get(*nth).cloned().ok_or_else(|| Error::msg("Invalid argument count."))?),
            Instruction::I32Const { val } => results.push(Value::S32(*val)),
            Instruction::Bitcasts { casts } => {
                for (cast, op) in casts.iter().zip(operands) {
                    match cast {
                        Bitcast::I32ToF32 => require_matches!(op, Value::S32(x), results.push(Value::F32(f32::from_bits(*x as u32)))),
                        Bitcast::F32ToI32 => require_matches!(op, Value::F32(x), results.push(Value::S32(x.to_bits() as i32))),
                        Bitcast::I64ToF64 => require_matches!(op, Value::S64(x), results.push(Value::F64(f64::from_bits(*x as u64)))),
                        Bitcast::F64ToI64 => require_matches!(op, Value::F64(x), results.push(Value::S64(x.to_bits() as i64))),
                        Bitcast::I32ToI64 => require_matches!(op, Value::S32(x), results.push(Value::S64(*x as i64))),
                        Bitcast::I64ToI32 => require_matches!(op, Value::S64(x), results.push(Value::S32(*x as i32))),
                        Bitcast::I64ToF32 => require_matches!(op, Value::S64(x), results.push(Value::F32(*x as f32))),
                        Bitcast::F32ToI64 => require_matches!(op, Value::F32(x), results.push(Value::S64(*x as i64))),
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
                        WasmType::F64 => results.push(Value::F64(0.0))
                    }
                }
            },
            Instruction::I32Load { offset } => require_matches!(operands.pop(), Some(Value::S32(x)), results.push(Value::S32(self.load((x as usize) + (*offset as usize))?))),
            Instruction::I32Load8U { offset } => require_matches!(operands.pop(), Some(Value::S32(x)), results.push(Value::S32(self.load::<u8>((x as usize) + (*offset as usize))? as i32))),
            Instruction::I32Load8S { offset } => require_matches!(operands.pop(), Some(Value::S32(x)), results.push(Value::S32(self.load::<i8>((x as usize) + (*offset as usize))? as i32))),
            Instruction::I32Load16U { offset } => require_matches!(operands.pop(), Some(Value::S32(x)), results.push(Value::S32(self.load::<u16>((x as usize) + (*offset as usize))? as i32))),
            Instruction::I32Load16S { offset } => require_matches!(operands.pop(), Some(Value::S32(x)), results.push(Value::S32(self.load::<i16>((x as usize) + (*offset as usize))? as i32))),
            Instruction::I64Load { offset } =>  require_matches!(operands.pop(), Some(Value::S32(x)), results.push(Value::S64(self.load((x as usize) + (*offset as usize))?))),
            Instruction::F32Load { offset } =>  require_matches!(operands.pop(), Some(Value::S32(x)), results.push(Value::F32(self.load((x as usize) + (*offset as usize))?))),
            Instruction::F64Load { offset } =>  require_matches!(operands.pop(), Some(Value::S32(x)), results.push(Value::F64(self.load((x as usize) + (*offset as usize))?))),
            Instruction::I32Store { offset } => require_matches!(operands.pop(), Some(Value::S32(address)), require_matches!(operands.pop(), Some(Value::S32(x)), self.store((address as usize) + (*offset as usize), x)?)),
            Instruction::I32Store8 { offset } => require_matches!(operands.pop(), Some(Value::S32(address)), require_matches!(operands.pop(), Some(Value::S32(x)), self.store((address as usize) + (*offset as usize), x as u8)?)),
            Instruction::I32Store16 { offset } => require_matches!(operands.pop(), Some(Value::S32(address)), require_matches!(operands.pop(), Some(Value::S32(x)), self.store((address as usize) + (*offset as usize), x as u16)?)),
            Instruction::I64Store { offset } => require_matches!(operands.pop(), Some(Value::S32(address)), require_matches!(operands.pop(), Some(Value::S64(x)), self.store((address as usize) + (*offset as usize), x)?)),
            Instruction::F32Store { offset } => require_matches!(operands.pop(), Some(Value::S32(address)), require_matches!(operands.pop(), Some(Value::F32(x)), self.store((address as usize) + (*offset as usize), x)?)),
            Instruction::F64Store { offset } => require_matches!(operands.pop(), Some(Value::S32(address)), require_matches!(operands.pop(), Some(Value::F64(x)), self.store((address as usize) + (*offset as usize), x)?)),
            Instruction::I32FromChar => require_matches!(operands.pop(), Some(Value::Char(x)), results.push(Value::S32(x as i32))),
            Instruction::I64FromU64 => require_matches!(operands.pop(), Some(Value::U64(x)), results.push(Value::S64(x as i64))),
            Instruction::I64FromS64 => require_matches!(operands.pop(), Some(Value::S64(x)), results.push(Value::S64(x))),
            Instruction::I32FromU32 => require_matches!(operands.pop(), Some(Value::U32(x)), results.push(Value::S32(x as i32))),
            Instruction::I32FromS32 => require_matches!(operands.pop(), Some(Value::S32(x)), results.push(Value::S32(x))),
            Instruction::I32FromU16 => require_matches!(operands.pop(), Some(Value::U16(x)), results.push(Value::S32(x as i32))),
            Instruction::I32FromS16 => require_matches!(operands.pop(), Some(Value::S16(x)), results.push(Value::S32(x as i32))),
            Instruction::I32FromU8 => require_matches!(operands.pop(), Some(Value::U8(x)), results.push(Value::S32(x as i32))),
            Instruction::I32FromS8 => require_matches!(operands.pop(), Some(Value::S8(x)), results.push(Value::S32(x as i32))),
            Instruction::F32FromFloat32 => require_matches!(operands.pop(), Some(Value::F32(x)), results.push(Value::F32(x))),
            Instruction::F64FromFloat64 => require_matches!(operands.pop(), Some(Value::F64(x)), results.push(Value::F64(x))),
            Instruction::S8FromI32 => require_matches!(operands.pop(), Some(Value::S32(x)), results.push(Value::S8(x as i8))),
            Instruction::U8FromI32 => require_matches!(operands.pop(), Some(Value::S32(x)), results.push(Value::U8(x as u8))),
            Instruction::S16FromI32 => require_matches!(operands.pop(), Some(Value::S32(x)), results.push(Value::S16(x as i16))),
            Instruction::U16FromI32 => require_matches!(operands.pop(), Some(Value::S32(x)), results.push(Value::U16(x as u16))),
            Instruction::S32FromI32 => require_matches!(operands.pop(), Some(Value::S32(x)), results.push(Value::S32(x))),
            Instruction::U32FromI32 => require_matches!(operands.pop(), Some(Value::S32(x)), results.push(Value::U32(x as u32))),
            Instruction::S64FromI64 => require_matches!(operands.pop(), Some(Value::S64(x)), results.push(Value::S64(x))),
            Instruction::U64FromI64 => require_matches!(operands.pop(), Some(Value::S64(x)), results.push(Value::U64(x as u64))),
            Instruction::CharFromI32 => require_matches!(operands.pop(), Some(Value::S32(x)), results.push(Value::Char(char::from_u32(x as u32).ok_or_else(|| Error::msg("Could not convert integer to char."))?))),
            Instruction::Float32FromF32 => require_matches!(operands.pop(), Some(Value::F32(x)), results.push(Value::F32(x))),
            Instruction::Float64FromF64 => require_matches!(operands.pop(), Some(Value::F64(x)), results.push(Value::F64(x))),
            Instruction::BoolFromI32 => require_matches!(operands.pop(), Some(Value::S32(x)), results.push(Value::Bool(if x > 0 { true } else { false }))),
            Instruction::I32FromBool => require_matches!(operands.pop(), Some(Value::Bool(x)), results.push(Value::S32(x as i32))),
            Instruction::StringLower { realloc } => {
                let encoded = require_matches!(operands.pop(), Some(Value::String(x)), match self.func.encoding {
                    StringEncoding::Utf8 => Vec::from_iter(x.bytes()),
                    StringEncoding::Utf16 | StringEncoding::CompactUtf16 => x.encode_utf16().flat_map(|a| a.to_le_bytes()).collect()
                });

                let realloc = self.func.realloc.as_ref().expect("No realloc.");
                let args = [wasm_runtime_layer::Value::I32(0), wasm_runtime_layer::Value::I32(0), wasm_runtime_layer::Value::I32(1), wasm_runtime_layer::Value::I32(encoded.len() as i32)];
                let mut res = [wasm_runtime_layer::Value::I32(0)];
                realloc.call(&mut self.ctx, &args, &mut res)?;
                let ptr = require_matches!(&res[0], wasm_runtime_layer::Value::I32(x), *x);

                let memory = self.func.memory.as_ref().expect("No memory.");
                memory.write(&mut self.ctx, ptr as usize, &encoded)?;

                results.push(Value::S32(ptr));
                results.push(Value::S32(encoded.len() as i32));
            },
            Instruction::ListLower { element, realloc, len } => {
                let list = require_matches!(operands.pop(), Some(Value::List(x)), x);
                let align = self.func.component.size_align.align(element);
                let size = self.func.component.size_align.size(element);

                let realloc = self.func.realloc.as_ref().expect("No realloc.");
                let args = [wasm_runtime_layer::Value::I32(0), wasm_runtime_layer::Value::I32(0), wasm_runtime_layer::Value::I32(align as i32), wasm_runtime_layer::Value::I32((list.len() * size) as i32)];
                let mut res = [wasm_runtime_layer::Value::I32(0)];
                realloc.call(&mut self.ctx, &args, &mut res)?;
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
                let memory = self.func.memory.as_ref().expect("No memory.");
                let mut len = require_matches!(operands.pop(), Some(Value::S32(len)), len) as usize;
                let mut result = vec!(0; len);
                require_matches!(operands.pop(), Some(Value::S32(ptr)), memory.read(&self.ctx, ptr as usize, &mut result));
                
                match self.func.encoding {
                    StringEncoding::Utf8 => results.push(Value::String(String::from_utf8(result)?.into())),
                    StringEncoding::Utf16 | StringEncoding::CompactUtf16 => {
                        ensure!(result.len() & 0b1 == 0, "Invalid string length");
                        results.push(Value::String(String::from_utf16(&result.chunks_exact(2)
                            .map(|e| u16::from_be_bytes(e.try_into().expect("All chunks must have size 2.")))
                            .collect::<Vec<_>>())?.into()));
                    },
                }
            },
            Instruction::ListLift { element, ty, len } => {
                let ty = self.func.component.types[ty.index()].clone();
                results.push(Value::List(List::new(require_matches!(ty, crate::types::ValueType::List(x), x), operands.drain(..))?));
            },
            Instruction::ReadI32 { value } => value.set(require_matches!(operands.pop(), Some(Value::S32(x)), x)),
            Instruction::RecordLower { record, name, ty } => {
                let official_ty = require_matches!(&self.func.component.types[ty.index()], ValueType::Record(x), x);
                let record = require_matches!(operands.pop(), Some(Value::Record(x)), x);
                ensure!(&record.ty() == official_ty, "Record types did not match.");
                
                for i in 0..record.fields().len() {
                    results.push(Value::Bool(false));
                }

                for (index, value) in official_ty.fields.iter().map(|x| x.0).zip(record.fields().map(|x| x.1)) {
                    results[index] = value;
                }
            },
            Instruction::RecordLift { record, name, ty } => {
                let official_ty = require_matches!(&self.func.component.types[ty.index()], ValueType::Record(x), x);
                ensure!(operands.len() == official_ty.fields().len(), "Record types did not match.");

                results.push(Value::Record(crate::values::Record::from_sorted(official_ty.clone(), official_ty.fields.iter().map(|(i, name, _)| (name.clone(), replace(&mut operands[*i], Value::Bool(false)))))));
                operands.clear();
            },
            Instruction::HandleLower { handle, name, ty } => bail!("Not yet implemented."),
            Instruction::HandleLift { handle, name, ty } => bail!("Not yet implemented."),
            Instruction::TupleLower { tuple, ty } => {
                let tuple = require_matches!(operands.pop(), Some(Value::Tuple(x)), x);
                results.extend(tuple.iter().cloned());
            },
            Instruction::TupleLift { tuple, ty } => {
                results.push(Value::Tuple(crate::values::Tuple::new_unchecked(require_matches!(&self.func.component.types[ty.index()], ValueType::Tuple(x), x.clone()), operands.drain(..))));
            },
            Instruction::FlagsLower { flags, name, ty } => {
                let flags = require_matches!(operands.pop(), Some(Value::Flags(x)), x);
                if flags.ty().names().len() > 0 {
                    results.extend(flags.as_u32_list().iter().map(|x| Value::S32(*x as i32)));
                }
            },
            Instruction::FlagsLift { flags, name, ty } => {
                let flags = require_matches!(&self.func.component.types[ty.index()], ValueType::Flags(x), x);

                let list = match operands.len() {
                    0 => FlagsList::Single(0),
                    1 => FlagsList::Single(require_matches!(operands.pop(), Some(Value::S32(x)), x as u32)),
                    _ => FlagsList::Multiple(Arc::new(operands.drain(..).map(|x| Ok(require_matches!(x, Value::S32(y), y) as u32)).collect::<Result<_>>()?))
                };

                results.push(Value::Flags(crate::values::Flags::new_unchecked(flags.clone(), list)));
            },
            Instruction::ExtractVariantDiscriminant { discriminant_value } => {
                let (discriminant, val) = match operands.pop().expect("No operand on stack for which to extract discriminant.") {
                    Value::Variant(x) => (x.discriminant(), x.value()),
                    Value::Enum(x) => (x.discriminant(), None),
                    Value::Union(x) => (x.discriminant(), Some(x.value())),
                    Value::Option(x) => (x.is_some().then_some(1).unwrap_or_default(), (*x).clone()),
                    Value::Result(x) => (x.is_err().then_some(1).unwrap_or_default(), match x { std::result::Result::Ok(y) => y, std::result::Result::Err(y) => y }),
                    _ => bail!("Invalid type for which to extract variant.")
                };

                if let Some(value) = val {
                    results.push(value);
                    discriminant_value.set((discriminant as i32, true));
                }
                else {
                    discriminant_value.set((discriminant as i32, false));
                }
            },
            Instruction::VariantPayloadName => todo!(),
            Instruction::VariantLower { variant, name, ty, results } => todo!(),
            Instruction::VariantLift { variant, name, ty, discriminant, .. } => {
                let variant_ty = require_matches!(&self.func.component.types[ty.index()], ValueType::Variant(x), x);
                results.push(Value::Variant(crate::values::Variant::new(variant_ty.clone(), *discriminant as usize, operands.pop())?));
            },
            Instruction::UnionLower { union, name, ty, results } => todo!(),
            Instruction::UnionLift { union, name, ty, discriminant } => {
                let union_ty = require_matches!(&self.func.component.types[ty.index()], ValueType::Union(x), x);
                let value = require_matches!(operands.pop(), Some(x), x);
                results.push(Value::Union(crate::values::Union::new(union_ty.clone(), *discriminant as usize, value)?));
            },
            Instruction::EnumLower { enum_, name, ty } => {
                let en = require_matches!(operands.pop(), Some(Value::Enum(x)), x);
                results.push(Value::S32(en.discriminant() as i32));
            },
            Instruction::EnumLift { enum_, name, ty, discriminant } => {
                let enum_ty = require_matches!(&self.func.component.types[ty.index()], ValueType::Enum(x), x);
                results.push(Value::Enum(crate::values::Enum::new(enum_ty.clone(), *discriminant as usize)?));
            },
            Instruction::OptionLower { payload, ty, results } => todo!(),
            Instruction::OptionLift { payload, ty, discriminant, .. } => {
                let option_ty = require_matches!(&self.func.component.types[ty.index()], ValueType::Option(x), x);
                results.push(Value::Option(OptionValue::new(option_ty.clone(), if *discriminant == 0 { None } else { Some(require_matches!(operands.pop(), Some(x), x)) })?));
            },
            Instruction::ResultLower { result, ty, results } => todo!(),
            Instruction::ResultLift { result, discriminant, ty, .. } => {
                let result_ty = require_matches!(&self.func.component.types[ty.index()], ValueType::Result(x), x);
                results.push(Value::Result(ResultValue::new(result_ty.clone(), if *discriminant == 0 { std::result::Result::Ok(operands.pop()) } else { std::result::Result::Err(operands.pop()) })?));
            },
            Instruction::CallWasm { name, sig } => {
                let args = operands.iter().map(TryFrom::try_from).collect::<Result<Vec<_>>>()?;
                self.flat_results = vec!(wasm_runtime_layer::Value::I32(0); sig.results.len());
                self.func.callee.call(&mut self.ctx, &args, &mut self.flat_results)?;
                results.extend(self.flat_results.iter().map(TryFrom::try_from).collect::<Result<Vec<_>>>()?);
            },
            Instruction::CallInterface { func } => todo!(),
            Instruction::Return { amt, func } => {
                if let Some(post) = &self.func.post_return {
                    post.call(&mut self.ctx, &self.flat_results, &mut [])?;
                }

                for (i, val) in operands.drain(..).enumerate() {
                    *self.results.get_mut(i).ok_or_else(|| Error::msg("Unexpected number of output arguments."))? = val;
                }
            },
            Instruction::Malloc { realloc, size, align } => {
                let realloc = self.func.realloc.as_ref().expect("No realloc.");
                let args = [wasm_runtime_layer::Value::I32(0), wasm_runtime_layer::Value::I32(0), wasm_runtime_layer::Value::I32(*align as i32), wasm_runtime_layer::Value::I32(*size as i32)];
                let mut res = [wasm_runtime_layer::Value::I32(0)];
                realloc.call(&mut self.ctx, &args, &mut res)?;
                require_matches!(&res[0], wasm_runtime_layer::Value::I32(x), results.push(Value::S32(*x)));
            },
            Instruction::GuestDeallocate { size, align } => unreachable!(),
            Instruction::GuestDeallocateString => unreachable!(),
            Instruction::GuestDeallocateList { element } => unreachable!(),
            Instruction::GuestDeallocateVariant { blocks } => unreachable!(),
        }

        Ok(())
    }

    fn sizes(&self) -> &SizeAlign {
        &self.func.component.size_align
    }
}

trait Blittable: Sized {
    type Array: ByteArray;

    fn from_bytes(array: Self::Array) -> Self;
    fn to_bytes(self) -> Self::Array;
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
        memory.read(ctx, offset, &mut res)?;
        Ok(res)
    }

    fn store(self, ctx: impl AsContextMut, memory: &Memory, offset: usize) -> Result<()> {
        memory.write(ctx, offset, &self)?;
        Ok(())
    }
}