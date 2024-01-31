//! Implements lifting lower traits for the dynamically typed `Value` enum, and related
//! newtypes.

use std::{slice, sync::Arc, vec};

use wasm_runtime_layer::{
    backend::{WasmEngine, WasmFunc, WasmGlobal, WasmMemory},
    Memory,
};
use wit_parser::Type;

use crate::{private::ListSpecialization, List, Record, Tuple, Value, ValueType, Variant};

use super::{
    align_to, alloc_list, ComponentType, Lift, LiftContext, Lower, LowerContext, PeekableIter,
};

impl ComponentType for List {
    fn size(&self) -> usize {
        4 + 4
    }

    fn align(&self) -> usize {
        4
    }
}

impl Lower for List {
    fn store<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        memory: &Memory,
        dst_ptr: usize,
    ) -> usize {
        let _span = tracing::info_span!("store list", list=?self).entered();
        let element_ty = self.ty().element_ty();
        let size = element_ty.size() * self.len();

        let ptr = alloc_list(cx, size as i32, 4).unwrap() as usize;
        let end_ptr = match self.values() {
            ListSpecialization::S32(v) => Lower::store_list(v, cx, memory, ptr as usize),
            ListSpecialization::Other(v) => Lower::store_list(v, cx, memory, ptr as usize),
            ListSpecialization::U32(v) => Lower::store_list(v, cx, memory, ptr as usize),
            ListSpecialization::S64(v) => Lower::store_list(v, cx, memory, ptr as usize),
            ListSpecialization::U64(v) => Lower::store_list(v, cx, memory, ptr as usize),
            ListSpecialization::F32(v) => Lower::store_list(v, cx, memory, ptr as usize),
            ListSpecialization::F64(v) => Lower::store_list(v, cx, memory, ptr as usize),
            ListSpecialization::Char(v) => Lower::store_list(v, cx, memory, ptr as usize),
            ListSpecialization::Bool(v) => Lower::store_list(v, cx, memory, ptr as usize),
            ListSpecialization::S8(v) => Lower::store_list(v, cx, memory, ptr as usize),
            ListSpecialization::U8(v) => Lower::store_list(v, cx, memory, ptr as usize),
            ListSpecialization::S16(v) => Lower::store_list(v, cx, memory, ptr as usize),
            ListSpecialization::U16(v) => Lower::store_list(v, cx, memory, ptr as usize),
        };

        // Checks that the size and align of the list element matches the stride of the list
        // members during conversion.
        // For example, this ensures a records or variants reported size matches the actual stride
        // by advancing the write ptr.
        debug_assert_eq!(
            ptr + size,
            end_ptr,
            "Actual bytes to list does not match element size\nExpected {} bytes, got {} bytes\nty: {:?}",
            element_ty.size(),
            (end_ptr - ptr) / self.len(),
            element_ty,
        );

        (ptr as i32, self.len() as i32).store(cx, memory, dst_ptr)
    }

    fn store_flat<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        dst_ptr: &mut Vec<wasm_runtime_layer::Value>,
    ) {
        let element_ty = self.ty().element_ty();
        let size = element_ty.size() * self.len();

        let memory = cx.memory.unwrap();

        let ptr = alloc_list(cx, size as i32, 8).unwrap();
        match self.values() {
            ListSpecialization::S32(v) => {
                Lower::store_list(v, cx, memory, ptr as usize);
            }
            ListSpecialization::Other(v) => {
                Lower::store_list(v, cx, memory, ptr as usize);
            }
            ListSpecialization::U32(v) => {
                Lower::store_list(v, cx, memory, ptr as usize);
            }
            ListSpecialization::S64(v) => {
                Lower::store_list(v, cx, memory, ptr as usize);
            }
            ListSpecialization::U64(v) => {
                Lower::store_list(v, cx, memory, ptr as usize);
            }
            ListSpecialization::F32(v) => {
                Lower::store_list(v, cx, memory, ptr as usize);
            }
            ListSpecialization::F64(v) => {
                Lower::store_list(v, cx, memory, ptr as usize);
            }
            ListSpecialization::Char(v) => {
                Lower::store_list(v, cx, memory, ptr as usize);
            }
            ListSpecialization::Bool(v) => {
                Lower::store_list(v, cx, memory, ptr as usize);
            }
            ListSpecialization::S8(v) => {
                Lower::store_list(v, cx, memory, ptr as usize);
            }
            ListSpecialization::U8(v) => {
                Lower::store_list(v, cx, memory, ptr as usize);
            }
            ListSpecialization::S16(v) => {
                Lower::store_list(v, cx, memory, ptr as usize);
            }
            ListSpecialization::U16(v) => {
                Lower::store_list(v, cx, memory, ptr as usize);
            }
        }

        (ptr, self.len() as i32).store_flat(cx, dst_ptr)
    }
}

impl ComponentType for Value {
    fn size(&self) -> usize {
        match self {
            Value::Bool(v) => v.size(),
            Value::S8(v) => v.size(),
            Value::U8(v) => v.size(),
            Value::S16(v) => v.size(),
            Value::U16(v) => v.size(),
            Value::S32(v) => v.size(),
            Value::U32(v) => v.size(),
            Value::S64(v) => v.size(),
            Value::U64(v) => v.size(),
            Value::F32(v) => v.size(),
            Value::F64(v) => v.size(),
            Value::Char(v) => v.size(),
            Value::String(v) => v.size(),
            Value::List(v) => v.size(),
            Value::Record(v) => v.size(),
            Value::Tuple(v) => v.size(),
            Value::Variant(v) => v.size(),
            Value::Enum(v) => v.size(),
            Value::Option(v) => v.size(),
            Value::Result(v) => v.size(),
            Value::Flags(v) => v.size(),
            Value::Own(v) => v.size(),
            Value::Borrow(v) => v.size(),
        }
    }

    fn align(&self) -> usize {
        match self {
            Value::S32(v) => v.align(),
            Value::Bool(v) => v.align(),
            Value::S8(v) => v.align(),
            Value::U8(v) => v.align(),
            Value::S16(v) => v.align(),
            Value::U16(v) => v.align(),
            Value::U32(v) => v.align(),
            Value::S64(v) => v.align(),
            Value::U64(v) => v.align(),
            Value::F32(v) => v.align(),
            Value::F64(v) => v.align(),
            Value::Char(v) => v.align(),
            Value::String(v) => v.align(),
            Value::List(v) => v.align(),
            Value::Record(v) => v.align(),
            Value::Tuple(v) => v.align(),
            Value::Variant(v) => v.align(),
            Value::Enum(v) => v.align(),
            Value::Option(v) => v.align(),
            Value::Result(v) => v.align(),
            Value::Flags(v) => v.align(),
            Value::Own(v) => v.align(),
            Value::Borrow(v) => v.align(),
        }
    }
}

impl Lower for Value {
    fn store<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        memory: &Memory,
        ptr: usize,
    ) -> usize {
        match self {
            Value::S32(v) => v.store(cx, memory, ptr),
            Value::String(v) => v.store(cx, memory, ptr),
            Value::List(v) => v.store(cx, memory, ptr),
            Value::Variant(v) => v.store(cx, memory, ptr),
            _ => {
                todo!()
            }
        }
    }

    fn store_flat<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        dst: &mut Vec<wasm_runtime_layer::Value>,
    ) {
        match self {
            Value::S32(v) => v.store_flat(cx, dst),
            Value::String(v) => v.store_flat(cx, dst),
            Value::List(v) => v.store_flat(cx, dst),
            Value::Variant(v) => v.store_flat(cx, dst),
            _ => {
                todo!()
            }
        }
    }
}

impl Lift for Value {
    fn load<E: WasmEngine, T>(
        cx: &mut LiftContext<'_, '_, E, T>,
        memory: &Memory,
        ty: &mut dyn PeekableIter<Item = &Type>,
        mut ptr: usize,
    ) -> (Self, usize) {
        match ty.next().unwrap() {
            Type::S32 => {
                let (v, ptr) = i32::load(cx, memory, ty, ptr);
                (Value::S32(v), ptr)
            }
            Type::String => {
                let (v, ptr) = String::load(cx, memory, ty, ptr);
                (Value::String(v.into()), ptr)
            }
            &Type::Id(id) => match &cx.resolve.types[id].kind {
                wit_parser::TypeDefKind::Record(v) => {
                    let mut args = Vec::new();
                    let mut ptr = ptr;

                    for field in v.fields.iter() {
                        let (v, p) = Value::load(
                            cx,
                            memory,
                            &mut slice::from_ref(&field.ty).iter().peekable(),
                            ptr,
                        );

                        args.push((Arc::from(field.name.as_str()), v));
                        ptr = p;
                    }

                    let ValueType::Record(ty) = &cx.types[id.index()] else {
                        panic!("Invalid type");
                    };

                    (
                        Value::Record(crate::Record::new(ty.clone(), args).unwrap()),
                        ptr,
                    )
                }
                wit_parser::TypeDefKind::Tuple(v) => {
                    let mut args = Vec::new();
                    let mut ptr = ptr;

                    for ty in v.types.iter() {
                        let (v, p) = Value::load(
                            cx,
                            memory,
                            &mut slice::from_ref(ty).iter().peekable(),
                            ptr,
                        );

                        args.push(v);
                        ptr = p;
                    }

                    let ValueType::Tuple(ty) = &cx.types[id.index()] else {
                        panic!("Invalid type");
                    };

                    (Value::Tuple(Tuple::new(ty.clone(), args).unwrap()), ptr)
                }
                wit_parser::TypeDefKind::List(list_ty) => {
                    let ((b_ptr, len), new_ptr) = <(i32, i32)>::load(cx, memory, ty, ptr);

                    let mut ptr = b_ptr as usize;
                    let values: Vec<_> = (0..len)
                        .map(|idx| {
                            tracing::debug!(?idx);
                            let (v, p) = Value::load(
                                cx,
                                memory,
                                &mut slice::from_ref(list_ty).iter().peekable(),
                                ptr,
                            );

                            ptr = p;

                            v
                        })
                        .collect();

                    let ValueType::List(ty) = &cx.types[id.index()] else {
                        panic!("Invalid type");
                    };

                    (Value::List(List::new(ty.clone(), values).unwrap()), new_ptr)
                }
                wit_parser::TypeDefKind::Variant(v) => {
                    let discriminant = match ((v.cases.len() as f32).log2() / 8.0).ceil() as u32 {
                        0 => {
                            let (v, new_ptr) = u8::load(cx, memory, ty, ptr);
                            ptr = new_ptr;
                            v as usize
                        }
                        1 => {
                            let (v, new_ptr) = u8::load(cx, memory, ty, ptr);
                            ptr = new_ptr;
                            v as usize
                        }
                        2 => {
                            let (v, new_ptr) = u16::load(cx, memory, ty, ptr);
                            ptr = new_ptr;
                            v as usize
                        }
                        3 => {
                            let (v, new_ptr) = u32::load(cx, memory, ty, ptr);
                            ptr = new_ptr;
                            v as usize
                        }
                        _ => unreachable!(),
                    };

                    let ValueType::Variant(ty) = &cx.types[id.index()] else {
                        panic!("Invalid type");
                    };

                    let case = &v.cases[discriminant];
                    if let Some(variant_ty) = case.ty {
                        let case = &ty.cases()[discriminant];
                        let ptr = align_to(ptr, case.ty().unwrap().align());
                        let (v, ptr) = Value::load(
                            cx,
                            memory,
                            &mut slice::from_ref(&variant_ty).iter().peekable(),
                            ptr,
                        );

                        (
                            Value::Variant(
                                crate::Variant::new(ty.clone(), discriminant, Some(v)).unwrap(),
                            ),
                            ptr,
                        )
                    } else {
                        (
                            Value::Variant(
                                crate::Variant::new(ty.clone(), discriminant, None).unwrap(),
                            ),
                            ptr,
                        )
                    }
                }
                wit_parser::TypeDefKind::Enum(_) => todo!(),
                wit_parser::TypeDefKind::Option(_) => todo!(),
                wit_parser::TypeDefKind::Result(_) => todo!(),
                wit_parser::TypeDefKind::Resource => todo!(),
                wit_parser::TypeDefKind::Handle(_) => todo!(),
                wit_parser::TypeDefKind::Flags(_) => todo!(),
                wit_parser::TypeDefKind::Future(_) => todo!(),
                wit_parser::TypeDefKind::Stream(_) => todo!(),
                wit_parser::TypeDefKind::Type(_) => todo!(),
                wit_parser::TypeDefKind::Unknown => todo!(),
            },
            _ => todo!(),
        }
    }

    fn load_flat<E: WasmEngine, T>(
        cx: &mut LiftContext<'_, '_, E, T>,
        ty: &mut dyn PeekableIter<Item = &Type>,
        args: &mut vec::IntoIter<wasm_runtime_layer::Value>,
    ) -> Self {
        match ty.next().unwrap() {
            Type::S32 => Value::S32(i32::load_flat(cx, ty, args)),
            Type::String => Value::String(String::load_flat(cx, ty, args).into()),
            &Type::Id(id) => match &cx.resolve.types[id].kind {
                wit_parser::TypeDefKind::Record(v) => {
                    let mut res = Vec::new();

                    for field in v.fields.iter() {
                        let v = Value::load_flat(
                            cx,
                            &mut slice::from_ref(&field.ty).iter().peekable(),
                            args,
                        );

                        res.push((Arc::from(field.name.as_str()), v));
                    }

                    let ValueType::Record(ty) = &cx.types[id.index()] else {
                        panic!("Invalid type");
                    };

                    Value::Record(crate::Record::new(ty.clone(), res).unwrap())
                }
                wit_parser::TypeDefKind::Tuple(v) => {
                    let mut res = Vec::new();

                    for ty in v.types.iter() {
                        let v =
                            Value::load_flat(cx, &mut slice::from_ref(ty).iter().peekable(), args);

                        res.push(v);
                    }

                    let ValueType::Tuple(ty) = &cx.types[id.index()] else {
                        panic!("Invalid type");
                    };

                    Value::Tuple(Tuple::new(ty.clone(), res).unwrap())
                }
                wit_parser::TypeDefKind::List(_) => unreachable!("A list can not be flat"),
                wit_parser::TypeDefKind::Variant(_) => unreachable!("A variant can not be flat"),
                wit_parser::TypeDefKind::Enum(_) => unreachable!("An enum can not be flat"),
                wit_parser::TypeDefKind::Option(_) => unreachable!("An option can not be flat"),
                wit_parser::TypeDefKind::Result(_) => unreachable!("A result can not be flat"),
                wit_parser::TypeDefKind::Resource => todo!(),
                wit_parser::TypeDefKind::Handle(_) => todo!(),
                wit_parser::TypeDefKind::Flags(_) => todo!(),
                wit_parser::TypeDefKind::Future(_) => todo!(),
                wit_parser::TypeDefKind::Stream(_) => todo!(),
                wit_parser::TypeDefKind::Type(_) => todo!(),
                wit_parser::TypeDefKind::Unknown => todo!(),
            },
            _ => todo!(),
        }
    }
}

impl ComponentType for Tuple {
    fn size(&self) -> usize {
        let mut s = 0;
        for v in self {
            s = align_to(s + v.size(), v.align());
        }

        s
    }

    fn align(&self) -> usize {
        self.into_iter().map(|v| v.align()).max().unwrap_or(0)
    }
}

impl ComponentType for Record {
    fn size(&self) -> usize {
        let mut s = 0;
        for (_, v) in self.fields() {
            s = align_to(s + v.size(), v.align());
        }

        s
    }

    fn align(&self) -> usize {
        self.fields().map(|(_, v)| v.align()).max().unwrap_or(0)
    }
}

impl ComponentType for Variant {
    fn size(&self) -> usize {
        self.ty().size()
    }

    fn align(&self) -> usize {
        self.ty().align()
    }
}

impl Lower for Variant {
    fn store<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        memory: &Memory,
        dst_ptr: usize,
    ) -> usize {
        let d = self.discriminant();
        let ptr = match ((self.ty().cases().len() as f32).log2() / 8.0).ceil() as u32 {
            0 => (d as u8).store(cx, memory, dst_ptr),
            1 => (d as u8).store(cx, memory, dst_ptr),
            2 => (d as u16).store(cx, memory, dst_ptr),
            3 => (d as u32).store(cx, memory, dst_ptr),
            _ => unreachable!(),
        };

        let ptr = align_to(ptr, self.ty().value_align());

        if let Some(v) = self.value() {
            let new_ptr = v.store(cx, memory, ptr);
            debug_assert!(new_ptr <= ptr + self.ty().value_size());
        }

        ptr + self.ty().value_size()
    }

    fn store_flat<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        dst: &mut Vec<wasm_runtime_layer::Value>,
    ) {
        (self.discriminant() as u32).store_flat(cx, dst);

        if let Some(value) = self.value() {
            value.store_flat(cx, dst);
        }
    }
}
