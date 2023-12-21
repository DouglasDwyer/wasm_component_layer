//! Conversion between guest and host types

use std::{iter::Peekable, slice, sync::Arc, vec};

use wasm_runtime_layer::{backend::WasmEngine, Func, Memory};
use wit_parser::{Resolve, Type};

use crate::{private::ListSpecialization, List, StoreContextMut, Tuple, Value, ValueType};

/// Implementation for native rust types
mod primitive_impls;

/// Utilizty trait for peekable iterator
pub trait PeekableIter: Iterator {
    /// Peeks the next item of the iterator
    fn peek(&mut self) -> Option<&Self::Item>;
}

impl<I> PeekableIter for Peekable<I>
where
    I: Iterator,
{
    fn peek(&mut self) -> Option<&Self::Item> {
        self.peek()
    }
}

/// A component type representation in guest memory
pub trait ComponentType {
    /// Returns the current type
    fn size(&self) -> usize;
    /// Returns the alignment of the type
    fn align(&self) -> usize;
    /// Returns the stride of the type
    fn stride(&self) -> usize {
        align_to(self.size(), self.align())
    }
}

/// Converts a value from the guest to the host
///
/// See:
/// <https://github.com/WebAssembly/component-model/blob/main/design/mvp/CanonicalABI.md#loading>
pub trait Lift: ComponentType {
    /// Reads the value from guest memory
    ///
    /// `ty` is used to infer the type to load for dynamically typed destination types, such as
    /// [`Value`].
    fn load<E: WasmEngine, T>(
        cx: &mut LiftContext<'_, '_, E, T>,
        memory: &Memory,
        ty: &mut dyn PeekableIter<Item = &Type>,
        ptr: usize,
    ) -> (Self, usize)
    where
        Self: Sized;

    /// Reads the value from flat arguments
    ///
    /// `ty` is used to infer the type to load for dynamically typed destination types, such as
    /// [`Value`].
    fn load_flat<E: WasmEngine, T>(
        cx: &mut LiftContext<'_, '_, E, T>,
        ty: &mut dyn PeekableIter<Item = &Type>,
        args: &mut vec::IntoIter<wasm_runtime_layer::Value>,
    ) -> Self;
}

/// Converts a value from the host to the guest
///
/// See:
/// <https://github.com/WebAssembly/component-model/blob/main/design/mvp/CanonicalABI.md#storing>
pub trait Lower: ComponentType {
    /// Lower the value to guest memory
    ///
    /// Returns a new pointer to the end of the written bytes.
    ///
    /// It is the responsibility of each type implementation to the correct stride is returned.
    ///
    /// The stride is not statically known to allow implementing for dynamically sized types and
    /// varying size enums.
    fn store<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        memory: &Memory,
        dst_ptr: usize,
    ) -> usize;

    /// Lower into guest function arguments
    fn store_flat<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        dst: &mut Vec<wasm_runtime_layer::Value>,
    );

    /// Stores a list of values.
    ///
    /// Has a default implementation but allows for specialization.
    fn store_list<E: WasmEngine, T>(
        items: &[Self],
        cx: &mut LowerContext<'_, '_, E, T>,
        memory: &Memory,
        mut dst_ptr: usize,
    ) -> usize
    where
        Self: Sized,
    {
        for item in items {
            dst_ptr = item.store(cx, memory, dst_ptr);
        }

        dst_ptr
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
        // let inner = &mut cx.store.inner;
        // let memory = &cx.memory;

        match self {
            Value::S32(v) => v.store_flat(cx, dst),
            Value::String(v) => v.store_flat(cx, dst),
            Value::List(v) => v.store_flat(cx, dst),
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
        ptr: usize,
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
                wit_parser::TypeDefKind::Variant(_) => todo!(),
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
                wit_parser::TypeDefKind::List(list_ty) => {
                    let memory = cx.memory.unwrap();

                    let (b_ptr, len) = <(i32, i32)>::load_flat(cx, ty, args);

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

                    Value::List(List::new(ty.clone(), values).unwrap())
                }
                wit_parser::TypeDefKind::Variant(_) => todo!(),
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
}

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
        let element_ty = self.ty().element_ty();
        let size = element_ty.size() * self.len();

        let ptr = alloc_list(cx, size as i32, 4).unwrap();
        match self.ty().element_ty() {
            ValueType::S32 => {
                i32::store_list(self.typed::<i32>().unwrap(), cx, memory, ptr as usize);
            }
            ValueType::U32 => todo!(),
            ValueType::S64 => todo!(),
            ValueType::U64 => todo!(),
            ValueType::F32 => todo!(),
            ValueType::F64 => todo!(),
            ValueType::Char => todo!(),
            ValueType::String => todo!(),
            ValueType::List(_) => todo!(),
            ValueType::Record(_) => todo!(),
            ValueType::Tuple(_) => todo!(),
            ValueType::Variant(_) => todo!(),
            ValueType::Enum(_) => todo!(),
            ValueType::Option(_) => todo!(),
            ValueType::Result(_) => todo!(),
            ValueType::Flags(_) => todo!(),
            ValueType::Own(_) => todo!(),
            ValueType::Borrow(_) => todo!(),
            ValueType::Bool => todo!(),
            ValueType::S8 => todo!(),
            ValueType::U8 => todo!(),
            ValueType::S16 => todo!(),
            ValueType::U16 => todo!(),
        }
        // (ptr, self.len() as i32).store(cx, memory, dst_ptr)
        (ptr, self.len() as i32).store(cx, memory, dst_ptr)
    }

    fn store_flat<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        dst: &mut Vec<wasm_runtime_layer::Value>,
    ) {
        let element_ty = self.ty().element_ty();
        let size = element_ty.size() * self.len();

        let memory = cx.memory.unwrap();

        let dst_ptr = alloc_list(cx, size as i32, 8).unwrap();
        match self.values() {
            ListSpecialization::S32(v) => {
                Lower::store_list(v, cx, memory, dst_ptr as usize);
            }
            ListSpecialization::Other(v) => {
                Lower::store_list(v, cx, memory, dst_ptr as usize);
            }
            ListSpecialization::U32(_) => todo!(),
            ListSpecialization::S64(_) => todo!(),
            ListSpecialization::U64(_) => todo!(),
            ListSpecialization::F32(_) => todo!(),
            ListSpecialization::F64(_) => todo!(),
            ListSpecialization::Char(_) => todo!(),
            ListSpecialization::Bool(_) => todo!(),
            ListSpecialization::S8(_) => todo!(),
            ListSpecialization::U8(_) => todo!(),
            ListSpecialization::S16(_) => todo!(),
            ListSpecialization::U16(_) => todo!(),
        }

        (dst_ptr, self.len() as i32).store_flat(cx, dst)
    }
}

impl ComponentType for Value {
    fn size(&self) -> usize {
        match self {
            Value::S32(v) => v.size(),
            _ => todo!(),
        }
    }

    fn align(&self) -> usize {
        match self {
            Value::S32(v) => v.align(),
            _ => todo!(),
        }
    }
}

/// Aligns a pointer to the given alignment
fn align_to(ptr: usize, align: usize) -> usize {
    // https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
    (ptr + (align - 1)) & !(align - 1)
}

/// Implement Lift and Lower for tuples
macro_rules! tuple_impl {
    ($($idx: tt $t: ident),*) => {
        impl<$($t: ComponentType,)*> ComponentType for ($($t,)*) {
            fn size(&self) -> usize {
                let mut s = 0;
                $(s = align_to(s + self.$idx.size(), self.$idx.align());)*

                s
            }

            fn align(&self) -> usize {
                0 $(.max(self.$idx.align()))*
            }
        }

        impl<$($t: Lower,)*> Lower for ($($t,)*) {
            fn store<En: WasmEngine, T>(
                &self,
                cx: &mut LowerContext<'_, '_, En, T>,
                memory: &Memory,
                mut ptr: usize,
            ) -> usize {
                $(ptr = self.$idx.store(cx, memory, ptr);)*

                ptr
            }

            fn store_flat<En: WasmEngine, T>(
                &self,
                cx: &mut LowerContext<'_, '_, En, T>,
                dst: &mut Vec<wasm_runtime_layer::Value>,
            ) {
                $(self.$idx.store_flat(cx, dst);)*
            }
        }

        impl<$($t: Lift,)*> Lift for ($($t,)*) {
            fn load<En: WasmEngine, T>(
                cx: &mut LiftContext<'_, '_, En, T>,
                memory: &Memory,
                ty: &mut dyn PeekableIter<Item = &Type>,
                ptr: usize,
            ) -> (Self, usize)
            where
                Self: Sized,
            {
                $(
                    #[allow(non_snake_case)]
                    let ($t, ptr) = $t::load(cx, memory, ty, ptr);
                )*

                (($($t,)*), ptr)
            }

            fn load_flat<En: WasmEngine, T>(
                cx: &mut LiftContext<'_, '_, En, T>,
                ty: &mut dyn PeekableIter<Item = &Type>,
                args: &mut vec::IntoIter<wasm_runtime_layer::Value>,
            ) -> Self {
                $(
                    #[allow(non_snake_case)]
                    let $t = $t::load_flat(cx, ty, args);
                )*

                ($($t,)*)
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
tuple_impl! { 0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I }
tuple_impl! { 0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K }
tuple_impl! { 0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K, 11 L }
tuple_impl! { 0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K, 11 L, 12 M }
tuple_impl! { 0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K, 11 L, 12 M, 13 N}
tuple_impl! { 0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K, 11 L, 12 M, 13 N, 14 O }
tuple_impl! { 0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K, 11 L, 12 M, 13 N, 14 O, 15 P }
tuple_impl! { 0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K, 11 L, 12 M, 13 N, 14 O, 15 P, 16 Q }

impl ComponentType for () {
    fn size(&self) -> usize {
        0
    }

    fn align(&self) -> usize {
        0
    }
}

impl Lift for () {
    fn load<E: WasmEngine, T>(
        cx: &mut LiftContext<'_, '_, E, T>,
        memory: &Memory,
        ty: &mut dyn PeekableIter<Item = &Type>,
        ptr: usize,
    ) -> (Self, usize)
    where
        Self: Sized,
    {
        ((), ptr)
    }

    fn load_flat<E: WasmEngine, T>(
        cx: &mut LiftContext<'_, '_, E, T>,
        ty: &mut dyn PeekableIter<Item = &Type>,
        args: &mut vec::IntoIter<wasm_runtime_layer::Value>,
    ) -> Self {
    }
}

/// Used when lowering into guest memory
pub struct LowerContext<'a, 't, E: WasmEngine, T> {
    /// The store context
    pub store: &'a mut StoreContextMut<'t, T, E>,
    /// Realloc function, if available
    pub realloc: Option<&'a Func>,
    /// The guest memory
    pub memory: Option<&'a Memory>,
}

/// Used when lifting from guest memory
pub struct LiftContext<'a, 't, E: WasmEngine, T> {
    /// WIT resolve context
    pub resolve: &'a Resolve,
    /// Wit types have been converted to crate types for convenience
    pub types: &'a [ValueType],
    /// The store context
    pub store: StoreContextMut<'t, T, E>,
    /// The guest memory
    ///
    /// It is not always available, such as during init
    pub memory: Option<&'a Memory>,
}

/// Allocate a block of memory in the guest for string and list lowering
pub(crate) fn alloc_list<E: WasmEngine, T>(
    cx: &mut LowerContext<'_, '_, E, T>,
    size: i32,
    align: i32,
) -> anyhow::Result<i32> {
    let mut res = [wasm_runtime_layer::Value::I32(0)];
    cx.realloc.unwrap().call(
        &mut cx.store.inner,
        // old_ptrmut , old_len, align, new_len
        &[
            wasm_runtime_layer::Value::I32(0),
            wasm_runtime_layer::Value::I32(0),
            wasm_runtime_layer::Value::I32(align),
            wasm_runtime_layer::Value::I32(size),
        ],
        &mut res,
    )?;

    let [wasm_runtime_layer::Value::I32(ptr)] = res else {
        panic!("Invalid return value")
    };

    Ok(ptr)
}
