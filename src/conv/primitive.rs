use std::{ops::Deref, sync::Arc};

use crate::func::{Blittable, ByteArray};
use wasm_runtime_layer::{backend::WasmEngine, Memory};
use wit_parser::Type;

use crate::conv::alloc_list;

use super::{ComponentType, Lift, LiftContext, Lower, LowerContext, PeekableIter};

macro_rules! integral_impl {
    ($ty: ty, $variant: ident, $flat_repr: ty ) => {
        impl ComponentType for $ty {
            fn size(&self) -> usize {
                std::mem::size_of::<$ty>()
            }

            fn align(&self) -> usize {
                std::mem::align_of::<$ty>()
            }
        }

        impl Lower for $ty {
            fn store<E: WasmEngine, T>(
                &self,
                cx: &mut LowerContext<'_, '_, E, T>,
                memory: &Memory,
                ptr: usize,
            ) -> usize {
                Blittable::to_bytes(*self)
                    .store(&mut cx.store, memory, ptr)
                    .unwrap();
                ptr + std::mem::size_of::<$ty>()
            }

            fn store_flat<E: WasmEngine, T>(
                &self,
                _: &mut LowerContext<'_, '_, E, T>,
                dst: &mut Vec<wasm_runtime_layer::Value>,
            ) {
                dst.push(wasm_runtime_layer::Value::$variant(*self as $flat_repr));
            }
        }

        impl Lift for $ty {
            fn load<E: WasmEngine, T>(
                cx: &mut LiftContext<'_, '_, E, T>,
                memory: &Memory,
                _: &mut dyn PeekableIter<Item = &Type>,
                ptr: usize,
            ) -> (Self, usize) {
                let v = <$ty as Blittable>::from_bytes(
                    ByteArray::load(&cx.store, memory, ptr).unwrap(),
                );
                tracing::debug!(?v);
                (v, ptr + std::mem::size_of::<$ty>())
            }

            fn load_flat<E: WasmEngine, T>(
                _: &mut LiftContext<'_, '_, E, T>,
                _: &mut dyn PeekableIter<Item = &Type>,
                args: &mut std::vec::IntoIter<wasm_runtime_layer::Value>,
            ) -> Self {
                let wasm_runtime_layer::Value::$variant(v) =
                    args.next().expect("too few arguments")
                else {
                    panic!("incorrect type, expected S32");
                };

                v as $ty
            }
        }
    };
}

integral_impl!(u8, I32, i32);
integral_impl!(i8, I32, i32);
integral_impl!(u16, I32, i32);
integral_impl!(i16, I32, i32);
integral_impl!(u32, I32, i32);
integral_impl!(i32, I32, i32);
integral_impl!(u64, I64, i64);
integral_impl!(i64, I64, i64);
integral_impl!(f32, F32, f32);
integral_impl!(f64, F64, f64);

impl ComponentType for bool {
    fn size(&self) -> usize {
        std::mem::size_of::<bool>()
    }
    fn align(&self) -> usize {
        std::mem::align_of::<bool>()
    }
}
impl Lower for bool {
    fn store<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        memory: &Memory,
        ptr: usize,
    ) -> usize {
        Blittable::to_bytes(*self)
            .store(&mut cx.store, memory, ptr)
            .unwrap();
        ptr + 4
    }
    fn store_flat<E: WasmEngine, T>(
        &self,
        _: &mut LowerContext<'_, '_, E, T>,
        dst: &mut Vec<wasm_runtime_layer::Value>,
    ) {
        dst.push(wasm_runtime_layer::Value::I32(*self as u32 as i32));
    }
}
impl Lift for bool {
    fn load<E: WasmEngine, T>(
        cx: &mut LiftContext<'_, '_, E, T>,
        memory: &Memory,
        _: &mut dyn PeekableIter<Item = &Type>,
        ptr: usize,
    ) -> (Self, usize) {
        let v = <bool as Blittable>::from_bytes(ByteArray::load(&cx.store, memory, ptr).unwrap());

        (v, ptr + 4)
    }
    fn load_flat<E: WasmEngine, T>(
        _: &mut LiftContext<'_, '_, E, T>,
        _: &mut dyn PeekableIter<Item = &Type>,
        args: &mut std::vec::IntoIter<wasm_runtime_layer::Value>,
    ) -> Self {
        let wasm_runtime_layer::Value::I32(v) = args.next().expect("too few arguments") else {
            panic!("incorrect type, expected S32");
        };

        match v {
            0 => false,
            1 => true,
            _ => panic!("invalid bool value"),
        }
    }
}

impl ComponentType for char {
    fn size(&self) -> usize {
        std::mem::size_of::<char>()
    }
    fn align(&self) -> usize {
        std::mem::align_of::<char>()
    }
}
impl Lower for char {
    fn store<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        memory: &Memory,
        ptr: usize,
    ) -> usize {
        Blittable::to_bytes(*self)
            .store(&mut cx.store, memory, ptr)
            .unwrap();

        ptr + 4
    }
    fn store_flat<E: WasmEngine, T>(
        &self,
        _: &mut LowerContext<'_, '_, E, T>,
        dst: &mut Vec<wasm_runtime_layer::Value>,
    ) {
        dst.push(wasm_runtime_layer::Value::I32(*self as u32 as i32));
    }
}
impl Lift for char {
    fn load<E: WasmEngine, T>(
        cx: &mut LiftContext<'_, '_, E, T>,
        memory: &Memory,
        _: &mut dyn PeekableIter<Item = &Type>,
        ptr: usize,
    ) -> (Self, usize) {
        let v = <char as Blittable>::from_bytes(ByteArray::load(&cx.store, memory, ptr).unwrap());
        tracing::debug!(?v);
        (v, ptr + 4)
    }
    fn load_flat<E: WasmEngine, T>(
        _: &mut LiftContext<'_, '_, E, T>,
        _: &mut dyn PeekableIter<Item = &Type>,
        args: &mut std::vec::IntoIter<wasm_runtime_layer::Value>,
    ) -> Self {
        let wasm_runtime_layer::Value::I32(v) = args.next().expect("too few arguments") else {
            panic!("incorrect type, expected S32");
        };

        char::from_u32(v as u32).expect("invalid char")
    }
}

impl ComponentType for str {
    fn size(&self) -> usize {
        // (i32, i32)
        4 + 4
    }

    fn align(&self) -> usize {
        4
    }
}

impl Lift for String {
    fn load<E: WasmEngine, T>(
        cx: &mut LiftContext<'_, '_, E, T>,
        memory: &Memory,
        ty: &mut dyn PeekableIter<Item = &Type>,
        ptr: usize,
    ) -> (Self, usize) {
        let ((s_ptr, len), new_ptr) = <(i32, i32)>::load(cx, memory, ty, ptr);

        let mut buf = vec![0u8; len as usize];

        memory
            .read(&mut cx.store.inner, s_ptr as usize, &mut buf)
            .unwrap();

        let s = String::from_utf8(buf).expect("Invalid UTF-8");

        (s, new_ptr)
    }

    fn load_flat<E: WasmEngine, T>(
        cx: &mut LiftContext<'_, '_, E, T>,
        ty: &mut dyn PeekableIter<Item = &Type>,
        args: &mut std::vec::IntoIter<wasm_runtime_layer::Value>,
    ) -> Self {
        let ptr = i32::load_flat(cx, ty, args);
        let len = i32::load_flat(cx, ty, args);

        let memory = cx.memory.unwrap();

        let mut buf = vec![0u8; len as usize];

        memory
            .read(&mut cx.store.inner, ptr as _, &mut buf)
            .unwrap();

        String::from_utf8(buf).expect("Invalid UTF-8")
    }
}

impl Lower for str {
    fn store<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        memory: &Memory,
        dst_ptr: usize,
    ) -> usize {
        // String::len returns the number of bytes, not the number of characters
        let byte_len: i32 = self.len().try_into().expect("string too long");

        let ptr = alloc_list(cx, byte_len, 1).expect("failed to allocate string");

        tracing::debug!(?ptr, ?byte_len, "allocated string");

        memory
            .write(&mut cx.store.inner, ptr as _, self.as_bytes())
            .unwrap();

        (ptr, byte_len).store(cx, memory, dst_ptr)
    }

    fn store_flat<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        dst: &mut Vec<wasm_runtime_layer::Value>,
    ) {
        // String::len returns the number of bytes, not the number of characters
        let byte_len: i32 = self.len().try_into().expect("string too long");

        let ptr = alloc_list(cx, byte_len, 1).expect("failed to allocate string");

        tracing::debug!(?ptr, ?byte_len, "allocated string");

        let memory = cx.memory.unwrap();

        memory
            .write(&mut cx.store.inner, ptr as _, self.as_bytes())
            .unwrap();

        (ptr, byte_len).store_flat(cx, dst);
    }
}

impl<V: ComponentType> ComponentType for [V] {
    fn size(&self) -> usize {
        4 + 4
    }

    fn align(&self) -> usize {
        4
    }
}

// Non-canonical lists
impl<V: Lower> Lower for [V] {
    fn store<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        memory: &Memory,
        dst_ptr: usize,
    ) -> usize {
        let byte_len: i32 = self.len().try_into().expect("list too long");

        let ptr = alloc_list(cx, byte_len, 1).expect("failed to allocate list");

        tracing::debug!(?ptr, ?byte_len, "allocated list");

        V::store_list(self, cx, memory, ptr as usize);

        (ptr, byte_len).store(cx, memory, dst_ptr)
    }

    fn store_flat<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        dst: &mut Vec<wasm_runtime_layer::Value>,
    ) {
        let byte_len: i32 = self.len().try_into().expect("list too long");

        let ptr = alloc_list(cx, byte_len, 1).expect("failed to allocate list");

        tracing::debug!(?ptr, ?byte_len, "allocated list");

        let memory = cx.memory.unwrap();

        V::store_list(self, cx, memory, ptr as usize);

        (ptr, byte_len).store_flat(cx, dst);
    }
}

impl<V: ComponentType> ComponentType for Vec<V> {
    fn size(&self) -> usize {
        self.as_slice().size()
    }

    fn align(&self) -> usize {
        self.as_slice().size()
    }
}

impl<V: Lift> Lift for Vec<V> {
    fn load<E: WasmEngine, T>(
        cx: &mut LiftContext<'_, '_, E, T>,
        memory: &Memory,
        ty: &mut dyn PeekableIter<Item = &Type>,
        ptr: usize,
    ) -> (Self, usize) {
        let ((b_ptr, len), new_ptr) = <(i32, i32)>::load(cx, memory, ty, ptr);

        let mut ptr = b_ptr as usize;

        let Some(Type::Id(id)) = ty.next() else {
            panic!("missing type");
        };

        let list_ty = match &cx.resolve.types[*id].kind {
            wit_parser::TypeDefKind::List(list_ty) => list_ty,
            ty => panic!("expected list type, found {ty:?}"),
        };

        let res = (0..len)
            .map(|idx| {
                tracing::debug!(?idx);
                let (v, p) = V::load(
                    cx,
                    memory,
                    &mut std::slice::from_ref(list_ty).into_iter().peekable(),
                    ptr,
                );
                ptr = p;

                v
            })
            .collect();

        (res, new_ptr)
    }

    fn load_flat<E: WasmEngine, T>(
        cx: &mut LiftContext<'_, '_, E, T>,
        ty: &mut dyn PeekableIter<Item = &Type>,
        args: &mut std::vec::IntoIter<wasm_runtime_layer::Value>,
    ) -> Self {
        let b_ptr = i32::load_flat(cx, ty, args);
        let len = i32::load_flat(cx, ty, args);

        let memory = cx.memory.unwrap();

        let mut ptr = b_ptr as usize;

        (0..len)
            .map(|idx| {
                tracing::debug!(?idx);
                let (v, p) = V::load(cx, memory, ty, ptr);
                ptr = p;

                v
            })
            .collect()
    }
}

impl<V: Lower> Lower for Vec<V> {
    fn store<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        memory: &Memory,
        dst_ptr: usize,
    ) -> usize {
        self.as_slice().store(cx, memory, dst_ptr)
    }

    fn store_flat<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        dst: &mut Vec<wasm_runtime_layer::Value>,
    ) {
        self.as_slice().store_flat(cx, dst)
    }
}

/// Allocate a block of memory in the guest for string and list lowering
macro_rules! auto_impl {
    ($ty: ty, [$($t: tt),*] $ptr: ty) => {

        impl<$($t: ComponentType,)*> ComponentType for $ptr {
            fn size(&self) -> usize {
                self.deref().size()
            }

            fn align(&self) -> usize {
                self.deref().align()
            }
        }

        impl<$($t: Lower,)*> Lower for $ptr {
            fn store<E: WasmEngine, T>(
                &self,
                cx: &mut LowerContext<'_, '_, E, T>,
                memory: &Memory,
                ptr: usize,
            ) -> usize {
                self.deref().store(cx, memory, ptr)
            }

            fn store_flat<E: WasmEngine, T>(
                &self,
                cx: &mut LowerContext<'_, '_, E, T>,
                dst: &mut Vec<wasm_runtime_layer::Value>,
            ) {
                self.deref().store_flat(cx, dst)
            }
        }
    };
    ($ty: ty => $([$($t: tt),*] $ptr: ty),*) => {
        $(auto_impl!($ty, [$($t),*] $ptr);)*
    };
}

auto_impl! { str => [] String, [] Box<str>, [] Arc<str>, [] &str }
auto_impl! { V => [V] Box<V>, [V] Arc<V>, [V] &[V] }
