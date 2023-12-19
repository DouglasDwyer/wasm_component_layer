use std::{ops::Deref, sync::Arc};

use wasm_runtime_layer::{backend::WasmEngine, Memory, Value};
use wit_parser::Type;

use super::{ComponentType, Lift, LiftContext, Lower, LowerContext, PeekableIter};

impl ComponentType for i32 {
    fn size(&self) -> usize {
        std::mem::size_of::<i32>()
    }

    fn align(&self) -> usize {
        std::mem::align_of::<i32>()
    }
}

impl Lower for i32 {
    fn store<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        _: &Memory,
        ptr: usize,
    ) -> usize {
        let inner = &mut cx.store.inner;
        let memory = &cx.memory.unwrap();
        memory.write(inner, ptr, &self.to_le_bytes()).unwrap();
        ptr + 4
    }

    fn store_flat<E: WasmEngine, T>(
        &self,
        _: &mut LowerContext<'_, '_, E, T>,
        dst: &mut Vec<wasm_runtime_layer::Value>,
    ) {
        dst.push(wasm_runtime_layer::Value::I32(*self))
    }
}

impl Lift for i32 {
    fn load<E: WasmEngine, T>(
        cx: &mut LiftContext<'_, '_, E, T>,
        _: &Memory,
        _: &mut dyn PeekableIter<Item = &Type>,
        ptr: usize,
    ) -> (Self, usize) {
        let memory = cx.memory.unwrap();
        let mut buf = [0u8; 4];
        memory
            .read(&mut cx.store.inner, ptr as usize, &mut buf)
            .unwrap();
        let v = i32::from_le_bytes(buf);
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

        v
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
        _: &mut LowerContext<'_, '_, E, T>,
        _: &Memory,
        _: usize,
    ) -> usize {
        todo!()
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

        let mut cur = ptr as usize;
        for item in self {
            // TODO: specialize using a default trait method :D
            cur = item.store(cx, memory, cur);
        }

        (ptr, byte_len).store_flat(cx, dst);
    }
}

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
            Value::I32(0),
            Value::I32(0),
            Value::I32(align),
            Value::I32(size),
        ],
        &mut res,
    )?;

    let [Value::I32(ptr)] = res else {
        panic!("Invalid return value")
    };

    Ok(ptr)
}
