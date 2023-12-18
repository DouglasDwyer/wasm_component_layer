use std::{ops::Deref, sync::Arc};

use wasm_runtime_layer::{backend::WasmEngine, Memory, Value};
use wit_parser::Type;

use super::{ComponentType, Lift, LiftContext, Lower, LowerContext};

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
        cx: LiftContext<'_, '_, E, T>,
        _: &Memory,
        _: Type,
        ptr: i32,
    ) -> Self {
        let memory = cx.memory.unwrap();
        let mut buf = [0u8; 4];
        memory.read(cx.store.inner, ptr as usize, &mut buf).unwrap();
        let v = i32::from_le_bytes(buf);
        tracing::debug!(?v);
        v
    }

    fn load_flat<E: WasmEngine, T>(
        _: &mut LiftContext<'_, '_, E, T>,
        _: &mut dyn Iterator<Item = &Type>,
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

impl ComponentType for Arc<str> {
    fn size(&self) -> usize {
        self.deref().size()
    }

    fn align(&self) -> usize {
        self.deref().align()
    }
}

impl ComponentType for String {
    fn size(&self) -> usize {
        self.deref().size()
    }

    fn align(&self) -> usize {
        self.deref().align()
    }
}

impl Lower for str {
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

        memory
            .write(&mut cx.store.inner, ptr as _, self.as_bytes())
            .unwrap();

        (ptr, byte_len).store_flat(cx, dst);
    }
}

impl Lower for Arc<str> {
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

impl Lower for String {
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

/// Allocate a block of memory in the guest for string and list lowering
fn alloc_list<E: WasmEngine, T>(
    cx: &mut LowerContext<'_, '_, E, T>,
    size: i32,
    align: i32,
) -> anyhow::Result<i32> {
    let mut res = [wasm_runtime_layer::Value::I32(0)];
    cx.realloc.unwrap().call(
        &mut cx.store.inner,
        // old_ptr, old_len, align, new_len
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
