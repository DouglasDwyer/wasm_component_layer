//! Conversion between guest and host types

use core::slice;
use std::vec;

use wasm_runtime_layer::{
    backend::{AsContext, WasmEngine, WasmMemory},
    Memory,
};
use wit_parser::Type;

use crate::{StoreContextMut, Value};

pub struct ComponentTypeInfo {
    size: i32,
    alignment: i32,
}

pub trait ComponentType {
    /// Returns the current type
    fn ty(&self) -> ComponentTypeInfo;
}

/// Converts a value from the guest to the host
///
/// See:
/// <https://github.com/WebAssembly/component-model/blob/main/design/mvp/CanonicalABI.md#loading>
pub trait Lift {
    /// Reads the value from guest memory
    ///
    /// `ty` is used to infer the type to load for dynamically typed destination types, such as
    /// [`Value`].
    fn load<E: WasmEngine, T>(cx: LiftContext<'_, '_, E, T>, ty: Type, ptr: i32) -> Self;

    /// Reads the value from flat arguments
    ///
    /// `ty` is used to infer the type to load for dynamically typed destination types, such as
    /// [`Value`].
    fn load_flat<E: WasmEngine, T>(
        cx: &mut LiftContext<'_, '_, E, T>,
        ty: &mut dyn Iterator<Item = &Type>,
        args: &mut vec::IntoIter<wasm_runtime_layer::Value>,
    ) -> Self;
}

/// Converts a value from the host to the guest
///
/// See:
/// <https://github.com/WebAssembly/component-model/blob/main/design/mvp/CanonicalABI.md#storing>
pub trait Lower {
    /// Lower the value to guest memory
    ///
    /// Returns a new pointer to the end of the written bytes.
    ///
    /// It is the responsibility of each type implementation to the correct stride is returned.
    ///
    /// The stride is not statically known to allow implementing for dynamically sized types and
    /// varying size enums.
    fn store<E: WasmEngine, T>(&self, cx: &mut LowerContext<'_, '_, E, T>, ptr: usize) -> usize;

    /// Lower into guest function arguments
    fn store_flat<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        dst: &mut Vec<wasm_runtime_layer::Value>,
    );
}

impl Lower for Value {
    fn store<E: WasmEngine, T>(&self, cx: &mut LowerContext<'_, '_, E, T>, ptr: usize) -> usize {
        let inner = &mut cx.store.inner;
        let memory = &cx.memory.unwrap();

        match self {
            Value::S32(v) => {
                memory.write(inner, ptr, &v.to_le_bytes()).unwrap();
                4
            }
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
            Value::S32(v) => dst.push(wasm_runtime_layer::Value::I32(*v)),
            _ => {
                todo!()
            }
        }
    }
}

impl Lift for Value {
    fn load<E: WasmEngine, T>(mut cx: LiftContext<'_, '_, E, T>, ty: Type, ptr: i32) -> Self {
        let inner = &mut cx.store.inner;

        match ty {
            Type::S32 => {
                let memory = cx.memory.unwrap();
                let mut buf = [0u8; 4];
                memory.read(inner, ptr as usize, &mut buf).unwrap();
                let v = i32::from_le_bytes(buf);
                tracing::debug!(?v);
                Value::S32(v)
            }
            _ => todo!(),
        }
    }

    fn load_flat<E: WasmEngine, T>(
        cx: &mut LiftContext<'_, '_, E, T>,
        ty: &mut dyn Iterator<Item = &Type>,
        args: &mut vec::IntoIter<wasm_runtime_layer::Value>,
    ) -> Self {
        match ty.next().unwrap() {
            Type::S32 => {
                let wasm_runtime_layer::Value::I32(v) = args.next().expect("too few arguments")
                else {
                    panic!("incorrect type, expected S32");
                };

                Value::S32(v)
            }
            _ => todo!(),
        }
    }
}

impl<A, B> Lower for (A, B)
where
    A: Lower,
    B: Lower,
{
    fn store<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        mut ptr: usize,
    ) -> usize {
        ptr = self.0.store(cx, ptr);
        ptr = self.1.store(cx, ptr);

        ptr
    }

    fn store_flat<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        dst: &mut Vec<wasm_runtime_layer::Value>,
    ) {
        self.0.store_flat(cx, dst);
        self.1.store_flat(cx, dst);
    }
}

/// Used when lowering into guest memory
pub struct LowerContext<'a, 't, E: WasmEngine, T> {
    /// The store context
    pub store: &'a mut StoreContextMut<'t, T, E>,
    /// The guest memory
    pub memory: Option<&'a Memory>,
}

/// Used when lifting from guest memory
pub struct LiftContext<'a, 't, E: WasmEngine, T> {
    /// The store context
    pub store: StoreContextMut<'t, T, E>,
    /// The guest memory
    ///
    /// It is not always available, such as during init
    pub memory: Option<&'a Memory>,
}
