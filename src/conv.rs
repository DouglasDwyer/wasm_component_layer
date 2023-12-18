//! Conversion between guest and host types

use std::vec;

use wasm_runtime_layer::{backend::WasmEngine, Func, Memory};
use wit_parser::Type;

use crate::{StoreContextMut, Value};

/// Implementation for native rust types
mod primitive_impls;

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
        cx: LiftContext<'_, '_, E, T>,
        memory: &Memory,
        ty: Type,
        ptr: i32,
    ) -> Self;

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
        ptr: usize,
    ) -> usize;

    /// Lower into guest function arguments
    fn store_flat<E: WasmEngine, T>(
        &self,
        cx: &mut LowerContext<'_, '_, E, T>,
        dst: &mut Vec<wasm_runtime_layer::Value>,
    );
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
            _ => {
                todo!()
            }
        }
    }
}

impl Lift for Value {
    fn load<E: WasmEngine, T>(
        cx: LiftContext<'_, '_, E, T>,
        memory: &Memory,
        ty: Type,
        ptr: i32,
    ) -> Self {
        match ty {
            Type::S32 => Value::S32(i32::load(cx, memory, ty, ptr)),
            _ => todo!(),
        }
    }

    fn load_flat<E: WasmEngine, T>(
        cx: &mut LiftContext<'_, '_, E, T>,
        ty: &mut dyn Iterator<Item = &Type>,
        args: &mut vec::IntoIter<wasm_runtime_layer::Value>,
    ) -> Self {
        match ty.next().unwrap() {
            Type::S32 => Value::S32(i32::load_flat(cx, ty, args)),
            _ => todo!(),
        }
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

impl<A: ComponentType, B: ComponentType> ComponentType for (A, B) {
    fn size(&self) -> usize {
        let mut s = 0;
        s = align_to(s + self.0.size(), self.0.align());
        s = align_to(s + self.1.size(), self.1.align());

        s
    }

    fn align(&self) -> usize {
        self.0.align().max(self.1.align())
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
        memory: &Memory,
        mut ptr: usize,
    ) -> usize {
        ptr = self.0.store(cx, memory, ptr);
        ptr = self.1.store(cx, memory, ptr);

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
    /// Realloc function, if available
    pub realloc: Option<&'a Func>,
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
