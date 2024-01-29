//! Implements lifting and lower traits for the dynamically typed `Value` enum, and related
//! newtypes.

use wasm_runtime_layer::{backend::WasmEngine, Memory};

use crate::{private::ListSpecialization, List, ValueType};

use super::{alloc_list, ComponentType, Lower, LowerContext};

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
            ListSpecialization::U32(v) => {
                Lower::store_list(v, cx, memory, dst_ptr as usize);
            }
            ListSpecialization::S64(v) => {
                Lower::store_list(v, cx, memory, dst_ptr as usize);
            }
            ListSpecialization::U64(v) => {
                Lower::store_list(v, cx, memory, dst_ptr as usize);
            }
            ListSpecialization::F32(v) => {
                Lower::store_list(v, cx, memory, dst_ptr as usize);
            }
            ListSpecialization::F64(v) => {
                Lower::store_list(v, cx, memory, dst_ptr as usize);
            }
            ListSpecialization::Char(v) => {
                Lower::store_list(v, cx, memory, dst_ptr as usize);
            }
            ListSpecialization::Bool(v) => {
                Lower::store_list(v, cx, memory, dst_ptr as usize);
            }
            ListSpecialization::S8(v) => {
                Lower::store_list(v, cx, memory, dst_ptr as usize);
            }
            ListSpecialization::U8(v) => {
                Lower::store_list(v, cx, memory, dst_ptr as usize);
            }
            ListSpecialization::S16(v) => {
                Lower::store_list(v, cx, memory, dst_ptr as usize);
            }
            ListSpecialization::U16(v) => {
                Lower::store_list(v, cx, memory, dst_ptr as usize);
            }
        }

        (dst_ptr, self.len() as i32).store_flat(cx, dst)
    }
}
