#![forbid(unused_results)]

use std::cell::*;

use anyhow::*;
pub use wit_parser::abi::{AbiVariant, WasmSignature, WasmType};
use wit_parser::*;

/// Joins two WASM types.
fn join(a: WasmType, b: WasmType) -> WasmType {
    use WasmType::*;

    match (a, b) {
        (I32, I32) | (I64, I64) | (F32, F32) | (F64, F64) => a,

        (I32, F32) | (F32, I32) => I32,

        (_, I64 | F64) | (I64 | F64, _) => I64,
    }
}

/// Aligns an address to a specific value.
fn align_to(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

/// Helper macro for defining instructions without having to have tons of
/// exhaustive `match` statements to update
macro_rules! def_instruction {
    (
        $( #[$enum_attr:meta] )*
        pub enum $name:ident<'a> {
            $(
                $( #[$attr:meta] )*
                $variant:ident $( {
                    $($field:ident : $field_ty:ty $(,)* )*
                } )?
                    :
                [$num_popped:expr] => [$num_pushed:expr],
            )*
        }
    ) => {
        $( #[$enum_attr] )*
        pub enum $name<'a> {
            $(
                $( #[$attr] )*
                $variant $( {
                    $(
                        $field : $field_ty,
                    )*
                } )? ,
            )*
        }

        impl $name<'_> {
            /// How many operands does this instruction pop from the stack?
            #[allow(unused_variables)]
            pub fn operands_len(&self) -> usize {
                match self {
                    $(
                        Self::$variant $( {
                            $(
                                $field,
                            )*
                        } )? => $num_popped,
                    )*
                }
            }

            /// How many results does this instruction push onto the stack?
            #[allow(unused_variables)]
            pub fn results_len(&self) -> usize {
                match self {
                    $(
                        Self::$variant $( {
                            $(
                                $field,
                            )*
                        } )? => $num_pushed,
                    )*
                }
            }
        }
    };
}

def_instruction! {
    /// Describes an action to take in the ABI stack machine.
    #[derive(Debug)]
    pub enum Instruction<'a> {
        /// Acquires the specified parameter and places it on the stack.
        /// Depending on the context this may refer to wasm parameters or
        /// interface types parameters.
        GetArg { nth: usize } : [0] => [1],

        // Integer const/manipulation instructions

        /// Pushes the constant `val` onto the stack.
        I32Const { val: i32 } : [0] => [1],
        /// Casts the top N items on the stack using the `Bitcast` enum
        /// provided. Consumes the same number of operands that this produces.
        Bitcasts { casts: &'a [Bitcast] } : [casts.len()] => [casts.len()],
        /// Pushes a number of constant zeros for each wasm type on the stack.
        ConstZero { tys: &'a [WasmType] } : [0] => [tys.len()],

        // Memory load/store instructions

        /// Pops an `i32` from the stack and loads a little-endian `i32` from
        /// it, using the specified constant offset.
        I32Load { offset: i32 } : [1] => [1],
        /// Pops an `i32` from the stack and loads a little-endian `i8` from
        /// it, using the specified constant offset. The value loaded is the
        /// zero-extended to 32-bits
        I32Load8U { offset: i32 } : [1] => [1],
        /// Pops an `i32` from the stack and loads a little-endian `i8` from
        /// it, using the specified constant offset. The value loaded is the
        /// sign-extended to 32-bits
        I32Load8S { offset: i32 } : [1] => [1],
        /// Pops an `i32` from the stack and loads a little-endian `i16` from
        /// it, using the specified constant offset. The value loaded is the
        /// zero-extended to 32-bits
        I32Load16U { offset: i32 } : [1] => [1],
        /// Pops an `i32` from the stack and loads a little-endian `i16` from
        /// it, using the specified constant offset. The value loaded is the
        /// sign-extended to 32-bits
        I32Load16S { offset: i32 } : [1] => [1],
        /// Pops an `i32` from the stack and loads a little-endian `i64` from
        /// it, using the specified constant offset.
        I64Load { offset: i32 } : [1] => [1],
        /// Pops an `i32` from the stack and loads a little-endian `f32` from
        /// it, using the specified constant offset.
        F32Load { offset: i32 } : [1] => [1],
        /// Pops an `i32` from the stack and loads a little-endian `f64` from
        /// it, using the specified constant offset.
        F64Load { offset: i32 } : [1] => [1],

        /// Pops an `i32` address from the stack and then an `i32` value.
        /// Stores the value in little-endian at the pointer specified plus the
        /// constant `offset`.
        I32Store { offset: i32 } : [2] => [0],
        /// Pops an `i32` address from the stack and then an `i32` value.
        /// Stores the low 8 bits of the value in little-endian at the pointer
        /// specified plus the constant `offset`.
        I32Store8 { offset: i32 } : [2] => [0],
        /// Pops an `i32` address from the stack and then an `i32` value.
        /// Stores the low 16 bits of the value in little-endian at the pointer
        /// specified plus the constant `offset`.
        I32Store16 { offset: i32 } : [2] => [0],
        /// Pops an `i32` address from the stack and then an `i64` value.
        /// Stores the value in little-endian at the pointer specified plus the
        /// constant `offset`.
        I64Store { offset: i32 } : [2] => [0],
        /// Pops an `i32` address from the stack and then an `f32` value.
        /// Stores the value in little-endian at the pointer specified plus the
        /// constant `offset`.
        F32Store { offset: i32 } : [2] => [0],
        /// Pops an `i32` address from the stack and then an `f64` value.
        /// Stores the value in little-endian at the pointer specified plus the
        /// constant `offset`.
        F64Store { offset: i32 } : [2] => [0],

        // Scalar lifting/lowering

        /// Converts an interface type `char` value to a 32-bit integer
        /// representing the unicode scalar value.
        I32FromChar : [1] => [1],
        /// Converts an interface type `u64` value to a wasm `i64`.
        I64FromU64 : [1] => [1],
        /// Converts an interface type `s64` value to a wasm `i64`.
        I64FromS64 : [1] => [1],
        /// Converts an interface type `u32` value to a wasm `i32`.
        I32FromU32 : [1] => [1],
        /// Converts an interface type `s32` value to a wasm `i32`.
        I32FromS32 : [1] => [1],
        /// Converts an interface type `u16` value to a wasm `i32`.
        I32FromU16 : [1] => [1],
        /// Converts an interface type `s16` value to a wasm `i32`.
        I32FromS16 : [1] => [1],
        /// Converts an interface type `u8` value to a wasm `i32`.
        I32FromU8 : [1] => [1],
        /// Converts an interface type `s8` value to a wasm `i32`.
        I32FromS8 : [1] => [1],
        /// Conversion an interface type `f32` value to a wasm `f32`.
        ///
        /// This may be a noop for some implementations, but it's here in case the
        /// native language representation of `f32` is different than the wasm
        /// representation of `f32`.
        F32FromFloat32 : [1] => [1],
        /// Conversion an interface type `f64` value to a wasm `f64`.
        ///
        /// This may be a noop for some implementations, but it's here in case the
        /// native language representation of `f64` is different than the wasm
        /// representation of `f64`.
        F64FromFloat64 : [1] => [1],

        /// Converts a native wasm `i32` to an interface type `s8`.
        ///
        /// This will truncate the upper bits of the `i32`.
        S8FromI32 : [1] => [1],
        /// Converts a native wasm `i32` to an interface type `u8`.
        ///
        /// This will truncate the upper bits of the `i32`.
        U8FromI32 : [1] => [1],
        /// Converts a native wasm `i32` to an interface type `s16`.
        ///
        /// This will truncate the upper bits of the `i32`.
        S16FromI32 : [1] => [1],
        /// Converts a native wasm `i32` to an interface type `u16`.
        ///
        /// This will truncate the upper bits of the `i32`.
        U16FromI32 : [1] => [1],
        /// Converts a native wasm `i32` to an interface type `s32`.
        S32FromI32 : [1] => [1],
        /// Converts a native wasm `i32` to an interface type `u32`.
        U32FromI32 : [1] => [1],
        /// Converts a native wasm `i64` to an interface type `s64`.
        S64FromI64 : [1] => [1],
        /// Converts a native wasm `i64` to an interface type `u64`.
        U64FromI64 : [1] => [1],
        /// Converts a native wasm `i32` to an interface type `char`.
        ///
        /// It's safe to assume that the `i32` is indeed a valid unicode code point.
        CharFromI32 : [1] => [1],
        /// Converts a native wasm `f32` to an interface type `f32`.
        Float32FromF32 : [1] => [1],
        /// Converts a native wasm `f64` to an interface type `f64`.
        Float64FromF64 : [1] => [1],

        /// Creates a `bool` from an `i32` input, trapping if the `i32` isn't
        /// zero or one.
        BoolFromI32 : [1] => [1],
        /// Creates an `i32` from a `bool` input, must return 0 or 1.
        I32FromBool : [1] => [1],

        /// Lowers a list where the element's layout in the native language is
        /// expected to match the canonical ABI definition of interface types.
        ///
        /// Pops a list value from the stack and pushes the pointer/length onto
        /// the stack. If `realloc` is set to `Some` then this is expected to
        /// *consume* the list which means that the data needs to be copied. An
        /// allocation/copy is expected when:
        ///
        /// * A host is calling a wasm export with a list (it needs to copy the
        ///   list in to the callee's module, allocating space with `realloc`)
        /// * A wasm export is returning a list (it's expected to use `realloc`
        ///   to give ownership of the list to the caller.
        /// * A host is returning a list in a import definition, meaning that
        ///   space needs to be allocated in the caller with `realloc`).
        ///
        /// A copy does not happen (e.g. `realloc` is `None`) when:
        ///
        /// * A wasm module calls an import with the list. In this situation
        ///   it's expected the caller will know how to access this module's
        ///   memory (e.g. the host has raw access or wasm-to-wasm communication
        ///   would copy the list).
        ///
        /// If `realloc` is `Some` then the adapter is not responsible for
        /// cleaning up this list because the other end is receiving the
        /// allocation. If `realloc` is `None` then the adapter is responsible
        /// for cleaning up any temporary allocation it created, if any.
        ListCanonLower {
            element: &'a Type,
            realloc: Option<&'a str>,
        } : [1] => [2],

        /// Pops a string from the stack, lowers it into guest memory, and pushes a pointer and length onto the stack.
        StringLower {
            realloc: Option<&'a str>,
        } : [1] => [2],

        /// Lowers a list where the element's layout in the native language is
        /// not expected to match the canonical ABI definition of interface
        /// types.
        ///
        /// Pops a list value from the stack, and pushes the following in order:
        ///
        /// - The pointer to the list
        /// - The length of the list
        /// - ...the values in the list
        /// - The length again
        ///
        /// The `realloc` field here is only set to `None` when a wasm module calls a declared import.
        /// Otherwise lowering in other contexts requires allocating memory for
        /// the receiver to own.
        ListLower {
            element: &'a Type,
            realloc: Option<&'a str>,
            len: Cell<i32>
        } : [1] => [(len.get() as usize) + 3],

        /// Lifts a list which has a canonical representation into an interface
        /// types value.
        ///
        /// The term "canonical" representation here means that the
        /// representation of the interface types value in the native language
        /// exactly matches the canonical ABI definition of the type.
        ///
        /// This will consume two `i32` values from the stack, a pointer and a
        /// length, and then produces an interface value list.
        ListCanonLift {
            element: &'a Type,
            ty: TypeId,
        } : [2] => [1],

        /// Pops a length and pointer off of the stack, and lifts a string.
        StringLift : [2] => [1],

        /// Lifts a list which into an interface types value.
        ///
        /// This will consume `len` operands of type `element` from the stack,
        /// concatenating them into a single list.
        ListLift {
            element: &'a Type,
            ty: TypeId,
            len: i32
        } : [*len as usize] => [1],

        /// Converts a 32-bit integer from a stack value, and stores it in the instruction.
        ReadI32 { value: Cell<i32> } : [1] => [0],

        // records and tuples

        /// Pops a record value off the stack, decomposes the record to all of
        /// its fields, and then pushes the fields onto the stack.
        RecordLower {
            record: &'a Record,
            ty: TypeId,
        } : [1] => [record.fields.len()],

        /// Pops all fields for a record off the stack and then composes them
        /// into a record.
        RecordLift {
            record: &'a Record,
            ty: TypeId,
        } : [record.fields.len()] => [1],

        /// Create an `i32` from a handle.
        HandleLower {
            handle: &'a Handle,
            ty: TypeId,
        } : [1] => [1],

        /// Create a handle from an `i32`.
        HandleLift {
            handle: &'a Handle,
            ty: TypeId,
        } : [1] => [1],

        /// Pops a tuple value off the stack, decomposes the tuple to all of
        /// its fields, and then pushes the fields onto the stack.
        TupleLower {
            tuple: &'a Tuple,
            ty: TypeId,
        } : [1] => [tuple.types.len()],

        /// Pops all fields for a tuple off the stack and then composes them
        /// into a tuple.
        TupleLift {
            tuple: &'a Tuple,
            ty: TypeId,
        } : [tuple.types.len()] => [1],

        /// Converts a language-specific record-of-bools to a list of `i32`.
        FlagsLower {
            flags: &'a Flags,
            ty: TypeId,
        } : [1] => [flags.repr().count()],
        /// Converts a list of native wasm `i32` to a language-specific
        /// record-of-bools.
        FlagsLift {
            flags: &'a Flags,
            ty: TypeId,
        } : [flags.repr().count()] => [1],

        // variants

        /// Pops a variant value from the stack, and stores the discriminant in the instruction.
        /// Pushes a value onto the stack if one exists.
        ExtractVariantDiscriminant { discriminant_value: Cell<(i32, bool)> } : [1] => [if discriminant_value.get().1 { 1 } else { 0 }],

        /// Pops an `i32` off the stack as well as `ty.cases.len()` blocks
        /// from the code generator. Uses each of those blocks and the value
        /// from the stack to produce a final variant.
        VariantLift {
            variant: &'a Variant,
            ty: TypeId,
            discriminant: i32,
            has_value: bool,
        } : [if *has_value { 1 } else { 0 }] => [1],

        /// Pops an enum off the stack and pushes the `i32` representation.
        EnumLower {
            enum_: &'a Enum,
            ty: TypeId,
        } : [1] => [1],

        /// Loads the specified discriminant into the `enum` specified.
        EnumLift {
            enum_: &'a Enum,
            ty: TypeId,
            discriminant: i32,
        } : [0] => [1],

        /// Specialization of `VariantLift` for specifically the `option<T>`
        /// type. Otherwise behaves the same as the `VariantLift` instruction
        /// with two blocks for the lift.
        OptionLift {
            payload: &'a Type,
            ty: TypeId,
            discriminant: i32,
            has_value: bool,
        } : [if *has_value { 1 } else { 0 }] => [1],

        /// Specialization of `VariantLift` for specifically the `result<T,
        /// E>` type. Otherwise behaves the same as the `VariantLift`
        /// instruction with two blocks for the lift.
        ResultLift {
            result: &'a Result_,
            ty: TypeId,
            discriminant: i32,
            has_value: bool,
        } : [if *has_value { 1 } else { 0 }] => [1],

        // calling/control flow

        /// Represents a call to a raw WebAssembly API. The module/name are
        /// provided inline as well as the types if necessary.
        CallWasm {
            name: &'a str,
            sig: &'a WasmSignature,
        } : [sig.params.len()] => [sig.results.len()],

        /// Same as `CallWasm`, except the dual where an interface is being
        /// called rather than a raw wasm function.
        ///
        /// Note that this will be used for async functions.
        CallInterface {
            func: &'a Function,
        } : [func.params.len()] => [func.results.len()],

        /// Returns `amt` values on the stack. This is always the last
        /// instruction.
        Return { amt: usize, func: &'a Function } : [*amt] => [0],

        /// Calls the `realloc` function specified in a malloc-like fashion
        /// allocating `size` bytes with alignment `align`.
        ///
        /// Pushes the returned pointer onto the stack.
        Malloc {
            realloc: &'static str,
            size: usize,
            align: usize,
        } : [0] => [1],
    }
}

/// Describes a casting operation that should happen on a primitive type.
#[derive(Debug, PartialEq)]
pub enum Bitcast {
    /// Casts an `f32` to an `i32`.
    F32ToI32,
    /// Casts an `f64` to an `i64`.
    F64ToI64,
    /// Casts an `i32` to an `i64`.
    I32ToI64,
    /// Casts an `f32` to an `i64`.
    F32ToI64,
    /// Casts an `i32` to an `f32`.
    I32ToF32,
    /// Casts an `i64` to an `f64`.
    I64ToF64,
    /// Casts an `i64` to an `i32`.
    I64ToI32,
    /// Casts an `i64` to an `f32`.
    I64ToF32,
    /// The identify operation.
    None,
}

/// Whether the glue code surrounding a call is lifting arguments and lowering
/// results or vice versa.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LiftLower {
    /// When the glue code lifts arguments and lowers results.
    ///
    /// ```text
    /// Wasm --lift-args--> SourceLanguage; call; SourceLanguage --lower-results--> Wasm
    /// ```
    LiftArgsLowerResults,
    /// When the glue code lowers arguments and lifts results.
    ///
    /// ```text
    /// SourceLanguage --lower-args--> Wasm; call; Wasm --lift-results--> SourceLanguage
    /// ```
    LowerArgsLiftResults,
}

/// Trait for language implementors to use to generate glue code between native
/// WebAssembly signatures and interface types signatures.
///
/// This is used as an implementation detail in interpreting the ABI between
/// interface types and wasm types. Eventually this will be driven by interface
/// types adapters themselves, but for now the ABI of a function dictates what
/// instructions are fed in.
///
/// Types implementing `Bindgen` are incrementally fed `Instruction` values to
/// generate code for. Instructions operate like a stack machine where each
/// instruction has a list of inputs and a list of outputs (provided by the
/// `emit` function).
pub trait Bindgen {
    /// The intermediate type for fragments of code for this type.
    ///
    /// For most languages `String` is a suitable intermediate type.
    type Operand: Clone + std::fmt::Debug;

    /// Emit code to implement the given instruction.
    ///
    /// Each operand is given in `operands` and can be popped off if ownership
    /// is required. It's guaranteed that `operands` has the appropriate length
    /// for the `inst` given, as specified with [`Instruction`].
    ///
    /// Each result variable should be pushed onto `results`. This function must
    /// push the appropriate number of results or binding generation will panic.
    fn emit(
        &mut self,
        resolve: &Resolve,
        inst: &Instruction<'_>,
        operands: &mut Vec<Self::Operand>,
        results: &mut Vec<Self::Operand>,
    ) -> Result<()>;

    /// Returns size information that was previously calculated for all types.
    fn sizes(&self) -> &SizeAlign;

    /// Determines whether a list's language-specific representation matches the canonical representation.
    fn is_list_canonical(&self, element: &Type) -> bool;
}

/// Generates stack-based instructions for using the canonical ABI.
///
/// This code was adapted from the `wit-parser` crate.
pub struct Generator<'a, B: Bindgen> {
    /// The ABI variant to use.
    variant: AbiVariant,
    /// The lift-lower type to use.
    lift_lower: LiftLower,
    /// The bindgen to use.
    bindgen: &'a mut B,
    /// The resolve to use.
    resolve: &'a Resolve,
    /// The set of operands for this call.
    operands: Vec<B::Operand>,
    /// The set of results for this call.
    results: Vec<B::Operand>,
    /// The full stack.
    stack: Vec<B::Operand>,
}

impl<'a, B: Bindgen> Generator<'a, B> {
    /// Creates a new generator.
    pub fn new(
        resolve: &'a Resolve,
        variant: AbiVariant,
        lift_lower: LiftLower,
        bindgen: &'a mut B,
    ) -> Generator<'a, B> {
        Generator {
            resolve,
            variant,
            lift_lower,
            bindgen,
            operands: Vec::new(),
            results: Vec::new(),
            stack: Vec::new(),
        }
    }

    /// Calls the provided function with the generator.
    pub fn call(&mut self, func: &Function) -> Result<()> {
        let sig = self.resolve.wasm_signature(self.variant, func);

        match self.lift_lower {
            LiftLower::LowerArgsLiftResults => {
                if !sig.indirect_params {
                    // If the parameters for this function aren't indirect
                    // (there aren't too many) then we simply do a normal lower
                    // operation for them all.
                    for (nth, (_, ty)) in func.params.iter().enumerate() {
                        self.emit(&Instruction::GetArg { nth })?;
                        self.lower(ty)?;
                    }
                } else {
                    // ... otherwise if parameters are indirect space is
                    // allocated from them and each argument is lowered
                    // individually into memory.
                    let (size, align) = self
                        .bindgen
                        .sizes()
                        .record(func.params.iter().map(|t| &t.1));
                    let ptr = match self.variant {
                        // When a wasm module calls an import it will provide
                        // space that isn't explicitly deallocated.
                        AbiVariant::GuestImport => unimplemented!(),
                        // When calling a wasm module from the outside, though,
                        // malloc needs to be called.
                        AbiVariant::GuestExport => {
                            self.emit(&Instruction::Malloc {
                                realloc: "cabi_realloc",
                                size,
                                align,
                            })?;
                            self.stack.pop().unwrap()
                        },
                    };
                    let mut offset = 0usize;
                    for (nth, (_, ty)) in func.params.iter().enumerate() {
                        self.emit(&Instruction::GetArg { nth })?;
                        offset = align_to(offset, self.bindgen.sizes().align(ty));
                        self.write_to_memory(ty, ptr.clone(), offset as i32)?;
                        offset += self.bindgen.sizes().size(ty);
                    }

                    self.stack.push(ptr);
                }

                // Now that all the wasm args are prepared we can call the
                // actual wasm function.
                assert_eq!(self.stack.len(), sig.params.len());
                self.emit(&Instruction::CallWasm {
                    name: &func.name,
                    sig: &sig,
                })?;

                if !sig.retptr {
                    // With no return pointer in use we can simply lift the
                    // result(s) of the function from the result of the core
                    // wasm function.
                    for ty in func.results.iter_types() {
                        self.lift(ty)?
                    }
                } else {
                    let ptr = self.stack.pop().unwrap();

                    self.read_results_from_memory(&func.results, ptr, 0)?;
                }

                self.emit(&Instruction::Return {
                    func,
                    amt: func.results.len(),
                })?;
            },
            LiftLower::LiftArgsLowerResults => {
                if !sig.indirect_params {
                    // If parameters are not passed indirectly then we lift each
                    // argument in succession from the component wasm types that
                    // make-up the type.
                    let mut offset = 0;
                    let mut temp = Vec::new();
                    for (_, ty) in func.params.iter() {
                        temp.truncate(0);
                        push_wasm(self.resolve, self.variant, ty, &mut temp);
                        for _ in 0..temp.len() {
                            self.emit(&Instruction::GetArg { nth: offset })?;
                            offset += 1;
                        }
                        self.lift(ty)?;
                    }
                } else {
                    // ... otherwise argument is read in succession from memory
                    // where the pointer to the arguments is the first argument
                    // to the function.
                    let mut offset = 0usize;
                    self.emit(&Instruction::GetArg { nth: 0 })?;
                    let ptr = self.stack.pop().unwrap();
                    for (_, ty) in func.params.iter() {
                        offset = align_to(offset, self.bindgen.sizes().align(ty));
                        self.read_from_memory(ty, ptr.clone(), offset as i32)?;
                        offset += self.bindgen.sizes().size(ty);
                    }
                }

                // ... and that allows us to call the interface types function
                self.emit(&Instruction::CallInterface { func })?;

                if !sig.retptr {
                    // With no return pointer in use we simply lower the
                    // result(s) and return that directly from the function.
                    let results = self
                        .stack
                        .drain(self.stack.len() - func.results.len()..)
                        .collect::<Vec<_>>();
                    for (ty, result) in func.results.iter_types().zip(results) {
                        self.stack.push(result);
                        self.lower(ty)?;
                    }
                } else {
                    match self.variant {
                        // When a function is imported to a guest this means
                        // it's a host providing the implementation of the
                        // import. The result is stored in the pointer
                        // specified in the last argument, so we get the
                        // pointer here and then write the return value into
                        // it.
                        AbiVariant::GuestImport => {
                            self.emit(&Instruction::GetArg {
                                nth: sig.params.len() - 1,
                            })?;
                            let ptr = self.stack.pop().unwrap();
                            self.write_params_to_memory(func.results.iter_types(), ptr, 0)?;
                        },

                        // For a guest import this is a function defined in
                        // wasm, so we're returning a pointer where the
                        // value was stored at. Allocate some space here
                        // (statically) and then write the result into that
                        // memory, returning the pointer at the end.
                        AbiVariant::GuestExport => {
                            unimplemented!()
                        },
                    }
                }

                self.emit(&Instruction::Return {
                    func,
                    amt: sig.results.len(),
                })?;
            },
        }

        assert!(
            self.stack.is_empty(),
            "stack has {} items remaining",
            self.stack.len()
        );

        Ok(())
    }

    /// Emits the provided instruction.
    fn emit(&mut self, inst: &Instruction<'_>) -> Result<()> {
        self.operands.clear();
        self.results.clear();

        let operands_len = inst.operands_len();
        assert!(
            self.stack.len() >= operands_len,
            "not enough operands on stack for {:?}",
            inst
        );
        self.operands
            .extend(self.stack.drain((self.stack.len() - operands_len)..));
        self.results.reserve(inst.results_len());

        self.bindgen
            .emit(self.resolve, inst, &mut self.operands, &mut self.results)?;

        assert_eq!(
            self.results.len(),
            inst.results_len(),
            "{:?} expected {} results, got {}",
            inst,
            inst.results_len(),
            self.results.len()
        );
        self.stack.append(&mut self.results);

        Ok(())
    }

    /// Lowers the given type.
    fn lower(&mut self, ty: &Type) -> Result<()> {
        use Instruction::*;

        match *ty {
            Type::Bool => self.emit(&I32FromBool),
            Type::S8 => self.emit(&I32FromS8),
            Type::U8 => self.emit(&I32FromU8),
            Type::S16 => self.emit(&I32FromS16),
            Type::U16 => self.emit(&I32FromU16),
            Type::S32 => self.emit(&I32FromS32),
            Type::U32 => self.emit(&I32FromU32),
            Type::S64 => self.emit(&I64FromS64),
            Type::U64 => self.emit(&I64FromU64),
            Type::Char => self.emit(&I32FromChar),
            Type::Float32 => self.emit(&F32FromFloat32),
            Type::Float64 => self.emit(&F64FromFloat64),
            Type::String => {
                let realloc = self.list_realloc();
                self.emit(&StringLower { realloc })
            },
            Type::Id(id) => match &self.resolve.types[id].kind {
                TypeDefKind::Type(t) => self.lower(t),
                TypeDefKind::List(element) => {
                    let realloc = self.list_realloc();
                    if self.bindgen.is_list_canonical(element) {
                        self.emit(&ListCanonLower { element, realloc })
                    } else {
                        let lower = ListLower {
                            element,
                            realloc,
                            len: Cell::default(),
                        };
                        self.emit(&lower)?;

                        let stride = self.bindgen.sizes().size(element) as i32;
                        let len = if let ListLower { len, .. } = lower {
                            len.get()
                        } else {
                            unreachable!()
                        };

                        let addr = self.stack.pop().unwrap();

                        for i in (0..len).rev() {
                            self.write_to_memory(element, addr.clone(), i * stride)?;
                        }
                        Ok(())
                    }
                },
                TypeDefKind::Handle(handle) => self.emit(&HandleLower { handle, ty: id }),
                TypeDefKind::Resource => {
                    todo!();
                },
                TypeDefKind::Record(record) => {
                    self.emit(&RecordLower { record, ty: id })?;
                    let values = self
                        .stack
                        .drain(self.stack.len() - record.fields.len()..)
                        .collect::<Vec<_>>();
                    for (field, value) in record.fields.iter().zip(values) {
                        self.stack.push(value);
                        self.lower(&field.ty)?;
                    }
                    Ok(())
                },
                TypeDefKind::Tuple(tuple) => {
                    self.emit(&TupleLower { tuple, ty: id })?;
                    let values = self
                        .stack
                        .drain(self.stack.len() - tuple.types.len()..)
                        .collect::<Vec<_>>();
                    for (ty, value) in tuple.types.iter().zip(values) {
                        self.stack.push(value);
                        self.lower(ty)?;
                    }
                    Ok(())
                },

                TypeDefKind::Flags(flags) => self.emit(&FlagsLower { flags, ty: id }),

                TypeDefKind::Variant(v) => {
                    self.lower_variant_arm(ty, v.cases.iter().map(|c| c.ty.as_ref()))
                },
                TypeDefKind::Enum(enum_) => self.emit(&EnumLower { enum_, ty: id }),
                TypeDefKind::Option(t) => self.lower_variant_arm(ty, [None, Some(t)]),
                TypeDefKind::Result(r) => {
                    self.lower_variant_arm(ty, [r.ok.as_ref(), r.err.as_ref()])
                },
                TypeDefKind::Future(_) => todo!("lower future"),
                TypeDefKind::Stream(_) => todo!("lower stream"),
                TypeDefKind::Unknown => unreachable!(),
            },
        }
    }

    /// Lowers the given variant arm.
    fn lower_variant_arm<'b>(
        &mut self,
        ty: &Type,
        cases: impl IntoIterator<Item = Option<&'b Type>>,
    ) -> Result<()> {
        use Instruction::*;
        let disc_val = ExtractVariantDiscriminant {
            discriminant_value: Cell::default(),
        };
        self.emit(&disc_val)?;

        let discriminant = if let ExtractVariantDiscriminant { discriminant_value } = disc_val {
            discriminant_value.get().0
        } else {
            unreachable!()
        };

        let mut results = Vec::new();
        let mut temp = Vec::new();
        let mut casts = Vec::new();
        push_wasm(self.resolve, self.variant, ty, &mut results);

        let payload_name = self.stack.pop().unwrap();
        self.emit(&I32Const { val: discriminant })?;
        let mut pushed = 1;
        if let Some(ty) = cases
            .into_iter()
            .nth(discriminant as usize)
            .ok_or_else(|| Error::msg("Invalid discriminator value."))?
        {
            // Using the payload of this block we lower the type to
            // raw wasm values.
            self.stack.push(payload_name);
            self.lower(ty)?;

            // Determine the types of all the wasm values we just
            // pushed, and record how many. If we pushed too few
            // then we'll need to push some zeros after this.
            temp.truncate(0);
            push_wasm(self.resolve, self.variant, ty, &mut temp);
            pushed += temp.len();

            // For all the types pushed we may need to insert some
            // bitcasts. This will go through and cast everything
            // to the right type to ensure all blocks produce the
            // same set of results.
            casts.truncate(0);
            for (actual, expected) in temp.iter().zip(&results[1..]) {
                casts.push(cast(*actual, *expected));
            }
            if casts.iter().any(|c| *c != Bitcast::None) {
                self.emit(&Bitcasts { casts: &casts })?;
            }
        }

        // If we haven't pushed enough items in this block to match
        // what other variants are pushing then we need to push
        // some zeros.
        if pushed < results.len() {
            self.emit(&ConstZero {
                tys: &results[pushed..],
            })?;
        }

        Ok(())
    }

    /// Reallocates a list.
    fn list_realloc(&self) -> Option<&'static str> {
        // Lowering parameters calling a wasm import means
        // we don't need to pass ownership, but we pass
        // ownership in all other cases.
        match (self.variant, self.lift_lower) {
            (AbiVariant::GuestImport, LiftLower::LowerArgsLiftResults) => None,
            _ => Some("cabi_realloc"),
        }
    }

    /// Note that in general everything in this function is the opposite of the
    /// `lower` function above. This is intentional and should be kept this way!
    fn lift(&mut self, ty: &Type) -> Result<()> {
        use Instruction::*;

        match *ty {
            Type::Bool => self.emit(&BoolFromI32),
            Type::S8 => self.emit(&S8FromI32),
            Type::U8 => self.emit(&U8FromI32),
            Type::S16 => self.emit(&S16FromI32),
            Type::U16 => self.emit(&U16FromI32),
            Type::S32 => self.emit(&S32FromI32),
            Type::U32 => self.emit(&U32FromI32),
            Type::S64 => self.emit(&S64FromI64),
            Type::U64 => self.emit(&U64FromI64),
            Type::Char => self.emit(&CharFromI32),
            Type::Float32 => self.emit(&Float32FromF32),
            Type::Float64 => self.emit(&Float64FromF64),
            Type::String => self.emit(&StringLift),
            Type::Id(id) => match &self.resolve.types[id].kind {
                TypeDefKind::Type(t) => self.lift(t),
                TypeDefKind::List(element) => {
                    if self.bindgen.is_list_canonical(element) {
                        self.emit(&ListCanonLift { element, ty: id })
                    } else {
                        let len = ReadI32 {
                            value: Cell::default(),
                        };
                        self.emit(&len)?;

                        let len = match len {
                            ReadI32 { value } => value.get(),
                            _ => unreachable!(),
                        };

                        let addr = self.stack.pop().unwrap();
                        let stride = self.bindgen.sizes().size(element) as i32;

                        for i in 0..len {
                            self.read_from_memory(element, addr.clone(), stride * i)?;
                        }

                        self.emit(&ListLift {
                            element,
                            ty: id,
                            len,
                        })
                    }
                },
                TypeDefKind::Handle(handle) => self.emit(&HandleLift { handle, ty: id }),
                TypeDefKind::Resource => {
                    todo!();
                },
                TypeDefKind::Record(record) => {
                    let mut temp = Vec::new();
                    push_wasm(self.resolve, self.variant, ty, &mut temp);
                    let mut args = self
                        .stack
                        .drain(self.stack.len() - temp.len()..)
                        .collect::<Vec<_>>();
                    for field in record.fields.iter() {
                        temp.truncate(0);
                        push_wasm(self.resolve, self.variant, &field.ty, &mut temp);
                        self.stack.extend(args.drain(..temp.len()));
                        self.lift(&field.ty)?;
                    }
                    self.emit(&RecordLift { record, ty: id })
                },
                TypeDefKind::Tuple(tuple) => {
                    let mut temp = Vec::new();
                    push_wasm(self.resolve, self.variant, ty, &mut temp);
                    let mut args = self
                        .stack
                        .drain(self.stack.len() - temp.len()..)
                        .collect::<Vec<_>>();
                    for ty in tuple.types.iter() {
                        temp.truncate(0);
                        push_wasm(self.resolve, self.variant, ty, &mut temp);
                        self.stack.extend(args.drain(..temp.len()));
                        self.lift(ty)?;
                    }
                    self.emit(&TupleLift { tuple, ty: id })
                },
                TypeDefKind::Flags(flags) => self.emit(&FlagsLift { flags, ty: id }),

                TypeDefKind::Variant(v) => {
                    let (discriminant, has_value) =
                        self.lift_variant_arm(ty, v.cases.iter().map(|c| c.ty.as_ref()))?;
                    self.emit(&VariantLift {
                        variant: v,
                        ty: id,
                        discriminant,
                        has_value,
                    })
                },

                TypeDefKind::Enum(enum_) => {
                    let variant = ReadI32 {
                        value: Cell::default(),
                    };
                    self.emit(&variant)?;
                    if let ReadI32 { value } = variant {
                        self.emit(&EnumLift {
                            enum_,
                            ty: id,
                            discriminant: value.get(),
                        })
                    } else {
                        unreachable!()
                    }
                },

                TypeDefKind::Option(t) => {
                    let (discriminant, has_value) = self.lift_variant_arm(ty, [None, Some(t)])?;
                    self.emit(&OptionLift {
                        payload: t,
                        ty: id,
                        discriminant,
                        has_value,
                    })
                },

                TypeDefKind::Result(r) => {
                    let (discriminant, has_value) =
                        self.lift_variant_arm(ty, [r.ok.as_ref(), r.err.as_ref()])?;
                    self.emit(&ResultLift {
                        result: r,
                        ty: id,
                        discriminant,
                        has_value,
                    })
                },

                TypeDefKind::Future(_) => todo!("lift future"),
                TypeDefKind::Stream(_) => todo!("lift stream"),
                TypeDefKind::Unknown => unreachable!(),
            },
        }
    }

    /// Lifts a variant arm.
    fn lift_variant_arm<'b>(
        &mut self,
        ty: &Type,
        cases: impl IntoIterator<Item = Option<&'b Type>>,
    ) -> Result<(i32, bool)> {
        let variant = Instruction::ReadI32 {
            value: Cell::default(),
        };
        self.emit(&variant)?;
        if let Instruction::ReadI32 { value } = variant {
            let discriminant = value.get();
            let mut params = Vec::new();
            let mut temp = Vec::new();
            let mut casts = Vec::new();
            push_wasm(self.resolve, self.variant, ty, &mut params);
            let block_inputs = self
                .stack
                .drain(self.stack.len() + 1 - params.len()..)
                .collect::<Vec<_>>();

            let has_value = if let Some(ty) = cases
                .into_iter()
                .nth(discriminant as usize)
                .ok_or_else(|| Error::msg("Invalid discriminant value."))?
            {
                // Push only the values we need for this variant onto
                // the stack.
                temp.truncate(0);
                push_wasm(self.resolve, self.variant, ty, &mut temp);
                self.stack
                    .extend(block_inputs[..temp.len()].iter().cloned());

                // Cast all the types we have on the stack to the actual
                // types needed for this variant, if necessary.
                casts.truncate(0);
                for (actual, expected) in temp.iter().zip(&params[1..]) {
                    casts.push(cast(*expected, *actual));
                }
                if casts.iter().any(|c| *c != Bitcast::None) {
                    self.emit(&Instruction::Bitcasts { casts: &casts })?;
                }

                // Then recursively lift this variant's payload.
                self.lift(ty)?;
                true
            } else {
                false
            };

            Ok((discriminant, has_value))
        } else {
            unreachable!()
        }
    }

    /// Writes a value to memory.
    fn write_to_memory(&mut self, ty: &Type, addr: B::Operand, offset: i32) -> Result<()> {
        use Instruction::*;

        match *ty {
            // Builtin types need different flavors of storage instructions
            // depending on the size of the value written.
            Type::Bool | Type::U8 | Type::S8 => {
                self.lower_and_emit(ty, addr, &I32Store8 { offset })
            },
            Type::U16 | Type::S16 => self.lower_and_emit(ty, addr, &I32Store16 { offset }),
            Type::U32 | Type::S32 | Type::Char => {
                self.lower_and_emit(ty, addr, &I32Store { offset })
            },
            Type::U64 | Type::S64 => self.lower_and_emit(ty, addr, &I64Store { offset }),
            Type::Float32 => self.lower_and_emit(ty, addr, &F32Store { offset }),
            Type::Float64 => self.lower_and_emit(ty, addr, &F64Store { offset }),
            Type::String => self.write_list_to_memory(ty, addr, offset),

            Type::Id(id) => match &self.resolve.types[id].kind {
                TypeDefKind::Type(t) => self.write_to_memory(t, addr, offset),
                TypeDefKind::List(_) => self.write_list_to_memory(ty, addr, offset),

                TypeDefKind::Handle(_) => self.lower_and_emit(ty, addr, &I32Store { offset }),

                // Decompose the record into its components and then write all
                // the components into memory one-by-one.
                TypeDefKind::Record(record) => {
                    self.emit(&RecordLower { record, ty: id })?;
                    self.write_fields_to_memory(record.fields.iter().map(|f| &f.ty), addr, offset)
                },
                TypeDefKind::Resource => {
                    todo!()
                },
                TypeDefKind::Tuple(tuple) => {
                    self.emit(&TupleLower { tuple, ty: id })?;
                    self.write_fields_to_memory(tuple.types.iter(), addr, offset)
                },

                TypeDefKind::Flags(f) => {
                    self.lower(ty)?;
                    match f.repr() {
                        FlagsRepr::U8 => {
                            self.stack.push(addr);
                            self.store_intrepr(offset, Int::U8)?;
                        },
                        FlagsRepr::U16 => {
                            self.stack.push(addr);
                            self.store_intrepr(offset, Int::U16)?;
                        },
                        FlagsRepr::U32(n) => {
                            for i in (0..n).rev() {
                                self.stack.push(addr.clone());
                                self.emit(&I32Store {
                                    offset: offset + (i as i32) * 4,
                                })?;
                            }
                        },
                    }

                    Ok(())
                },

                // Each case will get its own block, and the first item in each
                // case is writing the discriminant. After that if we have a
                // payload we write the payload after the discriminant, aligned up
                // to the type's alignment.
                TypeDefKind::Variant(v) => self.write_variant_arm_to_memory(
                    offset,
                    addr,
                    v.tag(),
                    v.cases.iter().map(|c| c.ty.as_ref()),
                ),

                TypeDefKind::Option(t) => {
                    self.write_variant_arm_to_memory(offset, addr, Int::U8, [None, Some(t)])
                },

                TypeDefKind::Result(r) => self.write_variant_arm_to_memory(
                    offset,
                    addr,
                    Int::U8,
                    [r.ok.as_ref(), r.err.as_ref()],
                ),

                TypeDefKind::Enum(e) => {
                    self.lower(ty)?;
                    self.stack.push(addr);
                    self.store_intrepr(offset, e.tag())
                },

                TypeDefKind::Future(_) => todo!("write future to memory"),
                TypeDefKind::Stream(_) => todo!("write stream to memory"),
                TypeDefKind::Unknown => unreachable!(),
            },
        }
    }

    /// Writes parameters to memory.
    fn write_params_to_memory<'b>(
        &mut self,
        params: impl IntoIterator<Item = &'b Type> + ExactSizeIterator,
        addr: B::Operand,
        offset: i32,
    ) -> Result<()> {
        self.write_fields_to_memory(params, addr, offset)
    }

    /// Writes a variant arm to memory.
    fn write_variant_arm_to_memory<'b>(
        &mut self,
        offset: i32,
        addr: B::Operand,
        tag: Int,
        cases: impl IntoIterator<Item = Option<&'b Type>> + Clone,
    ) -> Result<()> {
        let disc_val = Instruction::ExtractVariantDiscriminant {
            discriminant_value: Cell::default(),
        };
        self.emit(&disc_val)?;

        let discriminant =
            if let Instruction::ExtractVariantDiscriminant { discriminant_value } = disc_val {
                discriminant_value.get().0
            } else {
                unreachable!()
            };

        let payload_offset =
            offset + (self.bindgen.sizes().payload_offset(tag, cases.clone()) as i32);

        let payload_name = self.stack.pop().unwrap();
        self.emit(&Instruction::I32Const { val: discriminant })?;
        self.stack.push(addr.clone());
        self.store_intrepr(offset, tag)?;
        if let Some(ty) = cases
            .into_iter()
            .nth(discriminant as usize)
            .ok_or_else(|| Error::msg("Invalid discriminator value."))?
        {
            self.stack.push(payload_name);
            self.write_to_memory(ty, addr, payload_offset)?;
        }

        Ok(())
    }

    /// Writes a list to memory.
    fn write_list_to_memory(&mut self, ty: &Type, addr: B::Operand, offset: i32) -> Result<()> {
        // After lowering the list there's two i32 values on the stack
        // which we write into memory, writing the pointer into the low address
        // and the length into the high address.
        self.lower(ty)?;
        self.stack.push(addr.clone());
        self.emit(&Instruction::I32Store { offset: offset + 4 })?;
        self.stack.push(addr);
        self.emit(&Instruction::I32Store { offset })
    }

    /// Writes fields to memory.
    fn write_fields_to_memory<'b>(
        &mut self,
        tys: impl IntoIterator<Item = &'b Type> + ExactSizeIterator,
        addr: B::Operand,
        offset: i32,
    ) -> Result<()> {
        let fields = self
            .stack
            .drain(self.stack.len() - tys.len()..)
            .collect::<Vec<_>>();
        for ((field_offset, ty), op) in self
            .bindgen
            .sizes()
            .field_offsets(tys)
            .into_iter()
            .zip(fields)
        {
            self.stack.push(op);
            self.write_to_memory(ty, addr.clone(), offset + (field_offset as i32))?;
        }
        Ok(())
    }

    /// Lowers a type and emits an instruction.
    fn lower_and_emit(&mut self, ty: &Type, addr: B::Operand, instr: &Instruction) -> Result<()> {
        self.lower(ty)?;
        self.stack.push(addr);
        self.emit(instr)
    }

    /// Reads a value from memory.
    fn read_from_memory(&mut self, ty: &Type, addr: B::Operand, offset: i32) -> Result<()> {
        use Instruction::*;

        match *ty {
            Type::Bool => self.emit_and_lift(ty, addr, &I32Load8U { offset }),
            Type::U8 => self.emit_and_lift(ty, addr, &I32Load8U { offset }),
            Type::S8 => self.emit_and_lift(ty, addr, &I32Load8S { offset }),
            Type::U16 => self.emit_and_lift(ty, addr, &I32Load16U { offset }),
            Type::S16 => self.emit_and_lift(ty, addr, &I32Load16S { offset }),
            Type::U32 | Type::S32 | Type::Char => self.emit_and_lift(ty, addr, &I32Load { offset }),
            Type::U64 | Type::S64 => self.emit_and_lift(ty, addr, &I64Load { offset }),
            Type::Float32 => self.emit_and_lift(ty, addr, &F32Load { offset }),
            Type::Float64 => self.emit_and_lift(ty, addr, &F64Load { offset }),
            Type::String => self.read_list_from_memory(ty, addr, offset),

            Type::Id(id) => match &self.resolve.types[id].kind {
                TypeDefKind::Type(t) => self.read_from_memory(t, addr, offset),

                TypeDefKind::List(_) => self.read_list_from_memory(ty, addr, offset),

                TypeDefKind::Handle(_) => self.emit_and_lift(ty, addr, &I32Load { offset }),

                TypeDefKind::Resource => {
                    todo!();
                },

                // Read and lift each field individually, adjusting the offset
                // as we go along, then aggregate all the fields into the
                // record.
                TypeDefKind::Record(record) => {
                    self.read_fields_from_memory(
                        record.fields.iter().map(|f| &f.ty),
                        addr,
                        offset,
                    )?;
                    self.emit(&RecordLift { record, ty: id })
                },

                TypeDefKind::Tuple(tuple) => {
                    self.read_fields_from_memory(&tuple.types, addr, offset)?;
                    self.emit(&TupleLift { tuple, ty: id })
                },

                TypeDefKind::Flags(f) => {
                    match f.repr() {
                        FlagsRepr::U8 => {
                            self.stack.push(addr);
                            self.load_intrepr(offset, Int::U8)?;
                        },
                        FlagsRepr::U16 => {
                            self.stack.push(addr);
                            self.load_intrepr(offset, Int::U16)?;
                        },
                        FlagsRepr::U32(n) => {
                            for i in 0..n {
                                self.stack.push(addr.clone());
                                self.emit(&I32Load {
                                    offset: offset + (i as i32) * 4,
                                })?;
                            }
                        },
                    }
                    self.lift(ty)
                },

                // Each case will get its own block, and we'll dispatch to the
                // right block based on the `i32.load` we initially perform. Each
                // individual block is pretty simple and just reads the payload type
                // from the corresponding offset if one is available.
                TypeDefKind::Variant(variant) => {
                    let (discriminant, has_value) = self.read_variant_arm_from_memory(
                        offset,
                        addr,
                        variant.tag(),
                        variant.cases.iter().map(|c| c.ty.as_ref()),
                    )?;
                    self.emit(&VariantLift {
                        variant,
                        ty: id,
                        discriminant,
                        has_value,
                    })
                },

                TypeDefKind::Option(t) => {
                    let (discriminant, has_value) =
                        self.read_variant_arm_from_memory(offset, addr, Int::U8, [None, Some(t)])?;
                    self.emit(&OptionLift {
                        payload: t,
                        ty: id,
                        discriminant,
                        has_value,
                    })
                },

                TypeDefKind::Result(r) => {
                    let (discriminant, has_value) = self.read_variant_arm_from_memory(
                        offset,
                        addr,
                        Int::U8,
                        [r.ok.as_ref(), r.err.as_ref()],
                    )?;
                    self.emit(&ResultLift {
                        result: r,
                        discriminant,
                        has_value,
                        ty: id,
                    })
                },

                TypeDefKind::Enum(e) => {
                    self.stack.push(addr);
                    self.load_intrepr(offset, e.tag())?;
                    self.lift(ty)
                },

                TypeDefKind::Future(_) => todo!("read future from memory"),
                TypeDefKind::Stream(_) => todo!("read stream from memory"),
                TypeDefKind::Unknown => unreachable!(),
            },
        }
    }

    /// Reads results from memory.
    fn read_results_from_memory(
        &mut self,
        results: &Results,
        addr: B::Operand,
        offset: i32,
    ) -> Result<()> {
        self.read_fields_from_memory(results.iter_types(), addr, offset)
    }

    /// Reads a variant arm from memory.
    fn read_variant_arm_from_memory<'b>(
        &mut self,
        offset: i32,
        addr: B::Operand,
        tag: Int,
        cases: impl IntoIterator<Item = Option<&'b Type>> + Clone,
    ) -> Result<(i32, bool)> {
        self.stack.push(addr.clone());
        self.load_intrepr(offset, tag)?;
        let variant = Instruction::ReadI32 {
            value: Cell::default(),
        };
        self.emit(&variant)?;
        let payload_offset =
            offset + (self.bindgen.sizes().payload_offset(tag, cases.clone()) as i32);

        if let Instruction::ReadI32 { value } = variant {
            let disc = value.get();
            let has_value = if let Some(ty) = cases
                .into_iter()
                .nth(disc as usize)
                .ok_or_else(|| Error::msg("Invalid discriminant value."))?
            {
                self.read_from_memory(ty, addr, payload_offset)?;
                true
            } else {
                false
            };

            Ok((disc, has_value))
        } else {
            unreachable!()
        }
    }

    /// Reads a list from memory.
    fn read_list_from_memory(&mut self, ty: &Type, addr: B::Operand, offset: i32) -> Result<()> {
        // Read the pointer/len and then perform the standard lifting
        // proceses.
        self.stack.push(addr.clone());
        self.emit(&Instruction::I32Load { offset })?;
        self.stack.push(addr);
        self.emit(&Instruction::I32Load { offset: offset + 4 })?;
        self.lift(ty)
    }

    /// Reads the fields of a list from memory.
    fn read_fields_from_memory<'b>(
        &mut self,
        tys: impl IntoIterator<Item = &'b Type>,
        addr: B::Operand,
        offset: i32,
    ) -> Result<()> {
        for (field_offset, ty) in self.bindgen.sizes().field_offsets(tys).iter() {
            self.read_from_memory(ty, addr.clone(), offset + (*field_offset as i32))?;
        }
        Ok(())
    }

    /// Emits and lifts a variant.
    fn emit_and_lift(&mut self, ty: &Type, addr: B::Operand, instr: &Instruction) -> Result<()> {
        self.stack.push(addr);
        self.emit(instr)?;
        self.lift(ty)
    }

    /// Loads a representation.
    fn load_intrepr(&mut self, offset: i32, repr: Int) -> Result<()> {
        self.emit(&match repr {
            Int::U64 => Instruction::I64Load { offset },
            Int::U32 => Instruction::I32Load { offset },
            Int::U16 => Instruction::I32Load16U { offset },
            Int::U8 => Instruction::I32Load8U { offset },
        })
    }

    /// Stores a representation.
    fn store_intrepr(&mut self, offset: i32, repr: Int) -> Result<()> {
        self.emit(&match repr {
            Int::U64 => Instruction::I64Store { offset },
            Int::U32 => Instruction::I32Store { offset },
            Int::U16 => Instruction::I32Store16 { offset },
            Int::U8 => Instruction::I32Store8 { offset },
        })
    }
}

/// Generates a bitcast between two WASM types.
fn cast(from: WasmType, to: WasmType) -> Bitcast {
    use WasmType::*;

    match (from, to) {
        (I32, I32) | (I64, I64) | (F32, F32) | (F64, F64) => Bitcast::None,

        (I32, I64) => Bitcast::I32ToI64,
        (F32, I32) => Bitcast::F32ToI32,
        (F64, I64) => Bitcast::F64ToI64,

        (I64, I32) => Bitcast::I64ToI32,
        (I32, F32) => Bitcast::I32ToF32,
        (I64, F64) => Bitcast::I64ToF64,

        (F32, I64) => Bitcast::F32ToI64,
        (I64, F32) => Bitcast::I64ToF32,

        (F32, F64) | (F64, F32) | (F64, I32) | (I32, F64) => unreachable!(),
    }
}

/// Pushes the WASM types used to describe the given type into the result vector.
fn push_wasm(resolve: &Resolve, variant: AbiVariant, ty: &Type, result: &mut Vec<WasmType>) {
    match ty {
        Type::Bool
        | Type::S8
        | Type::U8
        | Type::S16
        | Type::U16
        | Type::S32
        | Type::U32
        | Type::Char => result.push(WasmType::I32),

        Type::U64 | Type::S64 => result.push(WasmType::I64),
        Type::Float32 => result.push(WasmType::F32),
        Type::Float64 => result.push(WasmType::F64),
        Type::String => {
            result.push(WasmType::I32);
            result.push(WasmType::I32);
        },

        Type::Id(id) => match &resolve.types[*id].kind {
            TypeDefKind::Type(t) => push_wasm(resolve, variant, t, result),

            TypeDefKind::Handle(Handle::Own(_) | Handle::Borrow(_)) => {
                result.push(WasmType::I32);
            },

            TypeDefKind::Resource => todo!(),

            TypeDefKind::Record(r) => {
                for field in r.fields.iter() {
                    push_wasm(resolve, variant, &field.ty, result);
                }
            },

            TypeDefKind::Tuple(t) => {
                for ty in t.types.iter() {
                    push_wasm(resolve, variant, ty, result);
                }
            },

            TypeDefKind::Flags(r) => {
                for _ in 0..r.repr().count() {
                    result.push(WasmType::I32);
                }
            },

            TypeDefKind::List(_) => {
                result.push(WasmType::I32);
                result.push(WasmType::I32);
            },

            TypeDefKind::Variant(v) => {
                result.push(v.tag().into());
                push_wasm_variants(
                    resolve,
                    variant,
                    v.cases.iter().map(|c| c.ty.as_ref()),
                    result,
                );
            },

            TypeDefKind::Enum(e) => result.push(e.tag().into()),

            TypeDefKind::Option(t) => {
                result.push(WasmType::I32);
                push_wasm_variants(resolve, variant, [None, Some(t)], result);
            },

            TypeDefKind::Result(r) => {
                result.push(WasmType::I32);
                push_wasm_variants(resolve, variant, [r.ok.as_ref(), r.err.as_ref()], result);
            },

            TypeDefKind::Future(_) => {
                result.push(WasmType::I32);
            },

            TypeDefKind::Stream(_) => {
                result.push(WasmType::I32);
            },

            TypeDefKind::Unknown => unreachable!(),
        },
    }
}

/// Pushes the WASM types used to represent variants into the given vector.
fn push_wasm_variants<'a>(
    resolve: &Resolve,
    variant: AbiVariant,
    tys: impl IntoIterator<Item = Option<&'a Type>>,
    result: &mut Vec<WasmType>,
) {
    let mut temp = Vec::new();
    let start = result.len();

    // Push each case's type onto a temporary vector, and then
    // merge that vector into our final list starting at
    // `start`. Note that this requires some degree of
    // "unification" so we can handle things like `Result<i32,
    // f32>` where that turns into `[i32 i32]` where the second
    // `i32` might be the `f32` bitcasted.
    for ty in tys.into_iter().flatten() {
        push_wasm(resolve, variant, ty, &mut temp);

        for (i, ty) in temp.drain(..).enumerate() {
            match result.get_mut(start + i) {
                Some(prev) => *prev = join(*prev, ty),
                None => result.push(ty),
            }
        }
    }
}
