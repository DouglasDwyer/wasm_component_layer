(component
  (type (;0;)
    (instance
      (type (;0;) (func (param "message" string)))
      (export (;0;) "log" (func (type 0)))
      (type (;1;) (option string))
      (type (;2;) (func (param "is-some" bool) (result 1)))
      (export (;1;) "result-option" (func (type 2)))
      (type (;3;) (result string (error string)))
      (type (;4;) (func (param "is-ok" bool) (result 3)))
      (export (;2;) "result-result" (func (type 4)))
      (type (;5;) (result string))
      (type (;6;) (func (param "is-ok" bool) (result 5)))
      (export (;3;) "result-result-ok" (func (type 6)))
      (type (;7;) (result (error string)))
      (type (;8;) (func (param "is-ok" bool) (result 7)))
      (export (;4;) "result-result-err" (func (type 8)))
      (type (;9;) (result))
      (type (;10;) (func (param "is-ok" bool) (result 9)))
      (export (;5;) "result-result-none" (func (type 10)))
    )
  )
  (import "test:guest/host" (instance (;0;) (type 0)))
  (core module (;0;)
    (type (;0;) (func (param i32 i32)))
    (type (;1;) (func (param i32 i32 i32) (result i32)))
    (type (;2;) (func (param i32 i32) (result i32)))
    (type (;3;) (func (param i32) (result i32)))
    (type (;4;) (func))
    (type (;5;) (func (param i32 i32 i32)))
    (type (;6;) (func (param i32 i32 i32 i32)))
    (type (;7;) (func (param i32 i32 i32 i32) (result i32)))
    (type (;8;) (func (param i32 i32 i32 i32 i32)))
    (type (;9;) (func (param i32)))
    (type (;10;) (func (result i32)))
    (type (;11;) (func (param i32 i32 i32 i32 i32) (result i32)))
    (type (;12;) (func (param i32 i32 i32 i32 i32 i32 i32)))
    (type (;13;) (func (param i32 i32 i32 i32 i32 i32) (result i32)))
    (type (;14;) (func (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)))
    (type (;15;) (func (param i32 i32 i32 i32 i32 i32 i32) (result i32)))
    (type (;16;) (func (param i64 i32 i32) (result i32)))
    (import "test:guest/host" "log" (func $_ZN17component_example4test5guest4host3log10wit_import17hd588ebb1b530ec2aE (;0;) (type 0)))
    (import "test:guest/host" "result-option" (func $_ZN17component_example4test5guest4host13result_option10wit_import17ha1bb84f1c0c19d52E (;1;) (type 0)))
    (import "test:guest/host" "result-result" (func $_ZN17component_example4test5guest4host13result_result10wit_import17ha859731761289ad4E (;2;) (type 0)))
    (import "test:guest/host" "result-result-ok" (func $_ZN17component_example4test5guest4host16result_result_ok10wit_import17h9b1549c51fb040fbE (;3;) (type 0)))
    (import "test:guest/host" "result-result-err" (func $_ZN17component_example4test5guest4host17result_result_err10wit_import17hebc19d5382075962E (;4;) (type 0)))
    (import "test:guest/host" "result-result-none" (func $_ZN17component_example4test5guest4host18result_result_none10wit_import17hd0c5bc84def681f4E (;5;) (type 3)))
    (func $__wasm_call_ctors (;6;) (type 4))
    (func $_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h4c1b6f1d0f27a9faE (;7;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 16
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=8
      local.get 4
      local.get 1
      i32.store offset=12
      local.get 0
      i32.load
      local.set 5
      local.get 5
      local.get 1
      call $_ZN45_$LT$$LP$$RP$$u20$as$u20$core..fmt..Debug$GT$3fmt17hbd1a05eae5c2387aE
      local.set 6
      i32.const 1
      local.set 7
      local.get 6
      local.get 7
      i32.and
      local.set 8
      i32.const 16
      local.set 9
      local.get 4
      local.get 9
      i32.add
      local.set 10
      local.get 10
      global.set $__stack_pointer
      local.get 8
      return
    )
    (func $_ZN4core3fmt9Arguments6new_v117h39b074a36c3a198bE (;8;) (type 5) (param i32 i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 3
      i32.const 16
      local.set 4
      local.get 3
      local.get 4
      i32.sub
      local.set 5
      local.get 5
      local.get 1
      i32.store offset=8
      local.get 5
      local.get 2
      i32.store offset=12
      local.get 0
      local.get 1
      i32.store
      i32.const 1
      local.set 6
      local.get 0
      local.get 6
      i32.store offset=4
      i32.const 0
      local.set 7
      local.get 7
      i32.load offset=1048576
      local.set 8
      i32.const 0
      local.set 9
      local.get 9
      i32.load offset=1048580
      local.set 10
      local.get 0
      local.get 8
      i32.store offset=16
      local.get 0
      local.get 10
      i32.store offset=20
      local.get 0
      local.get 2
      i32.store offset=8
      i32.const 1
      local.set 11
      local.get 0
      local.get 11
      i32.store offset=12
      return
    )
    (func $_ZN52_$LT$T$u20$as$u20$alloc..slice..hack..ConvertVec$GT$6to_vec17h7392a30f840e978bE (;9;) (type 5) (param i32 i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 3
      i32.const 80
      local.set 4
      local.get 3
      local.get 4
      i32.sub
      local.set 5
      local.get 5
      global.set $__stack_pointer
      local.get 5
      local.get 1
      i32.store offset=24
      local.get 5
      local.get 2
      i32.store offset=28
      local.get 5
      local.get 2
      i32.store offset=36
      i32.const 12
      local.set 6
      local.get 5
      local.get 6
      i32.add
      local.set 7
      local.get 7
      local.set 8
      i32.const 0
      local.set 9
      i32.const 1
      local.set 10
      local.get 9
      local.get 10
      i32.and
      local.set 11
      local.get 8
      local.get 2
      local.get 11
      call $_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$15try_allocate_in17ha3ac89577fa5bf59E
      local.get 5
      i32.load offset=12
      local.set 12
      block ;; label = @1
        block ;; label = @2
          local.get 12
          br_if 0 (;@2;)
          local.get 5
          i32.load offset=16
          local.set 13
          local.get 5
          i32.load offset=20
          local.set 14
          local.get 5
          local.get 13
          i32.store offset=40
          local.get 5
          local.get 14
          i32.store offset=44
          local.get 5
          local.get 13
          i32.store
          local.get 5
          local.get 14
          i32.store offset=4
          i32.const 0
          local.set 15
          local.get 5
          local.get 15
          i32.store offset=8
          local.get 5
          local.get 1
          i32.store offset=48
          local.get 5
          local.set 16
          local.get 5
          local.get 16
          i32.store offset=52
          local.get 5
          local.set 17
          local.get 5
          local.get 17
          i32.store offset=56
          local.get 5
          i32.load offset=4
          local.set 18
          local.get 5
          local.get 18
          i32.store offset=60
          local.get 5
          local.get 18
          i32.store offset=64
          br 1 (;@1;)
        end
        local.get 5
        i32.load offset=16
        local.set 19
        local.get 5
        i32.load offset=20
        local.set 20
        local.get 5
        local.get 19
        i32.store offset=72
        local.get 5
        local.get 20
        i32.store offset=76
        local.get 19
        local.get 20
        call $_ZN5alloc7raw_vec12handle_error17h5204850add322ec3E
        unreachable
      end
      i32.const 1
      local.set 21
      local.get 1
      local.get 18
      local.get 21
      local.get 21
      local.get 2
      call $_ZN4core10intrinsics19copy_nonoverlapping18precondition_check17h7a208a03a2265411E
      i32.const 0
      local.set 22
      local.get 2
      local.get 22
      i32.shl
      local.set 23
      local.get 18
      local.get 1
      local.get 23
      call $memcpy
      drop
      local.get 5
      local.set 24
      local.get 5
      local.get 24
      i32.store offset=68
      local.get 5
      local.get 2
      i32.store offset=8
      local.get 5
      i64.load align=4
      local.set 25
      local.get 0
      local.get 25
      i64.store align=4
      i32.const 8
      local.set 26
      local.get 0
      local.get 26
      i32.add
      local.set 27
      local.get 5
      local.get 26
      i32.add
      local.set 28
      local.get 28
      i32.load
      local.set 29
      local.get 27
      local.get 29
      i32.store
      i32.const 80
      local.set 30
      local.get 5
      local.get 30
      i32.add
      local.set 31
      local.get 31
      global.set $__stack_pointer
      return
    )
    (func $_ZN4core5slice3raw14from_raw_parts18precondition_check17h6e20dac50feedaf6E (;10;) (type 6) (param i32 i32 i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 4
      i32.const 64
      local.set 5
      local.get 4
      local.get 5
      i32.sub
      local.set 6
      local.get 6
      global.set $__stack_pointer
      i32.const 1048628
      local.set 7
      local.get 6
      local.get 7
      i32.store offset=4
      local.get 6
      local.get 0
      i32.store offset=36
      local.get 6
      local.get 1
      i32.store offset=40
      local.get 6
      local.get 2
      i32.store offset=44
      local.get 6
      local.get 3
      i32.store offset=48
      local.get 6
      local.get 0
      i32.store offset=52
      local.get 6
      local.get 0
      i32.store offset=56
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            local.get 0
            br_if 0 (;@3;)
            br 1 (;@2;)
          end
          local.get 2
          i32.popcnt
          local.set 8
          local.get 6
          local.get 8
          i32.store offset=60
          local.get 6
          i32.load offset=60
          local.set 9
          i32.const 1
          local.set 10
          local.get 9
          local.set 11
          local.get 10
          local.set 12
          local.get 11
          local.get 12
          i32.eq
          local.set 13
          i32.const 1
          local.set 14
          local.get 13
          local.get 14
          i32.and
          local.set 15
          block ;; label = @3
            block ;; label = @4
              local.get 15
              i32.eqz
              br_if 0 (;@4;)
              i32.const 1
              local.set 16
              local.get 2
              local.get 16
              i32.sub
              local.set 17
              local.get 0
              local.get 17
              i32.and
              local.set 18
              local.get 18
              i32.eqz
              br_if 1 (;@3;)
              br 2 (;@2;)
            end
            i32.const 1048628
            local.set 19
            local.get 6
            local.get 19
            i32.store offset=8
            i32.const 1
            local.set 20
            local.get 6
            local.get 20
            i32.store offset=12
            i32.const 0
            local.set 21
            local.get 21
            i32.load offset=1048896
            local.set 22
            i32.const 0
            local.set 23
            local.get 23
            i32.load offset=1048900
            local.set 24
            local.get 6
            local.get 22
            i32.store offset=24
            local.get 6
            local.get 24
            i32.store offset=28
            i32.const 4
            local.set 25
            local.get 6
            local.get 25
            i32.store offset=16
            i32.const 0
            local.set 26
            local.get 6
            local.get 26
            i32.store offset=20
            i32.const 8
            local.set 27
            local.get 6
            local.get 27
            i32.add
            local.set 28
            local.get 28
            local.set 29
            i32.const 1048988
            local.set 30
            local.get 29
            local.get 30
            call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
            unreachable
          end
          i32.const 0
          local.set 31
          local.get 1
          local.set 32
          local.get 31
          local.set 33
          local.get 32
          local.get 33
          i32.eq
          local.set 34
          block ;; label = @3
            block ;; label = @4
              local.get 1
              br_if 0 (;@4;)
              i32.const -1
              local.set 35
              local.get 6
              local.get 35
              i32.store offset=32
              br 1 (;@3;)
            end
            i32.const 1
            local.set 36
            local.get 34
            local.get 36
            i32.and
            local.set 37
            block ;; label = @4
              local.get 37
              br_if 0 (;@4;)
              i32.const 2147483647
              local.set 38
              local.get 38
              local.get 1
              i32.div_u
              local.set 39
              local.get 6
              local.get 39
              i32.store offset=32
              br 1 (;@3;)
            end
            i32.const 1048716
            local.set 40
            local.get 40
            call $_ZN4core9panicking11panic_const23panic_const_div_by_zero17hed37a86622bbbb5bE
            unreachable
          end
          local.get 6
          i32.load offset=32
          local.set 41
          local.get 3
          local.set 42
          local.get 41
          local.set 43
          local.get 42
          local.get 43
          i32.le_u
          local.set 44
          i32.const 1
          local.set 45
          local.get 44
          local.get 45
          i32.and
          local.set 46
          block ;; label = @3
            local.get 46
            br_if 0 (;@3;)
            br 2 (;@1;)
          end
          i32.const 64
          local.set 47
          local.get 6
          local.get 47
          i32.add
          local.set 48
          local.get 48
          global.set $__stack_pointer
          return
        end
      end
      i32.const 1048732
      local.set 49
      i32.const 162
      local.set 50
      local.get 49
      local.get 50
      call $_ZN4core9panicking14panic_nounwind17he2774a18f33c590cE
      unreachable
    )
    (func $_ZN4core9ub_checks17is_nonoverlapping7runtime17h4cf1e21c86cbfe41E (;11;) (type 7) (param i32 i32 i32 i32) (result i32)
      (local i32 i32 i32 i64 i64 i64 i64 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 4
      i32.const 64
      local.set 5
      local.get 4
      local.get 5
      i32.sub
      local.set 6
      local.get 6
      global.set $__stack_pointer
      local.get 6
      local.get 0
      i32.store offset=20
      local.get 6
      local.get 1
      i32.store offset=24
      local.get 6
      local.get 2
      i32.store offset=28
      local.get 6
      local.get 3
      i32.store offset=32
      local.get 6
      local.get 0
      i32.store offset=36
      local.get 6
      local.get 1
      i32.store offset=40
      local.get 3
      i64.extend_i32_u
      local.set 7
      local.get 2
      i64.extend_i32_u
      local.set 8
      local.get 8
      local.get 7
      i64.mul
      local.set 9
      i64.const 32
      local.set 10
      local.get 9
      local.get 10
      i64.shr_u
      local.set 11
      local.get 11
      i32.wrap_i64
      local.set 12
      i32.const 0
      local.set 13
      local.get 12
      local.get 13
      i32.ne
      local.set 14
      local.get 9
      i32.wrap_i64
      local.set 15
      local.get 6
      local.get 15
      i32.store offset=44
      i32.const 1
      local.set 16
      local.get 14
      local.get 16
      i32.and
      local.set 17
      local.get 6
      local.get 17
      i32.store8 offset=51
      local.get 6
      local.get 15
      i32.store offset=52
      i32.const 1
      local.set 18
      local.get 14
      local.get 18
      i32.and
      local.set 19
      local.get 6
      local.get 19
      i32.store8 offset=59
      local.get 6
      i32.load8_u offset=59
      local.set 20
      i32.const 1
      local.set 21
      local.get 20
      local.get 21
      i32.and
      local.set 22
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              local.get 22
              br_if 0 (;@4;)
              local.get 6
              local.get 15
              i32.store offset=12
              i32.const 1
              local.set 23
              local.get 6
              local.get 23
              i32.store offset=8
              local.get 6
              i32.load offset=12
              local.set 24
              local.get 6
              local.get 24
              i32.store offset=60
              local.get 0
              local.set 25
              local.get 1
              local.set 26
              local.get 25
              local.get 26
              i32.lt_u
              local.set 27
              i32.const 1
              local.set 28
              local.get 27
              local.get 28
              i32.and
              local.set 29
              local.get 29
              br_if 2 (;@2;)
              br 1 (;@3;)
            end
            i32.const 1049004
            local.set 30
            i32.const 61
            local.set 31
            local.get 30
            local.get 31
            call $_ZN4core9panicking14panic_nounwind17he2774a18f33c590cE
            unreachable
          end
          local.get 0
          local.get 1
          i32.sub
          local.set 32
          local.get 6
          local.get 32
          i32.store offset=16
          br 1 (;@1;)
        end
        local.get 1
        local.get 0
        i32.sub
        local.set 33
        local.get 6
        local.get 33
        i32.store offset=16
      end
      local.get 6
      i32.load offset=16
      local.set 34
      local.get 34
      local.set 35
      local.get 24
      local.set 36
      local.get 35
      local.get 36
      i32.ge_u
      local.set 37
      i32.const 1
      local.set 38
      local.get 37
      local.get 38
      i32.and
      local.set 39
      i32.const 64
      local.set 40
      local.get 6
      local.get 40
      i32.add
      local.set 41
      local.get 41
      global.set $__stack_pointer
      local.get 39
      return
    )
    (func $_ZN5alloc3fmt6format17h8c05e2c2d7c5fc16E (;12;) (type 0) (param i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i64 i32 i32 i32 i64 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 64
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 1
      i32.store offset=20
      local.get 1
      i32.load
      local.set 5
      local.get 1
      i32.load offset=4
      local.set 6
      local.get 1
      i32.load offset=12
      local.set 7
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  local.get 6
                  br_if 0 (;@6;)
                  local.get 7
                  i32.eqz
                  br_if 1 (;@5;)
                  br 4 (;@2;)
                end
                i32.const 1
                local.set 8
                local.get 6
                local.set 9
                local.get 8
                local.set 10
                local.get 9
                local.get 10
                i32.eq
                local.set 11
                i32.const 1
                local.set 12
                local.get 11
                local.get 12
                i32.and
                local.set 13
                local.get 13
                br_if 1 (;@4;)
                br 3 (;@2;)
              end
              i32.const 1
              local.set 14
              local.get 4
              local.get 14
              i32.store offset=4
              i32.const 0
              local.set 15
              local.get 4
              local.get 15
              i32.store offset=8
              local.get 4
              local.get 1
              i32.store offset=12
              local.get 4
              i32.load offset=12
              local.set 16
              local.get 4
              local.get 16
              i32.store offset=16
              br 1 (;@3;)
            end
            local.get 7
            br_if 1 (;@2;)
            local.get 4
            local.get 5
            i32.store offset=24
            local.get 5
            i32.load
            local.set 17
            local.get 5
            i32.load offset=4
            local.set 18
            local.get 4
            local.get 17
            i32.store offset=4
            local.get 4
            local.get 18
            i32.store offset=8
            local.get 4
            local.get 1
            i32.store offset=12
            local.get 4
            i32.load offset=12
            local.set 19
            local.get 4
            local.get 19
            i32.store offset=16
          end
          local.get 4
          i32.load offset=4
          local.set 20
          local.get 4
          i32.load offset=8
          local.set 21
          local.get 4
          local.get 20
          i32.store offset=28
          local.get 4
          local.get 21
          i32.store offset=32
          local.get 0
          local.get 20
          local.get 21
          call $_ZN4core3ops8function6FnOnce9call_once17h0e9ba0ef9fe25543E
          br 1 (;@1;)
        end
        i32.const 0
        local.set 22
        local.get 22
        i32.load offset=1049068
        local.set 23
        i32.const 0
        local.set 24
        local.get 24
        i32.load offset=1049072
        local.set 25
        local.get 4
        local.get 23
        i32.store offset=4
        local.get 4
        local.get 25
        i32.store offset=8
        local.get 4
        local.get 1
        i32.store offset=12
        local.get 4
        i32.load offset=12
        local.set 26
        local.get 4
        local.get 26
        i32.store offset=16
        local.get 4
        i32.load offset=12
        local.set 27
        local.get 4
        local.get 27
        i32.store offset=36
        i32.const 16
        local.set 28
        local.get 1
        local.get 28
        i32.add
        local.set 29
        local.get 29
        i64.load align=4
        local.set 30
        i32.const 40
        local.set 31
        local.get 4
        local.get 31
        i32.add
        local.set 32
        local.get 32
        local.get 28
        i32.add
        local.set 33
        local.get 33
        local.get 30
        i64.store
        i32.const 8
        local.set 34
        local.get 1
        local.get 34
        i32.add
        local.set 35
        local.get 35
        i64.load align=4
        local.set 36
        i32.const 40
        local.set 37
        local.get 4
        local.get 37
        i32.add
        local.set 38
        local.get 38
        local.get 34
        i32.add
        local.set 39
        local.get 39
        local.get 36
        i64.store
        local.get 1
        i64.load align=4
        local.set 40
        local.get 4
        local.get 40
        i64.store offset=40
        i32.const 40
        local.set 41
        local.get 4
        local.get 41
        i32.add
        local.set 42
        local.get 42
        local.set 43
        local.get 0
        local.get 43
        call $_ZN5alloc3fmt6format12format_inner17hddf591775ec9b0ebE
      end
      i32.const 64
      local.set 44
      local.get 4
      local.get 44
      i32.add
      local.set 45
      local.get 45
      global.set $__stack_pointer
      return
    )
    (func $_ZN5alloc3str56_$LT$impl$u20$alloc..borrow..ToOwned$u20$for$u20$str$GT$8to_owned17ha4066b8bbafecc01E (;13;) (type 5) (param i32 i32 i32)
      (local i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 3
      i32.const 32
      local.set 4
      local.get 3
      local.get 4
      i32.sub
      local.set 5
      local.get 5
      global.set $__stack_pointer
      local.get 5
      local.get 1
      i32.store offset=16
      local.get 5
      local.get 2
      i32.store offset=20
      local.get 5
      local.get 1
      i32.store offset=24
      local.get 5
      local.get 2
      i32.store offset=28
      i32.const 4
      local.set 6
      local.get 5
      local.get 6
      i32.add
      local.set 7
      local.get 7
      local.set 8
      local.get 8
      local.get 1
      local.get 2
      call $_ZN52_$LT$T$u20$as$u20$alloc..slice..hack..ConvertVec$GT$6to_vec17h7392a30f840e978bE
      local.get 5
      i64.load offset=4 align=4
      local.set 9
      local.get 0
      local.get 9
      i64.store align=4
      i32.const 8
      local.set 10
      local.get 0
      local.get 10
      i32.add
      local.set 11
      i32.const 4
      local.set 12
      local.get 5
      local.get 12
      i32.add
      local.set 13
      local.get 13
      local.get 10
      i32.add
      local.set 14
      local.get 14
      i32.load
      local.set 15
      local.get 11
      local.get 15
      i32.store
      i32.const 32
      local.set 16
      local.get 5
      local.get 16
      i32.add
      local.set 17
      local.get 17
      global.set $__stack_pointer
      return
    )
    (func $_ZN58_$LT$alloc..string..String$u20$as$u20$core..fmt..Debug$GT$3fmt17h4be1b83aa7ba5ee4E (;14;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 48
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=12
      local.get 4
      local.get 1
      i32.store offset=16
      local.get 4
      local.get 0
      i32.store offset=20
      local.get 4
      local.get 0
      i32.store offset=24
      local.get 0
      i32.load offset=4
      local.set 5
      local.get 4
      local.get 5
      i32.store offset=28
      local.get 4
      local.get 5
      i32.store offset=32
      local.get 0
      i32.load offset=8
      local.set 6
      local.get 4
      local.get 6
      i32.store offset=36
      i32.const 1
      local.set 7
      local.get 5
      local.get 7
      local.get 7
      local.get 6
      call $_ZN4core5slice3raw14from_raw_parts18precondition_check17h6e20dac50feedaf6E
      local.get 4
      local.get 5
      i32.store offset=40
      local.get 4
      local.get 6
      i32.store offset=44
      local.get 5
      local.get 6
      local.get 1
      call $_ZN40_$LT$str$u20$as$u20$core..fmt..Debug$GT$3fmt17h9c38ad0737c78d42E
      local.set 8
      i32.const 1
      local.set 9
      local.get 8
      local.get 9
      i32.and
      local.set 10
      i32.const 48
      local.set 11
      local.get 4
      local.get 11
      i32.add
      local.set 12
      local.get 12
      global.set $__stack_pointer
      local.get 10
      return
    )
    (func $_ZN65_$LT$alloc..string..String$u20$as$u20$core..ops..deref..Deref$GT$5deref17hd9f9c6da88ee9950E (;15;) (type 0) (param i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 32
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 1
      i32.store
      local.get 4
      local.get 1
      i32.store offset=4
      local.get 4
      local.get 1
      i32.store offset=8
      local.get 1
      i32.load offset=4
      local.set 5
      local.get 4
      local.get 5
      i32.store offset=12
      local.get 4
      local.get 5
      i32.store offset=16
      local.get 1
      i32.load offset=8
      local.set 6
      local.get 4
      local.get 6
      i32.store offset=20
      i32.const 1
      local.set 7
      local.get 5
      local.get 7
      local.get 7
      local.get 6
      call $_ZN4core5slice3raw14from_raw_parts18precondition_check17h6e20dac50feedaf6E
      local.get 4
      local.get 5
      i32.store offset=24
      local.get 4
      local.get 6
      i32.store offset=28
      local.get 0
      local.get 6
      i32.store offset=4
      local.get 0
      local.get 5
      i32.store
      i32.const 32
      local.set 8
      local.get 4
      local.get 8
      i32.add
      local.set 9
      local.get 9
      global.set $__stack_pointer
      return
    )
    (func $_ZN4core5alloc6layout6Layout5array5inner17h5633e9562e23f884E (;16;) (type 6) (param i32 i32 i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 4
      i32.const 48
      local.set 5
      local.get 4
      local.get 5
      i32.sub
      local.set 6
      local.get 6
      global.set $__stack_pointer
      local.get 6
      local.get 1
      i32.store offset=28
      local.get 6
      local.get 2
      i32.store offset=32
      local.get 6
      local.get 3
      i32.store offset=36
      block ;; label = @1
        block ;; label = @2
          local.get 1
          i32.eqz
          br_if 0 (;@2;)
          local.get 6
          local.get 2
          i32.store offset=20
          local.get 6
          i32.load offset=20
          local.set 7
          i32.const 1
          local.set 8
          local.get 7
          local.get 8
          i32.sub
          local.set 9
          i32.const 2147483647
          local.set 10
          local.get 10
          local.get 9
          i32.sub
          local.set 11
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                local.get 1
                i32.eqz
                br_if 0 (;@5;)
                local.get 11
                local.get 1
                i32.div_u
                local.set 12
                local.get 3
                local.set 13
                local.get 12
                local.set 14
                local.get 13
                local.get 14
                i32.gt_u
                local.set 15
                i32.const 1
                local.set 16
                local.get 15
                local.get 16
                i32.and
                local.set 17
                local.get 17
                br_if 2 (;@3;)
                br 1 (;@4;)
              end
              i32.const 1049156
              local.set 18
              local.get 18
              call $_ZN4core9panicking11panic_const23panic_const_div_by_zero17hed37a86622bbbb5bE
              unreachable
            end
            br 1 (;@2;)
          end
          i32.const 0
          local.set 19
          local.get 19
          i32.load offset=1049172
          local.set 20
          i32.const 0
          local.set 21
          local.get 21
          i32.load offset=1049176
          local.set 22
          local.get 6
          local.get 20
          i32.store offset=12
          local.get 6
          local.get 22
          i32.store offset=16
          br 1 (;@1;)
        end
        local.get 1
        local.get 3
        call $_ZN4core3num23_$LT$impl$u20$usize$GT$13unchecked_mul18precondition_check17h0769029f600eab28E
        local.get 1
        local.get 3
        i32.mul
        local.set 23
        local.get 6
        local.get 23
        i32.store offset=40
        local.get 6
        local.get 2
        i32.store offset=24
        local.get 6
        i32.load offset=24
        local.set 24
        local.get 6
        local.get 24
        i32.store offset=44
        local.get 6
        local.get 24
        i32.store offset=12
        local.get 6
        local.get 23
        i32.store offset=16
      end
      local.get 6
      i32.load offset=12
      local.set 25
      local.get 6
      i32.load offset=16
      local.set 26
      local.get 0
      local.get 26
      i32.store offset=4
      local.get 0
      local.get 25
      i32.store
      i32.const 48
      local.set 27
      local.get 6
      local.get 27
      i32.add
      local.set 28
      local.get 28
      global.set $__stack_pointer
      return
    )
    (func $_ZN5alloc5alloc5alloc17h84b3f5c063a5d867E (;17;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 32
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      i32.const 1056353
      local.set 5
      local.get 4
      local.get 5
      i32.store
      local.get 4
      local.get 0
      i32.store offset=4
      local.get 4
      local.get 1
      i32.store offset=8
      i32.const 1056353
      local.set 6
      i32.const 1
      local.set 7
      local.get 6
      local.get 7
      call $_ZN4core3ptr13read_volatile18precondition_check17h5b51455c8a7a3ebaE
      i32.const 0
      local.set 8
      local.get 8
      i32.load8_u offset=1056353
      local.set 9
      local.get 4
      local.get 9
      i32.store8 offset=19
      i32.const 4
      local.set 10
      local.get 4
      local.get 10
      i32.add
      local.set 11
      local.get 11
      local.set 12
      local.get 4
      local.get 12
      i32.store offset=20
      local.get 4
      i32.load offset=8
      local.set 13
      i32.const 4
      local.set 14
      local.get 4
      local.get 14
      i32.add
      local.set 15
      local.get 15
      local.set 16
      local.get 4
      local.get 16
      i32.store offset=24
      local.get 4
      i32.load offset=4
      local.set 17
      local.get 4
      local.get 17
      i32.store offset=28
      local.get 4
      local.get 17
      i32.store offset=12
      local.get 4
      i32.load offset=12
      local.set 18
      local.get 13
      local.get 18
      call $__rust_alloc
      local.set 19
      i32.const 32
      local.set 20
      local.get 4
      local.get 20
      i32.add
      local.set 21
      local.get 21
      global.set $__stack_pointer
      local.get 19
      return
    )
    (func $_ZN5alloc5alloc6Global10alloc_impl17h2268ba2ab3da212cE (;18;) (type 8) (param i32 i32 i32 i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 5
      i32.const 176
      local.set 6
      local.get 5
      local.get 6
      i32.sub
      local.set 7
      local.get 7
      global.set $__stack_pointer
      i32.const 0
      local.set 8
      local.get 7
      local.get 8
      i32.store offset=16
      i32.const 0
      local.set 9
      local.get 7
      local.get 9
      i32.store offset=20
      i32.const 0
      local.set 10
      local.get 7
      local.get 10
      i32.store offset=24
      local.get 7
      local.get 2
      i32.store offset=36
      local.get 7
      local.get 3
      i32.store offset=40
      local.get 7
      local.get 1
      i32.store offset=104
      local.get 4
      local.set 11
      local.get 7
      local.get 11
      i32.store8 offset=111
      i32.const 36
      local.set 12
      local.get 7
      local.get 12
      i32.add
      local.set 13
      local.get 13
      local.set 14
      local.get 7
      local.get 14
      i32.store offset=112
      local.get 7
      i32.load offset=40
      local.set 15
      local.get 7
      local.get 15
      i32.store offset=116
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  local.get 15
                  br_if 0 (;@6;)
                  i32.const 36
                  local.set 16
                  local.get 7
                  local.get 16
                  i32.add
                  local.set 17
                  local.get 17
                  local.set 18
                  local.get 7
                  local.get 18
                  i32.store offset=120
                  local.get 7
                  i32.load offset=36
                  local.set 19
                  local.get 7
                  local.get 19
                  i32.store offset=124
                  local.get 7
                  local.get 19
                  i32.store offset=80
                  local.get 7
                  i32.load offset=80
                  local.set 20
                  local.get 7
                  local.get 20
                  i32.store offset=128
                  i32.const 0
                  local.set 21
                  local.get 21
                  local.get 20
                  i32.add
                  local.set 22
                  local.get 7
                  local.get 22
                  i32.store offset=132
                  br 1 (;@5;)
                end
                local.get 4
                local.set 23
                local.get 23
                br_if 2 (;@3;)
                br 1 (;@4;)
              end
              local.get 22
              call $_ZN4core3ptr8non_null16NonNull$LT$T$GT$13new_unchecked18precondition_check17h19ea31e67e56ff85E
              local.get 7
              local.get 22
              i32.store offset=84
              local.get 7
              i32.load offset=84
              local.set 24
              local.get 7
              local.get 24
              i32.store offset=52
              local.get 7
              local.get 22
              i32.store offset=96
              local.get 7
              local.get 22
              i32.store offset=88
              i32.const 0
              local.set 25
              local.get 7
              local.get 25
              i32.store offset=92
              local.get 22
              call $_ZN4core3ptr8non_null16NonNull$LT$T$GT$13new_unchecked18precondition_check17h19ea31e67e56ff85E
              local.get 7
              i32.load offset=88
              local.set 26
              local.get 7
              i32.load offset=92
              local.set 27
              local.get 7
              local.get 26
              i32.store offset=44
              local.get 7
              local.get 27
              i32.store offset=48
              br 3 (;@1;)
            end
            local.get 7
            i32.load offset=36
            local.set 28
            local.get 7
            i32.load offset=40
            local.set 29
            local.get 28
            local.get 29
            call $_ZN5alloc5alloc5alloc17h84b3f5c063a5d867E
            local.set 30
            local.get 7
            local.get 30
            i32.store offset=56
            br 1 (;@2;)
          end
          local.get 7
          i32.load offset=36
          local.set 31
          local.get 7
          i32.load offset=40
          local.set 32
          local.get 7
          local.get 31
          i32.store offset=60
          local.get 7
          local.get 32
          i32.store offset=64
          i32.const 60
          local.set 33
          local.get 7
          local.get 33
          i32.add
          local.set 34
          local.get 34
          local.set 35
          local.get 7
          local.get 35
          i32.store offset=136
          i32.const 60
          local.set 36
          local.get 7
          local.get 36
          i32.add
          local.set 37
          local.get 37
          local.set 38
          local.get 7
          local.get 38
          i32.store offset=140
          local.get 7
          i32.load offset=36
          local.set 39
          local.get 7
          local.get 39
          i32.store offset=144
          local.get 7
          local.get 39
          i32.store offset=100
          local.get 7
          i32.load offset=100
          local.set 40
          local.get 15
          local.get 40
          call $__rust_alloc_zeroed
          local.set 41
          local.get 7
          local.get 41
          i32.store offset=56
        end
        local.get 7
        i32.load offset=56
        local.set 42
        local.get 7
        local.get 42
        i32.store offset=148
        block ;; label = @2
          local.get 42
          br_if 0 (;@2;)
          i32.const 0
          local.set 43
          local.get 7
          local.get 43
          i32.store offset=76
          i32.const 0
          local.set 44
          local.get 7
          local.get 44
          i32.store offset=72
          i32.const 0
          local.set 45
          local.get 45
          i32.load offset=1049180
          local.set 46
          i32.const 0
          local.set 47
          local.get 47
          i32.load offset=1049184
          local.set 48
          local.get 7
          local.get 46
          i32.store offset=44
          local.get 7
          local.get 48
          i32.store offset=48
          br 1 (;@1;)
        end
        local.get 42
        call $_ZN4core3ptr8non_null16NonNull$LT$T$GT$13new_unchecked18precondition_check17h19ea31e67e56ff85E
        local.get 7
        local.get 42
        i32.store offset=76
        local.get 7
        i32.load offset=76
        local.set 49
        local.get 7
        local.get 49
        i32.store offset=152
        local.get 7
        local.get 49
        i32.store offset=72
        local.get 7
        i32.load offset=72
        local.set 50
        local.get 7
        local.get 50
        i32.store offset=156
        local.get 7
        local.get 50
        i32.store offset=68
        local.get 7
        i32.load offset=68
        local.set 51
        local.get 7
        local.get 51
        i32.store offset=160
        local.get 7
        local.get 51
        i32.store offset=164
        local.get 7
        local.get 51
        i32.store offset=168
        local.get 7
        local.get 15
        i32.store offset=172
        local.get 51
        call $_ZN4core3ptr8non_null16NonNull$LT$T$GT$13new_unchecked18precondition_check17h19ea31e67e56ff85E
        local.get 7
        local.get 51
        i32.store offset=44
        local.get 7
        local.get 15
        i32.store offset=48
      end
      local.get 7
      i32.load offset=44
      local.set 52
      local.get 7
      i32.load offset=48
      local.set 53
      local.get 0
      local.get 53
      i32.store offset=4
      local.get 0
      local.get 52
      i32.store
      i32.const 176
      local.set 54
      local.get 7
      local.get 54
      i32.add
      local.set 55
      local.get 55
      global.set $__stack_pointer
      return
    )
    (func $_ZN63_$LT$alloc..alloc..Global$u20$as$u20$core..alloc..Allocator$GT$15allocate_zeroed17h660d3ae9c4c60b55E (;19;) (type 6) (param i32 i32 i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 4
      i32.const 32
      local.set 5
      local.get 4
      local.get 5
      i32.sub
      local.set 6
      local.get 6
      global.set $__stack_pointer
      local.get 6
      local.get 1
      i32.store offset=20
      local.get 6
      local.get 2
      i32.store offset=24
      local.get 6
      local.get 3
      i32.store offset=28
      i32.const 1
      local.set 7
      i32.const 8
      local.set 8
      local.get 6
      local.get 8
      i32.add
      local.set 9
      local.get 9
      local.get 1
      local.get 2
      local.get 3
      local.get 7
      call $_ZN5alloc5alloc6Global10alloc_impl17h2268ba2ab3da212cE
      local.get 6
      i32.load offset=8
      local.set 10
      local.get 6
      i32.load offset=12
      local.set 11
      local.get 0
      local.get 11
      i32.store offset=4
      local.get 0
      local.get 10
      i32.store
      i32.const 32
      local.set 12
      local.get 6
      local.get 12
      i32.add
      local.set 13
      local.get 13
      global.set $__stack_pointer
      return
    )
    (func $_ZN63_$LT$alloc..alloc..Global$u20$as$u20$core..alloc..Allocator$GT$8allocate17h33e56f560b57388eE (;20;) (type 6) (param i32 i32 i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 4
      i32.const 32
      local.set 5
      local.get 4
      local.get 5
      i32.sub
      local.set 6
      local.get 6
      global.set $__stack_pointer
      local.get 6
      local.get 1
      i32.store offset=20
      local.get 6
      local.get 2
      i32.store offset=24
      local.get 6
      local.get 3
      i32.store offset=28
      i32.const 0
      local.set 7
      i32.const 8
      local.set 8
      local.get 6
      local.get 8
      i32.add
      local.set 9
      local.get 9
      local.get 1
      local.get 2
      local.get 3
      local.get 7
      call $_ZN5alloc5alloc6Global10alloc_impl17h2268ba2ab3da212cE
      local.get 6
      i32.load offset=8
      local.set 10
      local.get 6
      i32.load offset=12
      local.set 11
      local.get 0
      local.get 11
      i32.store offset=4
      local.get 0
      local.get 10
      i32.store
      i32.const 32
      local.set 12
      local.get 6
      local.get 12
      i32.add
      local.set 13
      local.get 13
      global.set $__stack_pointer
      return
    )
    (func $_ZN11wit_bindgen2rt25invalid_enum_discriminant17h3050578c96a32865E (;21;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 32
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      local.get 3
      global.set $__stack_pointer
      i32.const 8
      local.set 4
      local.get 3
      local.get 4
      i32.add
      local.set 5
      local.get 5
      local.set 6
      i32.const 1049216
      local.set 7
      local.get 6
      local.get 7
      call $_ZN4core3fmt9Arguments9new_const17hd92b933da0585f96E
      i32.const 8
      local.set 8
      local.get 3
      local.get 8
      i32.add
      local.set 9
      local.get 9
      local.set 10
      i32.const 1049320
      local.set 11
      local.get 10
      local.get 11
      call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
      unreachable
    )
    (func $_ZN11wit_bindgen2rt25invalid_enum_discriminant17h41b3f6d22e6d50d1E (;22;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 32
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      local.get 3
      global.set $__stack_pointer
      i32.const 8
      local.set 4
      local.get 3
      local.get 4
      i32.add
      local.set 5
      local.get 5
      local.set 6
      i32.const 1049216
      local.set 7
      local.get 6
      local.get 7
      call $_ZN4core3fmt9Arguments9new_const17hd92b933da0585f96E
      i32.const 8
      local.set 8
      local.get 3
      local.get 8
      i32.add
      local.set 9
      local.get 9
      local.set 10
      i32.const 1049320
      local.set 11
      local.get 10
      local.get 11
      call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
      unreachable
    )
    (func $_ZN11wit_bindgen2rt25invalid_enum_discriminant17h4743fc51e5e64321E (;23;) (type 10) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 0
      i32.const 32
      local.set 1
      local.get 0
      local.get 1
      i32.sub
      local.set 2
      local.get 2
      global.set $__stack_pointer
      i32.const 8
      local.set 3
      local.get 2
      local.get 3
      i32.add
      local.set 4
      local.get 4
      local.set 5
      i32.const 1049216
      local.set 6
      local.get 5
      local.get 6
      call $_ZN4core3fmt9Arguments9new_const17hd92b933da0585f96E
      i32.const 8
      local.set 7
      local.get 2
      local.get 7
      i32.add
      local.set 8
      local.get 8
      local.set 9
      i32.const 1049320
      local.set 10
      local.get 9
      local.get 10
      call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
      unreachable
    )
    (func $_ZN11wit_bindgen2rt25invalid_enum_discriminant17h628a02db94cc7afbE (;24;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 32
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      local.get 3
      global.set $__stack_pointer
      i32.const 8
      local.set 4
      local.get 3
      local.get 4
      i32.add
      local.set 5
      local.get 5
      local.set 6
      i32.const 1049216
      local.set 7
      local.get 6
      local.get 7
      call $_ZN4core3fmt9Arguments9new_const17hd92b933da0585f96E
      i32.const 8
      local.set 8
      local.get 3
      local.get 8
      i32.add
      local.set 9
      local.get 9
      local.set 10
      i32.const 1049320
      local.set 11
      local.get 10
      local.get 11
      call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
      unreachable
    )
    (func $_ZN11wit_bindgen2rt25invalid_enum_discriminant17hb7683f0fece93b0dE (;25;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 32
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      local.get 3
      global.set $__stack_pointer
      i32.const 8
      local.set 4
      local.get 3
      local.get 4
      i32.add
      local.set 5
      local.get 5
      local.set 6
      i32.const 1049216
      local.set 7
      local.get 6
      local.get 7
      call $_ZN4core3fmt9Arguments9new_const17hd92b933da0585f96E
      i32.const 8
      local.set 8
      local.get 3
      local.get 8
      i32.add
      local.set 9
      local.get 9
      local.set 10
      i32.const 1049320
      local.set 11
      local.get 10
      local.get 11
      call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
      unreachable
    )
    (func $_ZN4core3ptr13read_volatile18precondition_check17h5b51455c8a7a3ebaE (;26;) (type 0) (param i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 48
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      i32.const 1049380
      local.set 5
      local.get 4
      local.get 5
      i32.store offset=4
      local.get 4
      local.get 0
      i32.store offset=32
      local.get 4
      local.get 1
      i32.store offset=36
      local.get 4
      local.get 0
      i32.store offset=40
      block ;; label = @1
        block ;; label = @2
          local.get 0
          br_if 0 (;@2;)
          br 1 (;@1;)
        end
        local.get 1
        i32.popcnt
        local.set 6
        local.get 4
        local.get 6
        i32.store offset=44
        local.get 4
        i32.load offset=44
        local.set 7
        i32.const 1
        local.set 8
        local.get 7
        local.set 9
        local.get 8
        local.set 10
        local.get 9
        local.get 10
        i32.eq
        local.set 11
        i32.const 1
        local.set 12
        local.get 11
        local.get 12
        i32.and
        local.set 13
        block ;; label = @2
          block ;; label = @3
            local.get 13
            i32.eqz
            br_if 0 (;@3;)
            i32.const 1
            local.set 14
            local.get 1
            local.get 14
            i32.sub
            local.set 15
            local.get 0
            local.get 15
            i32.and
            local.set 16
            local.get 16
            i32.eqz
            br_if 1 (;@2;)
            br 2 (;@1;)
          end
          i32.const 1049380
          local.set 17
          local.get 4
          local.get 17
          i32.store offset=8
          i32.const 1
          local.set 18
          local.get 4
          local.get 18
          i32.store offset=12
          i32.const 0
          local.set 19
          local.get 19
          i32.load offset=1049500
          local.set 20
          i32.const 0
          local.set 21
          local.get 21
          i32.load offset=1049504
          local.set 22
          local.get 4
          local.get 20
          i32.store offset=24
          local.get 4
          local.get 22
          i32.store offset=28
          i32.const 4
          local.set 23
          local.get 4
          local.get 23
          i32.store offset=16
          i32.const 0
          local.set 24
          local.get 4
          local.get 24
          i32.store offset=20
          i32.const 8
          local.set 25
          local.get 4
          local.get 25
          i32.add
          local.set 26
          local.get 26
          local.set 27
          i32.const 1049592
          local.set 28
          local.get 27
          local.get 28
          call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
          unreachable
        end
        i32.const 48
        local.set 29
        local.get 4
        local.get 29
        i32.add
        local.set 30
        local.get 30
        global.set $__stack_pointer
        return
      end
      i32.const 1049388
      local.set 31
      i32.const 110
      local.set 32
      local.get 31
      local.get 32
      call $_ZN4core9panicking14panic_nounwind17he2774a18f33c590cE
      unreachable
    )
    (func $_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$15try_allocate_in17ha3ac89577fa5bf59E (;27;) (type 5) (param i32 i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 3
      i32.const 160
      local.set 4
      local.get 3
      local.get 4
      i32.sub
      local.set 5
      local.get 5
      global.set $__stack_pointer
      i32.const 0
      local.set 6
      local.get 6
      i32.load offset=1049608
      local.set 7
      i32.const 0
      local.set 8
      local.get 8
      i32.load offset=1049612
      local.set 9
      local.get 5
      local.get 7
      i32.store offset=28
      local.get 5
      local.get 9
      i32.store offset=32
      i32.const 0
      local.set 10
      local.get 10
      i32.load offset=1049608
      local.set 11
      i32.const 0
      local.set 12
      local.get 12
      i32.load offset=1049612
      local.set 13
      local.get 5
      local.get 11
      i32.store offset=36
      local.get 5
      local.get 13
      i32.store offset=40
      i32.const 0
      local.set 14
      local.get 14
      i32.load offset=1049608
      local.set 15
      i32.const 0
      local.set 16
      local.get 16
      i32.load offset=1049612
      local.set 17
      local.get 5
      local.get 15
      i32.store offset=44
      local.get 5
      local.get 17
      i32.store offset=48
      i32.const 0
      local.set 18
      local.get 18
      i32.load offset=1049608
      local.set 19
      i32.const 0
      local.set 20
      local.get 20
      i32.load offset=1049612
      local.set 21
      local.get 5
      local.get 19
      i32.store offset=52
      local.get 5
      local.get 21
      i32.store offset=56
      local.get 2
      local.set 22
      local.get 5
      local.get 22
      i32.store8 offset=62
      local.get 5
      local.get 1
      i32.store offset=104
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              local.get 1
              br_if 0 (;@4;)
              i32.const 1
              local.set 23
              local.get 5
              local.get 23
              i32.store offset=152
              i32.const 0
              local.set 24
              i32.const 1
              local.set 25
              local.get 24
              local.get 25
              i32.add
              local.set 26
              local.get 5
              local.get 26
              i32.store offset=156
              br 1 (;@3;)
            end
            i32.const 1
            local.set 27
            local.get 5
            local.get 27
            i32.store offset=112
            i32.const 1
            local.set 28
            i32.const 16
            local.set 29
            local.get 5
            local.get 29
            i32.add
            local.set 30
            local.get 30
            local.get 28
            local.get 28
            local.get 1
            call $_ZN4core5alloc6layout6Layout5array5inner17h5633e9562e23f884E
            local.get 5
            i32.load offset=20
            local.set 31
            local.get 5
            i32.load offset=16
            local.set 32
            local.get 5
            local.get 32
            i32.store offset=72
            local.get 5
            local.get 31
            i32.store offset=76
            local.get 5
            i32.load offset=72
            local.set 33
            i32.const 1
            local.set 34
            i32.const 0
            local.set 35
            local.get 35
            local.get 34
            local.get 33
            select
            local.set 36
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  block ;; label = @7
                    block ;; label = @8
                      block ;; label = @9
                        block ;; label = @10
                          local.get 36
                          br_if 0 (;@10;)
                          local.get 5
                          i32.load offset=72
                          local.set 37
                          local.get 5
                          i32.load offset=76
                          local.set 38
                          local.get 5
                          local.get 37
                          i32.store offset=116
                          local.get 5
                          local.get 38
                          i32.store offset=120
                          local.get 5
                          local.get 37
                          i32.store offset=64
                          local.get 5
                          local.get 38
                          i32.store offset=68
                          i32.const 64
                          local.set 39
                          local.get 5
                          local.get 39
                          i32.add
                          local.set 40
                          local.get 40
                          local.set 41
                          local.get 5
                          local.get 41
                          i32.store offset=124
                          local.get 5
                          local.get 38
                          i32.store offset=128
                          i32.const 2147483647
                          local.set 42
                          local.get 38
                          local.set 43
                          local.get 42
                          local.set 44
                          local.get 43
                          local.get 44
                          i32.gt_u
                          local.set 45
                          i32.const 1
                          local.set 46
                          local.get 45
                          local.get 46
                          i32.and
                          local.set 47
                          local.get 47
                          br_if 2 (;@8;)
                          br 1 (;@9;)
                        end
                        i32.const 0
                        local.set 48
                        local.get 48
                        i32.load offset=1049608
                        local.set 49
                        i32.const 0
                        local.set 50
                        local.get 50
                        i32.load offset=1049612
                        local.set 51
                        local.get 0
                        local.get 49
                        i32.store offset=4
                        local.get 0
                        local.get 51
                        i32.store offset=8
                        i32.const 1
                        local.set 52
                        local.get 0
                        local.get 52
                        i32.store
                        br 5 (;@4;)
                      end
                      local.get 5
                      i32.load8_u offset=62
                      local.set 53
                      i32.const 1
                      local.set 54
                      local.get 53
                      local.get 54
                      i32.and
                      local.set 55
                      local.get 55
                      i32.eqz
                      br_if 1 (;@7;)
                      br 2 (;@6;)
                    end
                    i32.const 0
                    local.set 56
                    local.get 56
                    i32.load offset=1049608
                    local.set 57
                    i32.const 0
                    local.set 58
                    local.get 58
                    i32.load offset=1049612
                    local.set 59
                    local.get 5
                    local.get 57
                    i32.store offset=80
                    local.get 5
                    local.get 59
                    i32.store offset=84
                    local.get 5
                    i32.load offset=80
                    local.set 60
                    local.get 5
                    i32.load offset=84
                    local.set 61
                    local.get 5
                    local.get 60
                    i32.store offset=144
                    local.get 5
                    local.get 61
                    i32.store offset=148
                    local.get 0
                    local.get 60
                    i32.store offset=4
                    local.get 0
                    local.get 61
                    i32.store offset=8
                    i32.const 1
                    local.set 62
                    local.get 0
                    local.get 62
                    i32.store
                    br 3 (;@4;)
                  end
                  i32.const 63
                  local.set 63
                  local.get 5
                  local.get 63
                  i32.add
                  local.set 64
                  local.get 5
                  local.get 64
                  local.get 37
                  local.get 38
                  call $_ZN63_$LT$alloc..alloc..Global$u20$as$u20$core..alloc..Allocator$GT$8allocate17h33e56f560b57388eE
                  local.get 5
                  i32.load offset=4
                  local.set 65
                  local.get 5
                  i32.load
                  local.set 66
                  local.get 5
                  local.get 66
                  i32.store offset=88
                  local.get 5
                  local.get 65
                  i32.store offset=92
                  br 1 (;@5;)
                end
                i32.const 8
                local.set 67
                local.get 5
                local.get 67
                i32.add
                local.set 68
                i32.const 63
                local.set 69
                local.get 5
                local.get 69
                i32.add
                local.set 70
                local.get 68
                local.get 70
                local.get 37
                local.get 38
                call $_ZN63_$LT$alloc..alloc..Global$u20$as$u20$core..alloc..Allocator$GT$15allocate_zeroed17h660d3ae9c4c60b55E
                local.get 5
                i32.load offset=12
                local.set 71
                local.get 5
                i32.load offset=8
                local.set 72
                local.get 5
                local.get 72
                i32.store offset=88
                local.get 5
                local.get 71
                i32.store offset=92
              end
              local.get 5
              i32.load offset=88
              local.set 73
              i32.const 1
              local.set 74
              i32.const 0
              local.set 75
              local.get 75
              local.get 74
              local.get 73
              select
              local.set 76
              block ;; label = @5
                local.get 76
                br_if 0 (;@5;)
                local.get 5
                i32.load offset=88
                local.set 77
                local.get 5
                i32.load offset=92
                local.set 78
                local.get 5
                local.get 77
                i32.store offset=132
                local.get 5
                local.get 78
                i32.store offset=136
                local.get 5
                local.get 77
                i32.store offset=140
                local.get 0
                local.get 1
                i32.store offset=4
                local.get 0
                local.get 77
                i32.store offset=8
                i32.const 0
                local.set 79
                local.get 0
                local.get 79
                i32.store
                br 3 (;@2;)
              end
              local.get 5
              local.get 37
              i32.store offset=96
              local.get 5
              local.get 38
              i32.store offset=100
              local.get 5
              i32.load offset=96
              local.set 80
              local.get 5
              i32.load offset=100
              local.set 81
              local.get 0
              local.get 80
              i32.store offset=4
              local.get 0
              local.get 81
              i32.store offset=8
              i32.const 1
              local.set 82
              local.get 0
              local.get 82
              i32.store
            end
            br 2 (;@1;)
          end
          i32.const 0
          local.set 83
          i32.const 1
          local.set 84
          local.get 83
          local.get 84
          i32.add
          local.set 85
          local.get 85
          call $_ZN4core3ptr8non_null16NonNull$LT$T$GT$13new_unchecked18precondition_check17h19ea31e67e56ff85E
          i32.const 0
          local.set 86
          local.get 0
          local.get 86
          i32.store offset=4
          i32.const 0
          local.set 87
          i32.const 1
          local.set 88
          local.get 87
          local.get 88
          i32.add
          local.set 89
          local.get 0
          local.get 89
          i32.store offset=8
          i32.const 0
          local.set 90
          local.get 0
          local.get 90
          i32.store
        end
      end
      i32.const 160
      local.set 91
      local.get 5
      local.get 91
      i32.add
      local.set 92
      local.get 92
      global.set $__stack_pointer
      return
    )
    (func $_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$17from_raw_parts_in17h31459d0e31f464c6E (;28;) (type 5) (param i32 i32 i32)
      (local i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 3
      i32.const 16
      local.set 4
      local.get 3
      local.get 4
      i32.sub
      local.set 5
      local.get 5
      global.set $__stack_pointer
      local.get 5
      local.get 1
      i32.store offset=4
      local.get 5
      local.get 2
      i32.store offset=8
      local.get 5
      local.get 2
      i32.store
      local.get 1
      call $_ZN4core3ptr8non_null16NonNull$LT$T$GT$13new_unchecked18precondition_check17h19ea31e67e56ff85E
      local.get 5
      i32.load
      local.set 6
      local.get 0
      local.get 1
      i32.store offset=4
      local.get 0
      local.get 6
      i32.store
      i32.const 16
      local.set 7
      local.get 5
      local.get 7
      i32.add
      local.set 8
      local.get 8
      global.set $__stack_pointer
      return
    )
    (func $_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17hce44cecf0f119939E (;29;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 16
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=8
      local.get 4
      local.get 1
      i32.store offset=12
      local.get 0
      i32.load
      local.set 5
      local.get 5
      local.get 1
      call $_ZN58_$LT$alloc..string..String$u20$as$u20$core..fmt..Debug$GT$3fmt17h4be1b83aa7ba5ee4E
      local.set 6
      i32.const 1
      local.set 7
      local.get 6
      local.get 7
      i32.and
      local.set 8
      i32.const 16
      local.set 9
      local.get 4
      local.get 9
      i32.add
      local.set 10
      local.get 10
      global.set $__stack_pointer
      local.get 8
      return
    )
    (func $_ZN4core3ops8function6FnOnce9call_once17h0e9ba0ef9fe25543E (;30;) (type 5) (param i32 i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 3
      i32.const 16
      local.set 4
      local.get 3
      local.get 4
      i32.sub
      local.set 5
      local.get 5
      global.set $__stack_pointer
      local.get 5
      local.get 1
      i32.store offset=4
      local.get 5
      local.get 2
      i32.store offset=8
      local.get 5
      i32.load offset=4
      local.set 6
      local.get 5
      i32.load offset=8
      local.set 7
      local.get 0
      local.get 6
      local.get 7
      call $_ZN5alloc3str56_$LT$impl$u20$alloc..borrow..ToOwned$u20$for$u20$str$GT$8to_owned17ha4066b8bbafecc01E
      i32.const 16
      local.set 8
      local.get 5
      local.get 8
      i32.add
      local.set 9
      local.get 9
      global.set $__stack_pointer
      return
    )
    (func $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17he3a070ea415b5a8dE (;31;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 16
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      local.get 3
      global.set $__stack_pointer
      local.get 3
      local.get 0
      i32.store offset=12
      local.get 0
      call $_ZN4core3ptr46drop_in_place$LT$alloc..vec..Vec$LT$u8$GT$$GT$17h57b6af354035ef20E
      i32.const 16
      local.set 4
      local.get 3
      local.get 4
      i32.add
      local.set 5
      local.get 5
      global.set $__stack_pointer
      return
    )
    (func $_ZN4core3ptr70drop_in_place$LT$core..option..Option$LT$alloc..string..String$GT$$GT$17he19cfb52433c8da3E (;32;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 16
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      local.get 3
      global.set $__stack_pointer
      local.get 3
      local.get 0
      i32.store offset=12
      local.get 0
      i32.load
      local.set 4
      i32.const -2147483648
      local.set 5
      local.get 4
      local.set 6
      local.get 5
      local.set 7
      local.get 6
      local.get 7
      i32.eq
      local.set 8
      i32.const 0
      local.set 9
      i32.const 1
      local.set 10
      i32.const 1
      local.set 11
      local.get 8
      local.get 11
      i32.and
      local.set 12
      local.get 9
      local.get 10
      local.get 12
      select
      local.set 13
      block ;; label = @1
        local.get 13
        i32.eqz
        br_if 0 (;@1;)
        local.get 0
        call $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17he3a070ea415b5a8dE
      end
      i32.const 16
      local.set 14
      local.get 3
      local.get 14
      i32.add
      local.set 15
      local.get 15
      global.set $__stack_pointer
      return
    )
    (func $_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$$LP$$RP$$C$alloc..string..String$GT$$GT$17h747b2ded2b20ac66E (;33;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 16
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      local.get 3
      global.set $__stack_pointer
      local.get 3
      local.get 0
      i32.store offset=12
      local.get 0
      i32.load
      local.set 4
      i32.const -2147483648
      local.set 5
      local.get 4
      local.set 6
      local.get 5
      local.set 7
      local.get 6
      local.get 7
      i32.eq
      local.set 8
      i32.const 0
      local.set 9
      i32.const 1
      local.set 10
      i32.const 1
      local.set 11
      local.get 8
      local.get 11
      i32.and
      local.set 12
      local.get 9
      local.get 10
      local.get 12
      select
      local.set 13
      block ;; label = @1
        local.get 13
        i32.eqz
        br_if 0 (;@1;)
        local.get 0
        call $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17he3a070ea415b5a8dE
      end
      i32.const 16
      local.set 14
      local.get 3
      local.get 14
      i32.add
      local.set 15
      local.get 15
      global.set $__stack_pointer
      return
    )
    (func $_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$alloc..string..String$C$$LP$$RP$$GT$$GT$17h640b2ff63ffdbd3fE (;34;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 16
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      local.get 3
      global.set $__stack_pointer
      local.get 3
      local.get 0
      i32.store offset=12
      local.get 0
      i32.load
      local.set 4
      i32.const -2147483648
      local.set 5
      local.get 4
      local.set 6
      local.get 5
      local.set 7
      local.get 6
      local.get 7
      i32.eq
      local.set 8
      i32.const 1
      local.set 9
      i32.const 0
      local.set 10
      i32.const 1
      local.set 11
      local.get 8
      local.get 11
      i32.and
      local.set 12
      local.get 9
      local.get 10
      local.get 12
      select
      local.set 13
      block ;; label = @1
        local.get 13
        br_if 0 (;@1;)
        local.get 0
        call $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17he3a070ea415b5a8dE
      end
      i32.const 16
      local.set 14
      local.get 3
      local.get 14
      i32.add
      local.set 15
      local.get 15
      global.set $__stack_pointer
      return
    )
    (func $_ZN4core3ptr94drop_in_place$LT$core..result..Result$LT$alloc..string..String$C$alloc..string..String$GT$$GT$17hb27e0d70cac2ed41E (;35;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 16
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      local.get 3
      global.set $__stack_pointer
      local.get 3
      local.get 0
      i32.store offset=12
      local.get 0
      i32.load
      local.set 4
      block ;; label = @1
        block ;; label = @2
          local.get 4
          br_if 0 (;@2;)
          i32.const 4
          local.set 5
          local.get 0
          local.get 5
          i32.add
          local.set 6
          local.get 6
          call $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17he3a070ea415b5a8dE
          br 1 (;@1;)
        end
        i32.const 4
        local.set 7
        local.get 0
        local.get 7
        i32.add
        local.set 8
        local.get 8
        call $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17he3a070ea415b5a8dE
      end
      i32.const 16
      local.set 9
      local.get 3
      local.get 9
      i32.add
      local.set 10
      local.get 10
      global.set $__stack_pointer
      return
    )
    (func $_ZN4core3ptr8non_null16NonNull$LT$T$GT$13new_unchecked18precondition_check17h19ea31e67e56ff85E (;36;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 16
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      local.get 3
      global.set $__stack_pointer
      local.get 3
      local.get 0
      i32.store offset=8
      local.get 3
      local.get 0
      i32.store offset=12
      block ;; label = @1
        local.get 0
        br_if 0 (;@1;)
        i32.const 1049616
        local.set 4
        i32.const 93
        local.set 5
        local.get 4
        local.get 5
        call $_ZN4core9panicking14panic_nounwind17he2774a18f33c590cE
        unreachable
      end
      i32.const 16
      local.set 6
      local.get 3
      local.get 6
      i32.add
      local.set 7
      local.get 7
      global.set $__stack_pointer
      return
    )
    (func $_ZN70_$LT$core..result..Result$LT$T$C$E$GT$$u20$as$u20$core..fmt..Debug$GT$3fmt17h2148cc03c39dba01E (;37;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 32
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=24
      local.get 4
      local.get 1
      i32.store offset=28
      local.get 0
      i32.load
      local.set 5
      block ;; label = @1
        block ;; label = @2
          local.get 5
          br_if 0 (;@2;)
          i32.const 4
          local.set 6
          local.get 0
          local.get 6
          i32.add
          local.set 7
          local.get 4
          local.get 7
          i32.store offset=16
          i32.const 1049728
          local.set 8
          i32.const 2
          local.set 9
          i32.const 16
          local.set 10
          local.get 4
          local.get 10
          i32.add
          local.set 11
          local.get 11
          local.set 12
          i32.const 1049712
          local.set 13
          local.get 1
          local.get 8
          local.get 9
          local.get 12
          local.get 13
          call $_ZN4core3fmt9Formatter25debug_tuple_field1_finish17h60a7f9de909b7316E
          local.set 14
          i32.const 1
          local.set 15
          local.get 14
          local.get 15
          i32.and
          local.set 16
          local.get 4
          local.get 16
          i32.store8 offset=15
          br 1 (;@1;)
        end
        i32.const 4
        local.set 17
        local.get 0
        local.get 17
        i32.add
        local.set 18
        local.get 4
        local.get 18
        i32.store offset=20
        i32.const 1049730
        local.set 19
        i32.const 3
        local.set 20
        i32.const 20
        local.set 21
        local.get 4
        local.get 21
        i32.add
        local.set 22
        local.get 22
        local.set 23
        i32.const 1049712
        local.set 24
        local.get 1
        local.get 19
        local.get 20
        local.get 23
        local.get 24
        call $_ZN4core3fmt9Formatter25debug_tuple_field1_finish17h60a7f9de909b7316E
        local.set 25
        i32.const 1
        local.set 26
        local.get 25
        local.get 26
        i32.and
        local.set 27
        local.get 4
        local.get 27
        i32.store8 offset=15
      end
      local.get 4
      i32.load8_u offset=15
      local.set 28
      i32.const 1
      local.set 29
      local.get 28
      local.get 29
      i32.and
      local.set 30
      i32.const 32
      local.set 31
      local.get 4
      local.get 31
      i32.add
      local.set 32
      local.get 32
      global.set $__stack_pointer
      local.get 30
      return
    )
    (func $_ZN70_$LT$core..result..Result$LT$T$C$E$GT$$u20$as$u20$core..fmt..Debug$GT$3fmt17h65c3f310f2d696a0E (;38;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 32
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=24
      local.get 4
      local.get 1
      i32.store offset=28
      local.get 0
      i32.load
      local.set 5
      i32.const -2147483648
      local.set 6
      local.get 5
      local.set 7
      local.get 6
      local.set 8
      local.get 7
      local.get 8
      i32.eq
      local.set 9
      i32.const 1
      local.set 10
      i32.const 0
      local.set 11
      i32.const 1
      local.set 12
      local.get 9
      local.get 12
      i32.and
      local.set 13
      local.get 10
      local.get 11
      local.get 13
      select
      local.set 14
      block ;; label = @1
        block ;; label = @2
          local.get 14
          br_if 0 (;@2;)
          local.get 4
          local.get 0
          i32.store offset=16
          i32.const 1049728
          local.set 15
          i32.const 2
          local.set 16
          i32.const 16
          local.set 17
          local.get 4
          local.get 17
          i32.add
          local.set 18
          local.get 18
          local.set 19
          i32.const 1049712
          local.set 20
          local.get 1
          local.get 15
          local.get 16
          local.get 19
          local.get 20
          call $_ZN4core3fmt9Formatter25debug_tuple_field1_finish17h60a7f9de909b7316E
          local.set 21
          i32.const 1
          local.set 22
          local.get 21
          local.get 22
          i32.and
          local.set 23
          local.get 4
          local.get 23
          i32.store8 offset=15
          br 1 (;@1;)
        end
        local.get 4
        local.get 0
        i32.store offset=20
        i32.const 1049730
        local.set 24
        i32.const 3
        local.set 25
        i32.const 20
        local.set 26
        local.get 4
        local.get 26
        i32.add
        local.set 27
        local.get 27
        local.set 28
        i32.const 1049736
        local.set 29
        local.get 1
        local.get 24
        local.get 25
        local.get 28
        local.get 29
        call $_ZN4core3fmt9Formatter25debug_tuple_field1_finish17h60a7f9de909b7316E
        local.set 30
        i32.const 1
        local.set 31
        local.get 30
        local.get 31
        i32.and
        local.set 32
        local.get 4
        local.get 32
        i32.store8 offset=15
      end
      local.get 4
      i32.load8_u offset=15
      local.set 33
      i32.const 1
      local.set 34
      local.get 33
      local.get 34
      i32.and
      local.set 35
      i32.const 32
      local.set 36
      local.get 4
      local.get 36
      i32.add
      local.set 37
      local.get 37
      global.set $__stack_pointer
      local.get 35
      return
    )
    (func $_ZN70_$LT$core..result..Result$LT$T$C$E$GT$$u20$as$u20$core..fmt..Debug$GT$3fmt17h691212d9090277eaE (;39;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 32
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=24
      local.get 4
      local.get 1
      i32.store offset=28
      local.get 0
      i32.load
      local.set 5
      i32.const -2147483648
      local.set 6
      local.get 5
      local.set 7
      local.get 6
      local.set 8
      local.get 7
      local.get 8
      i32.eq
      local.set 9
      i32.const 0
      local.set 10
      i32.const 1
      local.set 11
      i32.const 1
      local.set 12
      local.get 9
      local.get 12
      i32.and
      local.set 13
      local.get 10
      local.get 11
      local.get 13
      select
      local.set 14
      block ;; label = @1
        block ;; label = @2
          local.get 14
          br_if 0 (;@2;)
          local.get 4
          local.get 0
          i32.store offset=16
          i32.const 1049728
          local.set 15
          i32.const 2
          local.set 16
          i32.const 16
          local.set 17
          local.get 4
          local.get 17
          i32.add
          local.set 18
          local.get 18
          local.set 19
          i32.const 1049736
          local.set 20
          local.get 1
          local.get 15
          local.get 16
          local.get 19
          local.get 20
          call $_ZN4core3fmt9Formatter25debug_tuple_field1_finish17h60a7f9de909b7316E
          local.set 21
          i32.const 1
          local.set 22
          local.get 21
          local.get 22
          i32.and
          local.set 23
          local.get 4
          local.get 23
          i32.store8 offset=15
          br 1 (;@1;)
        end
        local.get 4
        local.get 0
        i32.store offset=20
        i32.const 1049730
        local.set 24
        i32.const 3
        local.set 25
        i32.const 20
        local.set 26
        local.get 4
        local.get 26
        i32.add
        local.set 27
        local.get 27
        local.set 28
        i32.const 1049712
        local.set 29
        local.get 1
        local.get 24
        local.get 25
        local.get 28
        local.get 29
        call $_ZN4core3fmt9Formatter25debug_tuple_field1_finish17h60a7f9de909b7316E
        local.set 30
        i32.const 1
        local.set 31
        local.get 30
        local.get 31
        i32.and
        local.set 32
        local.get 4
        local.get 32
        i32.store8 offset=15
      end
      local.get 4
      i32.load8_u offset=15
      local.set 33
      i32.const 1
      local.set 34
      local.get 33
      local.get 34
      i32.and
      local.set 35
      i32.const 32
      local.set 36
      local.get 4
      local.get 36
      i32.add
      local.set 37
      local.get 37
      global.set $__stack_pointer
      local.get 35
      return
    )
    (func $_ZN70_$LT$core..result..Result$LT$T$C$E$GT$$u20$as$u20$core..fmt..Debug$GT$3fmt17he46203b2fce9b184E (;40;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 32
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=24
      local.get 4
      local.get 1
      i32.store offset=28
      local.get 0
      i32.load8_u
      local.set 5
      i32.const 1
      local.set 6
      local.get 5
      local.get 6
      i32.and
      local.set 7
      block ;; label = @1
        block ;; label = @2
          local.get 7
          br_if 0 (;@2;)
          i32.const 1
          local.set 8
          local.get 0
          local.get 8
          i32.add
          local.set 9
          local.get 4
          local.get 9
          i32.store offset=16
          i32.const 1049728
          local.set 10
          i32.const 2
          local.set 11
          i32.const 16
          local.set 12
          local.get 4
          local.get 12
          i32.add
          local.set 13
          local.get 13
          local.set 14
          i32.const 1049736
          local.set 15
          local.get 1
          local.get 10
          local.get 11
          local.get 14
          local.get 15
          call $_ZN4core3fmt9Formatter25debug_tuple_field1_finish17h60a7f9de909b7316E
          local.set 16
          i32.const 1
          local.set 17
          local.get 16
          local.get 17
          i32.and
          local.set 18
          local.get 4
          local.get 18
          i32.store8 offset=15
          br 1 (;@1;)
        end
        i32.const 1
        local.set 19
        local.get 0
        local.get 19
        i32.add
        local.set 20
        local.get 4
        local.get 20
        i32.store offset=20
        i32.const 1049730
        local.set 21
        i32.const 3
        local.set 22
        i32.const 20
        local.set 23
        local.get 4
        local.get 23
        i32.add
        local.set 24
        local.get 24
        local.set 25
        i32.const 1049736
        local.set 26
        local.get 1
        local.get 21
        local.get 22
        local.get 25
        local.get 26
        call $_ZN4core3fmt9Formatter25debug_tuple_field1_finish17h60a7f9de909b7316E
        local.set 27
        i32.const 1
        local.set 28
        local.get 27
        local.get 28
        i32.and
        local.set 29
        local.get 4
        local.get 29
        i32.store8 offset=15
      end
      local.get 4
      i32.load8_u offset=15
      local.set 30
      i32.const 1
      local.set 31
      local.get 30
      local.get 31
      i32.and
      local.set 32
      i32.const 32
      local.set 33
      local.get 4
      local.get 33
      i32.add
      local.set 34
      local.get 34
      global.set $__stack_pointer
      local.get 32
      return
    )
    (func $_ZN5alloc3vec12Vec$LT$T$GT$14from_raw_parts17h6e095f0370ac34f4E (;41;) (type 6) (param i32 i32 i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 4
      i32.const 32
      local.set 5
      local.get 4
      local.get 5
      i32.sub
      local.set 6
      local.get 6
      global.set $__stack_pointer
      local.get 6
      local.get 1
      i32.store offset=20
      local.get 6
      local.get 2
      i32.store offset=24
      local.get 6
      local.get 3
      i32.store offset=28
      i32.const 8
      local.set 7
      local.get 6
      local.get 7
      i32.add
      local.set 8
      local.get 8
      local.get 1
      local.get 3
      call $_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$17from_raw_parts_in17h31459d0e31f464c6E
      local.get 6
      i32.load offset=12
      local.set 9
      local.get 6
      i32.load offset=8
      local.set 10
      local.get 0
      local.get 10
      i32.store
      local.get 0
      local.get 9
      i32.store offset=4
      local.get 0
      local.get 2
      i32.store offset=8
      i32.const 32
      local.set 11
      local.get 6
      local.get 11
      i32.add
      local.set 12
      local.get 12
      global.set $__stack_pointer
      return
    )
    (func $_ZN66_$LT$core..option..Option$LT$T$GT$$u20$as$u20$core..fmt..Debug$GT$3fmt17hd92efa95cfc864f3E (;42;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 16
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=8
      local.get 4
      local.get 1
      i32.store offset=12
      local.get 0
      i32.load
      local.set 5
      i32.const -2147483648
      local.set 6
      local.get 5
      local.set 7
      local.get 6
      local.set 8
      local.get 7
      local.get 8
      i32.eq
      local.set 9
      i32.const 0
      local.set 10
      i32.const 1
      local.set 11
      i32.const 1
      local.set 12
      local.get 9
      local.get 12
      i32.and
      local.set 13
      local.get 10
      local.get 11
      local.get 13
      select
      local.set 14
      block ;; label = @1
        block ;; label = @2
          local.get 14
          br_if 0 (;@2;)
          i32.const 1049752
          local.set 15
          i32.const 4
          local.set 16
          local.get 1
          local.get 15
          local.get 16
          call $_ZN4core3fmt9Formatter9write_str17h8905e7a9d9d32ff1E
          local.set 17
          i32.const 1
          local.set 18
          local.get 17
          local.get 18
          i32.and
          local.set 19
          local.get 4
          local.get 19
          i32.store8 offset=3
          br 1 (;@1;)
        end
        local.get 4
        local.get 0
        i32.store offset=4
        i32.const 1049772
        local.set 20
        i32.const 4
        local.set 21
        i32.const 4
        local.set 22
        local.get 4
        local.get 22
        i32.add
        local.set 23
        local.get 23
        local.set 24
        i32.const 1049756
        local.set 25
        local.get 1
        local.get 20
        local.get 21
        local.get 24
        local.get 25
        call $_ZN4core3fmt9Formatter25debug_tuple_field1_finish17h60a7f9de909b7316E
        local.set 26
        i32.const 1
        local.set 27
        local.get 26
        local.get 27
        i32.and
        local.set 28
        local.get 4
        local.get 28
        i32.store8 offset=3
      end
      local.get 4
      i32.load8_u offset=3
      local.set 29
      i32.const 1
      local.set 30
      local.get 29
      local.get 30
      i32.and
      local.set 31
      i32.const 16
      local.set 32
      local.get 4
      local.get 32
      i32.add
      local.set 33
      local.get 33
      global.set $__stack_pointer
      local.get 31
      return
    )
    (func $_ZN4core3str21_$LT$impl$u20$str$GT$3len17h582e63060e713246E (;43;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 16
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      local.get 0
      i32.store offset=8
      local.get 4
      local.get 1
      i32.store offset=12
      local.get 1
      return
    )
    (func $_ZN94_$LT$component_example..Run$u20$as$u20$component_example..exports..test..guest..run..Guest$GT$5start17h2dd88ef91cd4a0f0E (;44;) (type 4)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 0
      i32.const 1072
      local.set 1
      local.get 0
      local.get 1
      i32.sub
      local.set 2
      local.get 2
      global.set $__stack_pointer
      i32.const 156
      local.set 3
      local.get 2
      local.get 3
      i32.add
      local.set 4
      local.get 4
      local.set 5
      i32.const 1
      local.set 6
      i32.const 1
      local.set 7
      local.get 6
      local.get 7
      i32.and
      local.set 8
      local.get 5
      local.get 8
      call $_ZN17component_example4test5guest4host13result_option17h4747fdc126c7a265E
      i32.const 156
      local.set 9
      local.get 2
      local.get 9
      i32.add
      local.set 10
      local.get 10
      local.set 11
      local.get 2
      local.get 11
      i32.store offset=900
      i32.const 3
      local.set 12
      local.get 2
      local.get 12
      i32.store offset=904
      i32.const 156
      local.set 13
      local.get 2
      local.get 13
      i32.add
      local.set 14
      local.get 14
      local.set 15
      local.get 2
      local.get 15
      i32.store offset=908
      i32.const 156
      local.set 16
      local.get 2
      local.get 16
      i32.add
      local.set 17
      local.get 17
      local.set 18
      local.get 2
      local.get 18
      i32.store offset=892
      i32.const 3
      local.set 19
      local.get 2
      local.get 19
      i32.store offset=896
      local.get 2
      i64.load offset=892 align=4
      local.set 20
      local.get 2
      local.get 20
      i64.store offset=144
      i32.const 136
      local.set 21
      local.get 2
      local.get 21
      i32.add
      local.set 22
      local.get 22
      local.set 23
      local.get 2
      i64.load offset=144 align=4
      local.set 24
      local.get 23
      local.get 24
      i64.store align=4
      i32.const 112
      local.set 25
      local.get 2
      local.get 25
      i32.add
      local.set 26
      local.get 26
      local.set 27
      i32.const 1049820
      local.set 28
      i32.const 136
      local.set 29
      local.get 2
      local.get 29
      i32.add
      local.set 30
      local.get 30
      local.set 31
      local.get 27
      local.get 28
      local.get 31
      call $_ZN4core3fmt9Arguments6new_v117h39b074a36c3a198bE
      i32.const 100
      local.set 32
      local.get 2
      local.get 32
      i32.add
      local.set 33
      local.get 33
      local.set 34
      i32.const 112
      local.set 35
      local.get 2
      local.get 35
      i32.add
      local.set 36
      local.get 36
      local.set 37
      local.get 34
      local.get 37
      call $_ZN5alloc3fmt6format17h8c05e2c2d7c5fc16E
      i32.const 156
      local.set 38
      local.get 2
      local.get 38
      i32.add
      local.set 39
      local.get 39
      local.set 40
      local.get 40
      call $_ZN4core3ptr70drop_in_place$LT$core..option..Option$LT$alloc..string..String$GT$$GT$17he19cfb52433c8da3E
      i32.const 8
      local.set 41
      i32.const 88
      local.set 42
      local.get 2
      local.get 42
      i32.add
      local.set 43
      local.get 43
      local.get 41
      i32.add
      local.set 44
      i32.const 100
      local.set 45
      local.get 2
      local.get 45
      i32.add
      local.set 46
      local.get 46
      local.get 41
      i32.add
      local.set 47
      local.get 47
      i32.load
      local.set 48
      local.get 44
      local.get 48
      i32.store
      local.get 2
      i64.load offset=100 align=4
      local.set 49
      local.get 2
      local.get 49
      i64.store offset=88
      i32.const 8
      local.set 50
      local.get 2
      local.get 50
      i32.add
      local.set 51
      i32.const 88
      local.set 52
      local.get 2
      local.get 52
      i32.add
      local.set 53
      local.get 51
      local.get 53
      call $_ZN65_$LT$alloc..string..String$u20$as$u20$core..ops..deref..Deref$GT$5deref17hd9f9c6da88ee9950E
      local.get 2
      i32.load offset=12
      local.set 54
      local.get 2
      i32.load offset=8
      local.set 55
      local.get 55
      local.get 54
      call $_ZN17component_example4test5guest4host3log17h8adef950ed24064bE
      i32.const 88
      local.set 56
      local.get 2
      local.get 56
      i32.add
      local.set 57
      local.get 57
      local.set 58
      local.get 58
      call $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17he3a070ea415b5a8dE
      i32.const 236
      local.set 59
      local.get 2
      local.get 59
      i32.add
      local.set 60
      local.get 60
      local.set 61
      i32.const 0
      local.set 62
      i32.const 1
      local.set 63
      local.get 62
      local.get 63
      i32.and
      local.set 64
      local.get 61
      local.get 64
      call $_ZN17component_example4test5guest4host13result_option17h4747fdc126c7a265E
      i32.const 236
      local.set 65
      local.get 2
      local.get 65
      i32.add
      local.set 66
      local.get 66
      local.set 67
      local.get 2
      local.get 67
      i32.store offset=880
      i32.const 3
      local.set 68
      local.get 2
      local.get 68
      i32.store offset=884
      i32.const 236
      local.set 69
      local.get 2
      local.get 69
      i32.add
      local.set 70
      local.get 70
      local.set 71
      local.get 2
      local.get 71
      i32.store offset=888
      i32.const 236
      local.set 72
      local.get 2
      local.get 72
      i32.add
      local.set 73
      local.get 73
      local.set 74
      local.get 2
      local.get 74
      i32.store offset=872
      i32.const 3
      local.set 75
      local.get 2
      local.get 75
      i32.store offset=876
      local.get 2
      i64.load offset=872 align=4
      local.set 76
      local.get 2
      local.get 76
      i64.store offset=224
      i32.const 216
      local.set 77
      local.get 2
      local.get 77
      i32.add
      local.set 78
      local.get 78
      local.set 79
      local.get 2
      i64.load offset=224 align=4
      local.set 80
      local.get 79
      local.get 80
      i64.store align=4
      i32.const 192
      local.set 81
      local.get 2
      local.get 81
      i32.add
      local.set 82
      local.get 82
      local.set 83
      i32.const 1049868
      local.set 84
      i32.const 216
      local.set 85
      local.get 2
      local.get 85
      i32.add
      local.set 86
      local.get 86
      local.set 87
      local.get 83
      local.get 84
      local.get 87
      call $_ZN4core3fmt9Arguments6new_v117h39b074a36c3a198bE
      i32.const 180
      local.set 88
      local.get 2
      local.get 88
      i32.add
      local.set 89
      local.get 89
      local.set 90
      i32.const 192
      local.set 91
      local.get 2
      local.get 91
      i32.add
      local.set 92
      local.get 92
      local.set 93
      local.get 90
      local.get 93
      call $_ZN5alloc3fmt6format17h8c05e2c2d7c5fc16E
      i32.const 236
      local.set 94
      local.get 2
      local.get 94
      i32.add
      local.set 95
      local.get 95
      local.set 96
      local.get 96
      call $_ZN4core3ptr70drop_in_place$LT$core..option..Option$LT$alloc..string..String$GT$$GT$17he19cfb52433c8da3E
      i32.const 8
      local.set 97
      i32.const 168
      local.set 98
      local.get 2
      local.get 98
      i32.add
      local.set 99
      local.get 99
      local.get 97
      i32.add
      local.set 100
      i32.const 180
      local.set 101
      local.get 2
      local.get 101
      i32.add
      local.set 102
      local.get 102
      local.get 97
      i32.add
      local.set 103
      local.get 103
      i32.load
      local.set 104
      local.get 100
      local.get 104
      i32.store
      local.get 2
      i64.load offset=180 align=4
      local.set 105
      local.get 2
      local.get 105
      i64.store offset=168
      i32.const 16
      local.set 106
      local.get 2
      local.get 106
      i32.add
      local.set 107
      i32.const 168
      local.set 108
      local.get 2
      local.get 108
      i32.add
      local.set 109
      local.get 107
      local.get 109
      call $_ZN65_$LT$alloc..string..String$u20$as$u20$core..ops..deref..Deref$GT$5deref17hd9f9c6da88ee9950E
      local.get 2
      i32.load offset=20
      local.set 110
      local.get 2
      i32.load offset=16
      local.set 111
      local.get 111
      local.get 110
      call $_ZN17component_example4test5guest4host3log17h8adef950ed24064bE
      i32.const 168
      local.set 112
      local.get 2
      local.get 112
      i32.add
      local.set 113
      local.get 113
      local.set 114
      local.get 114
      call $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17he3a070ea415b5a8dE
      i32.const 312
      local.set 115
      local.get 2
      local.get 115
      i32.add
      local.set 116
      local.get 116
      local.set 117
      i32.const 1
      local.set 118
      i32.const 1
      local.set 119
      local.get 118
      local.get 119
      i32.and
      local.set 120
      local.get 117
      local.get 120
      call $_ZN17component_example4test5guest4host13result_result17hde906f70cc48d572E
      i32.const 312
      local.set 121
      local.get 2
      local.get 121
      i32.add
      local.set 122
      local.get 122
      local.set 123
      local.get 2
      local.get 123
      i32.store offset=940
      i32.const 4
      local.set 124
      local.get 2
      local.get 124
      i32.store offset=944
      i32.const 312
      local.set 125
      local.get 2
      local.get 125
      i32.add
      local.set 126
      local.get 126
      local.set 127
      local.get 2
      local.get 127
      i32.store offset=948
      i32.const 312
      local.set 128
      local.get 2
      local.get 128
      i32.add
      local.set 129
      local.get 129
      local.set 130
      local.get 2
      local.get 130
      i32.store offset=932
      i32.const 4
      local.set 131
      local.get 2
      local.get 131
      i32.store offset=936
      local.get 2
      i64.load offset=932 align=4
      local.set 132
      local.get 2
      local.get 132
      i64.store offset=304
      i32.const 296
      local.set 133
      local.get 2
      local.get 133
      i32.add
      local.set 134
      local.get 134
      local.set 135
      local.get 2
      i64.load offset=304 align=4
      local.set 136
      local.get 135
      local.get 136
      i64.store align=4
      i32.const 272
      local.set 137
      local.get 2
      local.get 137
      i32.add
      local.set 138
      local.get 138
      local.set 139
      i32.const 1049920
      local.set 140
      i32.const 296
      local.set 141
      local.get 2
      local.get 141
      i32.add
      local.set 142
      local.get 142
      local.set 143
      local.get 139
      local.get 140
      local.get 143
      call $_ZN4core3fmt9Arguments6new_v117h39b074a36c3a198bE
      i32.const 260
      local.set 144
      local.get 2
      local.get 144
      i32.add
      local.set 145
      local.get 145
      local.set 146
      i32.const 272
      local.set 147
      local.get 2
      local.get 147
      i32.add
      local.set 148
      local.get 148
      local.set 149
      local.get 146
      local.get 149
      call $_ZN5alloc3fmt6format17h8c05e2c2d7c5fc16E
      i32.const 312
      local.set 150
      local.get 2
      local.get 150
      i32.add
      local.set 151
      local.get 151
      local.set 152
      local.get 152
      call $_ZN4core3ptr94drop_in_place$LT$core..result..Result$LT$alloc..string..String$C$alloc..string..String$GT$$GT$17hb27e0d70cac2ed41E
      i32.const 8
      local.set 153
      i32.const 248
      local.set 154
      local.get 2
      local.get 154
      i32.add
      local.set 155
      local.get 155
      local.get 153
      i32.add
      local.set 156
      i32.const 260
      local.set 157
      local.get 2
      local.get 157
      i32.add
      local.set 158
      local.get 158
      local.get 153
      i32.add
      local.set 159
      local.get 159
      i32.load
      local.set 160
      local.get 156
      local.get 160
      i32.store
      local.get 2
      i64.load offset=260 align=4
      local.set 161
      local.get 2
      local.get 161
      i64.store offset=248
      i32.const 24
      local.set 162
      local.get 2
      local.get 162
      i32.add
      local.set 163
      i32.const 248
      local.set 164
      local.get 2
      local.get 164
      i32.add
      local.set 165
      local.get 163
      local.get 165
      call $_ZN65_$LT$alloc..string..String$u20$as$u20$core..ops..deref..Deref$GT$5deref17hd9f9c6da88ee9950E
      local.get 2
      i32.load offset=28
      local.set 166
      local.get 2
      i32.load offset=24
      local.set 167
      local.get 167
      local.get 166
      call $_ZN17component_example4test5guest4host3log17h8adef950ed24064bE
      i32.const 248
      local.set 168
      local.get 2
      local.get 168
      i32.add
      local.set 169
      local.get 169
      local.set 170
      local.get 170
      call $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17he3a070ea415b5a8dE
      i32.const 392
      local.set 171
      local.get 2
      local.get 171
      i32.add
      local.set 172
      local.get 172
      local.set 173
      i32.const 0
      local.set 174
      i32.const 1
      local.set 175
      local.get 174
      local.get 175
      i32.and
      local.set 176
      local.get 173
      local.get 176
      call $_ZN17component_example4test5guest4host13result_result17hde906f70cc48d572E
      i32.const 392
      local.set 177
      local.get 2
      local.get 177
      i32.add
      local.set 178
      local.get 178
      local.set 179
      local.get 2
      local.get 179
      i32.store offset=920
      i32.const 4
      local.set 180
      local.get 2
      local.get 180
      i32.store offset=924
      i32.const 392
      local.set 181
      local.get 2
      local.get 181
      i32.add
      local.set 182
      local.get 182
      local.set 183
      local.get 2
      local.get 183
      i32.store offset=928
      i32.const 392
      local.set 184
      local.get 2
      local.get 184
      i32.add
      local.set 185
      local.get 185
      local.set 186
      local.get 2
      local.get 186
      i32.store offset=912
      i32.const 4
      local.set 187
      local.get 2
      local.get 187
      i32.store offset=916
      local.get 2
      i64.load offset=912 align=4
      local.set 188
      local.get 2
      local.get 188
      i64.store offset=384
      i32.const 376
      local.set 189
      local.get 2
      local.get 189
      i32.add
      local.set 190
      local.get 190
      local.set 191
      local.get 2
      i64.load offset=384 align=4
      local.set 192
      local.get 191
      local.get 192
      i64.store align=4
      i32.const 352
      local.set 193
      local.get 2
      local.get 193
      i32.add
      local.set 194
      local.get 194
      local.set 195
      i32.const 1049972
      local.set 196
      i32.const 376
      local.set 197
      local.get 2
      local.get 197
      i32.add
      local.set 198
      local.get 198
      local.set 199
      local.get 195
      local.get 196
      local.get 199
      call $_ZN4core3fmt9Arguments6new_v117h39b074a36c3a198bE
      i32.const 340
      local.set 200
      local.get 2
      local.get 200
      i32.add
      local.set 201
      local.get 201
      local.set 202
      i32.const 352
      local.set 203
      local.get 2
      local.get 203
      i32.add
      local.set 204
      local.get 204
      local.set 205
      local.get 202
      local.get 205
      call $_ZN5alloc3fmt6format17h8c05e2c2d7c5fc16E
      i32.const 392
      local.set 206
      local.get 2
      local.get 206
      i32.add
      local.set 207
      local.get 207
      local.set 208
      local.get 208
      call $_ZN4core3ptr94drop_in_place$LT$core..result..Result$LT$alloc..string..String$C$alloc..string..String$GT$$GT$17hb27e0d70cac2ed41E
      i32.const 8
      local.set 209
      i32.const 328
      local.set 210
      local.get 2
      local.get 210
      i32.add
      local.set 211
      local.get 211
      local.get 209
      i32.add
      local.set 212
      i32.const 340
      local.set 213
      local.get 2
      local.get 213
      i32.add
      local.set 214
      local.get 214
      local.get 209
      i32.add
      local.set 215
      local.get 215
      i32.load
      local.set 216
      local.get 212
      local.get 216
      i32.store
      local.get 2
      i64.load offset=340 align=4
      local.set 217
      local.get 2
      local.get 217
      i64.store offset=328
      i32.const 32
      local.set 218
      local.get 2
      local.get 218
      i32.add
      local.set 219
      i32.const 328
      local.set 220
      local.get 2
      local.get 220
      i32.add
      local.set 221
      local.get 219
      local.get 221
      call $_ZN65_$LT$alloc..string..String$u20$as$u20$core..ops..deref..Deref$GT$5deref17hd9f9c6da88ee9950E
      local.get 2
      i32.load offset=36
      local.set 222
      local.get 2
      i32.load offset=32
      local.set 223
      local.get 223
      local.get 222
      call $_ZN17component_example4test5guest4host3log17h8adef950ed24064bE
      i32.const 328
      local.set 224
      local.get 2
      local.get 224
      i32.add
      local.set 225
      local.get 225
      local.set 226
      local.get 226
      call $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17he3a070ea415b5a8dE
      i32.const 476
      local.set 227
      local.get 2
      local.get 227
      i32.add
      local.set 228
      local.get 228
      local.set 229
      i32.const 1
      local.set 230
      i32.const 1
      local.set 231
      local.get 230
      local.get 231
      i32.and
      local.set 232
      local.get 229
      local.get 232
      call $_ZN17component_example4test5guest4host16result_result_ok17hd8699ce2421ad21dE
      i32.const 476
      local.set 233
      local.get 2
      local.get 233
      i32.add
      local.set 234
      local.get 234
      local.set 235
      local.get 2
      local.get 235
      i32.store offset=1020
      i32.const 5
      local.set 236
      local.get 2
      local.get 236
      i32.store offset=1024
      i32.const 476
      local.set 237
      local.get 2
      local.get 237
      i32.add
      local.set 238
      local.get 238
      local.set 239
      local.get 2
      local.get 239
      i32.store offset=1028
      i32.const 476
      local.set 240
      local.get 2
      local.get 240
      i32.add
      local.set 241
      local.get 241
      local.set 242
      local.get 2
      local.get 242
      i32.store offset=1012
      i32.const 5
      local.set 243
      local.get 2
      local.get 243
      i32.store offset=1016
      local.get 2
      i64.load offset=1012 align=4
      local.set 244
      local.get 2
      local.get 244
      i64.store offset=464
      i32.const 456
      local.set 245
      local.get 2
      local.get 245
      i32.add
      local.set 246
      local.get 246
      local.set 247
      local.get 2
      i64.load offset=464 align=4
      local.set 248
      local.get 247
      local.get 248
      i64.store align=4
      i32.const 432
      local.set 249
      local.get 2
      local.get 249
      i32.add
      local.set 250
      local.get 250
      local.set 251
      i32.const 1050028
      local.set 252
      i32.const 456
      local.set 253
      local.get 2
      local.get 253
      i32.add
      local.set 254
      local.get 254
      local.set 255
      local.get 251
      local.get 252
      local.get 255
      call $_ZN4core3fmt9Arguments6new_v117h39b074a36c3a198bE
      i32.const 420
      local.set 256
      local.get 2
      local.get 256
      i32.add
      local.set 257
      local.get 257
      local.set 258
      i32.const 432
      local.set 259
      local.get 2
      local.get 259
      i32.add
      local.set 260
      local.get 260
      local.set 261
      local.get 258
      local.get 261
      call $_ZN5alloc3fmt6format17h8c05e2c2d7c5fc16E
      i32.const 476
      local.set 262
      local.get 2
      local.get 262
      i32.add
      local.set 263
      local.get 263
      local.set 264
      local.get 264
      call $_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$alloc..string..String$C$$LP$$RP$$GT$$GT$17h640b2ff63ffdbd3fE
      i32.const 8
      local.set 265
      i32.const 408
      local.set 266
      local.get 2
      local.get 266
      i32.add
      local.set 267
      local.get 267
      local.get 265
      i32.add
      local.set 268
      i32.const 420
      local.set 269
      local.get 2
      local.get 269
      i32.add
      local.set 270
      local.get 270
      local.get 265
      i32.add
      local.set 271
      local.get 271
      i32.load
      local.set 272
      local.get 268
      local.get 272
      i32.store
      local.get 2
      i64.load offset=420 align=4
      local.set 273
      local.get 2
      local.get 273
      i64.store offset=408
      i32.const 40
      local.set 274
      local.get 2
      local.get 274
      i32.add
      local.set 275
      i32.const 408
      local.set 276
      local.get 2
      local.get 276
      i32.add
      local.set 277
      local.get 275
      local.get 277
      call $_ZN65_$LT$alloc..string..String$u20$as$u20$core..ops..deref..Deref$GT$5deref17hd9f9c6da88ee9950E
      local.get 2
      i32.load offset=44
      local.set 278
      local.get 2
      i32.load offset=40
      local.set 279
      local.get 279
      local.get 278
      call $_ZN17component_example4test5guest4host3log17h8adef950ed24064bE
      i32.const 408
      local.set 280
      local.get 2
      local.get 280
      i32.add
      local.set 281
      local.get 281
      local.set 282
      local.get 282
      call $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17he3a070ea415b5a8dE
      i32.const 556
      local.set 283
      local.get 2
      local.get 283
      i32.add
      local.set 284
      local.get 284
      local.set 285
      i32.const 0
      local.set 286
      i32.const 1
      local.set 287
      local.get 286
      local.get 287
      i32.and
      local.set 288
      local.get 285
      local.get 288
      call $_ZN17component_example4test5guest4host16result_result_ok17hd8699ce2421ad21dE
      i32.const 556
      local.set 289
      local.get 2
      local.get 289
      i32.add
      local.set 290
      local.get 290
      local.set 291
      local.get 2
      local.get 291
      i32.store offset=1000
      i32.const 5
      local.set 292
      local.get 2
      local.get 292
      i32.store offset=1004
      i32.const 556
      local.set 293
      local.get 2
      local.get 293
      i32.add
      local.set 294
      local.get 294
      local.set 295
      local.get 2
      local.get 295
      i32.store offset=1008
      i32.const 556
      local.set 296
      local.get 2
      local.get 296
      i32.add
      local.set 297
      local.get 297
      local.set 298
      local.get 2
      local.get 298
      i32.store offset=992
      i32.const 5
      local.set 299
      local.get 2
      local.get 299
      i32.store offset=996
      local.get 2
      i64.load offset=992 align=4
      local.set 300
      local.get 2
      local.get 300
      i64.store offset=544
      i32.const 536
      local.set 301
      local.get 2
      local.get 301
      i32.add
      local.set 302
      local.get 302
      local.set 303
      local.get 2
      i64.load offset=544 align=4
      local.set 304
      local.get 303
      local.get 304
      i64.store align=4
      i32.const 512
      local.set 305
      local.get 2
      local.get 305
      i32.add
      local.set 306
      local.get 306
      local.set 307
      i32.const 1050080
      local.set 308
      i32.const 536
      local.set 309
      local.get 2
      local.get 309
      i32.add
      local.set 310
      local.get 310
      local.set 311
      local.get 307
      local.get 308
      local.get 311
      call $_ZN4core3fmt9Arguments6new_v117h39b074a36c3a198bE
      i32.const 500
      local.set 312
      local.get 2
      local.get 312
      i32.add
      local.set 313
      local.get 313
      local.set 314
      i32.const 512
      local.set 315
      local.get 2
      local.get 315
      i32.add
      local.set 316
      local.get 316
      local.set 317
      local.get 314
      local.get 317
      call $_ZN5alloc3fmt6format17h8c05e2c2d7c5fc16E
      i32.const 556
      local.set 318
      local.get 2
      local.get 318
      i32.add
      local.set 319
      local.get 319
      local.set 320
      local.get 320
      call $_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$alloc..string..String$C$$LP$$RP$$GT$$GT$17h640b2ff63ffdbd3fE
      i32.const 8
      local.set 321
      i32.const 488
      local.set 322
      local.get 2
      local.get 322
      i32.add
      local.set 323
      local.get 323
      local.get 321
      i32.add
      local.set 324
      i32.const 500
      local.set 325
      local.get 2
      local.get 325
      i32.add
      local.set 326
      local.get 326
      local.get 321
      i32.add
      local.set 327
      local.get 327
      i32.load
      local.set 328
      local.get 324
      local.get 328
      i32.store
      local.get 2
      i64.load offset=500 align=4
      local.set 329
      local.get 2
      local.get 329
      i64.store offset=488
      i32.const 48
      local.set 330
      local.get 2
      local.get 330
      i32.add
      local.set 331
      i32.const 488
      local.set 332
      local.get 2
      local.get 332
      i32.add
      local.set 333
      local.get 331
      local.get 333
      call $_ZN65_$LT$alloc..string..String$u20$as$u20$core..ops..deref..Deref$GT$5deref17hd9f9c6da88ee9950E
      local.get 2
      i32.load offset=52
      local.set 334
      local.get 2
      i32.load offset=48
      local.set 335
      local.get 335
      local.get 334
      call $_ZN17component_example4test5guest4host3log17h8adef950ed24064bE
      i32.const 488
      local.set 336
      local.get 2
      local.get 336
      i32.add
      local.set 337
      local.get 337
      local.set 338
      local.get 338
      call $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17he3a070ea415b5a8dE
      i32.const 636
      local.set 339
      local.get 2
      local.get 339
      i32.add
      local.set 340
      local.get 340
      local.set 341
      i32.const 1
      local.set 342
      i32.const 1
      local.set 343
      local.get 342
      local.get 343
      i32.and
      local.set 344
      local.get 341
      local.get 344
      call $_ZN17component_example4test5guest4host17result_result_err17h062c169bf525362aE
      i32.const 636
      local.set 345
      local.get 2
      local.get 345
      i32.add
      local.set 346
      local.get 346
      local.set 347
      local.get 2
      local.get 347
      i32.store offset=980
      i32.const 6
      local.set 348
      local.get 2
      local.get 348
      i32.store offset=984
      i32.const 636
      local.set 349
      local.get 2
      local.get 349
      i32.add
      local.set 350
      local.get 350
      local.set 351
      local.get 2
      local.get 351
      i32.store offset=988
      i32.const 636
      local.set 352
      local.get 2
      local.get 352
      i32.add
      local.set 353
      local.get 353
      local.set 354
      local.get 2
      local.get 354
      i32.store offset=972
      i32.const 6
      local.set 355
      local.get 2
      local.get 355
      i32.store offset=976
      local.get 2
      i64.load offset=972 align=4
      local.set 356
      local.get 2
      local.get 356
      i64.store offset=624
      i32.const 616
      local.set 357
      local.get 2
      local.get 357
      i32.add
      local.set 358
      local.get 358
      local.set 359
      local.get 2
      i64.load offset=624 align=4
      local.set 360
      local.get 359
      local.get 360
      i64.store align=4
      i32.const 592
      local.set 361
      local.get 2
      local.get 361
      i32.add
      local.set 362
      local.get 362
      local.set 363
      i32.const 1050132
      local.set 364
      i32.const 616
      local.set 365
      local.get 2
      local.get 365
      i32.add
      local.set 366
      local.get 366
      local.set 367
      local.get 363
      local.get 364
      local.get 367
      call $_ZN4core3fmt9Arguments6new_v117h39b074a36c3a198bE
      i32.const 580
      local.set 368
      local.get 2
      local.get 368
      i32.add
      local.set 369
      local.get 369
      local.set 370
      i32.const 592
      local.set 371
      local.get 2
      local.get 371
      i32.add
      local.set 372
      local.get 372
      local.set 373
      local.get 370
      local.get 373
      call $_ZN5alloc3fmt6format17h8c05e2c2d7c5fc16E
      i32.const 636
      local.set 374
      local.get 2
      local.get 374
      i32.add
      local.set 375
      local.get 375
      local.set 376
      local.get 376
      call $_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$$LP$$RP$$C$alloc..string..String$GT$$GT$17h747b2ded2b20ac66E
      i32.const 8
      local.set 377
      i32.const 568
      local.set 378
      local.get 2
      local.get 378
      i32.add
      local.set 379
      local.get 379
      local.get 377
      i32.add
      local.set 380
      i32.const 580
      local.set 381
      local.get 2
      local.get 381
      i32.add
      local.set 382
      local.get 382
      local.get 377
      i32.add
      local.set 383
      local.get 383
      i32.load
      local.set 384
      local.get 380
      local.get 384
      i32.store
      local.get 2
      i64.load offset=580 align=4
      local.set 385
      local.get 2
      local.get 385
      i64.store offset=568
      i32.const 56
      local.set 386
      local.get 2
      local.get 386
      i32.add
      local.set 387
      i32.const 568
      local.set 388
      local.get 2
      local.get 388
      i32.add
      local.set 389
      local.get 387
      local.get 389
      call $_ZN65_$LT$alloc..string..String$u20$as$u20$core..ops..deref..Deref$GT$5deref17hd9f9c6da88ee9950E
      local.get 2
      i32.load offset=60
      local.set 390
      local.get 2
      i32.load offset=56
      local.set 391
      local.get 391
      local.get 390
      call $_ZN17component_example4test5guest4host3log17h8adef950ed24064bE
      i32.const 568
      local.set 392
      local.get 2
      local.get 392
      i32.add
      local.set 393
      local.get 393
      local.set 394
      local.get 394
      call $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17he3a070ea415b5a8dE
      i32.const 716
      local.set 395
      local.get 2
      local.get 395
      i32.add
      local.set 396
      local.get 396
      local.set 397
      i32.const 0
      local.set 398
      i32.const 1
      local.set 399
      local.get 398
      local.get 399
      i32.and
      local.set 400
      local.get 397
      local.get 400
      call $_ZN17component_example4test5guest4host17result_result_err17h062c169bf525362aE
      i32.const 716
      local.set 401
      local.get 2
      local.get 401
      i32.add
      local.set 402
      local.get 402
      local.set 403
      local.get 2
      local.get 403
      i32.store offset=960
      i32.const 6
      local.set 404
      local.get 2
      local.get 404
      i32.store offset=964
      i32.const 716
      local.set 405
      local.get 2
      local.get 405
      i32.add
      local.set 406
      local.get 406
      local.set 407
      local.get 2
      local.get 407
      i32.store offset=968
      i32.const 716
      local.set 408
      local.get 2
      local.get 408
      i32.add
      local.set 409
      local.get 409
      local.set 410
      local.get 2
      local.get 410
      i32.store offset=952
      i32.const 6
      local.set 411
      local.get 2
      local.get 411
      i32.store offset=956
      local.get 2
      i64.load offset=952 align=4
      local.set 412
      local.get 2
      local.get 412
      i64.store offset=704
      i32.const 696
      local.set 413
      local.get 2
      local.get 413
      i32.add
      local.set 414
      local.get 414
      local.set 415
      local.get 2
      i64.load offset=704 align=4
      local.set 416
      local.get 415
      local.get 416
      i64.store align=4
      i32.const 672
      local.set 417
      local.get 2
      local.get 417
      i32.add
      local.set 418
      local.get 418
      local.set 419
      i32.const 1050188
      local.set 420
      i32.const 696
      local.set 421
      local.get 2
      local.get 421
      i32.add
      local.set 422
      local.get 422
      local.set 423
      local.get 419
      local.get 420
      local.get 423
      call $_ZN4core3fmt9Arguments6new_v117h39b074a36c3a198bE
      i32.const 660
      local.set 424
      local.get 2
      local.get 424
      i32.add
      local.set 425
      local.get 425
      local.set 426
      i32.const 672
      local.set 427
      local.get 2
      local.get 427
      i32.add
      local.set 428
      local.get 428
      local.set 429
      local.get 426
      local.get 429
      call $_ZN5alloc3fmt6format17h8c05e2c2d7c5fc16E
      i32.const 716
      local.set 430
      local.get 2
      local.get 430
      i32.add
      local.set 431
      local.get 431
      local.set 432
      local.get 432
      call $_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$$LP$$RP$$C$alloc..string..String$GT$$GT$17h747b2ded2b20ac66E
      i32.const 8
      local.set 433
      i32.const 648
      local.set 434
      local.get 2
      local.get 434
      i32.add
      local.set 435
      local.get 435
      local.get 433
      i32.add
      local.set 436
      i32.const 660
      local.set 437
      local.get 2
      local.get 437
      i32.add
      local.set 438
      local.get 438
      local.get 433
      i32.add
      local.set 439
      local.get 439
      i32.load
      local.set 440
      local.get 436
      local.get 440
      i32.store
      local.get 2
      i64.load offset=660 align=4
      local.set 441
      local.get 2
      local.get 441
      i64.store offset=648
      i32.const 64
      local.set 442
      local.get 2
      local.get 442
      i32.add
      local.set 443
      i32.const 648
      local.set 444
      local.get 2
      local.get 444
      i32.add
      local.set 445
      local.get 443
      local.get 445
      call $_ZN65_$LT$alloc..string..String$u20$as$u20$core..ops..deref..Deref$GT$5deref17hd9f9c6da88ee9950E
      local.get 2
      i32.load offset=68
      local.set 446
      local.get 2
      i32.load offset=64
      local.set 447
      local.get 447
      local.get 446
      call $_ZN17component_example4test5guest4host3log17h8adef950ed24064bE
      i32.const 648
      local.set 448
      local.get 2
      local.get 448
      i32.add
      local.set 449
      local.get 449
      local.set 450
      local.get 450
      call $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17he3a070ea415b5a8dE
      i32.const 1
      local.set 451
      i32.const 1
      local.set 452
      local.get 451
      local.get 452
      i32.and
      local.set 453
      local.get 453
      call $_ZN17component_example4test5guest4host18result_result_none17h10d78c2b9efa93ebE
      local.set 454
      i32.const 1
      local.set 455
      local.get 454
      local.get 455
      i32.and
      local.set 456
      local.get 2
      local.get 456
      i32.store8 offset=799
      i32.const 799
      local.set 457
      local.get 2
      local.get 457
      i32.add
      local.set 458
      local.get 458
      local.set 459
      local.get 2
      local.get 459
      i32.store offset=1060
      i32.const 7
      local.set 460
      local.get 2
      local.get 460
      i32.store offset=1064
      i32.const 799
      local.set 461
      local.get 2
      local.get 461
      i32.add
      local.set 462
      local.get 462
      local.set 463
      local.get 2
      local.get 463
      i32.store offset=1068
      i32.const 799
      local.set 464
      local.get 2
      local.get 464
      i32.add
      local.set 465
      local.get 465
      local.set 466
      local.get 2
      local.get 466
      i32.store offset=1052
      i32.const 7
      local.set 467
      local.get 2
      local.get 467
      i32.store offset=1056
      local.get 2
      i64.load offset=1052 align=4
      local.set 468
      local.get 2
      local.get 468
      i64.store offset=784
      i32.const 776
      local.set 469
      local.get 2
      local.get 469
      i32.add
      local.set 470
      local.get 470
      local.set 471
      local.get 2
      i64.load offset=784 align=4
      local.set 472
      local.get 471
      local.get 472
      i64.store align=4
      i32.const 752
      local.set 473
      local.get 2
      local.get 473
      i32.add
      local.set 474
      local.get 474
      local.set 475
      i32.const 1050244
      local.set 476
      i32.const 776
      local.set 477
      local.get 2
      local.get 477
      i32.add
      local.set 478
      local.get 478
      local.set 479
      local.get 475
      local.get 476
      local.get 479
      call $_ZN4core3fmt9Arguments6new_v117h39b074a36c3a198bE
      i32.const 740
      local.set 480
      local.get 2
      local.get 480
      i32.add
      local.set 481
      local.get 481
      local.set 482
      i32.const 752
      local.set 483
      local.get 2
      local.get 483
      i32.add
      local.set 484
      local.get 484
      local.set 485
      local.get 482
      local.get 485
      call $_ZN5alloc3fmt6format17h8c05e2c2d7c5fc16E
      i32.const 8
      local.set 486
      i32.const 728
      local.set 487
      local.get 2
      local.get 487
      i32.add
      local.set 488
      local.get 488
      local.get 486
      i32.add
      local.set 489
      i32.const 740
      local.set 490
      local.get 2
      local.get 490
      i32.add
      local.set 491
      local.get 491
      local.get 486
      i32.add
      local.set 492
      local.get 492
      i32.load
      local.set 493
      local.get 489
      local.get 493
      i32.store
      local.get 2
      i64.load offset=740 align=4
      local.set 494
      local.get 2
      local.get 494
      i64.store offset=728
      i32.const 72
      local.set 495
      local.get 2
      local.get 495
      i32.add
      local.set 496
      i32.const 728
      local.set 497
      local.get 2
      local.get 497
      i32.add
      local.set 498
      local.get 496
      local.get 498
      call $_ZN65_$LT$alloc..string..String$u20$as$u20$core..ops..deref..Deref$GT$5deref17hd9f9c6da88ee9950E
      local.get 2
      i32.load offset=76
      local.set 499
      local.get 2
      i32.load offset=72
      local.set 500
      local.get 500
      local.get 499
      call $_ZN17component_example4test5guest4host3log17h8adef950ed24064bE
      i32.const 728
      local.set 501
      local.get 2
      local.get 501
      i32.add
      local.set 502
      local.get 502
      local.set 503
      local.get 503
      call $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17he3a070ea415b5a8dE
      i32.const 0
      local.set 504
      i32.const 1
      local.set 505
      local.get 504
      local.get 505
      i32.and
      local.set 506
      local.get 506
      call $_ZN17component_example4test5guest4host18result_result_none17h10d78c2b9efa93ebE
      local.set 507
      i32.const 1
      local.set 508
      local.get 507
      local.get 508
      i32.and
      local.set 509
      local.get 2
      local.get 509
      i32.store8 offset=871
      i32.const 871
      local.set 510
      local.get 2
      local.get 510
      i32.add
      local.set 511
      local.get 511
      local.set 512
      local.get 2
      local.get 512
      i32.store offset=1040
      i32.const 7
      local.set 513
      local.get 2
      local.get 513
      i32.store offset=1044
      i32.const 871
      local.set 514
      local.get 2
      local.get 514
      i32.add
      local.set 515
      local.get 515
      local.set 516
      local.get 2
      local.get 516
      i32.store offset=1048
      i32.const 871
      local.set 517
      local.get 2
      local.get 517
      i32.add
      local.set 518
      local.get 518
      local.set 519
      local.get 2
      local.get 519
      i32.store offset=1032
      i32.const 7
      local.set 520
      local.get 2
      local.get 520
      i32.store offset=1036
      local.get 2
      i64.load offset=1032 align=4
      local.set 521
      local.get 2
      local.get 521
      i64.store offset=856
      i32.const 848
      local.set 522
      local.get 2
      local.get 522
      i32.add
      local.set 523
      local.get 523
      local.set 524
      local.get 2
      i64.load offset=856 align=4
      local.set 525
      local.get 524
      local.get 525
      i64.store align=4
      i32.const 824
      local.set 526
      local.get 2
      local.get 526
      i32.add
      local.set 527
      local.get 527
      local.set 528
      i32.const 1050300
      local.set 529
      i32.const 848
      local.set 530
      local.get 2
      local.get 530
      i32.add
      local.set 531
      local.get 531
      local.set 532
      local.get 528
      local.get 529
      local.get 532
      call $_ZN4core3fmt9Arguments6new_v117h39b074a36c3a198bE
      i32.const 812
      local.set 533
      local.get 2
      local.get 533
      i32.add
      local.set 534
      local.get 534
      local.set 535
      i32.const 824
      local.set 536
      local.get 2
      local.get 536
      i32.add
      local.set 537
      local.get 537
      local.set 538
      local.get 535
      local.get 538
      call $_ZN5alloc3fmt6format17h8c05e2c2d7c5fc16E
      i32.const 8
      local.set 539
      i32.const 800
      local.set 540
      local.get 2
      local.get 540
      i32.add
      local.set 541
      local.get 541
      local.get 539
      i32.add
      local.set 542
      i32.const 812
      local.set 543
      local.get 2
      local.get 543
      i32.add
      local.set 544
      local.get 544
      local.get 539
      i32.add
      local.set 545
      local.get 545
      i32.load
      local.set 546
      local.get 542
      local.get 546
      i32.store
      local.get 2
      i64.load offset=812 align=4
      local.set 547
      local.get 2
      local.get 547
      i64.store offset=800
      i32.const 80
      local.set 548
      local.get 2
      local.get 548
      i32.add
      local.set 549
      i32.const 800
      local.set 550
      local.get 2
      local.get 550
      i32.add
      local.set 551
      local.get 549
      local.get 551
      call $_ZN65_$LT$alloc..string..String$u20$as$u20$core..ops..deref..Deref$GT$5deref17hd9f9c6da88ee9950E
      local.get 2
      i32.load offset=84
      local.set 552
      local.get 2
      i32.load offset=80
      local.set 553
      local.get 553
      local.get 552
      call $_ZN17component_example4test5guest4host3log17h8adef950ed24064bE
      i32.const 800
      local.set 554
      local.get 2
      local.get 554
      i32.add
      local.set 555
      local.get 555
      local.set 556
      local.get 556
      call $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17he3a070ea415b5a8dE
      i32.const 1072
      local.set 557
      local.get 2
      local.get 557
      i32.add
      local.set 558
      local.get 558
      global.set $__stack_pointer
      return
    )
    (func $_ZN4core10intrinsics19copy_nonoverlapping18precondition_check17h7a208a03a2265411E (;45;) (type 8) (param i32 i32 i32 i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 5
      i32.const 96
      local.set 6
      local.get 5
      local.get 6
      i32.sub
      local.set 7
      local.get 7
      global.set $__stack_pointer
      i32.const 1050352
      local.set 8
      local.get 7
      local.get 8
      i32.store
      i32.const 1050352
      local.set 9
      local.get 7
      local.get 9
      i32.store offset=4
      local.get 7
      local.get 0
      i32.store offset=56
      local.get 7
      local.get 1
      i32.store offset=60
      local.get 7
      local.get 2
      i32.store offset=64
      local.get 7
      local.get 3
      i32.store offset=68
      local.get 7
      local.get 4
      i32.store offset=72
      local.get 7
      local.get 0
      i32.store offset=76
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            local.get 0
            br_if 0 (;@3;)
            br 1 (;@2;)
          end
          local.get 3
          i32.popcnt
          local.set 10
          local.get 7
          local.get 10
          i32.store offset=80
          local.get 7
          i32.load offset=80
          local.set 11
          i32.const 1
          local.set 12
          local.get 11
          local.set 13
          local.get 12
          local.set 14
          local.get 13
          local.get 14
          i32.eq
          local.set 15
          i32.const 1
          local.set 16
          local.get 15
          local.get 16
          i32.and
          local.set 17
          block ;; label = @3
            block ;; label = @4
              local.get 17
              i32.eqz
              br_if 0 (;@4;)
              i32.const 1
              local.set 18
              local.get 3
              local.get 18
              i32.sub
              local.set 19
              local.get 0
              local.get 19
              i32.and
              local.set 20
              local.get 20
              i32.eqz
              br_if 1 (;@3;)
              br 2 (;@2;)
            end
            i32.const 1050352
            local.set 21
            local.get 7
            local.get 21
            i32.store offset=8
            i32.const 1
            local.set 22
            local.get 7
            local.get 22
            i32.store offset=12
            i32.const 0
            local.set 23
            local.get 23
            i32.load offset=1050528
            local.set 24
            i32.const 0
            local.set 25
            local.get 25
            i32.load offset=1050532
            local.set 26
            local.get 7
            local.get 24
            i32.store offset=24
            local.get 7
            local.get 26
            i32.store offset=28
            i32.const 4
            local.set 27
            local.get 7
            local.get 27
            i32.store offset=16
            i32.const 0
            local.set 28
            local.get 7
            local.get 28
            i32.store offset=20
            i32.const 8
            local.set 29
            local.get 7
            local.get 29
            i32.add
            local.set 30
            local.get 30
            local.set 31
            i32.const 1050620
            local.set 32
            local.get 31
            local.get 32
            call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
            unreachable
          end
          local.get 7
          local.get 1
          i32.store offset=84
          local.get 7
          local.get 1
          i32.store offset=88
          block ;; label = @3
            block ;; label = @4
              local.get 1
              br_if 0 (;@4;)
              br 1 (;@3;)
            end
            local.get 3
            i32.popcnt
            local.set 33
            local.get 7
            local.get 33
            i32.store offset=92
            local.get 7
            i32.load offset=92
            local.set 34
            i32.const 1
            local.set 35
            local.get 34
            local.set 36
            local.get 35
            local.set 37
            local.get 36
            local.get 37
            i32.eq
            local.set 38
            i32.const 1
            local.set 39
            local.get 38
            local.get 39
            i32.and
            local.set 40
            block ;; label = @4
              block ;; label = @5
                local.get 40
                i32.eqz
                br_if 0 (;@5;)
                i32.const 1
                local.set 41
                local.get 3
                local.get 41
                i32.sub
                local.set 42
                local.get 1
                local.get 42
                i32.and
                local.set 43
                local.get 43
                i32.eqz
                br_if 1 (;@4;)
                br 2 (;@3;)
              end
              i32.const 1050352
              local.set 44
              local.get 7
              local.get 44
              i32.store offset=32
              i32.const 1
              local.set 45
              local.get 7
              local.get 45
              i32.store offset=36
              i32.const 0
              local.set 46
              local.get 46
              i32.load offset=1050528
              local.set 47
              i32.const 0
              local.set 48
              local.get 48
              i32.load offset=1050532
              local.set 49
              local.get 7
              local.get 47
              i32.store offset=48
              local.get 7
              local.get 49
              i32.store offset=52
              i32.const 4
              local.set 50
              local.get 7
              local.get 50
              i32.store offset=40
              i32.const 0
              local.set 51
              local.get 7
              local.get 51
              i32.store offset=44
              i32.const 32
              local.set 52
              local.get 7
              local.get 52
              i32.add
              local.set 53
              local.get 53
              local.set 54
              i32.const 1050620
              local.set 55
              local.get 54
              local.get 55
              call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
              unreachable
            end
            local.get 0
            local.get 1
            local.get 2
            local.get 4
            call $_ZN4core9ub_checks17is_nonoverlapping7runtime17h4cf1e21c86cbfe41E
            local.set 56
            i32.const 1
            local.set 57
            local.get 56
            local.get 57
            i32.and
            local.set 58
            block ;; label = @4
              local.get 58
              br_if 0 (;@4;)
              br 3 (;@1;)
            end
            i32.const 96
            local.set 59
            local.get 7
            local.get 59
            i32.add
            local.set 60
            local.get 60
            global.set $__stack_pointer
            return
          end
          br 1 (;@1;)
        end
      end
      i32.const 1050360
      local.set 61
      i32.const 166
      local.set 62
      local.get 61
      local.get 62
      call $_ZN4core9panicking14panic_nounwind17he2774a18f33c590cE
      unreachable
    )
    (func $_ZN45_$LT$$LP$$RP$$u20$as$u20$core..fmt..Debug$GT$3fmt17hbd1a05eae5c2387aE (;46;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 16
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=8
      local.get 4
      local.get 1
      i32.store offset=12
      i32.const 1050636
      local.set 5
      i32.const 2
      local.set 6
      local.get 1
      local.get 5
      local.get 6
      call $_ZN4core3fmt9Formatter3pad17hce9cc0d410ecbe47E
      local.set 7
      i32.const 1
      local.set 8
      local.get 7
      local.get 8
      i32.and
      local.set 9
      i32.const 16
      local.set 10
      local.get 4
      local.get 10
      i32.add
      local.set 11
      local.get 11
      global.set $__stack_pointer
      local.get 9
      return
    )
    (func $_ZN4core3num23_$LT$impl$u20$usize$GT$13unchecked_mul18precondition_check17h0769029f600eab28E (;47;) (type 0) (param i32 i32)
      (local i32 i32 i32 i64 i64 i64 i64 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 16
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store
      local.get 4
      local.get 1
      i32.store offset=4
      local.get 1
      i64.extend_i32_u
      local.set 5
      local.get 0
      i64.extend_i32_u
      local.set 6
      local.get 6
      local.get 5
      i64.mul
      local.set 7
      i64.const 32
      local.set 8
      local.get 7
      local.get 8
      i64.shr_u
      local.set 9
      local.get 9
      i32.wrap_i64
      local.set 10
      i32.const 0
      local.set 11
      local.get 10
      local.get 11
      i32.ne
      local.set 12
      local.get 7
      i32.wrap_i64
      local.set 13
      local.get 4
      local.get 13
      i32.store offset=8
      i32.const 1
      local.set 14
      local.get 12
      local.get 14
      i32.and
      local.set 15
      local.get 4
      local.get 15
      i32.store8 offset=15
      i32.const 1
      local.set 16
      local.get 12
      local.get 16
      i32.and
      local.set 17
      block ;; label = @1
        local.get 17
        br_if 0 (;@1;)
        i32.const 16
        local.set 18
        local.get 4
        local.get 18
        i32.add
        local.set 19
        local.get 19
        global.set $__stack_pointer
        return
      end
      i32.const 1050638
      local.set 20
      i32.const 69
      local.set 21
      local.get 20
      local.get 21
      call $_ZN4core9panicking14panic_nounwind17he2774a18f33c590cE
      unreachable
    )
    (func $test:guest/run#start (;48;) (type 4)
      call $_ZN11wit_bindgen2rt14run_ctors_once17h2d2a1af2565a62bcE
      call $_ZN94_$LT$component_example..Run$u20$as$u20$component_example..exports..test..guest..run..Guest$GT$5start17h2dd88ef91cd4a0f0E
      return
    )
    (func $_ZN17component_example4test5guest4host3log17h8adef950ed24064bE (;49;) (type 0) (param i32 i32)
      (local i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 32
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=8
      local.get 4
      local.get 1
      i32.store offset=12
      local.get 4
      local.get 0
      i32.store offset=24
      local.get 4
      local.get 1
      i32.store offset=28
      local.get 4
      local.get 0
      i32.store offset=16
      local.get 0
      local.get 1
      call $_ZN4core3str21_$LT$impl$u20$str$GT$3len17h582e63060e713246E
      local.set 5
      local.get 4
      local.get 5
      i32.store offset=20
      local.get 0
      local.get 5
      call $_ZN17component_example4test5guest4host3log10wit_import17hd588ebb1b530ec2aE
      i32.const 32
      local.set 6
      local.get 4
      local.get 6
      i32.add
      local.set 7
      local.get 7
      global.set $__stack_pointer
      return
    )
    (func $_ZN17component_example4test5guest4host13result_option17h4747fdc126c7a265E (;50;) (type 0) (param i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 80
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 1
      local.set 5
      local.get 4
      local.get 5
      i32.store8 offset=51
      i32.const 8
      local.set 6
      local.get 4
      local.get 6
      i32.add
      local.set 7
      local.get 7
      local.set 8
      local.get 4
      local.get 8
      i32.store offset=72
      i32.const 8
      local.set 9
      local.get 4
      local.get 9
      i32.add
      local.set 10
      local.get 10
      local.set 11
      local.get 4
      local.get 11
      i32.store offset=52
      local.get 1
      local.set 12
      block ;; label = @1
        block ;; label = @2
          local.get 12
          br_if 0 (;@2;)
          i32.const 0
          local.set 13
          local.get 4
          local.get 13
          i32.store offset=20
          br 1 (;@1;)
        end
        i32.const 1
        local.set 14
        local.get 4
        local.get 14
        i32.store offset=20
      end
      local.get 4
      i32.load offset=20
      local.set 15
      local.get 15
      local.get 11
      call $_ZN17component_example4test5guest4host13result_option10wit_import17ha1bb84f1c0c19d52E
      i32.const 0
      local.set 16
      local.get 11
      local.set 17
      i32.const 1
      local.set 18
      local.get 16
      local.get 18
      i32.and
      local.set 19
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                local.get 19
                br_if 0 (;@5;)
                local.get 17
                i32.load8_u
                local.set 20
                local.get 4
                local.get 20
                i32.store8 offset=79
                local.get 4
                local.get 20
                i32.store offset=56
                i32.const 1
                local.set 21
                local.get 20
                local.get 21
                i32.gt_u
                drop
                local.get 20
                br_table 2 (;@3;) 3 (;@2;) 1 (;@4;)
              end
              i32.const 1050720
              local.set 22
              local.get 22
              call $_ZN4core9panicking11panic_const24panic_const_add_overflow17h3b4e4f99c6b52bd0E
              unreachable
            end
            local.get 0
            call $_ZN11wit_bindgen2rt25invalid_enum_discriminant17hb7683f0fece93b0dE
            br 2 (;@1;)
          end
          i32.const -2147483648
          local.set 23
          local.get 0
          local.get 23
          i32.store
          br 1 (;@1;)
        end
        i32.const 4
        local.set 24
        local.get 11
        local.get 24
        i32.add
        local.set 25
        local.get 25
        local.get 11
        i32.lt_s
        local.set 26
        i32.const 1
        local.set 27
        local.get 26
        local.get 27
        i32.and
        local.set 28
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  block ;; label = @7
                    block ;; label = @8
                      local.get 28
                      br_if 0 (;@8;)
                      i32.const 3
                      local.set 29
                      local.get 25
                      local.get 29
                      i32.and
                      local.set 30
                      local.get 30
                      i32.eqz
                      br_if 1 (;@7;)
                      br 2 (;@6;)
                    end
                    i32.const 1050720
                    local.set 31
                    local.get 31
                    call $_ZN4core9panicking11panic_const24panic_const_add_overflow17h3b4e4f99c6b52bd0E
                    unreachable
                  end
                  local.get 25
                  i32.load
                  local.set 32
                  local.get 4
                  local.get 32
                  i32.store offset=60
                  i32.const 8
                  local.set 33
                  local.get 11
                  local.get 33
                  i32.add
                  local.set 34
                  local.get 34
                  local.get 11
                  i32.lt_s
                  local.set 35
                  i32.const 1
                  local.set 36
                  local.get 35
                  local.get 36
                  i32.and
                  local.set 37
                  local.get 37
                  br_if 2 (;@4;)
                  br 1 (;@5;)
                end
                i32.const 4
                local.set 38
                i32.const 1050720
                local.set 39
                local.get 38
                local.get 25
                local.get 39
                call $_ZN4core9panicking36panic_misaligned_pointer_dereference17hb39f65062076de38E
                unreachable
              end
              i32.const 3
              local.set 40
              local.get 34
              local.get 40
              i32.and
              local.set 41
              local.get 41
              i32.eqz
              br_if 1 (;@3;)
              br 2 (;@2;)
            end
            i32.const 1050720
            local.set 42
            local.get 42
            call $_ZN4core9panicking11panic_const24panic_const_add_overflow17h3b4e4f99c6b52bd0E
            unreachable
          end
          local.get 34
          i32.load
          local.set 43
          local.get 4
          local.get 43
          i32.store offset=64
          local.get 4
          local.get 43
          i32.store offset=68
          i32.const 36
          local.set 44
          local.get 4
          local.get 44
          i32.add
          local.set 45
          local.get 45
          local.set 46
          local.get 46
          local.get 32
          local.get 43
          local.get 43
          call $_ZN5alloc3vec12Vec$LT$T$GT$14from_raw_parts17h6e095f0370ac34f4E
          i32.const 24
          local.set 47
          local.get 4
          local.get 47
          i32.add
          local.set 48
          local.get 48
          local.set 49
          i32.const 36
          local.set 50
          local.get 4
          local.get 50
          i32.add
          local.set 51
          local.get 51
          local.set 52
          local.get 49
          local.get 52
          call $_ZN11wit_bindgen2rt11string_lift17h09e85ffb5d1e1eadE
          local.get 4
          i64.load offset=24 align=4
          local.set 53
          local.get 0
          local.get 53
          i64.store align=4
          i32.const 8
          local.set 54
          local.get 0
          local.get 54
          i32.add
          local.set 55
          i32.const 24
          local.set 56
          local.get 4
          local.get 56
          i32.add
          local.set 57
          local.get 57
          local.get 54
          i32.add
          local.set 58
          local.get 58
          i32.load
          local.set 59
          local.get 55
          local.get 59
          i32.store
          br 1 (;@1;)
        end
        i32.const 4
        local.set 60
        i32.const 1050720
        local.set 61
        local.get 60
        local.get 34
        local.get 61
        call $_ZN4core9panicking36panic_misaligned_pointer_dereference17hb39f65062076de38E
        unreachable
      end
      i32.const 80
      local.set 62
      local.get 4
      local.get 62
      i32.add
      local.set 63
      local.get 63
      global.set $__stack_pointer
      return
    )
    (func $_ZN17component_example4test5guest4host13result_result17hde906f70cc48d572E (;51;) (type 0) (param i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 112
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 1
      local.set 5
      local.get 4
      local.get 5
      i32.store8 offset=71
      i32.const 4
      local.set 6
      local.get 4
      local.get 6
      i32.add
      local.set 7
      local.get 7
      local.set 8
      local.get 4
      local.get 8
      i32.store offset=104
      i32.const 4
      local.set 9
      local.get 4
      local.get 9
      i32.add
      local.set 10
      local.get 10
      local.set 11
      local.get 4
      local.get 11
      i32.store offset=72
      local.get 1
      local.set 12
      block ;; label = @1
        block ;; label = @2
          local.get 12
          br_if 0 (;@2;)
          i32.const 0
          local.set 13
          local.get 4
          local.get 13
          i32.store offset=16
          br 1 (;@1;)
        end
        i32.const 1
        local.set 14
        local.get 4
        local.get 14
        i32.store offset=16
      end
      local.get 4
      i32.load offset=16
      local.set 15
      local.get 15
      local.get 11
      call $_ZN17component_example4test5guest4host13result_result10wit_import17ha859731761289ad4E
      i32.const 0
      local.set 16
      local.get 11
      local.set 17
      i32.const 1
      local.set 18
      local.get 16
      local.get 18
      i32.and
      local.set 19
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  block ;; label = @7
                    block ;; label = @8
                      block ;; label = @9
                        block ;; label = @10
                          block ;; label = @11
                            block ;; label = @12
                              block ;; label = @13
                                block ;; label = @14
                                  block ;; label = @15
                                    block ;; label = @16
                                      block ;; label = @17
                                        block ;; label = @18
                                          block ;; label = @19
                                            block ;; label = @20
                                              block ;; label = @21
                                                local.get 19
                                                br_if 0 (;@21;)
                                                local.get 17
                                                i32.load8_u
                                                local.set 20
                                                local.get 4
                                                local.get 20
                                                i32.store8 offset=111
                                                local.get 4
                                                local.get 20
                                                i32.store offset=76
                                                i32.const 1
                                                local.set 21
                                                local.get 20
                                                local.get 21
                                                i32.gt_u
                                                drop
                                                local.get 20
                                                br_table 2 (;@19;) 3 (;@18;) 1 (;@20;)
                                              end
                                              i32.const 1050720
                                              local.set 22
                                              local.get 22
                                              call $_ZN4core9panicking11panic_const24panic_const_add_overflow17h3b4e4f99c6b52bd0E
                                              unreachable
                                            end
                                            local.get 0
                                            call $_ZN11wit_bindgen2rt25invalid_enum_discriminant17h41b3f6d22e6d50d1E
                                            br 18 (;@1;)
                                          end
                                          i32.const 4
                                          local.set 23
                                          local.get 11
                                          local.get 23
                                          i32.add
                                          local.set 24
                                          local.get 24
                                          local.get 11
                                          i32.lt_s
                                          local.set 25
                                          i32.const 1
                                          local.set 26
                                          local.get 25
                                          local.get 26
                                          i32.and
                                          local.set 27
                                          local.get 27
                                          br_if 2 (;@16;)
                                          br 1 (;@17;)
                                        end
                                        i32.const 4
                                        local.set 28
                                        local.get 11
                                        local.get 28
                                        i32.add
                                        local.set 29
                                        local.get 29
                                        local.get 11
                                        i32.lt_s
                                        local.set 30
                                        i32.const 1
                                        local.set 31
                                        local.get 30
                                        local.get 31
                                        i32.and
                                        local.set 32
                                        local.get 32
                                        br_if 9 (;@8;)
                                        br 8 (;@9;)
                                      end
                                      i32.const 3
                                      local.set 33
                                      local.get 24
                                      local.get 33
                                      i32.and
                                      local.set 34
                                      local.get 34
                                      i32.eqz
                                      br_if 1 (;@15;)
                                      br 2 (;@14;)
                                    end
                                    i32.const 1050720
                                    local.set 35
                                    local.get 35
                                    call $_ZN4core9panicking11panic_const24panic_const_add_overflow17h3b4e4f99c6b52bd0E
                                    unreachable
                                  end
                                  local.get 24
                                  i32.load
                                  local.set 36
                                  local.get 4
                                  local.get 36
                                  i32.store offset=80
                                  i32.const 8
                                  local.set 37
                                  local.get 11
                                  local.get 37
                                  i32.add
                                  local.set 38
                                  local.get 38
                                  local.get 11
                                  i32.lt_s
                                  local.set 39
                                  i32.const 1
                                  local.set 40
                                  local.get 39
                                  local.get 40
                                  i32.and
                                  local.set 41
                                  local.get 41
                                  br_if 2 (;@12;)
                                  br 1 (;@13;)
                                end
                                i32.const 4
                                local.set 42
                                i32.const 1050720
                                local.set 43
                                local.get 42
                                local.get 24
                                local.get 43
                                call $_ZN4core9panicking36panic_misaligned_pointer_dereference17hb39f65062076de38E
                                unreachable
                              end
                              i32.const 3
                              local.set 44
                              local.get 38
                              local.get 44
                              i32.and
                              local.set 45
                              local.get 45
                              i32.eqz
                              br_if 1 (;@11;)
                              br 2 (;@10;)
                            end
                            i32.const 1050720
                            local.set 46
                            local.get 46
                            call $_ZN4core9panicking11panic_const24panic_const_add_overflow17h3b4e4f99c6b52bd0E
                            unreachable
                          end
                          local.get 38
                          i32.load
                          local.set 47
                          local.get 4
                          local.get 47
                          i32.store offset=84
                          local.get 4
                          local.get 47
                          i32.store offset=88
                          i32.const 32
                          local.set 48
                          local.get 4
                          local.get 48
                          i32.add
                          local.set 49
                          local.get 49
                          local.set 50
                          local.get 50
                          local.get 36
                          local.get 47
                          local.get 47
                          call $_ZN5alloc3vec12Vec$LT$T$GT$14from_raw_parts17h6e095f0370ac34f4E
                          i32.const 20
                          local.set 51
                          local.get 4
                          local.get 51
                          i32.add
                          local.set 52
                          local.get 52
                          local.set 53
                          i32.const 32
                          local.set 54
                          local.get 4
                          local.get 54
                          i32.add
                          local.set 55
                          local.get 55
                          local.set 56
                          local.get 53
                          local.get 56
                          call $_ZN11wit_bindgen2rt11string_lift17h09e85ffb5d1e1eadE
                          i32.const 4
                          local.set 57
                          local.get 0
                          local.get 57
                          i32.add
                          local.set 58
                          local.get 4
                          i64.load offset=20 align=4
                          local.set 59
                          local.get 58
                          local.get 59
                          i64.store align=4
                          i32.const 8
                          local.set 60
                          local.get 58
                          local.get 60
                          i32.add
                          local.set 61
                          i32.const 20
                          local.set 62
                          local.get 4
                          local.get 62
                          i32.add
                          local.set 63
                          local.get 63
                          local.get 60
                          i32.add
                          local.set 64
                          local.get 64
                          i32.load
                          local.set 65
                          local.get 61
                          local.get 65
                          i32.store
                          i32.const 0
                          local.set 66
                          local.get 0
                          local.get 66
                          i32.store
                          br 9 (;@1;)
                        end
                        i32.const 4
                        local.set 67
                        i32.const 1050720
                        local.set 68
                        local.get 67
                        local.get 38
                        local.get 68
                        call $_ZN4core9panicking36panic_misaligned_pointer_dereference17hb39f65062076de38E
                        unreachable
                      end
                      i32.const 3
                      local.set 69
                      local.get 29
                      local.get 69
                      i32.and
                      local.set 70
                      local.get 70
                      i32.eqz
                      br_if 1 (;@7;)
                      br 2 (;@6;)
                    end
                    i32.const 1050720
                    local.set 71
                    local.get 71
                    call $_ZN4core9panicking11panic_const24panic_const_add_overflow17h3b4e4f99c6b52bd0E
                    unreachable
                  end
                  local.get 29
                  i32.load
                  local.set 72
                  local.get 4
                  local.get 72
                  i32.store offset=92
                  i32.const 8
                  local.set 73
                  local.get 11
                  local.get 73
                  i32.add
                  local.set 74
                  local.get 74
                  local.get 11
                  i32.lt_s
                  local.set 75
                  i32.const 1
                  local.set 76
                  local.get 75
                  local.get 76
                  i32.and
                  local.set 77
                  local.get 77
                  br_if 2 (;@4;)
                  br 1 (;@5;)
                end
                i32.const 4
                local.set 78
                i32.const 1050720
                local.set 79
                local.get 78
                local.get 29
                local.get 79
                call $_ZN4core9panicking36panic_misaligned_pointer_dereference17hb39f65062076de38E
                unreachable
              end
              i32.const 3
              local.set 80
              local.get 74
              local.get 80
              i32.and
              local.set 81
              local.get 81
              i32.eqz
              br_if 1 (;@3;)
              br 2 (;@2;)
            end
            i32.const 1050720
            local.set 82
            local.get 82
            call $_ZN4core9panicking11panic_const24panic_const_add_overflow17h3b4e4f99c6b52bd0E
            unreachable
          end
          local.get 74
          i32.load
          local.set 83
          local.get 4
          local.get 83
          i32.store offset=96
          local.get 4
          local.get 83
          i32.store offset=100
          i32.const 56
          local.set 84
          local.get 4
          local.get 84
          i32.add
          local.set 85
          local.get 85
          local.set 86
          local.get 86
          local.get 72
          local.get 83
          local.get 83
          call $_ZN5alloc3vec12Vec$LT$T$GT$14from_raw_parts17h6e095f0370ac34f4E
          i32.const 44
          local.set 87
          local.get 4
          local.get 87
          i32.add
          local.set 88
          local.get 88
          local.set 89
          i32.const 56
          local.set 90
          local.get 4
          local.get 90
          i32.add
          local.set 91
          local.get 91
          local.set 92
          local.get 89
          local.get 92
          call $_ZN11wit_bindgen2rt11string_lift17h09e85ffb5d1e1eadE
          i32.const 4
          local.set 93
          local.get 0
          local.get 93
          i32.add
          local.set 94
          local.get 4
          i64.load offset=44 align=4
          local.set 95
          local.get 94
          local.get 95
          i64.store align=4
          i32.const 8
          local.set 96
          local.get 94
          local.get 96
          i32.add
          local.set 97
          i32.const 44
          local.set 98
          local.get 4
          local.get 98
          i32.add
          local.set 99
          local.get 99
          local.get 96
          i32.add
          local.set 100
          local.get 100
          i32.load
          local.set 101
          local.get 97
          local.get 101
          i32.store
          i32.const 1
          local.set 102
          local.get 0
          local.get 102
          i32.store
          br 1 (;@1;)
        end
        i32.const 4
        local.set 103
        i32.const 1050720
        local.set 104
        local.get 103
        local.get 74
        local.get 104
        call $_ZN4core9panicking36panic_misaligned_pointer_dereference17hb39f65062076de38E
        unreachable
      end
      i32.const 112
      local.set 105
      local.get 4
      local.get 105
      i32.add
      local.set 106
      local.get 106
      global.set $__stack_pointer
      return
    )
    (func $_ZN17component_example4test5guest4host16result_result_ok17hd8699ce2421ad21dE (;52;) (type 0) (param i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 80
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 1
      local.set 5
      local.get 4
      local.get 5
      i32.store8 offset=51
      i32.const 8
      local.set 6
      local.get 4
      local.get 6
      i32.add
      local.set 7
      local.get 7
      local.set 8
      local.get 4
      local.get 8
      i32.store offset=72
      i32.const 8
      local.set 9
      local.get 4
      local.get 9
      i32.add
      local.set 10
      local.get 10
      local.set 11
      local.get 4
      local.get 11
      i32.store offset=52
      local.get 1
      local.set 12
      block ;; label = @1
        block ;; label = @2
          local.get 12
          br_if 0 (;@2;)
          i32.const 0
          local.set 13
          local.get 4
          local.get 13
          i32.store offset=20
          br 1 (;@1;)
        end
        i32.const 1
        local.set 14
        local.get 4
        local.get 14
        i32.store offset=20
      end
      local.get 4
      i32.load offset=20
      local.set 15
      local.get 15
      local.get 11
      call $_ZN17component_example4test5guest4host16result_result_ok10wit_import17h9b1549c51fb040fbE
      i32.const 0
      local.set 16
      local.get 11
      local.set 17
      i32.const 1
      local.set 18
      local.get 16
      local.get 18
      i32.and
      local.set 19
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  block ;; label = @7
                    block ;; label = @8
                      block ;; label = @9
                        block ;; label = @10
                          block ;; label = @11
                            block ;; label = @12
                              block ;; label = @13
                                local.get 19
                                br_if 0 (;@13;)
                                local.get 17
                                i32.load8_u
                                local.set 20
                                local.get 4
                                local.get 20
                                i32.store8 offset=79
                                local.get 4
                                local.get 20
                                i32.store offset=56
                                i32.const 1
                                local.set 21
                                local.get 20
                                local.get 21
                                i32.gt_u
                                drop
                                local.get 20
                                br_table 2 (;@11;) 3 (;@10;) 1 (;@12;)
                              end
                              i32.const 1050720
                              local.set 22
                              local.get 22
                              call $_ZN4core9panicking11panic_const24panic_const_add_overflow17h3b4e4f99c6b52bd0E
                              unreachable
                            end
                            local.get 0
                            call $_ZN11wit_bindgen2rt25invalid_enum_discriminant17h3050578c96a32865E
                            br 10 (;@1;)
                          end
                          i32.const 4
                          local.set 23
                          local.get 11
                          local.get 23
                          i32.add
                          local.set 24
                          local.get 24
                          local.get 11
                          i32.lt_s
                          local.set 25
                          i32.const 1
                          local.set 26
                          local.get 25
                          local.get 26
                          i32.and
                          local.set 27
                          local.get 27
                          br_if 2 (;@8;)
                          br 1 (;@9;)
                        end
                        i32.const -2147483648
                        local.set 28
                        local.get 0
                        local.get 28
                        i32.store
                        br 8 (;@1;)
                      end
                      i32.const 3
                      local.set 29
                      local.get 24
                      local.get 29
                      i32.and
                      local.set 30
                      local.get 30
                      i32.eqz
                      br_if 1 (;@7;)
                      br 2 (;@6;)
                    end
                    i32.const 1050720
                    local.set 31
                    local.get 31
                    call $_ZN4core9panicking11panic_const24panic_const_add_overflow17h3b4e4f99c6b52bd0E
                    unreachable
                  end
                  local.get 24
                  i32.load
                  local.set 32
                  local.get 4
                  local.get 32
                  i32.store offset=60
                  i32.const 8
                  local.set 33
                  local.get 11
                  local.get 33
                  i32.add
                  local.set 34
                  local.get 34
                  local.get 11
                  i32.lt_s
                  local.set 35
                  i32.const 1
                  local.set 36
                  local.get 35
                  local.get 36
                  i32.and
                  local.set 37
                  local.get 37
                  br_if 2 (;@4;)
                  br 1 (;@5;)
                end
                i32.const 4
                local.set 38
                i32.const 1050720
                local.set 39
                local.get 38
                local.get 24
                local.get 39
                call $_ZN4core9panicking36panic_misaligned_pointer_dereference17hb39f65062076de38E
                unreachable
              end
              i32.const 3
              local.set 40
              local.get 34
              local.get 40
              i32.and
              local.set 41
              local.get 41
              i32.eqz
              br_if 1 (;@3;)
              br 2 (;@2;)
            end
            i32.const 1050720
            local.set 42
            local.get 42
            call $_ZN4core9panicking11panic_const24panic_const_add_overflow17h3b4e4f99c6b52bd0E
            unreachable
          end
          local.get 34
          i32.load
          local.set 43
          local.get 4
          local.get 43
          i32.store offset=64
          local.get 4
          local.get 43
          i32.store offset=68
          i32.const 36
          local.set 44
          local.get 4
          local.get 44
          i32.add
          local.set 45
          local.get 45
          local.set 46
          local.get 46
          local.get 32
          local.get 43
          local.get 43
          call $_ZN5alloc3vec12Vec$LT$T$GT$14from_raw_parts17h6e095f0370ac34f4E
          i32.const 24
          local.set 47
          local.get 4
          local.get 47
          i32.add
          local.set 48
          local.get 48
          local.set 49
          i32.const 36
          local.set 50
          local.get 4
          local.get 50
          i32.add
          local.set 51
          local.get 51
          local.set 52
          local.get 49
          local.get 52
          call $_ZN11wit_bindgen2rt11string_lift17h09e85ffb5d1e1eadE
          local.get 4
          i64.load offset=24 align=4
          local.set 53
          local.get 0
          local.get 53
          i64.store align=4
          i32.const 8
          local.set 54
          local.get 0
          local.get 54
          i32.add
          local.set 55
          i32.const 24
          local.set 56
          local.get 4
          local.get 56
          i32.add
          local.set 57
          local.get 57
          local.get 54
          i32.add
          local.set 58
          local.get 58
          i32.load
          local.set 59
          local.get 55
          local.get 59
          i32.store
          br 1 (;@1;)
        end
        i32.const 4
        local.set 60
        i32.const 1050720
        local.set 61
        local.get 60
        local.get 34
        local.get 61
        call $_ZN4core9panicking36panic_misaligned_pointer_dereference17hb39f65062076de38E
        unreachable
      end
      i32.const 80
      local.set 62
      local.get 4
      local.get 62
      i32.add
      local.set 63
      local.get 63
      global.set $__stack_pointer
      return
    )
    (func $_ZN17component_example4test5guest4host17result_result_err17h062c169bf525362aE (;53;) (type 0) (param i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 80
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 1
      local.set 5
      local.get 4
      local.get 5
      i32.store8 offset=51
      i32.const 8
      local.set 6
      local.get 4
      local.get 6
      i32.add
      local.set 7
      local.get 7
      local.set 8
      local.get 4
      local.get 8
      i32.store offset=72
      i32.const 8
      local.set 9
      local.get 4
      local.get 9
      i32.add
      local.set 10
      local.get 10
      local.set 11
      local.get 4
      local.get 11
      i32.store offset=52
      local.get 1
      local.set 12
      block ;; label = @1
        block ;; label = @2
          local.get 12
          br_if 0 (;@2;)
          i32.const 0
          local.set 13
          local.get 4
          local.get 13
          i32.store offset=20
          br 1 (;@1;)
        end
        i32.const 1
        local.set 14
        local.get 4
        local.get 14
        i32.store offset=20
      end
      local.get 4
      i32.load offset=20
      local.set 15
      local.get 15
      local.get 11
      call $_ZN17component_example4test5guest4host17result_result_err10wit_import17hebc19d5382075962E
      i32.const 0
      local.set 16
      local.get 11
      local.set 17
      i32.const 1
      local.set 18
      local.get 16
      local.get 18
      i32.and
      local.set 19
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                local.get 19
                br_if 0 (;@5;)
                local.get 17
                i32.load8_u
                local.set 20
                local.get 4
                local.get 20
                i32.store8 offset=79
                local.get 4
                local.get 20
                i32.store offset=56
                i32.const 1
                local.set 21
                local.get 20
                local.get 21
                i32.gt_u
                drop
                local.get 20
                br_table 2 (;@3;) 3 (;@2;) 1 (;@4;)
              end
              i32.const 1050720
              local.set 22
              local.get 22
              call $_ZN4core9panicking11panic_const24panic_const_add_overflow17h3b4e4f99c6b52bd0E
              unreachable
            end
            local.get 0
            call $_ZN11wit_bindgen2rt25invalid_enum_discriminant17h628a02db94cc7afbE
            br 2 (;@1;)
          end
          i32.const -2147483648
          local.set 23
          local.get 0
          local.get 23
          i32.store
          br 1 (;@1;)
        end
        i32.const 4
        local.set 24
        local.get 11
        local.get 24
        i32.add
        local.set 25
        local.get 25
        local.get 11
        i32.lt_s
        local.set 26
        i32.const 1
        local.set 27
        local.get 26
        local.get 27
        i32.and
        local.set 28
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  block ;; label = @7
                    block ;; label = @8
                      local.get 28
                      br_if 0 (;@8;)
                      i32.const 3
                      local.set 29
                      local.get 25
                      local.get 29
                      i32.and
                      local.set 30
                      local.get 30
                      i32.eqz
                      br_if 1 (;@7;)
                      br 2 (;@6;)
                    end
                    i32.const 1050720
                    local.set 31
                    local.get 31
                    call $_ZN4core9panicking11panic_const24panic_const_add_overflow17h3b4e4f99c6b52bd0E
                    unreachable
                  end
                  local.get 25
                  i32.load
                  local.set 32
                  local.get 4
                  local.get 32
                  i32.store offset=60
                  i32.const 8
                  local.set 33
                  local.get 11
                  local.get 33
                  i32.add
                  local.set 34
                  local.get 34
                  local.get 11
                  i32.lt_s
                  local.set 35
                  i32.const 1
                  local.set 36
                  local.get 35
                  local.get 36
                  i32.and
                  local.set 37
                  local.get 37
                  br_if 2 (;@4;)
                  br 1 (;@5;)
                end
                i32.const 4
                local.set 38
                i32.const 1050720
                local.set 39
                local.get 38
                local.get 25
                local.get 39
                call $_ZN4core9panicking36panic_misaligned_pointer_dereference17hb39f65062076de38E
                unreachable
              end
              i32.const 3
              local.set 40
              local.get 34
              local.get 40
              i32.and
              local.set 41
              local.get 41
              i32.eqz
              br_if 1 (;@3;)
              br 2 (;@2;)
            end
            i32.const 1050720
            local.set 42
            local.get 42
            call $_ZN4core9panicking11panic_const24panic_const_add_overflow17h3b4e4f99c6b52bd0E
            unreachable
          end
          local.get 34
          i32.load
          local.set 43
          local.get 4
          local.get 43
          i32.store offset=64
          local.get 4
          local.get 43
          i32.store offset=68
          i32.const 36
          local.set 44
          local.get 4
          local.get 44
          i32.add
          local.set 45
          local.get 45
          local.set 46
          local.get 46
          local.get 32
          local.get 43
          local.get 43
          call $_ZN5alloc3vec12Vec$LT$T$GT$14from_raw_parts17h6e095f0370ac34f4E
          i32.const 24
          local.set 47
          local.get 4
          local.get 47
          i32.add
          local.set 48
          local.get 48
          local.set 49
          i32.const 36
          local.set 50
          local.get 4
          local.get 50
          i32.add
          local.set 51
          local.get 51
          local.set 52
          local.get 49
          local.get 52
          call $_ZN11wit_bindgen2rt11string_lift17h09e85ffb5d1e1eadE
          local.get 4
          i64.load offset=24 align=4
          local.set 53
          local.get 0
          local.get 53
          i64.store align=4
          i32.const 8
          local.set 54
          local.get 0
          local.get 54
          i32.add
          local.set 55
          i32.const 24
          local.set 56
          local.get 4
          local.get 56
          i32.add
          local.set 57
          local.get 57
          local.get 54
          i32.add
          local.set 58
          local.get 58
          i32.load
          local.set 59
          local.get 55
          local.get 59
          i32.store
          br 1 (;@1;)
        end
        i32.const 4
        local.set 60
        i32.const 1050720
        local.set 61
        local.get 60
        local.get 34
        local.get 61
        call $_ZN4core9panicking36panic_misaligned_pointer_dereference17hb39f65062076de38E
        unreachable
      end
      i32.const 80
      local.set 62
      local.get 4
      local.get 62
      i32.add
      local.set 63
      local.get 63
      global.set $__stack_pointer
      return
    )
    (func $_ZN17component_example4test5guest4host18result_result_none17h10d78c2b9efa93ebE (;54;) (type 3) (param i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 16
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      local.get 3
      global.set $__stack_pointer
      local.get 0
      local.set 4
      local.get 3
      local.get 4
      i32.store8 offset=11
      local.get 0
      local.set 5
      block ;; label = @1
        block ;; label = @2
          local.get 5
          br_if 0 (;@2;)
          i32.const 0
          local.set 6
          local.get 3
          local.get 6
          i32.store offset=4
          br 1 (;@1;)
        end
        i32.const 1
        local.set 7
        local.get 3
        local.get 7
        i32.store offset=4
      end
      local.get 3
      i32.load offset=4
      local.set 8
      local.get 8
      call $_ZN17component_example4test5guest4host18result_result_none10wit_import17hd0c5bc84def681f4E
      local.set 9
      local.get 3
      local.get 9
      i32.store offset=12
      i32.const 1
      local.set 10
      local.get 9
      local.get 10
      i32.gt_u
      drop
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              local.get 9
              br_table 1 (;@3;) 2 (;@2;) 0 (;@4;)
            end
            call $_ZN11wit_bindgen2rt25invalid_enum_discriminant17h4743fc51e5e64321E
            local.set 11
            i32.const 1
            local.set 12
            local.get 11
            local.get 12
            i32.and
            local.set 13
            local.get 3
            local.get 13
            i32.store8 offset=3
            br 2 (;@1;)
          end
          i32.const 0
          local.set 14
          local.get 3
          local.get 14
          i32.store8 offset=3
          br 1 (;@1;)
        end
        i32.const 1
        local.set 15
        local.get 3
        local.get 15
        i32.store8 offset=3
      end
      local.get 3
      i32.load8_u offset=3
      local.set 16
      i32.const 1
      local.set 17
      local.get 16
      local.get 17
      i32.and
      local.set 18
      i32.const 16
      local.set 19
      local.get 3
      local.get 19
      i32.add
      local.set 20
      local.get 20
      global.set $__stack_pointer
      local.get 18
      return
    )
    (func $__rust_alloc (;55;) (type 2) (param i32 i32) (result i32)
      (local i32)
      local.get 0
      local.get 1
      call $__rdl_alloc
      local.set 2
      local.get 2
      return
    )
    (func $__rust_dealloc (;56;) (type 5) (param i32 i32 i32)
      local.get 0
      local.get 1
      local.get 2
      call $__rdl_dealloc
      return
    )
    (func $__rust_realloc (;57;) (type 7) (param i32 i32 i32 i32) (result i32)
      (local i32)
      local.get 0
      local.get 1
      local.get 2
      local.get 3
      call $__rdl_realloc
      local.set 4
      local.get 4
      return
    )
    (func $__rust_alloc_zeroed (;58;) (type 2) (param i32 i32) (result i32)
      (local i32)
      local.get 0
      local.get 1
      call $__rdl_alloc_zeroed
      local.set 2
      local.get 2
      return
    )
    (func $__rust_alloc_error_handler (;59;) (type 0) (param i32 i32)
      local.get 0
      local.get 1
      call $__rg_oom
      return
    )
    (func $_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h6ca084259debba81E (;60;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 16
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=8
      local.get 4
      local.get 1
      i32.store offset=12
      local.get 0
      i32.load
      local.set 5
      local.get 5
      local.get 1
      call $_ZN4core3fmt3num49_$LT$impl$u20$core..fmt..Debug$u20$for$u20$u8$GT$3fmt17h3d489bfa366bebedE
      local.set 6
      i32.const 1
      local.set 7
      local.get 6
      local.get 7
      i32.and
      local.set 8
      i32.const 16
      local.set 9
      local.get 4
      local.get 9
      i32.add
      local.set 10
      local.get 10
      global.set $__stack_pointer
      local.get 8
      return
    )
    (func $_ZN4core3fmt3num49_$LT$impl$u20$core..fmt..Debug$u20$for$u20$u8$GT$3fmt17h3d489bfa366bebedE (;61;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 16
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=8
      local.get 4
      local.get 1
      i32.store offset=12
      local.get 1
      i32.load offset=28
      local.set 5
      i32.const 16
      local.set 6
      local.get 5
      local.get 6
      i32.and
      local.set 7
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                local.get 7
                br_if 0 (;@5;)
                local.get 1
                i32.load offset=28
                local.set 8
                i32.const 32
                local.set 9
                local.get 8
                local.get 9
                i32.and
                local.set 10
                local.get 10
                i32.eqz
                br_if 1 (;@4;)
                br 2 (;@3;)
              end
              local.get 0
              local.get 1
              call $_ZN4core3fmt3num52_$LT$impl$u20$core..fmt..LowerHex$u20$for$u20$i8$GT$3fmt17he6877f4a3b3ac905E
              local.set 11
              i32.const 1
              local.set 12
              local.get 11
              local.get 12
              i32.and
              local.set 13
              local.get 4
              local.get 13
              i32.store8 offset=7
              br 3 (;@1;)
            end
            local.get 0
            local.get 1
            call $_ZN4core3fmt3num3imp51_$LT$impl$u20$core..fmt..Display$u20$for$u20$u8$GT$3fmt17h2c58f1fef76e6ba7E
            local.set 14
            i32.const 1
            local.set 15
            local.get 14
            local.get 15
            i32.and
            local.set 16
            local.get 4
            local.get 16
            i32.store8 offset=7
            br 1 (;@2;)
          end
          local.get 0
          local.get 1
          call $_ZN4core3fmt3num52_$LT$impl$u20$core..fmt..UpperHex$u20$for$u20$i8$GT$3fmt17h2f283451cba24807E
          local.set 17
          i32.const 1
          local.set 18
          local.get 17
          local.get 18
          i32.and
          local.set 19
          local.get 4
          local.get 19
          i32.store8 offset=7
        end
      end
      local.get 4
      i32.load8_u offset=7
      local.set 20
      i32.const 1
      local.set 21
      local.get 20
      local.get 21
      i32.and
      local.set 22
      i32.const 16
      local.set 23
      local.get 4
      local.get 23
      i32.add
      local.set 24
      local.get 24
      global.set $__stack_pointer
      local.get 22
      return
    )
    (func $_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h9c2833f5366bf8eaE (;62;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 16
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=8
      local.get 4
      local.get 1
      i32.store offset=12
      local.get 0
      i32.load
      local.set 5
      local.get 5
      local.get 1
      call $_ZN64_$LT$core..str..error..Utf8Error$u20$as$u20$core..fmt..Debug$GT$3fmt17h20d9d95780b19257E
      local.set 6
      i32.const 1
      local.set 7
      local.get 6
      local.get 7
      i32.and
      local.set 8
      i32.const 16
      local.set 9
      local.get 4
      local.get 9
      i32.add
      local.set 10
      local.get 10
      global.set $__stack_pointer
      local.get 8
      return
    )
    (func $_ZN64_$LT$core..str..error..Utf8Error$u20$as$u20$core..fmt..Debug$GT$3fmt17h20d9d95780b19257E (;63;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 16
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=8
      local.get 4
      local.get 1
      i32.store offset=12
      i32.const 4
      local.set 5
      local.get 0
      local.get 5
      i32.add
      local.set 6
      local.get 4
      local.get 6
      i32.store offset=4
      i32.const 1051460
      local.set 7
      i32.const 9
      local.set 8
      i32.const 1051469
      local.set 9
      i32.const 11
      local.set 10
      i32.const 1051428
      local.set 11
      i32.const 1051480
      local.set 12
      i32.const 4
      local.set 13
      local.get 4
      local.get 13
      i32.add
      local.set 14
      local.get 14
      local.set 15
      i32.const 1051444
      local.set 16
      local.get 1
      local.get 7
      local.get 8
      local.get 9
      local.get 10
      local.get 0
      local.get 11
      local.get 12
      local.get 8
      local.get 15
      local.get 16
      call $_ZN4core3fmt9Formatter26debug_struct_field2_finish17h73c9201d8434494dE
      local.set 17
      i32.const 1
      local.set 18
      local.get 17
      local.get 18
      i32.and
      local.set 19
      i32.const 16
      local.set 20
      local.get 4
      local.get 20
      i32.add
      local.set 21
      local.get 21
      global.set $__stack_pointer
      local.get 19
      return
    )
    (func $_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17ha6c7b58c4bb7816eE (;64;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 16
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=8
      local.get 4
      local.get 1
      i32.store offset=12
      local.get 0
      i32.load
      local.set 5
      local.get 5
      local.get 1
      call $_ZN4core3fmt3num52_$LT$impl$u20$core..fmt..Debug$u20$for$u20$usize$GT$3fmt17h6d706772c335a50cE
      local.set 6
      i32.const 1
      local.set 7
      local.get 6
      local.get 7
      i32.and
      local.set 8
      i32.const 16
      local.set 9
      local.get 4
      local.get 9
      i32.add
      local.set 10
      local.get 10
      global.set $__stack_pointer
      local.get 8
      return
    )
    (func $_ZN4core3fmt3num52_$LT$impl$u20$core..fmt..Debug$u20$for$u20$usize$GT$3fmt17h6d706772c335a50cE (;65;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 16
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=8
      local.get 4
      local.get 1
      i32.store offset=12
      local.get 1
      i32.load offset=28
      local.set 5
      i32.const 16
      local.set 6
      local.get 5
      local.get 6
      i32.and
      local.set 7
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                local.get 7
                br_if 0 (;@5;)
                local.get 1
                i32.load offset=28
                local.set 8
                i32.const 32
                local.set 9
                local.get 8
                local.get 9
                i32.and
                local.set 10
                local.get 10
                i32.eqz
                br_if 1 (;@4;)
                br 2 (;@3;)
              end
              local.get 0
              local.get 1
              call $_ZN4core3fmt3num53_$LT$impl$u20$core..fmt..LowerHex$u20$for$u20$i32$GT$3fmt17hc7fe287f99d5f9c4E
              local.set 11
              i32.const 1
              local.set 12
              local.get 11
              local.get 12
              i32.and
              local.set 13
              local.get 4
              local.get 13
              i32.store8 offset=7
              br 3 (;@1;)
            end
            local.get 0
            local.get 1
            call $_ZN4core3fmt3num3imp52_$LT$impl$u20$core..fmt..Display$u20$for$u20$u32$GT$3fmt17h0baf20a96941cbedE
            local.set 14
            i32.const 1
            local.set 15
            local.get 14
            local.get 15
            i32.and
            local.set 16
            local.get 4
            local.get 16
            i32.store8 offset=7
            br 1 (;@2;)
          end
          local.get 0
          local.get 1
          call $_ZN4core3fmt3num53_$LT$impl$u20$core..fmt..UpperHex$u20$for$u20$i32$GT$3fmt17hde5ba4f379c8f81cE
          local.set 17
          i32.const 1
          local.set 18
          local.get 17
          local.get 18
          i32.and
          local.set 19
          local.get 4
          local.get 19
          i32.store8 offset=7
        end
      end
      local.get 4
      i32.load8_u offset=7
      local.set 20
      i32.const 1
      local.set 21
      local.get 20
      local.get 21
      i32.and
      local.set 22
      i32.const 16
      local.set 23
      local.get 4
      local.get 23
      i32.add
      local.set 24
      local.get 24
      global.set $__stack_pointer
      local.get 22
      return
    )
    (func $_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17he31bc5b485e387bbE (;66;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 16
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=8
      local.get 4
      local.get 1
      i32.store offset=12
      local.get 0
      i32.load
      local.set 5
      local.get 5
      local.get 1
      call $_ZN66_$LT$core..option..Option$LT$T$GT$$u20$as$u20$core..fmt..Debug$GT$3fmt17hfbccc50c9575f5f3E
      local.set 6
      i32.const 1
      local.set 7
      local.get 6
      local.get 7
      i32.and
      local.set 8
      i32.const 16
      local.set 9
      local.get 4
      local.get 9
      i32.add
      local.set 10
      local.get 10
      global.set $__stack_pointer
      local.get 8
      return
    )
    (func $_ZN66_$LT$core..option..Option$LT$T$GT$$u20$as$u20$core..fmt..Debug$GT$3fmt17hfbccc50c9575f5f3E (;67;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 16
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=8
      local.get 4
      local.get 1
      i32.store offset=12
      local.get 0
      i32.load8_u
      local.set 5
      i32.const 1
      local.set 6
      local.get 5
      local.get 6
      i32.and
      local.set 7
      block ;; label = @1
        block ;; label = @2
          local.get 7
          br_if 0 (;@2;)
          i32.const 1051547
          local.set 8
          i32.const 4
          local.set 9
          local.get 1
          local.get 8
          local.get 9
          call $_ZN4core3fmt9Formatter9write_str17h8905e7a9d9d32ff1E
          local.set 10
          i32.const 1
          local.set 11
          local.get 10
          local.get 11
          i32.and
          local.set 12
          local.get 4
          local.get 12
          i32.store8 offset=3
          br 1 (;@1;)
        end
        i32.const 1
        local.set 13
        local.get 0
        local.get 13
        i32.add
        local.set 14
        local.get 4
        local.get 14
        i32.store offset=4
        i32.const 1051551
        local.set 15
        i32.const 4
        local.set 16
        i32.const 4
        local.set 17
        local.get 4
        local.get 17
        i32.add
        local.set 18
        local.get 18
        local.set 19
        i32.const 1050736
        local.set 20
        local.get 1
        local.get 15
        local.get 16
        local.get 19
        local.get 20
        call $_ZN4core3fmt9Formatter25debug_tuple_field1_finish17h60a7f9de909b7316E
        local.set 21
        i32.const 1
        local.set 22
        local.get 21
        local.get 22
        i32.and
        local.set 23
        local.get 4
        local.get 23
        i32.store8 offset=3
      end
      local.get 4
      i32.load8_u offset=3
      local.set 24
      i32.const 1
      local.set 25
      local.get 24
      local.get 25
      i32.and
      local.set 26
      i32.const 16
      local.set 27
      local.get 4
      local.get 27
      i32.add
      local.set 28
      local.get 28
      global.set $__stack_pointer
      local.get 26
      return
    )
    (func $_ZN48_$LT$$u5b$T$u5d$$u20$as$u20$core..fmt..Debug$GT$3fmt17hf3236d8eb423c151E (;68;) (type 1) (param i32 i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 3
      i32.const 48
      local.set 4
      local.get 3
      local.get 4
      i32.sub
      local.set 5
      local.get 5
      global.set $__stack_pointer
      local.get 5
      local.get 0
      i32.store offset=16
      local.get 5
      local.get 1
      i32.store offset=20
      local.get 5
      local.get 2
      i32.store offset=24
      i32.const 4
      local.set 6
      local.get 5
      local.get 6
      i32.add
      local.set 7
      local.get 7
      local.set 8
      local.get 8
      local.get 2
      call $_ZN4core3fmt9Formatter10debug_list17h53cbf25952dc9840E
      local.get 5
      local.get 1
      i32.store offset=28
      local.get 5
      local.get 0
      i32.store offset=32
      local.get 5
      local.get 1
      i32.store offset=36
      local.get 5
      local.get 0
      i32.store offset=40
      local.get 5
      local.get 0
      i32.store offset=44
      local.get 0
      local.get 1
      i32.add
      local.set 9
      local.get 5
      local.get 9
      i32.store offset=12
      local.get 5
      i32.load offset=12
      local.set 10
      i32.const 4
      local.set 11
      local.get 5
      local.get 11
      i32.add
      local.set 12
      local.get 12
      local.set 13
      local.get 13
      local.get 0
      local.get 10
      call $_ZN4core3fmt8builders9DebugList7entries17h6253e4b422ecd942E
      local.set 14
      local.get 14
      call $_ZN4core3fmt8builders9DebugList6finish17h184ffc6ee21bdc8bE
      local.set 15
      i32.const 1
      local.set 16
      local.get 15
      local.get 16
      i32.and
      local.set 17
      i32.const 48
      local.set 18
      local.get 5
      local.get 18
      i32.add
      local.set 19
      local.get 19
      global.set $__stack_pointer
      local.get 17
      return
    )
    (func $_ZN4core3fmt8builders9DebugList7entries17h6253e4b422ecd942E (;69;) (type 1) (param i32 i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 3
      i32.const 48
      local.set 4
      local.get 3
      local.get 4
      i32.sub
      local.set 5
      local.get 5
      global.set $__stack_pointer
      local.get 5
      local.get 0
      i32.store offset=36
      local.get 5
      local.get 1
      i32.store offset=40
      local.get 5
      local.get 2
      i32.store offset=44
      i32.const 8
      local.set 6
      local.get 5
      local.get 6
      i32.add
      local.set 7
      local.get 7
      local.get 1
      local.get 2
      call $_ZN63_$LT$I$u20$as$u20$core..iter..traits..collect..IntoIterator$GT$9into_iter17ha1611e094fe7bf10E
      local.get 5
      i32.load offset=12
      local.set 8
      local.get 5
      i32.load offset=8
      local.set 9
      local.get 5
      local.get 9
      i32.store offset=20
      local.get 5
      local.get 8
      i32.store offset=24
      loop (result i32) ;; label = @1
        i32.const 20
        local.set 10
        local.get 5
        local.get 10
        i32.add
        local.set 11
        local.get 11
        local.set 12
        local.get 12
        call $_ZN91_$LT$core..slice..iter..Iter$LT$T$GT$$u20$as$u20$core..iter..traits..iterator..Iterator$GT$4next17h8eeabbd44510a810E
        local.set 13
        local.get 5
        local.get 13
        i32.store offset=28
        local.get 5
        i32.load offset=28
        local.set 14
        i32.const 0
        local.set 15
        i32.const 1
        local.set 16
        local.get 16
        local.get 15
        local.get 14
        select
        local.set 17
        block ;; label = @2
          local.get 17
          br_if 0 (;@2;)
          i32.const 48
          local.set 18
          local.get 5
          local.get 18
          i32.add
          local.set 19
          local.get 19
          global.set $__stack_pointer
          local.get 0
          return
        end
        local.get 5
        i32.load offset=28
        local.set 20
        local.get 5
        local.get 20
        i32.store offset=32
        i32.const 32
        local.set 21
        local.get 5
        local.get 21
        i32.add
        local.set 22
        local.get 22
        local.set 23
        i32.const 1050736
        local.set 24
        local.get 0
        local.get 23
        local.get 24
        call $_ZN4core3fmt8builders8DebugSet5entry17hb6091ce1c7746e19E
        drop
        br 0 (;@1;)
      end
    )
    (func $_ZN63_$LT$I$u20$as$u20$core..iter..traits..collect..IntoIterator$GT$9into_iter17ha1611e094fe7bf10E (;70;) (type 5) (param i32 i32 i32)
      (local i32 i32 i32)
      global.get $__stack_pointer
      local.set 3
      i32.const 16
      local.set 4
      local.get 3
      local.get 4
      i32.sub
      local.set 5
      local.get 5
      local.get 1
      i32.store offset=8
      local.get 5
      local.get 2
      i32.store offset=12
      local.get 0
      local.get 2
      i32.store offset=4
      local.get 0
      local.get 1
      i32.store
      return
    )
    (func $_ZN91_$LT$core..slice..iter..Iter$LT$T$GT$$u20$as$u20$core..iter..traits..iterator..Iterator$GT$4next17h8eeabbd44510a810E (;71;) (type 3) (param i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 64
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      i32.const 1
      local.set 4
      local.get 3
      local.get 4
      i32.store
      i32.const 1
      local.set 5
      local.get 3
      local.get 5
      i32.store offset=4
      local.get 3
      local.get 0
      i32.store offset=24
      i32.const 4
      local.set 6
      local.get 0
      local.get 6
      i32.add
      local.set 7
      local.get 3
      local.get 7
      i32.store offset=28
      local.get 0
      i32.load offset=4
      local.set 8
      local.get 3
      local.get 8
      i32.store offset=16
      local.get 3
      local.get 0
      i32.store offset=32
      i32.const 16
      local.set 9
      local.get 3
      local.get 9
      i32.add
      local.set 10
      local.get 10
      local.set 11
      local.get 3
      local.get 11
      i32.store offset=36
      local.get 0
      i32.load
      local.set 12
      local.get 3
      local.get 12
      i32.store offset=40
      local.get 3
      i32.load offset=16
      local.set 13
      local.get 12
      local.set 14
      local.get 13
      local.set 15
      local.get 14
      local.get 15
      i32.eq
      local.set 16
      i32.const 1
      local.set 17
      local.get 16
      local.get 17
      i32.and
      local.set 18
      local.get 3
      local.get 18
      i32.store8 offset=15
      local.get 3
      i32.load8_u offset=15
      local.set 19
      i32.const 1
      local.set 20
      local.get 19
      local.get 20
      i32.and
      local.set 21
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            local.get 21
            br_if 0 (;@3;)
            local.get 0
            i32.load
            local.set 22
            local.get 3
            local.get 22
            i32.store offset=20
            br 1 (;@2;)
          end
          i32.const 0
          local.set 23
          local.get 3
          local.get 23
          i32.store offset=8
          br 1 (;@1;)
        end
        i32.const 4
        local.set 24
        local.get 0
        local.get 24
        i32.add
        local.set 25
        local.get 3
        local.get 25
        i32.store offset=44
        local.get 3
        local.get 25
        i32.store offset=48
        local.get 0
        i32.load
        local.set 26
        local.get 3
        local.get 26
        i32.store offset=52
        i32.const 1
        local.set 27
        local.get 26
        local.get 27
        i32.add
        local.set 28
        local.get 0
        local.get 28
        i32.store
        i32.const 20
        local.set 29
        local.get 3
        local.get 29
        i32.add
        local.set 30
        local.get 30
        local.set 31
        local.get 3
        local.get 31
        i32.store offset=56
        local.get 3
        i32.load offset=20
        local.set 32
        local.get 3
        local.get 32
        i32.store offset=60
        local.get 3
        local.get 32
        i32.store offset=8
      end
      local.get 3
      i32.load offset=8
      local.set 33
      local.get 33
      return
    )
    (func $_ZN4core3num23_$LT$impl$u20$usize$GT$13unchecked_mul18precondition_check17h2a301cdfefe2a7e9E (;72;) (type 0) (param i32 i32)
      (local i32 i32 i32 i64 i64 i64 i64 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 16
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store
      local.get 4
      local.get 1
      i32.store offset=4
      local.get 1
      i64.extend_i32_u
      local.set 5
      local.get 0
      i64.extend_i32_u
      local.set 6
      local.get 6
      local.get 5
      i64.mul
      local.set 7
      i64.const 32
      local.set 8
      local.get 7
      local.get 8
      i64.shr_u
      local.set 9
      local.get 9
      i32.wrap_i64
      local.set 10
      i32.const 0
      local.set 11
      local.get 10
      local.get 11
      i32.ne
      local.set 12
      local.get 7
      i32.wrap_i64
      local.set 13
      local.get 4
      local.get 13
      i32.store offset=8
      i32.const 1
      local.set 14
      local.get 12
      local.get 14
      i32.and
      local.set 15
      local.get 4
      local.get 15
      i32.store8 offset=15
      i32.const 1
      local.set 16
      local.get 12
      local.get 16
      i32.and
      local.set 17
      block ;; label = @1
        local.get 17
        br_if 0 (;@1;)
        i32.const 16
        local.set 18
        local.get 4
        local.get 18
        i32.add
        local.set 19
        local.get 19
        global.set $__stack_pointer
        return
      end
      i32.const 1050752
      local.set 20
      i32.const 69
      local.set 21
      local.get 20
      local.get 21
      call $_ZN4core9panicking14panic_nounwind17he2774a18f33c590cE
      unreachable
    )
    (func $_ZN4core3ptr13read_volatile18precondition_check17h046302b28e109536E (;73;) (type 0) (param i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 48
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      i32.const 1050864
      local.set 5
      local.get 4
      local.get 5
      i32.store offset=4
      local.get 4
      local.get 0
      i32.store offset=32
      local.get 4
      local.get 1
      i32.store offset=36
      local.get 4
      local.get 0
      i32.store offset=40
      block ;; label = @1
        block ;; label = @2
          local.get 0
          br_if 0 (;@2;)
          br 1 (;@1;)
        end
        local.get 1
        i32.popcnt
        local.set 6
        local.get 4
        local.get 6
        i32.store offset=44
        local.get 4
        i32.load offset=44
        local.set 7
        i32.const 1
        local.set 8
        local.get 7
        local.set 9
        local.get 8
        local.set 10
        local.get 9
        local.get 10
        i32.eq
        local.set 11
        i32.const 1
        local.set 12
        local.get 11
        local.get 12
        i32.and
        local.set 13
        block ;; label = @2
          block ;; label = @3
            local.get 13
            i32.eqz
            br_if 0 (;@3;)
            i32.const 1
            local.set 14
            local.get 1
            local.get 14
            i32.sub
            local.set 15
            local.get 0
            local.get 15
            i32.and
            local.set 16
            local.get 16
            i32.eqz
            br_if 1 (;@2;)
            br 2 (;@1;)
          end
          i32.const 1050864
          local.set 17
          local.get 4
          local.get 17
          i32.store offset=8
          i32.const 1
          local.set 18
          local.get 4
          local.get 18
          i32.store offset=12
          i32.const 0
          local.set 19
          local.get 19
          i32.load offset=1050984
          local.set 20
          i32.const 0
          local.set 21
          local.get 21
          i32.load offset=1050988
          local.set 22
          local.get 4
          local.get 20
          i32.store offset=24
          local.get 4
          local.get 22
          i32.store offset=28
          i32.const 4
          local.set 23
          local.get 4
          local.get 23
          i32.store offset=16
          i32.const 0
          local.set 24
          local.get 4
          local.get 24
          i32.store offset=20
          i32.const 8
          local.set 25
          local.get 4
          local.get 25
          i32.add
          local.set 26
          local.get 26
          local.set 27
          i32.const 1051076
          local.set 28
          local.get 27
          local.get 28
          call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
          unreachable
        end
        i32.const 48
        local.set 29
        local.get 4
        local.get 29
        i32.add
        local.set 30
        local.get 30
        global.set $__stack_pointer
        return
      end
      i32.const 1050872
      local.set 31
      i32.const 110
      local.set 32
      local.get 31
      local.get 32
      call $_ZN4core9panicking14panic_nounwind17he2774a18f33c590cE
      unreachable
    )
    (func $_ZN4core3ptr46drop_in_place$LT$alloc..vec..Vec$LT$u8$GT$$GT$17h57b6af354035ef20E (;74;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 16
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      local.get 3
      global.set $__stack_pointer
      local.get 3
      local.get 0
      i32.store offset=12
      local.get 0
      call $_ZN70_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17he0822737c6b55594E
      local.get 0
      call $_ZN4core3ptr53drop_in_place$LT$alloc..raw_vec..RawVec$LT$u8$GT$$GT$17h7ffa8a9abedbdc8fE
      i32.const 16
      local.set 4
      local.get 3
      local.get 4
      i32.add
      local.set 5
      local.get 5
      global.set $__stack_pointer
      return
    )
    (func $_ZN70_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17he0822737c6b55594E (;75;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 32
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      local.get 3
      local.get 0
      i32.store offset=12
      local.get 3
      local.get 0
      i32.store offset=16
      local.get 0
      i32.load offset=4
      local.set 4
      local.get 3
      local.get 4
      i32.store offset=20
      local.get 3
      local.get 4
      i32.store offset=24
      local.get 0
      i32.load offset=8
      local.set 5
      local.get 3
      local.get 5
      i32.store offset=28
      i32.const 0
      local.set 6
      local.get 3
      local.get 6
      i32.store offset=8
      block ;; label = @1
        loop ;; label = @2
          local.get 3
          i32.load offset=8
          local.set 7
          local.get 7
          local.set 8
          local.get 5
          local.set 9
          local.get 8
          local.get 9
          i32.eq
          local.set 10
          i32.const 1
          local.set 11
          local.get 10
          local.get 11
          i32.and
          local.set 12
          local.get 12
          br_if 1 (;@1;)
          local.get 3
          i32.load offset=8
          local.set 13
          i32.const 1
          local.set 14
          local.get 13
          local.get 14
          i32.add
          local.set 15
          local.get 3
          local.get 15
          i32.store offset=8
          br 0 (;@2;)
        end
      end
      return
    )
    (func $_ZN4core3ptr53drop_in_place$LT$alloc..raw_vec..RawVec$LT$u8$GT$$GT$17h7ffa8a9abedbdc8fE (;76;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 16
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      local.get 3
      global.set $__stack_pointer
      local.get 3
      local.get 0
      i32.store offset=12
      local.get 0
      call $_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17hb10823387c9a77bdE
      i32.const 16
      local.set 4
      local.get 3
      local.get 4
      i32.add
      local.set 5
      local.get 5
      global.set $__stack_pointer
      return
    )
    (func $_ZN4core3ptr49drop_in_place$LT$alloc..string..FromUtf8Error$GT$17hec3a267b9bd19dd8E (;77;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 16
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      local.get 3
      global.set $__stack_pointer
      local.get 3
      local.get 0
      i32.store offset=12
      local.get 0
      call $_ZN4core3ptr46drop_in_place$LT$alloc..vec..Vec$LT$u8$GT$$GT$17h57b6af354035ef20E
      i32.const 16
      local.set 4
      local.get 3
      local.get 4
      i32.add
      local.set 5
      local.get 5
      global.set $__stack_pointer
      return
    )
    (func $_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17hb10823387c9a77bdE (;78;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 32
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      local.get 3
      global.set $__stack_pointer
      local.get 3
      local.get 0
      i32.store offset=16
      i32.const 4
      local.set 4
      local.get 3
      local.get 4
      i32.add
      local.set 5
      local.get 5
      local.set 6
      local.get 6
      local.get 0
      call $_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$14current_memory17h719515fd4baea9ecE
      local.get 3
      i32.load offset=8
      local.set 7
      i32.const 0
      local.set 8
      i32.const 1
      local.set 9
      local.get 9
      local.get 8
      local.get 7
      select
      local.set 10
      i32.const 1
      local.set 11
      local.get 10
      local.set 12
      local.get 11
      local.set 13
      local.get 12
      local.get 13
      i32.eq
      local.set 14
      i32.const 1
      local.set 15
      local.get 14
      local.get 15
      i32.and
      local.set 16
      block ;; label = @1
        local.get 16
        i32.eqz
        br_if 0 (;@1;)
        local.get 3
        i32.load offset=4
        local.set 17
        local.get 3
        local.get 17
        i32.store offset=20
        local.get 3
        i32.load offset=8
        local.set 18
        local.get 3
        i32.load offset=12
        local.set 19
        local.get 3
        local.get 18
        i32.store offset=24
        local.get 3
        local.get 19
        i32.store offset=28
        i32.const 8
        local.set 20
        local.get 0
        local.get 20
        i32.add
        local.set 21
        local.get 21
        local.get 17
        local.get 18
        local.get 19
        call $_ZN63_$LT$alloc..alloc..Global$u20$as$u20$core..alloc..Allocator$GT$10deallocate17h26f17aa909e649d7E
      end
      i32.const 32
      local.set 22
      local.get 3
      local.get 22
      i32.add
      local.set 23
      local.get 23
      global.set $__stack_pointer
      return
    )
    (func $_ZN4core3ptr7mut_ptr31_$LT$impl$u20$$BP$mut$u20$T$GT$7is_null17h7dd7c30dd534ee73E (;79;) (type 3) (param i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 1
      i32.const 16
      local.set 2
      local.get 1
      local.get 2
      i32.sub
      local.set 3
      local.get 3
      local.get 0
      i32.store offset=8
      local.get 3
      local.get 0
      i32.store offset=12
      i32.const 0
      local.set 4
      local.get 0
      local.set 5
      local.get 4
      local.set 6
      local.get 5
      local.get 6
      i32.eq
      local.set 7
      i32.const 1
      local.set 8
      local.get 7
      local.get 8
      i32.and
      local.set 9
      local.get 9
      return
    )
    (func $_ZN4core5alloc6layout6Layout25from_size_align_unchecked17h3f4cfcb7d6ae7554E (;80;) (type 5) (param i32 i32 i32)
      (local i32 i32 i32)
      global.get $__stack_pointer
      local.set 3
      i32.const 16
      local.set 4
      local.get 3
      local.get 4
      i32.sub
      local.set 5
      local.get 5
      local.get 1
      i32.store offset=8
      local.get 5
      local.get 2
      i32.store offset=12
      local.get 0
      local.get 1
      i32.store offset=4
      local.get 0
      local.get 2
      i32.store
      return
    )
    (func $_ZN4core5slice3raw14from_raw_parts18precondition_check17h772c2c63148a002fE (;81;) (type 6) (param i32 i32 i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 4
      i32.const 64
      local.set 5
      local.get 4
      local.get 5
      i32.sub
      local.set 6
      local.get 6
      global.set $__stack_pointer
      i32.const 1050864
      local.set 7
      local.get 6
      local.get 7
      i32.store offset=4
      local.get 6
      local.get 0
      i32.store offset=36
      local.get 6
      local.get 1
      i32.store offset=40
      local.get 6
      local.get 2
      i32.store offset=44
      local.get 6
      local.get 3
      i32.store offset=48
      local.get 6
      local.get 0
      i32.store offset=52
      local.get 6
      local.get 0
      i32.store offset=56
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            local.get 0
            br_if 0 (;@3;)
            br 1 (;@2;)
          end
          local.get 2
          i32.popcnt
          local.set 8
          local.get 6
          local.get 8
          i32.store offset=60
          local.get 6
          i32.load offset=60
          local.set 9
          i32.const 1
          local.set 10
          local.get 9
          local.set 11
          local.get 10
          local.set 12
          local.get 11
          local.get 12
          i32.eq
          local.set 13
          i32.const 1
          local.set 14
          local.get 13
          local.get 14
          i32.and
          local.set 15
          block ;; label = @3
            block ;; label = @4
              local.get 15
              i32.eqz
              br_if 0 (;@4;)
              i32.const 1
              local.set 16
              local.get 2
              local.get 16
              i32.sub
              local.set 17
              local.get 0
              local.get 17
              i32.and
              local.set 18
              local.get 18
              i32.eqz
              br_if 1 (;@3;)
              br 2 (;@2;)
            end
            i32.const 1050864
            local.set 19
            local.get 6
            local.get 19
            i32.store offset=8
            i32.const 1
            local.set 20
            local.get 6
            local.get 20
            i32.store offset=12
            i32.const 0
            local.set 21
            local.get 21
            i32.load offset=1050984
            local.set 22
            i32.const 0
            local.set 23
            local.get 23
            i32.load offset=1050988
            local.set 24
            local.get 6
            local.get 22
            i32.store offset=24
            local.get 6
            local.get 24
            i32.store offset=28
            i32.const 4
            local.set 25
            local.get 6
            local.get 25
            i32.store offset=16
            i32.const 0
            local.set 26
            local.get 6
            local.get 26
            i32.store offset=20
            i32.const 8
            local.set 27
            local.get 6
            local.get 27
            i32.add
            local.set 28
            local.get 28
            local.set 29
            i32.const 1051076
            local.set 30
            local.get 29
            local.get 30
            call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
            unreachable
          end
          i32.const 0
          local.set 31
          local.get 1
          local.set 32
          local.get 31
          local.set 33
          local.get 32
          local.get 33
          i32.eq
          local.set 34
          block ;; label = @3
            block ;; label = @4
              local.get 1
              br_if 0 (;@4;)
              i32.const -1
              local.set 35
              local.get 6
              local.get 35
              i32.store offset=32
              br 1 (;@3;)
            end
            i32.const 1
            local.set 36
            local.get 34
            local.get 36
            i32.and
            local.set 37
            block ;; label = @4
              local.get 37
              br_if 0 (;@4;)
              i32.const 2147483647
              local.set 38
              local.get 38
              local.get 1
              i32.div_u
              local.set 39
              local.get 6
              local.get 39
              i32.store offset=32
              br 1 (;@3;)
            end
            i32.const 1051172
            local.set 40
            local.get 40
            call $_ZN4core9panicking11panic_const23panic_const_div_by_zero17hed37a86622bbbb5bE
            unreachable
          end
          local.get 6
          i32.load offset=32
          local.set 41
          local.get 3
          local.set 42
          local.get 41
          local.set 43
          local.get 42
          local.get 43
          i32.le_u
          local.set 44
          i32.const 1
          local.set 45
          local.get 44
          local.get 45
          i32.and
          local.set 46
          block ;; label = @3
            local.get 46
            br_if 0 (;@3;)
            br 2 (;@1;)
          end
          i32.const 64
          local.set 47
          local.get 6
          local.get 47
          i32.add
          local.set 48
          local.get 48
          global.set $__stack_pointer
          return
        end
      end
      i32.const 1051188
      local.set 49
      i32.const 162
      local.set 50
      local.get 49
      local.get 50
      call $_ZN4core9panicking14panic_nounwind17he2774a18f33c590cE
      unreachable
    )
    (func $_ZN4core9panicking13assert_failed17hdcd41e4667a8ffb5E (;82;) (type 8) (param i32 i32 i32 i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 5
      i32.const 16
      local.set 6
      local.get 5
      local.get 6
      i32.sub
      local.set 7
      local.get 7
      global.set $__stack_pointer
      local.get 7
      local.get 1
      i32.store offset=4
      local.get 7
      local.get 2
      i32.store offset=8
      local.get 7
      local.get 0
      i32.store8 offset=15
      i32.const 4
      local.set 8
      local.get 7
      local.get 8
      i32.add
      local.set 9
      local.get 9
      local.set 10
      i32.const 1051412
      local.set 11
      i32.const 8
      local.set 12
      local.get 7
      local.get 12
      i32.add
      local.set 13
      local.get 13
      local.set 14
      local.get 0
      local.get 10
      local.get 11
      local.get 14
      local.get 11
      local.get 3
      local.get 4
      call $_ZN4core9panicking19assert_failed_inner17h162ff0d740a1cd68E
      unreachable
    )
    (func $_ZN5alloc5alloc5alloc17hba97f008fb15ed2bE (;83;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 32
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      i32.const 1056353
      local.set 5
      local.get 4
      local.get 5
      i32.store
      local.get 4
      local.get 0
      i32.store offset=4
      local.get 4
      local.get 1
      i32.store offset=8
      i32.const 1056353
      local.set 6
      i32.const 1
      local.set 7
      local.get 6
      local.get 7
      call $_ZN4core3ptr13read_volatile18precondition_check17h046302b28e109536E
      i32.const 0
      local.set 8
      local.get 8
      i32.load8_u offset=1056353
      local.set 9
      local.get 4
      local.get 9
      i32.store8 offset=19
      i32.const 4
      local.set 10
      local.get 4
      local.get 10
      i32.add
      local.set 11
      local.get 11
      local.set 12
      local.get 4
      local.get 12
      i32.store offset=20
      local.get 4
      i32.load offset=8
      local.set 13
      i32.const 4
      local.set 14
      local.get 4
      local.get 14
      i32.add
      local.set 15
      local.get 15
      local.set 16
      local.get 4
      local.get 16
      i32.store offset=24
      local.get 4
      i32.load offset=4
      local.set 17
      local.get 4
      local.get 17
      i32.store offset=28
      local.get 4
      local.get 17
      i32.store offset=12
      local.get 4
      i32.load offset=12
      local.set 18
      local.get 13
      local.get 18
      call $__rust_alloc
      local.set 19
      i32.const 32
      local.set 20
      local.get 4
      local.get 20
      i32.add
      local.set 21
      local.get 21
      global.set $__stack_pointer
      local.get 19
      return
    )
    (func $_ZN5alloc5alloc7realloc17h67b0606c850eb3a0E (;84;) (type 7) (param i32 i32 i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 4
      i32.const 32
      local.set 5
      local.get 4
      local.get 5
      i32.sub
      local.set 6
      local.get 6
      global.set $__stack_pointer
      local.get 6
      local.get 1
      i32.store
      local.get 6
      local.get 2
      i32.store offset=4
      local.get 6
      local.get 0
      i32.store offset=12
      local.get 6
      local.get 3
      i32.store offset=16
      local.get 6
      local.set 7
      local.get 6
      local.get 7
      i32.store offset=20
      local.get 6
      i32.load offset=4
      local.set 8
      local.get 6
      local.set 9
      local.get 6
      local.get 9
      i32.store offset=24
      local.get 6
      i32.load
      local.set 10
      local.get 6
      local.get 10
      i32.store offset=28
      local.get 6
      local.get 10
      i32.store offset=8
      local.get 6
      i32.load offset=8
      local.set 11
      local.get 0
      local.get 8
      local.get 11
      local.get 3
      call $__rust_realloc
      local.set 12
      i32.const 32
      local.set 13
      local.get 6
      local.get 13
      i32.add
      local.set 14
      local.get 14
      global.set $__stack_pointer
      local.get 12
      return
    )
    (func $_ZN5alloc6string6String9from_utf817hf9c24943a7ea7d3dE (;85;) (type 0) (param i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i64 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 80
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 1
      i32.store offset=60
      local.get 4
      local.get 1
      i32.store offset=64
      local.get 1
      i32.load offset=4
      local.set 5
      local.get 4
      local.get 5
      i32.store offset=68
      local.get 4
      local.get 5
      i32.store offset=72
      local.get 1
      i32.load offset=8
      local.set 6
      local.get 4
      local.get 6
      i32.store offset=76
      i32.const 1
      local.set 7
      local.get 5
      local.get 7
      local.get 7
      local.get 6
      call $_ZN4core5slice3raw14from_raw_parts18precondition_check17h772c2c63148a002fE
      i32.const 4
      local.set 8
      local.get 4
      local.get 8
      i32.add
      local.set 9
      local.get 9
      local.set 10
      local.get 10
      local.get 5
      local.get 6
      call $_ZN4core3str8converts9from_utf817he2399b8738172384E
      local.get 4
      i32.load offset=4
      local.set 11
      block ;; label = @1
        block ;; label = @2
          local.get 11
          br_if 0 (;@2;)
          i32.const 8
          local.set 12
          local.get 1
          local.get 12
          i32.add
          local.set 13
          local.get 13
          i32.load
          local.set 14
          i32.const 16
          local.set 15
          local.get 4
          local.get 15
          i32.add
          local.set 16
          local.get 16
          local.get 12
          i32.add
          local.set 17
          local.get 17
          local.get 14
          i32.store
          local.get 1
          i64.load align=4
          local.set 18
          local.get 4
          local.get 18
          i64.store offset=16
          i32.const 4
          local.set 19
          local.get 0
          local.get 19
          i32.add
          local.set 20
          local.get 4
          i64.load offset=16 align=4
          local.set 21
          local.get 20
          local.get 21
          i64.store align=4
          i32.const 8
          local.set 22
          local.get 20
          local.get 22
          i32.add
          local.set 23
          i32.const 16
          local.set 24
          local.get 4
          local.get 24
          i32.add
          local.set 25
          local.get 25
          local.get 22
          i32.add
          local.set 26
          local.get 26
          i32.load
          local.set 27
          local.get 23
          local.get 27
          i32.store
          i32.const -2147483648
          local.set 28
          local.get 0
          local.get 28
          i32.store
          br 1 (;@1;)
        end
        i32.const 4
        local.set 29
        local.get 4
        local.get 29
        i32.add
        local.set 30
        local.get 30
        local.set 31
        i32.const 4
        local.set 32
        local.get 31
        local.get 32
        i32.add
        local.set 33
        local.get 33
        i64.load align=4
        local.set 34
        local.get 4
        local.get 34
        i64.store offset=32
        i32.const 8
        local.set 35
        local.get 1
        local.get 35
        i32.add
        local.set 36
        local.get 36
        i32.load
        local.set 37
        i32.const 40
        local.set 38
        local.get 4
        local.get 38
        i32.add
        local.set 39
        local.get 39
        local.get 35
        i32.add
        local.set 40
        local.get 40
        local.get 37
        i32.store
        local.get 1
        i64.load align=4
        local.set 41
        local.get 4
        local.get 41
        i64.store offset=40
        i32.const 40
        local.set 42
        local.get 4
        local.get 42
        i32.add
        local.set 43
        local.get 43
        local.set 44
        i32.const 12
        local.set 45
        local.get 44
        local.get 45
        i32.add
        local.set 46
        local.get 4
        i64.load offset=32 align=4
        local.set 47
        local.get 46
        local.get 47
        i64.store align=4
        local.get 4
        i64.load offset=40 align=4
        local.set 48
        local.get 0
        local.get 48
        i64.store align=4
        i32.const 16
        local.set 49
        local.get 0
        local.get 49
        i32.add
        local.set 50
        i32.const 40
        local.set 51
        local.get 4
        local.get 51
        i32.add
        local.set 52
        local.get 52
        local.get 49
        i32.add
        local.set 53
        local.get 53
        i32.load
        local.set 54
        local.get 50
        local.get 54
        i32.store
        i32.const 8
        local.set 55
        local.get 0
        local.get 55
        i32.add
        local.set 56
        i32.const 40
        local.set 57
        local.get 4
        local.get 57
        i32.add
        local.set 58
        local.get 58
        local.get 55
        i32.add
        local.set 59
        local.get 59
        i64.load align=4
        local.set 60
        local.get 56
        local.get 60
        i64.store align=4
      end
      i32.const 80
      local.set 61
      local.get 4
      local.get 61
      i32.add
      local.set 62
      local.get 62
      global.set $__stack_pointer
      return
    )
    (func $_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$14current_memory17h719515fd4baea9ecE (;86;) (type 0) (param i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 48
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 1
      i32.store offset=12
      local.get 1
      i32.load
      local.set 5
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              local.get 5
              br_if 0 (;@4;)
              br 1 (;@3;)
            end
            i32.const 1
            local.set 6
            local.get 4
            local.get 6
            i32.store offset=16
            i32.const 1
            local.set 7
            local.get 4
            local.get 7
            i32.store offset=20
            local.get 1
            i32.load
            local.set 8
            local.get 4
            local.get 8
            i32.store offset=24
            br 1 (;@2;)
          end
          i32.const 0
          local.set 9
          local.get 0
          local.get 9
          i32.store offset=4
          br 1 (;@1;)
        end
        i32.const 1
        local.set 10
        local.get 10
        local.get 8
        call $_ZN4core3num23_$LT$impl$u20$usize$GT$13unchecked_mul18precondition_check17h2a301cdfefe2a7e9E
        i32.const 0
        local.set 11
        local.get 8
        local.get 11
        i32.shl
        local.set 12
        local.get 4
        local.get 12
        i32.store offset=28
        i32.const 1
        local.set 13
        local.get 4
        local.get 13
        i32.store offset=32
        local.get 4
        local.get 12
        i32.store offset=36
        local.get 1
        i32.load offset=4
        local.set 14
        local.get 4
        local.get 14
        i32.store offset=40
        local.get 4
        local.get 14
        i32.store offset=44
        local.get 4
        local.get 14
        i32.store
        i32.const 1
        local.set 15
        local.get 4
        local.get 15
        i32.store offset=4
        local.get 4
        local.get 12
        i32.store offset=8
        local.get 4
        i64.load align=4
        local.set 16
        local.get 0
        local.get 16
        i64.store align=4
        i32.const 8
        local.set 17
        local.get 0
        local.get 17
        i32.add
        local.set 18
        local.get 4
        local.get 17
        i32.add
        local.set 19
        local.get 19
        i32.load
        local.set 20
        local.get 18
        local.get 20
        i32.store
      end
      i32.const 48
      local.set 21
      local.get 4
      local.get 21
      i32.add
      local.set 22
      local.get 22
      global.set $__stack_pointer
      return
    )
    (func $_ZN63_$LT$alloc..alloc..Global$u20$as$u20$core..alloc..Allocator$GT$10deallocate17h26f17aa909e649d7E (;87;) (type 6) (param i32 i32 i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 4
      i32.const 48
      local.set 5
      local.get 4
      local.get 5
      i32.sub
      local.set 6
      local.get 6
      global.set $__stack_pointer
      local.get 6
      local.get 2
      i32.store
      local.get 6
      local.get 3
      i32.store offset=4
      local.get 6
      local.get 0
      i32.store offset=20
      local.get 6
      local.get 1
      i32.store offset=24
      local.get 6
      local.set 7
      local.get 6
      local.get 7
      i32.store offset=28
      local.get 6
      i32.load offset=4
      local.set 8
      block ;; label = @1
        local.get 8
        i32.eqz
        br_if 0 (;@1;)
        local.get 6
        local.get 1
        i32.store offset=32
        local.get 6
        i32.load
        local.set 9
        local.get 6
        i32.load offset=4
        local.set 10
        local.get 6
        local.get 9
        i32.store offset=8
        local.get 6
        local.get 10
        i32.store offset=12
        i32.const 8
        local.set 11
        local.get 6
        local.get 11
        i32.add
        local.set 12
        local.get 12
        local.set 13
        local.get 6
        local.get 13
        i32.store offset=36
        i32.const 8
        local.set 14
        local.get 6
        local.get 14
        i32.add
        local.set 15
        local.get 15
        local.set 16
        local.get 6
        local.get 16
        i32.store offset=40
        local.get 6
        i32.load
        local.set 17
        local.get 6
        local.get 17
        i32.store offset=44
        local.get 6
        local.get 17
        i32.store offset=16
        local.get 6
        i32.load offset=16
        local.set 18
        local.get 1
        local.get 8
        local.get 18
        call $__rust_dealloc
      end
      i32.const 48
      local.set 19
      local.get 6
      local.get 19
      i32.add
      local.set 20
      local.get 20
      global.set $__stack_pointer
      return
    )
    (func $_ZN65_$LT$alloc..string..FromUtf8Error$u20$as$u20$core..fmt..Debug$GT$3fmt17hc6b77939b21326f3E (;88;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 16
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=8
      local.get 4
      local.get 1
      i32.store offset=12
      i32.const 12
      local.set 5
      local.get 0
      local.get 5
      i32.add
      local.set 6
      local.get 4
      local.get 6
      i32.store offset=4
      i32.const 1051524
      local.set 7
      i32.const 13
      local.set 8
      i32.const 1051537
      local.set 9
      i32.const 5
      local.set 10
      i32.const 1051492
      local.set 11
      i32.const 1051542
      local.set 12
      i32.const 4
      local.set 13
      local.get 4
      local.get 13
      i32.add
      local.set 14
      local.get 14
      local.set 15
      i32.const 1051508
      local.set 16
      local.get 1
      local.get 7
      local.get 8
      local.get 9
      local.get 10
      local.get 0
      local.get 11
      local.get 12
      local.get 10
      local.get 15
      local.get 16
      call $_ZN4core3fmt9Formatter26debug_struct_field2_finish17h73c9201d8434494dE
      local.set 17
      i32.const 1
      local.set 18
      local.get 17
      local.get 18
      i32.and
      local.set 19
      i32.const 16
      local.set 20
      local.get 4
      local.get 20
      i32.add
      local.set 21
      local.get 21
      global.set $__stack_pointer
      local.get 19
      return
    )
    (func $_ZN65_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$core..fmt..Debug$GT$3fmt17h940d068692ae5b09E (;89;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 32
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      local.get 4
      local.get 0
      i32.store offset=8
      local.get 4
      local.get 1
      i32.store offset=12
      local.get 4
      local.get 0
      i32.store offset=16
      local.get 0
      i32.load offset=4
      local.set 5
      local.get 4
      local.get 5
      i32.store offset=20
      local.get 4
      local.get 5
      i32.store offset=24
      local.get 0
      i32.load offset=8
      local.set 6
      local.get 4
      local.get 6
      i32.store offset=28
      i32.const 1
      local.set 7
      local.get 5
      local.get 7
      local.get 7
      local.get 6
      call $_ZN4core5slice3raw14from_raw_parts18precondition_check17h772c2c63148a002fE
      local.get 5
      local.get 6
      local.get 1
      call $_ZN48_$LT$$u5b$T$u5d$$u20$as$u20$core..fmt..Debug$GT$3fmt17hf3236d8eb423c151E
      local.set 8
      i32.const 1
      local.set 9
      local.get 8
      local.get 9
      i32.and
      local.set 10
      i32.const 32
      local.set 11
      local.get 4
      local.get 11
      i32.add
      local.set 12
      local.get 12
      global.set $__stack_pointer
      local.get 10
      return
    )
    (func $_ZN11wit_bindgen2rt14run_ctors_once17h2d2a1af2565a62bcE (;90;) (type 4)
      (local i32 i32 i32 i32 i32 i32)
      i32.const 0
      local.set 0
      local.get 0
      i32.load8_u offset=1056354
      local.set 1
      i32.const 1
      local.set 2
      local.get 1
      local.get 2
      i32.and
      local.set 3
      block ;; label = @1
        local.get 3
        br_if 0 (;@1;)
        call $__wasm_call_ctors
        i32.const 1
        local.set 4
        i32.const 0
        local.set 5
        local.get 5
        local.get 4
        i32.store8 offset=1056354
      end
      return
    )
    (func $cabi_realloc (;91;) (type 7) (param i32 i32 i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i64 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 4
      i32.const 112
      local.set 5
      local.get 4
      local.get 5
      i32.sub
      local.set 6
      local.get 6
      global.set $__stack_pointer
      local.get 6
      local.get 3
      i32.store offset=16
      local.get 6
      local.get 0
      i32.store offset=92
      local.get 6
      local.get 1
      i32.store offset=96
      local.get 6
      local.get 2
      i32.store offset=100
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  block ;; label = @7
                    local.get 1
                    br_if 0 (;@7;)
                    local.get 6
                    i32.load offset=16
                    local.set 7
                    local.get 7
                    i32.eqz
                    br_if 1 (;@6;)
                    br 2 (;@5;)
                  end
                  i32.const 16
                  local.set 8
                  local.get 6
                  local.get 8
                  i32.add
                  local.set 9
                  local.get 9
                  local.set 10
                  local.get 6
                  local.get 10
                  i32.store offset=104
                  i32.const 1051556
                  local.set 11
                  local.get 6
                  local.get 11
                  i32.store offset=108
                  local.get 6
                  i32.load offset=16
                  local.set 12
                  i32.const 0
                  local.set 13
                  local.get 13
                  i32.load offset=1051556
                  local.set 14
                  local.get 12
                  local.set 15
                  local.get 14
                  local.set 16
                  local.get 15
                  local.get 16
                  i32.eq
                  local.set 17
                  i32.const 1
                  local.set 18
                  local.get 17
                  local.get 18
                  i32.and
                  local.set 19
                  local.get 19
                  br_if 3 (;@3;)
                  br 2 (;@4;)
                end
                local.get 6
                local.get 2
                i32.store offset=20
                br 4 (;@1;)
              end
              local.get 6
              i32.load offset=16
              local.set 20
              local.get 6
              local.get 20
              local.get 2
              call $_ZN4core5alloc6layout6Layout25from_size_align_unchecked17h3f4cfcb7d6ae7554E
              local.get 6
              i32.load offset=4
              local.set 21
              local.get 6
              i32.load
              local.set 22
              local.get 6
              local.get 22
              i32.store offset=24
              local.get 6
              local.get 21
              i32.store offset=28
              local.get 6
              i32.load offset=24
              local.set 23
              local.get 6
              i32.load offset=28
              local.set 24
              local.get 23
              local.get 24
              call $_ZN5alloc5alloc5alloc17hba97f008fb15ed2bE
              local.set 25
              local.get 6
              local.get 25
              i32.store offset=32
              br 2 (;@2;)
            end
            i32.const 8
            local.set 26
            local.get 6
            local.get 26
            i32.add
            local.set 27
            local.get 27
            local.get 1
            local.get 2
            call $_ZN4core5alloc6layout6Layout25from_size_align_unchecked17h3f4cfcb7d6ae7554E
            local.get 6
            i32.load offset=12
            local.set 28
            local.get 6
            i32.load offset=8
            local.set 29
            local.get 6
            local.get 29
            i32.store offset=24
            local.get 6
            local.get 28
            i32.store offset=28
            local.get 6
            i32.load offset=24
            local.set 30
            local.get 6
            i32.load offset=28
            local.set 31
            local.get 6
            i32.load offset=16
            local.set 32
            local.get 0
            local.get 30
            local.get 31
            local.get 32
            call $_ZN5alloc5alloc7realloc17h67b0606c850eb3a0E
            local.set 33
            local.get 6
            local.get 33
            i32.store offset=32
            br 1 (;@2;)
          end
          i32.const 1
          local.set 34
          local.get 6
          local.get 34
          i32.store8 offset=39
          i32.const 68
          local.set 35
          local.get 6
          local.get 35
          i32.add
          local.set 36
          local.get 36
          local.set 37
          i32.const 1051604
          local.set 38
          local.get 37
          local.get 38
          call $_ZN4core3fmt9Arguments9new_const17hd92b933da0585f96E
          i32.const 16
          local.set 39
          i32.const 40
          local.set 40
          local.get 6
          local.get 40
          i32.add
          local.set 41
          local.get 41
          local.get 39
          i32.add
          local.set 42
          i32.const 68
          local.set 43
          local.get 6
          local.get 43
          i32.add
          local.set 44
          local.get 44
          local.get 39
          i32.add
          local.set 45
          local.get 45
          i64.load align=4
          local.set 46
          local.get 42
          local.get 46
          i64.store
          i32.const 8
          local.set 47
          i32.const 40
          local.set 48
          local.get 6
          local.get 48
          i32.add
          local.set 49
          local.get 49
          local.get 47
          i32.add
          local.set 50
          i32.const 68
          local.set 51
          local.get 6
          local.get 51
          i32.add
          local.set 52
          local.get 52
          local.get 47
          i32.add
          local.set 53
          local.get 53
          i64.load align=4
          local.set 54
          local.get 50
          local.get 54
          i64.store
          local.get 6
          i64.load offset=68 align=4
          local.set 55
          local.get 6
          local.get 55
          i64.store offset=40
          local.get 6
          i32.load8_u offset=39
          local.set 56
          i32.const 16
          local.set 57
          local.get 6
          local.get 57
          i32.add
          local.set 58
          local.get 58
          local.set 59
          i32.const 1051556
          local.set 60
          i32.const 40
          local.set 61
          local.get 6
          local.get 61
          i32.add
          local.set 62
          local.get 62
          local.set 63
          i32.const 1051708
          local.set 64
          local.get 56
          local.get 59
          local.get 60
          local.get 63
          local.get 64
          call $_ZN4core9panicking13assert_failed17hdcd41e4667a8ffb5E
          unreachable
        end
        local.get 6
        i32.load offset=32
        local.set 65
        local.get 65
        call $_ZN4core3ptr7mut_ptr31_$LT$impl$u20$$BP$mut$u20$T$GT$7is_null17h7dd7c30dd534ee73E
        local.set 66
        i32.const 1
        local.set 67
        local.get 66
        local.get 67
        i32.and
        local.set 68
        block ;; label = @2
          local.get 68
          br_if 0 (;@2;)
          local.get 6
          i32.load offset=32
          local.set 69
          local.get 6
          local.get 69
          i32.store offset=20
          br 1 (;@1;)
        end
        local.get 6
        i32.load offset=24
        local.set 70
        local.get 6
        i32.load offset=28
        local.set 71
        local.get 70
        local.get 71
        call $_ZN5alloc5alloc18handle_alloc_error17h0ba28a7c65be46c8E
        unreachable
      end
      local.get 6
      i32.load offset=20
      local.set 72
      i32.const 112
      local.set 73
      local.get 6
      local.get 73
      i32.add
      local.set 74
      local.get 74
      global.set $__stack_pointer
      local.get 72
      return
    )
    (func $_ZN11wit_bindgen2rt11string_lift17h09e85ffb5d1e1eadE (;92;) (type 0) (param i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i64 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 48
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      global.set $__stack_pointer
      i32.const 4
      local.set 5
      local.get 4
      local.get 5
      i32.add
      local.set 6
      local.get 6
      local.set 7
      local.get 7
      local.get 1
      call $_ZN5alloc6string6String9from_utf817hf9c24943a7ea7d3dE
      local.get 4
      i32.load offset=4
      local.set 8
      i32.const -2147483648
      local.set 9
      local.get 8
      local.set 10
      local.get 9
      local.set 11
      local.get 10
      local.get 11
      i32.eq
      local.set 12
      i32.const 1
      local.set 13
      local.get 12
      local.get 13
      i32.and
      local.set 14
      block ;; label = @1
        local.get 14
        br_if 0 (;@1;)
        i32.const 16
        local.set 15
        i32.const 24
        local.set 16
        local.get 4
        local.get 16
        i32.add
        local.set 17
        local.get 17
        local.get 15
        i32.add
        local.set 18
        i32.const 4
        local.set 19
        local.get 4
        local.get 19
        i32.add
        local.set 20
        local.get 20
        local.get 15
        i32.add
        local.set 21
        local.get 21
        i32.load
        local.set 22
        local.get 18
        local.get 22
        i32.store
        i32.const 8
        local.set 23
        i32.const 24
        local.set 24
        local.get 4
        local.get 24
        i32.add
        local.set 25
        local.get 25
        local.get 23
        i32.add
        local.set 26
        i32.const 4
        local.set 27
        local.get 4
        local.get 27
        i32.add
        local.set 28
        local.get 28
        local.get 23
        i32.add
        local.set 29
        local.get 29
        i64.load align=4
        local.set 30
        local.get 26
        local.get 30
        i64.store
        local.get 4
        i64.load offset=4 align=4
        local.set 31
        local.get 4
        local.get 31
        i64.store offset=24
        i32.const 1051368
        local.set 32
        i32.const 43
        local.set 33
        i32.const 24
        local.set 34
        local.get 4
        local.get 34
        i32.add
        local.set 35
        local.get 35
        local.set 36
        i32.const 1051352
        local.set 37
        i32.const 1051724
        local.set 38
        local.get 32
        local.get 33
        local.get 36
        local.get 37
        local.get 38
        call $_ZN4core6result13unwrap_failed17h3e6036b583f82d93E
        unreachable
      end
      i32.const 4
      local.set 39
      local.get 4
      local.get 39
      i32.add
      local.set 40
      local.get 40
      local.set 41
      i32.const 4
      local.set 42
      local.get 41
      local.get 42
      i32.add
      local.set 43
      local.get 43
      i64.load align=4
      local.set 44
      local.get 0
      local.get 44
      i64.store align=4
      i32.const 8
      local.set 45
      local.get 0
      local.get 45
      i32.add
      local.set 46
      local.get 43
      local.get 45
      i32.add
      local.set 47
      local.get 47
      i32.load
      local.set 48
      local.get 46
      local.get 48
      i32.store
      i32.const 48
      local.set 49
      local.get 4
      local.get 49
      i32.add
      local.set 50
      local.get 50
      global.set $__stack_pointer
      return
    )
    (func $_ZN4core3fmt9Arguments9new_const17hd92b933da0585f96E (;93;) (type 0) (param i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      local.set 2
      i32.const 16
      local.set 3
      local.get 2
      local.get 3
      i32.sub
      local.set 4
      local.get 4
      local.get 1
      i32.store offset=12
      local.get 0
      local.get 1
      i32.store
      i32.const 1
      local.set 5
      local.get 0
      local.get 5
      i32.store offset=4
      i32.const 0
      local.set 6
      local.get 6
      i32.load offset=1051740
      local.set 7
      i32.const 0
      local.set 8
      local.get 8
      i32.load offset=1051744
      local.set 9
      local.get 0
      local.get 7
      i32.store offset=16
      local.get 0
      local.get 9
      i32.store offset=20
      i32.const 4
      local.set 10
      local.get 0
      local.get 10
      i32.store offset=8
      i32.const 0
      local.set 11
      local.get 0
      local.get 11
      i32.store offset=12
      return
    )
    (func $_ZN36_$LT$T$u20$as$u20$core..any..Any$GT$7type_id17hd1cbeb6a70b6771bE (;94;) (type 0) (param i32 i32)
      local.get 0
      i64.const 7199936582794304877
      i64.store offset=8
      local.get 0
      i64.const -5076933981314334344
      i64.store
    )
    (func $_ZN36_$LT$T$u20$as$u20$core..any..Any$GT$7type_id17hf8c8bae692844a7bE (;95;) (type 0) (param i32 i32)
      local.get 0
      i64.const 6736479778479879123
      i64.store offset=8
      local.get 0
      i64.const -4024166894029875428
      i64.store
    )
    (func $_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$7reserve21do_reserve_and_handle17hd2a95c7b218ae967E (;96;) (type 5) (param i32 i32 i32)
      (local i32 i32 i32)
      global.get $__stack_pointer
      i32.const 32
      i32.sub
      local.tee 3
      global.set $__stack_pointer
      block ;; label = @1
        local.get 1
        local.get 2
        i32.add
        local.tee 2
        local.get 1
        i32.ge_u
        br_if 0 (;@1;)
        i32.const 0
        i32.const 0
        call $_ZN5alloc7raw_vec12handle_error17h5204850add322ec3E
        unreachable
      end
      i32.const 1
      local.set 4
      local.get 0
      i32.load
      local.tee 5
      i32.const 1
      i32.shl
      local.tee 1
      local.get 2
      local.get 1
      local.get 2
      i32.gt_u
      select
      local.tee 1
      i32.const 8
      local.get 1
      i32.const 8
      i32.gt_u
      select
      local.tee 1
      i32.const -1
      i32.xor
      i32.const 31
      i32.shr_u
      local.set 2
      block ;; label = @1
        block ;; label = @2
          local.get 5
          br_if 0 (;@2;)
          i32.const 0
          local.set 4
          br 1 (;@1;)
        end
        local.get 3
        local.get 5
        i32.store offset=28
        local.get 3
        local.get 0
        i32.load offset=4
        i32.store offset=20
      end
      local.get 3
      local.get 4
      i32.store offset=24
      local.get 3
      i32.const 8
      i32.add
      local.get 2
      local.get 1
      local.get 3
      i32.const 20
      i32.add
      call $_ZN5alloc7raw_vec11finish_grow17h6cc141763e8807d2E
      block ;; label = @1
        local.get 3
        i32.load offset=8
        i32.eqz
        br_if 0 (;@1;)
        local.get 3
        i32.load offset=12
        local.get 3
        i32.load offset=16
        call $_ZN5alloc7raw_vec12handle_error17h5204850add322ec3E
        unreachable
      end
      local.get 3
      i32.load offset=12
      local.set 2
      local.get 0
      local.get 1
      i32.store
      local.get 0
      local.get 2
      i32.store offset=4
      local.get 3
      i32.const 32
      i32.add
      global.set $__stack_pointer
    )
    (func $_ZN4core3fmt5Write9write_fmt17h8e7667e000b2e00bE (;97;) (type 2) (param i32 i32) (result i32)
      local.get 0
      i32.const 1051748
      local.get 1
      call $_ZN4core3fmt5write17h43164ada91fcaaeeE
    )
    (func $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17ha5fb9ff401ff50deE (;98;) (type 9) (param i32)
      (local i32)
      block ;; label = @1
        local.get 0
        i32.load
        local.tee 1
        i32.eqz
        br_if 0 (;@1;)
        local.get 0
        i32.load offset=4
        local.get 1
        i32.const 1
        call $__rust_dealloc
      end
    )
    (func $_ZN4core3ptr77drop_in_place$LT$std..panicking..begin_panic_handler..FormatStringPayload$GT$17h32febad92427b942E (;99;) (type 9) (param i32)
      (local i32)
      block ;; label = @1
        local.get 0
        i32.load
        local.tee 1
        i32.const -2147483648
        i32.or
        i32.const -2147483648
        i32.eq
        br_if 0 (;@1;)
        local.get 0
        i32.load offset=4
        local.get 1
        i32.const 1
        call $__rust_dealloc
      end
    )
    (func $_ZN4core5panic12PanicPayload6as_str17haccd18006a9ab86bE (;100;) (type 0) (param i32 i32)
      local.get 0
      i32.const 0
      i32.store
    )
    (func $_ZN58_$LT$alloc..string..String$u20$as$u20$core..fmt..Write$GT$10write_char17h4c0947de865746f3E (;101;) (type 2) (param i32 i32) (result i32)
      (local i32 i32)
      global.get $__stack_pointer
      i32.const 16
      i32.sub
      local.tee 2
      global.set $__stack_pointer
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              local.get 1
              i32.const 128
              i32.lt_u
              br_if 0 (;@4;)
              local.get 2
              i32.const 0
              i32.store offset=12
              local.get 1
              i32.const 2048
              i32.lt_u
              br_if 1 (;@3;)
              block ;; label = @5
                local.get 1
                i32.const 65536
                i32.ge_u
                br_if 0 (;@5;)
                local.get 2
                local.get 1
                i32.const 63
                i32.and
                i32.const 128
                i32.or
                i32.store8 offset=14
                local.get 2
                local.get 1
                i32.const 12
                i32.shr_u
                i32.const 224
                i32.or
                i32.store8 offset=12
                local.get 2
                local.get 1
                i32.const 6
                i32.shr_u
                i32.const 63
                i32.and
                i32.const 128
                i32.or
                i32.store8 offset=13
                i32.const 3
                local.set 1
                br 3 (;@2;)
              end
              local.get 2
              local.get 1
              i32.const 63
              i32.and
              i32.const 128
              i32.or
              i32.store8 offset=15
              local.get 2
              local.get 1
              i32.const 6
              i32.shr_u
              i32.const 63
              i32.and
              i32.const 128
              i32.or
              i32.store8 offset=14
              local.get 2
              local.get 1
              i32.const 12
              i32.shr_u
              i32.const 63
              i32.and
              i32.const 128
              i32.or
              i32.store8 offset=13
              local.get 2
              local.get 1
              i32.const 18
              i32.shr_u
              i32.const 7
              i32.and
              i32.const 240
              i32.or
              i32.store8 offset=12
              i32.const 4
              local.set 1
              br 2 (;@2;)
            end
            block ;; label = @4
              local.get 0
              i32.load offset=8
              local.tee 3
              local.get 0
              i32.load
              i32.ne
              br_if 0 (;@4;)
              local.get 0
              call $_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17hd0d85d0ee7ed23abE
            end
            local.get 0
            local.get 3
            i32.const 1
            i32.add
            i32.store offset=8
            local.get 0
            i32.load offset=4
            local.get 3
            i32.add
            local.get 1
            i32.store8
            br 2 (;@1;)
          end
          local.get 2
          local.get 1
          i32.const 63
          i32.and
          i32.const 128
          i32.or
          i32.store8 offset=13
          local.get 2
          local.get 1
          i32.const 6
          i32.shr_u
          i32.const 192
          i32.or
          i32.store8 offset=12
          i32.const 2
          local.set 1
        end
        block ;; label = @2
          local.get 0
          i32.load
          local.get 0
          i32.load offset=8
          local.tee 3
          i32.sub
          local.get 1
          i32.ge_u
          br_if 0 (;@2;)
          local.get 0
          local.get 3
          local.get 1
          call $_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$7reserve21do_reserve_and_handle17hd2a95c7b218ae967E
          local.get 0
          i32.load offset=8
          local.set 3
        end
        local.get 0
        i32.load offset=4
        local.get 3
        i32.add
        local.get 2
        i32.const 12
        i32.add
        local.get 1
        call $memcpy
        drop
        local.get 0
        local.get 3
        local.get 1
        i32.add
        i32.store offset=8
      end
      local.get 2
      i32.const 16
      i32.add
      global.set $__stack_pointer
      i32.const 0
    )
    (func $_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17hd0d85d0ee7ed23abE (;102;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      i32.const 32
      i32.sub
      local.tee 1
      global.set $__stack_pointer
      block ;; label = @1
        local.get 0
        i32.load
        local.tee 2
        i32.const -1
        i32.ne
        br_if 0 (;@1;)
        i32.const 0
        i32.const 0
        call $_ZN5alloc7raw_vec12handle_error17h5204850add322ec3E
        unreachable
      end
      i32.const 1
      local.set 3
      local.get 2
      i32.const 1
      i32.shl
      local.tee 4
      local.get 2
      i32.const 1
      i32.add
      local.tee 5
      local.get 4
      local.get 5
      i32.gt_u
      select
      local.tee 4
      i32.const 8
      local.get 4
      i32.const 8
      i32.gt_u
      select
      local.tee 4
      i32.const -1
      i32.xor
      i32.const 31
      i32.shr_u
      local.set 5
      block ;; label = @1
        block ;; label = @2
          local.get 2
          br_if 0 (;@2;)
          i32.const 0
          local.set 3
          br 1 (;@1;)
        end
        local.get 1
        local.get 2
        i32.store offset=28
        local.get 1
        local.get 0
        i32.load offset=4
        i32.store offset=20
      end
      local.get 1
      local.get 3
      i32.store offset=24
      local.get 1
      i32.const 8
      i32.add
      local.get 5
      local.get 4
      local.get 1
      i32.const 20
      i32.add
      call $_ZN5alloc7raw_vec11finish_grow17h6cc141763e8807d2E
      block ;; label = @1
        local.get 1
        i32.load offset=8
        i32.eqz
        br_if 0 (;@1;)
        local.get 1
        i32.load offset=12
        local.get 1
        i32.load offset=16
        call $_ZN5alloc7raw_vec12handle_error17h5204850add322ec3E
        unreachable
      end
      local.get 1
      i32.load offset=12
      local.set 2
      local.get 0
      local.get 4
      i32.store
      local.get 0
      local.get 2
      i32.store offset=4
      local.get 1
      i32.const 32
      i32.add
      global.set $__stack_pointer
    )
    (func $_ZN58_$LT$alloc..string..String$u20$as$u20$core..fmt..Write$GT$9write_str17h8c93e1ada73d40b2E (;103;) (type 1) (param i32 i32 i32) (result i32)
      (local i32)
      block ;; label = @1
        local.get 0
        i32.load
        local.get 0
        i32.load offset=8
        local.tee 3
        i32.sub
        local.get 2
        i32.ge_u
        br_if 0 (;@1;)
        local.get 0
        local.get 3
        local.get 2
        call $_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$7reserve21do_reserve_and_handle17hd2a95c7b218ae967E
        local.get 0
        i32.load offset=8
        local.set 3
      end
      local.get 0
      i32.load offset=4
      local.get 3
      i32.add
      local.get 1
      local.get 2
      call $memcpy
      drop
      local.get 0
      local.get 3
      local.get 2
      i32.add
      i32.store offset=8
      i32.const 0
    )
    (func $_ZN5alloc7raw_vec11finish_grow17h6cc141763e8807d2E (;104;) (type 6) (param i32 i32 i32 i32)
      (local i32)
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            local.get 1
            i32.eqz
            br_if 0 (;@3;)
            local.get 2
            i32.const 0
            i32.lt_s
            br_if 1 (;@2;)
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  local.get 3
                  i32.load offset=4
                  i32.eqz
                  br_if 0 (;@6;)
                  block ;; label = @7
                    local.get 3
                    i32.load offset=8
                    local.tee 4
                    br_if 0 (;@7;)
                    block ;; label = @8
                      local.get 2
                      br_if 0 (;@8;)
                      local.get 1
                      local.set 3
                      br 4 (;@4;)
                    end
                    i32.const 0
                    i32.load8_u offset=1056353
                    drop
                    br 2 (;@5;)
                  end
                  local.get 3
                  i32.load
                  local.get 4
                  local.get 1
                  local.get 2
                  call $__rust_realloc
                  local.set 3
                  br 2 (;@4;)
                end
                block ;; label = @6
                  local.get 2
                  br_if 0 (;@6;)
                  local.get 1
                  local.set 3
                  br 2 (;@4;)
                end
                i32.const 0
                i32.load8_u offset=1056353
                drop
              end
              local.get 2
              local.get 1
              call $__rust_alloc
              local.set 3
            end
            block ;; label = @4
              local.get 3
              i32.eqz
              br_if 0 (;@4;)
              local.get 0
              local.get 2
              i32.store offset=8
              local.get 0
              local.get 3
              i32.store offset=4
              local.get 0
              i32.const 0
              i32.store
              return
            end
            local.get 0
            local.get 2
            i32.store offset=8
            local.get 0
            local.get 1
            i32.store offset=4
            br 2 (;@1;)
          end
          local.get 0
          i32.const 0
          i32.store offset=4
          br 1 (;@1;)
        end
        local.get 0
        i32.const 0
        i32.store offset=4
      end
      local.get 0
      i32.const 1
      i32.store
    )
    (func $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$12unlink_chunk17hc085d75ef8348387E (;105;) (type 0) (param i32 i32)
      (local i32 i32 i32 i32)
      local.get 0
      i32.load offset=12
      local.set 2
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            local.get 1
            i32.const 256
            i32.lt_u
            br_if 0 (;@3;)
            local.get 0
            i32.load offset=24
            local.set 3
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  local.get 2
                  local.get 0
                  i32.ne
                  br_if 0 (;@6;)
                  local.get 0
                  i32.const 20
                  i32.const 16
                  local.get 0
                  i32.load offset=20
                  local.tee 2
                  select
                  i32.add
                  i32.load
                  local.tee 1
                  br_if 1 (;@5;)
                  i32.const 0
                  local.set 2
                  br 2 (;@4;)
                end
                local.get 0
                i32.load offset=8
                local.tee 1
                local.get 2
                i32.store offset=12
                local.get 2
                local.get 1
                i32.store offset=8
                br 1 (;@4;)
              end
              local.get 0
              i32.const 20
              i32.add
              local.get 0
              i32.const 16
              i32.add
              local.get 2
              select
              local.set 4
              loop ;; label = @5
                local.get 4
                local.set 5
                local.get 1
                local.tee 2
                i32.const 20
                i32.add
                local.get 2
                i32.const 16
                i32.add
                local.get 2
                i32.load offset=20
                local.tee 1
                select
                local.set 4
                local.get 2
                i32.const 20
                i32.const 16
                local.get 1
                select
                i32.add
                i32.load
                local.tee 1
                br_if 0 (;@5;)
              end
              local.get 5
              i32.const 0
              i32.store
            end
            local.get 3
            i32.eqz
            br_if 2 (;@1;)
            block ;; label = @4
              local.get 0
              i32.load offset=28
              i32.const 2
              i32.shl
              i32.const 1056376
              i32.add
              local.tee 1
              i32.load
              local.get 0
              i32.eq
              br_if 0 (;@4;)
              local.get 3
              i32.const 16
              i32.const 20
              local.get 3
              i32.load offset=16
              local.get 0
              i32.eq
              select
              i32.add
              local.get 2
              i32.store
              local.get 2
              i32.eqz
              br_if 3 (;@1;)
              br 2 (;@2;)
            end
            local.get 1
            local.get 2
            i32.store
            local.get 2
            br_if 1 (;@2;)
            i32.const 0
            i32.const 0
            i32.load offset=1056788
            i32.const -2
            local.get 0
            i32.load offset=28
            i32.rotl
            i32.and
            i32.store offset=1056788
            br 2 (;@1;)
          end
          block ;; label = @3
            local.get 2
            local.get 0
            i32.load offset=8
            local.tee 4
            i32.eq
            br_if 0 (;@3;)
            local.get 4
            local.get 2
            i32.store offset=12
            local.get 2
            local.get 4
            i32.store offset=8
            return
          end
          i32.const 0
          i32.const 0
          i32.load offset=1056784
          i32.const -2
          local.get 1
          i32.const 3
          i32.shr_u
          i32.rotl
          i32.and
          i32.store offset=1056784
          return
        end
        local.get 2
        local.get 3
        i32.store offset=24
        block ;; label = @2
          local.get 0
          i32.load offset=16
          local.tee 1
          i32.eqz
          br_if 0 (;@2;)
          local.get 2
          local.get 1
          i32.store offset=16
          local.get 1
          local.get 2
          i32.store offset=24
        end
        local.get 0
        i32.load offset=20
        local.tee 1
        i32.eqz
        br_if 0 (;@1;)
        local.get 2
        local.get 1
        i32.store offset=20
        local.get 1
        local.get 2
        i32.store offset=24
        return
      end
    )
    (func $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$13dispose_chunk17h812266503d4b53e1E (;106;) (type 0) (param i32 i32)
      (local i32 i32)
      local.get 0
      local.get 1
      i32.add
      local.set 2
      block ;; label = @1
        block ;; label = @2
          local.get 0
          i32.load offset=4
          local.tee 3
          i32.const 1
          i32.and
          br_if 0 (;@2;)
          local.get 3
          i32.const 2
          i32.and
          i32.eqz
          br_if 1 (;@1;)
          local.get 0
          i32.load
          local.tee 3
          local.get 1
          i32.add
          local.set 1
          block ;; label = @3
            local.get 0
            local.get 3
            i32.sub
            local.tee 0
            i32.const 0
            i32.load offset=1056800
            i32.ne
            br_if 0 (;@3;)
            local.get 2
            i32.load offset=4
            i32.const 3
            i32.and
            i32.const 3
            i32.ne
            br_if 1 (;@2;)
            i32.const 0
            local.get 1
            i32.store offset=1056792
            local.get 2
            local.get 2
            i32.load offset=4
            i32.const -2
            i32.and
            i32.store offset=4
            local.get 0
            local.get 1
            i32.const 1
            i32.or
            i32.store offset=4
            local.get 2
            local.get 1
            i32.store
            br 2 (;@1;)
          end
          local.get 0
          local.get 3
          call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$12unlink_chunk17hc085d75ef8348387E
        end
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                local.get 2
                i32.load offset=4
                local.tee 3
                i32.const 2
                i32.and
                br_if 0 (;@5;)
                local.get 2
                i32.const 0
                i32.load offset=1056804
                i32.eq
                br_if 2 (;@3;)
                local.get 2
                i32.const 0
                i32.load offset=1056800
                i32.eq
                br_if 3 (;@2;)
                local.get 2
                local.get 3
                i32.const -8
                i32.and
                local.tee 3
                call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$12unlink_chunk17hc085d75ef8348387E
                local.get 0
                local.get 3
                local.get 1
                i32.add
                local.tee 1
                i32.const 1
                i32.or
                i32.store offset=4
                local.get 0
                local.get 1
                i32.add
                local.get 1
                i32.store
                local.get 0
                i32.const 0
                i32.load offset=1056800
                i32.ne
                br_if 1 (;@4;)
                i32.const 0
                local.get 1
                i32.store offset=1056792
                return
              end
              local.get 2
              local.get 3
              i32.const -2
              i32.and
              i32.store offset=4
              local.get 0
              local.get 1
              i32.const 1
              i32.or
              i32.store offset=4
              local.get 0
              local.get 1
              i32.add
              local.get 1
              i32.store
            end
            block ;; label = @4
              local.get 1
              i32.const 256
              i32.lt_u
              br_if 0 (;@4;)
              local.get 0
              local.get 1
              call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$18insert_large_chunk17h6d1973057372779eE
              return
            end
            local.get 1
            i32.const -8
            i32.and
            i32.const 1056520
            i32.add
            local.set 2
            block ;; label = @4
              block ;; label = @5
                i32.const 0
                i32.load offset=1056784
                local.tee 3
                i32.const 1
                local.get 1
                i32.const 3
                i32.shr_u
                i32.shl
                local.tee 1
                i32.and
                br_if 0 (;@5;)
                i32.const 0
                local.get 3
                local.get 1
                i32.or
                i32.store offset=1056784
                local.get 2
                local.set 1
                br 1 (;@4;)
              end
              local.get 2
              i32.load offset=8
              local.set 1
            end
            local.get 2
            local.get 0
            i32.store offset=8
            local.get 1
            local.get 0
            i32.store offset=12
            local.get 0
            local.get 2
            i32.store offset=12
            local.get 0
            local.get 1
            i32.store offset=8
            return
          end
          i32.const 0
          local.get 0
          i32.store offset=1056804
          i32.const 0
          i32.const 0
          i32.load offset=1056796
          local.get 1
          i32.add
          local.tee 1
          i32.store offset=1056796
          local.get 0
          local.get 1
          i32.const 1
          i32.or
          i32.store offset=4
          local.get 0
          i32.const 0
          i32.load offset=1056800
          i32.ne
          br_if 1 (;@1;)
          i32.const 0
          i32.const 0
          i32.store offset=1056792
          i32.const 0
          i32.const 0
          i32.store offset=1056800
          return
        end
        i32.const 0
        local.get 0
        i32.store offset=1056800
        i32.const 0
        i32.const 0
        i32.load offset=1056792
        local.get 1
        i32.add
        local.tee 1
        i32.store offset=1056792
        local.get 0
        local.get 1
        i32.const 1
        i32.or
        i32.store offset=4
        local.get 0
        local.get 1
        i32.add
        local.get 1
        i32.store
        return
      end
    )
    (func $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$18insert_large_chunk17h6d1973057372779eE (;107;) (type 0) (param i32 i32)
      (local i32 i32 i32 i32)
      i32.const 0
      local.set 2
      block ;; label = @1
        local.get 1
        i32.const 256
        i32.lt_u
        br_if 0 (;@1;)
        i32.const 31
        local.set 2
        local.get 1
        i32.const 16777215
        i32.gt_u
        br_if 0 (;@1;)
        local.get 1
        i32.const 6
        local.get 1
        i32.const 8
        i32.shr_u
        i32.clz
        local.tee 2
        i32.sub
        i32.shr_u
        i32.const 1
        i32.and
        local.get 2
        i32.const 1
        i32.shl
        i32.sub
        i32.const 62
        i32.add
        local.set 2
      end
      local.get 0
      i64.const 0
      i64.store offset=16 align=4
      local.get 0
      local.get 2
      i32.store offset=28
      local.get 2
      i32.const 2
      i32.shl
      i32.const 1056376
      i32.add
      local.set 3
      block ;; label = @1
        i32.const 0
        i32.load offset=1056788
        i32.const 1
        local.get 2
        i32.shl
        local.tee 4
        i32.and
        br_if 0 (;@1;)
        local.get 3
        local.get 0
        i32.store
        local.get 0
        local.get 3
        i32.store offset=24
        local.get 0
        local.get 0
        i32.store offset=12
        local.get 0
        local.get 0
        i32.store offset=8
        i32.const 0
        i32.const 0
        i32.load offset=1056788
        local.get 4
        i32.or
        i32.store offset=1056788
        return
      end
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            local.get 3
            i32.load
            local.tee 4
            i32.load offset=4
            i32.const -8
            i32.and
            local.get 1
            i32.ne
            br_if 0 (;@3;)
            local.get 4
            local.set 2
            br 1 (;@2;)
          end
          local.get 1
          i32.const 0
          i32.const 25
          local.get 2
          i32.const 1
          i32.shr_u
          i32.sub
          local.get 2
          i32.const 31
          i32.eq
          select
          i32.shl
          local.set 3
          loop ;; label = @3
            local.get 4
            local.get 3
            i32.const 29
            i32.shr_u
            i32.const 4
            i32.and
            i32.add
            i32.const 16
            i32.add
            local.tee 5
            i32.load
            local.tee 2
            i32.eqz
            br_if 2 (;@1;)
            local.get 3
            i32.const 1
            i32.shl
            local.set 3
            local.get 2
            local.set 4
            local.get 2
            i32.load offset=4
            i32.const -8
            i32.and
            local.get 1
            i32.ne
            br_if 0 (;@3;)
          end
        end
        local.get 2
        i32.load offset=8
        local.tee 3
        local.get 0
        i32.store offset=12
        local.get 2
        local.get 0
        i32.store offset=8
        local.get 0
        i32.const 0
        i32.store offset=24
        local.get 0
        local.get 2
        i32.store offset=12
        local.get 0
        local.get 3
        i32.store offset=8
        return
      end
      local.get 5
      local.get 0
      i32.store
      local.get 0
      local.get 4
      i32.store offset=24
      local.get 0
      local.get 0
      i32.store offset=12
      local.get 0
      local.get 0
      i32.store offset=8
    )
    (func $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$4free17h092132f18ab141f8E (;108;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32)
      local.get 0
      i32.const -8
      i32.add
      local.tee 1
      local.get 0
      i32.const -4
      i32.add
      i32.load
      local.tee 2
      i32.const -8
      i32.and
      local.tee 0
      i32.add
      local.set 3
      block ;; label = @1
        block ;; label = @2
          local.get 2
          i32.const 1
          i32.and
          br_if 0 (;@2;)
          local.get 2
          i32.const 2
          i32.and
          i32.eqz
          br_if 1 (;@1;)
          local.get 1
          i32.load
          local.tee 2
          local.get 0
          i32.add
          local.set 0
          block ;; label = @3
            local.get 1
            local.get 2
            i32.sub
            local.tee 1
            i32.const 0
            i32.load offset=1056800
            i32.ne
            br_if 0 (;@3;)
            local.get 3
            i32.load offset=4
            i32.const 3
            i32.and
            i32.const 3
            i32.ne
            br_if 1 (;@2;)
            i32.const 0
            local.get 0
            i32.store offset=1056792
            local.get 3
            local.get 3
            i32.load offset=4
            i32.const -2
            i32.and
            i32.store offset=4
            local.get 1
            local.get 0
            i32.const 1
            i32.or
            i32.store offset=4
            local.get 3
            local.get 0
            i32.store
            return
          end
          local.get 1
          local.get 2
          call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$12unlink_chunk17hc085d75ef8348387E
        end
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  block ;; label = @7
                    local.get 3
                    i32.load offset=4
                    local.tee 2
                    i32.const 2
                    i32.and
                    br_if 0 (;@7;)
                    local.get 3
                    i32.const 0
                    i32.load offset=1056804
                    i32.eq
                    br_if 2 (;@5;)
                    local.get 3
                    i32.const 0
                    i32.load offset=1056800
                    i32.eq
                    br_if 3 (;@4;)
                    local.get 3
                    local.get 2
                    i32.const -8
                    i32.and
                    local.tee 2
                    call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$12unlink_chunk17hc085d75ef8348387E
                    local.get 1
                    local.get 2
                    local.get 0
                    i32.add
                    local.tee 0
                    i32.const 1
                    i32.or
                    i32.store offset=4
                    local.get 1
                    local.get 0
                    i32.add
                    local.get 0
                    i32.store
                    local.get 1
                    i32.const 0
                    i32.load offset=1056800
                    i32.ne
                    br_if 1 (;@6;)
                    i32.const 0
                    local.get 0
                    i32.store offset=1056792
                    return
                  end
                  local.get 3
                  local.get 2
                  i32.const -2
                  i32.and
                  i32.store offset=4
                  local.get 1
                  local.get 0
                  i32.const 1
                  i32.or
                  i32.store offset=4
                  local.get 1
                  local.get 0
                  i32.add
                  local.get 0
                  i32.store
                end
                local.get 0
                i32.const 256
                i32.lt_u
                br_if 2 (;@3;)
                local.get 1
                local.get 0
                call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$18insert_large_chunk17h6d1973057372779eE
                i32.const 0
                local.set 1
                i32.const 0
                i32.const 0
                i32.load offset=1056824
                i32.const -1
                i32.add
                local.tee 0
                i32.store offset=1056824
                local.get 0
                br_if 4 (;@1;)
                block ;; label = @6
                  i32.const 0
                  i32.load offset=1056512
                  local.tee 0
                  i32.eqz
                  br_if 0 (;@6;)
                  i32.const 0
                  local.set 1
                  loop ;; label = @7
                    local.get 1
                    i32.const 1
                    i32.add
                    local.set 1
                    local.get 0
                    i32.load offset=8
                    local.tee 0
                    br_if 0 (;@7;)
                  end
                end
                i32.const 0
                local.get 1
                i32.const 4095
                local.get 1
                i32.const 4095
                i32.gt_u
                select
                i32.store offset=1056824
                return
              end
              i32.const 0
              local.get 1
              i32.store offset=1056804
              i32.const 0
              i32.const 0
              i32.load offset=1056796
              local.get 0
              i32.add
              local.tee 0
              i32.store offset=1056796
              local.get 1
              local.get 0
              i32.const 1
              i32.or
              i32.store offset=4
              block ;; label = @5
                local.get 1
                i32.const 0
                i32.load offset=1056800
                i32.ne
                br_if 0 (;@5;)
                i32.const 0
                i32.const 0
                i32.store offset=1056792
                i32.const 0
                i32.const 0
                i32.store offset=1056800
              end
              local.get 0
              i32.const 0
              i32.load offset=1056816
              local.tee 4
              i32.le_u
              br_if 3 (;@1;)
              i32.const 0
              i32.load offset=1056804
              local.tee 0
              i32.eqz
              br_if 3 (;@1;)
              i32.const 0
              local.set 2
              i32.const 0
              i32.load offset=1056796
              local.tee 5
              i32.const 41
              i32.lt_u
              br_if 2 (;@2;)
              i32.const 1056504
              local.set 1
              loop ;; label = @5
                block ;; label = @6
                  local.get 1
                  i32.load
                  local.tee 3
                  local.get 0
                  i32.gt_u
                  br_if 0 (;@6;)
                  local.get 0
                  local.get 3
                  local.get 1
                  i32.load offset=4
                  i32.add
                  i32.lt_u
                  br_if 4 (;@2;)
                end
                local.get 1
                i32.load offset=8
                local.set 1
                br 0 (;@5;)
              end
            end
            i32.const 0
            local.get 1
            i32.store offset=1056800
            i32.const 0
            i32.const 0
            i32.load offset=1056792
            local.get 0
            i32.add
            local.tee 0
            i32.store offset=1056792
            local.get 1
            local.get 0
            i32.const 1
            i32.or
            i32.store offset=4
            local.get 1
            local.get 0
            i32.add
            local.get 0
            i32.store
            return
          end
          local.get 0
          i32.const -8
          i32.and
          i32.const 1056520
          i32.add
          local.set 3
          block ;; label = @3
            block ;; label = @4
              i32.const 0
              i32.load offset=1056784
              local.tee 2
              i32.const 1
              local.get 0
              i32.const 3
              i32.shr_u
              i32.shl
              local.tee 0
              i32.and
              br_if 0 (;@4;)
              i32.const 0
              local.get 2
              local.get 0
              i32.or
              i32.store offset=1056784
              local.get 3
              local.set 0
              br 1 (;@3;)
            end
            local.get 3
            i32.load offset=8
            local.set 0
          end
          local.get 3
          local.get 1
          i32.store offset=8
          local.get 0
          local.get 1
          i32.store offset=12
          local.get 1
          local.get 3
          i32.store offset=12
          local.get 1
          local.get 0
          i32.store offset=8
          return
        end
        block ;; label = @2
          i32.const 0
          i32.load offset=1056512
          local.tee 1
          i32.eqz
          br_if 0 (;@2;)
          i32.const 0
          local.set 2
          loop ;; label = @3
            local.get 2
            i32.const 1
            i32.add
            local.set 2
            local.get 1
            i32.load offset=8
            local.tee 1
            br_if 0 (;@3;)
          end
        end
        i32.const 0
        local.get 2
        i32.const 4095
        local.get 2
        i32.const 4095
        i32.gt_u
        select
        i32.store offset=1056824
        local.get 5
        local.get 4
        i32.le_u
        br_if 0 (;@1;)
        i32.const 0
        i32.const -1
        i32.store offset=1056816
      end
    )
    (func $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$6malloc17h78827ecb73929b04E (;109;) (type 3) (param i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i64)
      global.get $__stack_pointer
      i32.const 16
      i32.sub
      local.tee 1
      global.set $__stack_pointer
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  block ;; label = @7
                    block ;; label = @8
                      local.get 0
                      i32.const 245
                      i32.lt_u
                      br_if 0 (;@8;)
                      i32.const 0
                      local.set 2
                      local.get 0
                      i32.const -65587
                      i32.ge_u
                      br_if 7 (;@1;)
                      local.get 0
                      i32.const 11
                      i32.add
                      local.tee 0
                      i32.const -8
                      i32.and
                      local.set 3
                      i32.const 0
                      i32.load offset=1056788
                      local.tee 4
                      i32.eqz
                      br_if 4 (;@4;)
                      i32.const 0
                      local.set 5
                      block ;; label = @9
                        local.get 3
                        i32.const 256
                        i32.lt_u
                        br_if 0 (;@9;)
                        i32.const 31
                        local.set 5
                        local.get 3
                        i32.const 16777215
                        i32.gt_u
                        br_if 0 (;@9;)
                        local.get 3
                        i32.const 6
                        local.get 0
                        i32.const 8
                        i32.shr_u
                        i32.clz
                        local.tee 0
                        i32.sub
                        i32.shr_u
                        i32.const 1
                        i32.and
                        local.get 0
                        i32.const 1
                        i32.shl
                        i32.sub
                        i32.const 62
                        i32.add
                        local.set 5
                      end
                      i32.const 0
                      local.get 3
                      i32.sub
                      local.set 2
                      block ;; label = @9
                        local.get 5
                        i32.const 2
                        i32.shl
                        i32.const 1056376
                        i32.add
                        i32.load
                        local.tee 6
                        br_if 0 (;@9;)
                        i32.const 0
                        local.set 0
                        i32.const 0
                        local.set 7
                        br 2 (;@7;)
                      end
                      i32.const 0
                      local.set 0
                      local.get 3
                      i32.const 0
                      i32.const 25
                      local.get 5
                      i32.const 1
                      i32.shr_u
                      i32.sub
                      local.get 5
                      i32.const 31
                      i32.eq
                      select
                      i32.shl
                      local.set 8
                      i32.const 0
                      local.set 7
                      loop ;; label = @9
                        block ;; label = @10
                          local.get 6
                          local.tee 6
                          i32.load offset=4
                          i32.const -8
                          i32.and
                          local.tee 9
                          local.get 3
                          i32.lt_u
                          br_if 0 (;@10;)
                          local.get 9
                          local.get 3
                          i32.sub
                          local.tee 9
                          local.get 2
                          i32.ge_u
                          br_if 0 (;@10;)
                          local.get 9
                          local.set 2
                          local.get 6
                          local.set 7
                          local.get 9
                          br_if 0 (;@10;)
                          i32.const 0
                          local.set 2
                          local.get 6
                          local.set 7
                          local.get 6
                          local.set 0
                          br 4 (;@6;)
                        end
                        local.get 6
                        i32.load offset=20
                        local.tee 9
                        local.get 0
                        local.get 9
                        local.get 6
                        local.get 8
                        i32.const 29
                        i32.shr_u
                        i32.const 4
                        i32.and
                        i32.add
                        i32.const 16
                        i32.add
                        i32.load
                        local.tee 6
                        i32.ne
                        select
                        local.get 0
                        local.get 9
                        select
                        local.set 0
                        local.get 8
                        i32.const 1
                        i32.shl
                        local.set 8
                        local.get 6
                        i32.eqz
                        br_if 2 (;@7;)
                        br 0 (;@9;)
                      end
                    end
                    block ;; label = @8
                      i32.const 0
                      i32.load offset=1056784
                      local.tee 6
                      i32.const 16
                      local.get 0
                      i32.const 11
                      i32.add
                      i32.const 504
                      i32.and
                      local.get 0
                      i32.const 11
                      i32.lt_u
                      select
                      local.tee 3
                      i32.const 3
                      i32.shr_u
                      local.tee 2
                      i32.shr_u
                      local.tee 0
                      i32.const 3
                      i32.and
                      i32.eqz
                      br_if 0 (;@8;)
                      block ;; label = @9
                        block ;; label = @10
                          local.get 0
                          i32.const -1
                          i32.xor
                          i32.const 1
                          i32.and
                          local.get 2
                          i32.add
                          local.tee 3
                          i32.const 3
                          i32.shl
                          local.tee 0
                          i32.const 1056520
                          i32.add
                          local.tee 2
                          local.get 0
                          i32.const 1056528
                          i32.add
                          i32.load
                          local.tee 0
                          i32.load offset=8
                          local.tee 7
                          i32.eq
                          br_if 0 (;@10;)
                          local.get 7
                          local.get 2
                          i32.store offset=12
                          local.get 2
                          local.get 7
                          i32.store offset=8
                          br 1 (;@9;)
                        end
                        i32.const 0
                        local.get 6
                        i32.const -2
                        local.get 3
                        i32.rotl
                        i32.and
                        i32.store offset=1056784
                      end
                      local.get 0
                      i32.const 8
                      i32.add
                      local.set 2
                      local.get 0
                      local.get 3
                      i32.const 3
                      i32.shl
                      local.tee 3
                      i32.const 3
                      i32.or
                      i32.store offset=4
                      local.get 0
                      local.get 3
                      i32.add
                      local.tee 0
                      local.get 0
                      i32.load offset=4
                      i32.const 1
                      i32.or
                      i32.store offset=4
                      br 7 (;@1;)
                    end
                    local.get 3
                    i32.const 0
                    i32.load offset=1056792
                    i32.le_u
                    br_if 3 (;@4;)
                    block ;; label = @8
                      block ;; label = @9
                        block ;; label = @10
                          local.get 0
                          br_if 0 (;@10;)
                          i32.const 0
                          i32.load offset=1056788
                          local.tee 0
                          i32.eqz
                          br_if 6 (;@4;)
                          local.get 0
                          i32.ctz
                          i32.const 2
                          i32.shl
                          i32.const 1056376
                          i32.add
                          i32.load
                          local.tee 7
                          i32.load offset=4
                          i32.const -8
                          i32.and
                          local.get 3
                          i32.sub
                          local.set 2
                          local.get 7
                          local.set 6
                          loop ;; label = @11
                            block ;; label = @12
                              local.get 7
                              i32.load offset=16
                              local.tee 0
                              br_if 0 (;@12;)
                              local.get 7
                              i32.load offset=20
                              local.tee 0
                              br_if 0 (;@12;)
                              local.get 6
                              i32.load offset=24
                              local.set 5
                              block ;; label = @13
                                block ;; label = @14
                                  block ;; label = @15
                                    local.get 6
                                    i32.load offset=12
                                    local.tee 0
                                    local.get 6
                                    i32.ne
                                    br_if 0 (;@15;)
                                    local.get 6
                                    i32.const 20
                                    i32.const 16
                                    local.get 6
                                    i32.load offset=20
                                    local.tee 0
                                    select
                                    i32.add
                                    i32.load
                                    local.tee 7
                                    br_if 1 (;@14;)
                                    i32.const 0
                                    local.set 0
                                    br 2 (;@13;)
                                  end
                                  local.get 6
                                  i32.load offset=8
                                  local.tee 7
                                  local.get 0
                                  i32.store offset=12
                                  local.get 0
                                  local.get 7
                                  i32.store offset=8
                                  br 1 (;@13;)
                                end
                                local.get 6
                                i32.const 20
                                i32.add
                                local.get 6
                                i32.const 16
                                i32.add
                                local.get 0
                                select
                                local.set 8
                                loop ;; label = @14
                                  local.get 8
                                  local.set 9
                                  local.get 7
                                  local.tee 0
                                  i32.const 20
                                  i32.add
                                  local.get 0
                                  i32.const 16
                                  i32.add
                                  local.get 0
                                  i32.load offset=20
                                  local.tee 7
                                  select
                                  local.set 8
                                  local.get 0
                                  i32.const 20
                                  i32.const 16
                                  local.get 7
                                  select
                                  i32.add
                                  i32.load
                                  local.tee 7
                                  br_if 0 (;@14;)
                                end
                                local.get 9
                                i32.const 0
                                i32.store
                              end
                              local.get 5
                              i32.eqz
                              br_if 4 (;@8;)
                              block ;; label = @13
                                local.get 6
                                i32.load offset=28
                                i32.const 2
                                i32.shl
                                i32.const 1056376
                                i32.add
                                local.tee 7
                                i32.load
                                local.get 6
                                i32.eq
                                br_if 0 (;@13;)
                                local.get 5
                                i32.const 16
                                i32.const 20
                                local.get 5
                                i32.load offset=16
                                local.get 6
                                i32.eq
                                select
                                i32.add
                                local.get 0
                                i32.store
                                local.get 0
                                i32.eqz
                                br_if 5 (;@8;)
                                br 4 (;@9;)
                              end
                              local.get 7
                              local.get 0
                              i32.store
                              local.get 0
                              br_if 3 (;@9;)
                              i32.const 0
                              i32.const 0
                              i32.load offset=1056788
                              i32.const -2
                              local.get 6
                              i32.load offset=28
                              i32.rotl
                              i32.and
                              i32.store offset=1056788
                              br 4 (;@8;)
                            end
                            local.get 0
                            i32.load offset=4
                            i32.const -8
                            i32.and
                            local.get 3
                            i32.sub
                            local.tee 7
                            local.get 2
                            local.get 7
                            local.get 2
                            i32.lt_u
                            local.tee 7
                            select
                            local.set 2
                            local.get 0
                            local.get 6
                            local.get 7
                            select
                            local.set 6
                            local.get 0
                            local.set 7
                            br 0 (;@11;)
                          end
                        end
                        block ;; label = @10
                          block ;; label = @11
                            local.get 0
                            local.get 2
                            i32.shl
                            i32.const 2
                            local.get 2
                            i32.shl
                            local.tee 0
                            i32.const 0
                            local.get 0
                            i32.sub
                            i32.or
                            i32.and
                            i32.ctz
                            local.tee 2
                            i32.const 3
                            i32.shl
                            local.tee 0
                            i32.const 1056520
                            i32.add
                            local.tee 7
                            local.get 0
                            i32.const 1056528
                            i32.add
                            i32.load
                            local.tee 0
                            i32.load offset=8
                            local.tee 8
                            i32.eq
                            br_if 0 (;@11;)
                            local.get 8
                            local.get 7
                            i32.store offset=12
                            local.get 7
                            local.get 8
                            i32.store offset=8
                            br 1 (;@10;)
                          end
                          i32.const 0
                          local.get 6
                          i32.const -2
                          local.get 2
                          i32.rotl
                          i32.and
                          i32.store offset=1056784
                        end
                        local.get 0
                        local.get 3
                        i32.const 3
                        i32.or
                        i32.store offset=4
                        local.get 0
                        local.get 3
                        i32.add
                        local.tee 8
                        local.get 2
                        i32.const 3
                        i32.shl
                        local.tee 2
                        local.get 3
                        i32.sub
                        local.tee 7
                        i32.const 1
                        i32.or
                        i32.store offset=4
                        local.get 0
                        local.get 2
                        i32.add
                        local.get 7
                        i32.store
                        block ;; label = @10
                          i32.const 0
                          i32.load offset=1056792
                          local.tee 6
                          i32.eqz
                          br_if 0 (;@10;)
                          local.get 6
                          i32.const -8
                          i32.and
                          i32.const 1056520
                          i32.add
                          local.set 2
                          i32.const 0
                          i32.load offset=1056800
                          local.set 3
                          block ;; label = @11
                            block ;; label = @12
                              i32.const 0
                              i32.load offset=1056784
                              local.tee 9
                              i32.const 1
                              local.get 6
                              i32.const 3
                              i32.shr_u
                              i32.shl
                              local.tee 6
                              i32.and
                              br_if 0 (;@12;)
                              i32.const 0
                              local.get 9
                              local.get 6
                              i32.or
                              i32.store offset=1056784
                              local.get 2
                              local.set 6
                              br 1 (;@11;)
                            end
                            local.get 2
                            i32.load offset=8
                            local.set 6
                          end
                          local.get 2
                          local.get 3
                          i32.store offset=8
                          local.get 6
                          local.get 3
                          i32.store offset=12
                          local.get 3
                          local.get 2
                          i32.store offset=12
                          local.get 3
                          local.get 6
                          i32.store offset=8
                        end
                        local.get 0
                        i32.const 8
                        i32.add
                        local.set 2
                        i32.const 0
                        local.get 8
                        i32.store offset=1056800
                        i32.const 0
                        local.get 7
                        i32.store offset=1056792
                        br 8 (;@1;)
                      end
                      local.get 0
                      local.get 5
                      i32.store offset=24
                      block ;; label = @9
                        local.get 6
                        i32.load offset=16
                        local.tee 7
                        i32.eqz
                        br_if 0 (;@9;)
                        local.get 0
                        local.get 7
                        i32.store offset=16
                        local.get 7
                        local.get 0
                        i32.store offset=24
                      end
                      local.get 6
                      i32.load offset=20
                      local.tee 7
                      i32.eqz
                      br_if 0 (;@8;)
                      local.get 0
                      local.get 7
                      i32.store offset=20
                      local.get 7
                      local.get 0
                      i32.store offset=24
                    end
                    block ;; label = @8
                      block ;; label = @9
                        block ;; label = @10
                          local.get 2
                          i32.const 16
                          i32.lt_u
                          br_if 0 (;@10;)
                          local.get 6
                          local.get 3
                          i32.const 3
                          i32.or
                          i32.store offset=4
                          local.get 6
                          local.get 3
                          i32.add
                          local.tee 3
                          local.get 2
                          i32.const 1
                          i32.or
                          i32.store offset=4
                          local.get 3
                          local.get 2
                          i32.add
                          local.get 2
                          i32.store
                          i32.const 0
                          i32.load offset=1056792
                          local.tee 8
                          i32.eqz
                          br_if 1 (;@9;)
                          local.get 8
                          i32.const -8
                          i32.and
                          i32.const 1056520
                          i32.add
                          local.set 7
                          i32.const 0
                          i32.load offset=1056800
                          local.set 0
                          block ;; label = @11
                            block ;; label = @12
                              i32.const 0
                              i32.load offset=1056784
                              local.tee 9
                              i32.const 1
                              local.get 8
                              i32.const 3
                              i32.shr_u
                              i32.shl
                              local.tee 8
                              i32.and
                              br_if 0 (;@12;)
                              i32.const 0
                              local.get 9
                              local.get 8
                              i32.or
                              i32.store offset=1056784
                              local.get 7
                              local.set 8
                              br 1 (;@11;)
                            end
                            local.get 7
                            i32.load offset=8
                            local.set 8
                          end
                          local.get 7
                          local.get 0
                          i32.store offset=8
                          local.get 8
                          local.get 0
                          i32.store offset=12
                          local.get 0
                          local.get 7
                          i32.store offset=12
                          local.get 0
                          local.get 8
                          i32.store offset=8
                          br 1 (;@9;)
                        end
                        local.get 6
                        local.get 2
                        local.get 3
                        i32.add
                        local.tee 0
                        i32.const 3
                        i32.or
                        i32.store offset=4
                        local.get 6
                        local.get 0
                        i32.add
                        local.tee 0
                        local.get 0
                        i32.load offset=4
                        i32.const 1
                        i32.or
                        i32.store offset=4
                        br 1 (;@8;)
                      end
                      i32.const 0
                      local.get 3
                      i32.store offset=1056800
                      i32.const 0
                      local.get 2
                      i32.store offset=1056792
                    end
                    local.get 6
                    i32.const 8
                    i32.add
                    local.set 2
                    br 6 (;@1;)
                  end
                  block ;; label = @7
                    local.get 0
                    local.get 7
                    i32.or
                    br_if 0 (;@7;)
                    i32.const 0
                    local.set 7
                    i32.const 2
                    local.get 5
                    i32.shl
                    local.tee 0
                    i32.const 0
                    local.get 0
                    i32.sub
                    i32.or
                    local.get 4
                    i32.and
                    local.tee 0
                    i32.eqz
                    br_if 3 (;@4;)
                    local.get 0
                    i32.ctz
                    i32.const 2
                    i32.shl
                    i32.const 1056376
                    i32.add
                    i32.load
                    local.set 0
                  end
                  local.get 0
                  i32.eqz
                  br_if 1 (;@5;)
                end
                loop ;; label = @6
                  local.get 0
                  local.get 7
                  local.get 0
                  i32.load offset=4
                  i32.const -8
                  i32.and
                  local.tee 6
                  local.get 3
                  i32.sub
                  local.tee 9
                  local.get 2
                  i32.lt_u
                  local.tee 5
                  select
                  local.set 4
                  local.get 6
                  local.get 3
                  i32.lt_u
                  local.set 8
                  local.get 9
                  local.get 2
                  local.get 5
                  select
                  local.set 9
                  block ;; label = @7
                    local.get 0
                    i32.load offset=16
                    local.tee 6
                    br_if 0 (;@7;)
                    local.get 0
                    i32.load offset=20
                    local.set 6
                  end
                  local.get 7
                  local.get 4
                  local.get 8
                  select
                  local.set 7
                  local.get 2
                  local.get 9
                  local.get 8
                  select
                  local.set 2
                  local.get 6
                  local.set 0
                  local.get 6
                  br_if 0 (;@6;)
                end
              end
              local.get 7
              i32.eqz
              br_if 0 (;@4;)
              block ;; label = @5
                i32.const 0
                i32.load offset=1056792
                local.tee 0
                local.get 3
                i32.lt_u
                br_if 0 (;@5;)
                local.get 2
                local.get 0
                local.get 3
                i32.sub
                i32.ge_u
                br_if 1 (;@4;)
              end
              local.get 7
              i32.load offset=24
              local.set 5
              block ;; label = @5
                block ;; label = @6
                  block ;; label = @7
                    local.get 7
                    i32.load offset=12
                    local.tee 0
                    local.get 7
                    i32.ne
                    br_if 0 (;@7;)
                    local.get 7
                    i32.const 20
                    i32.const 16
                    local.get 7
                    i32.load offset=20
                    local.tee 0
                    select
                    i32.add
                    i32.load
                    local.tee 6
                    br_if 1 (;@6;)
                    i32.const 0
                    local.set 0
                    br 2 (;@5;)
                  end
                  local.get 7
                  i32.load offset=8
                  local.tee 6
                  local.get 0
                  i32.store offset=12
                  local.get 0
                  local.get 6
                  i32.store offset=8
                  br 1 (;@5;)
                end
                local.get 7
                i32.const 20
                i32.add
                local.get 7
                i32.const 16
                i32.add
                local.get 0
                select
                local.set 8
                loop ;; label = @6
                  local.get 8
                  local.set 9
                  local.get 6
                  local.tee 0
                  i32.const 20
                  i32.add
                  local.get 0
                  i32.const 16
                  i32.add
                  local.get 0
                  i32.load offset=20
                  local.tee 6
                  select
                  local.set 8
                  local.get 0
                  i32.const 20
                  i32.const 16
                  local.get 6
                  select
                  i32.add
                  i32.load
                  local.tee 6
                  br_if 0 (;@6;)
                end
                local.get 9
                i32.const 0
                i32.store
              end
              local.get 5
              i32.eqz
              br_if 2 (;@2;)
              block ;; label = @5
                local.get 7
                i32.load offset=28
                i32.const 2
                i32.shl
                i32.const 1056376
                i32.add
                local.tee 6
                i32.load
                local.get 7
                i32.eq
                br_if 0 (;@5;)
                local.get 5
                i32.const 16
                i32.const 20
                local.get 5
                i32.load offset=16
                local.get 7
                i32.eq
                select
                i32.add
                local.get 0
                i32.store
                local.get 0
                i32.eqz
                br_if 3 (;@2;)
                br 2 (;@3;)
              end
              local.get 6
              local.get 0
              i32.store
              local.get 0
              br_if 1 (;@3;)
              i32.const 0
              i32.const 0
              i32.load offset=1056788
              i32.const -2
              local.get 7
              i32.load offset=28
              i32.rotl
              i32.and
              i32.store offset=1056788
              br 2 (;@2;)
            end
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  block ;; label = @7
                    block ;; label = @8
                      block ;; label = @9
                        i32.const 0
                        i32.load offset=1056792
                        local.tee 0
                        local.get 3
                        i32.ge_u
                        br_if 0 (;@9;)
                        block ;; label = @10
                          i32.const 0
                          i32.load offset=1056796
                          local.tee 0
                          local.get 3
                          i32.gt_u
                          br_if 0 (;@10;)
                          local.get 1
                          i32.const 4
                          i32.add
                          i32.const 1056828
                          local.get 3
                          i32.const 65583
                          i32.add
                          i32.const -65536
                          i32.and
                          call $_ZN61_$LT$dlmalloc..sys..System$u20$as$u20$dlmalloc..Allocator$GT$5alloc17h282328c9a7a8a484E
                          block ;; label = @11
                            local.get 1
                            i32.load offset=4
                            local.tee 6
                            br_if 0 (;@11;)
                            i32.const 0
                            local.set 2
                            br 10 (;@1;)
                          end
                          local.get 1
                          i32.load offset=12
                          local.set 5
                          i32.const 0
                          i32.const 0
                          i32.load offset=1056808
                          local.get 1
                          i32.load offset=8
                          local.tee 9
                          i32.add
                          local.tee 0
                          i32.store offset=1056808
                          i32.const 0
                          i32.const 0
                          i32.load offset=1056812
                          local.tee 2
                          local.get 0
                          local.get 2
                          local.get 0
                          i32.gt_u
                          select
                          i32.store offset=1056812
                          block ;; label = @11
                            block ;; label = @12
                              block ;; label = @13
                                i32.const 0
                                i32.load offset=1056804
                                local.tee 2
                                i32.eqz
                                br_if 0 (;@13;)
                                i32.const 1056504
                                local.set 0
                                loop ;; label = @14
                                  local.get 6
                                  local.get 0
                                  i32.load
                                  local.tee 7
                                  local.get 0
                                  i32.load offset=4
                                  local.tee 8
                                  i32.add
                                  i32.eq
                                  br_if 2 (;@12;)
                                  local.get 0
                                  i32.load offset=8
                                  local.tee 0
                                  br_if 0 (;@14;)
                                  br 3 (;@11;)
                                end
                              end
                              block ;; label = @13
                                block ;; label = @14
                                  i32.const 0
                                  i32.load offset=1056820
                                  local.tee 0
                                  i32.eqz
                                  br_if 0 (;@14;)
                                  local.get 6
                                  local.get 0
                                  i32.ge_u
                                  br_if 1 (;@13;)
                                end
                                i32.const 0
                                local.get 6
                                i32.store offset=1056820
                              end
                              i32.const 0
                              i32.const 4095
                              i32.store offset=1056824
                              i32.const 0
                              local.get 5
                              i32.store offset=1056516
                              i32.const 0
                              local.get 9
                              i32.store offset=1056508
                              i32.const 0
                              local.get 6
                              i32.store offset=1056504
                              i32.const 0
                              i32.const 1056520
                              i32.store offset=1056532
                              i32.const 0
                              i32.const 1056528
                              i32.store offset=1056540
                              i32.const 0
                              i32.const 1056520
                              i32.store offset=1056528
                              i32.const 0
                              i32.const 1056536
                              i32.store offset=1056548
                              i32.const 0
                              i32.const 1056528
                              i32.store offset=1056536
                              i32.const 0
                              i32.const 1056544
                              i32.store offset=1056556
                              i32.const 0
                              i32.const 1056536
                              i32.store offset=1056544
                              i32.const 0
                              i32.const 1056552
                              i32.store offset=1056564
                              i32.const 0
                              i32.const 1056544
                              i32.store offset=1056552
                              i32.const 0
                              i32.const 1056560
                              i32.store offset=1056572
                              i32.const 0
                              i32.const 1056552
                              i32.store offset=1056560
                              i32.const 0
                              i32.const 1056568
                              i32.store offset=1056580
                              i32.const 0
                              i32.const 1056560
                              i32.store offset=1056568
                              i32.const 0
                              i32.const 1056576
                              i32.store offset=1056588
                              i32.const 0
                              i32.const 1056568
                              i32.store offset=1056576
                              i32.const 0
                              i32.const 1056584
                              i32.store offset=1056596
                              i32.const 0
                              i32.const 1056576
                              i32.store offset=1056584
                              i32.const 0
                              i32.const 1056584
                              i32.store offset=1056592
                              i32.const 0
                              i32.const 1056592
                              i32.store offset=1056604
                              i32.const 0
                              i32.const 1056592
                              i32.store offset=1056600
                              i32.const 0
                              i32.const 1056600
                              i32.store offset=1056612
                              i32.const 0
                              i32.const 1056600
                              i32.store offset=1056608
                              i32.const 0
                              i32.const 1056608
                              i32.store offset=1056620
                              i32.const 0
                              i32.const 1056608
                              i32.store offset=1056616
                              i32.const 0
                              i32.const 1056616
                              i32.store offset=1056628
                              i32.const 0
                              i32.const 1056616
                              i32.store offset=1056624
                              i32.const 0
                              i32.const 1056624
                              i32.store offset=1056636
                              i32.const 0
                              i32.const 1056624
                              i32.store offset=1056632
                              i32.const 0
                              i32.const 1056632
                              i32.store offset=1056644
                              i32.const 0
                              i32.const 1056632
                              i32.store offset=1056640
                              i32.const 0
                              i32.const 1056640
                              i32.store offset=1056652
                              i32.const 0
                              i32.const 1056640
                              i32.store offset=1056648
                              i32.const 0
                              i32.const 1056648
                              i32.store offset=1056660
                              i32.const 0
                              i32.const 1056656
                              i32.store offset=1056668
                              i32.const 0
                              i32.const 1056648
                              i32.store offset=1056656
                              i32.const 0
                              i32.const 1056664
                              i32.store offset=1056676
                              i32.const 0
                              i32.const 1056656
                              i32.store offset=1056664
                              i32.const 0
                              i32.const 1056672
                              i32.store offset=1056684
                              i32.const 0
                              i32.const 1056664
                              i32.store offset=1056672
                              i32.const 0
                              i32.const 1056680
                              i32.store offset=1056692
                              i32.const 0
                              i32.const 1056672
                              i32.store offset=1056680
                              i32.const 0
                              i32.const 1056688
                              i32.store offset=1056700
                              i32.const 0
                              i32.const 1056680
                              i32.store offset=1056688
                              i32.const 0
                              i32.const 1056696
                              i32.store offset=1056708
                              i32.const 0
                              i32.const 1056688
                              i32.store offset=1056696
                              i32.const 0
                              i32.const 1056704
                              i32.store offset=1056716
                              i32.const 0
                              i32.const 1056696
                              i32.store offset=1056704
                              i32.const 0
                              i32.const 1056712
                              i32.store offset=1056724
                              i32.const 0
                              i32.const 1056704
                              i32.store offset=1056712
                              i32.const 0
                              i32.const 1056720
                              i32.store offset=1056732
                              i32.const 0
                              i32.const 1056712
                              i32.store offset=1056720
                              i32.const 0
                              i32.const 1056728
                              i32.store offset=1056740
                              i32.const 0
                              i32.const 1056720
                              i32.store offset=1056728
                              i32.const 0
                              i32.const 1056736
                              i32.store offset=1056748
                              i32.const 0
                              i32.const 1056728
                              i32.store offset=1056736
                              i32.const 0
                              i32.const 1056744
                              i32.store offset=1056756
                              i32.const 0
                              i32.const 1056736
                              i32.store offset=1056744
                              i32.const 0
                              i32.const 1056752
                              i32.store offset=1056764
                              i32.const 0
                              i32.const 1056744
                              i32.store offset=1056752
                              i32.const 0
                              i32.const 1056760
                              i32.store offset=1056772
                              i32.const 0
                              i32.const 1056752
                              i32.store offset=1056760
                              i32.const 0
                              i32.const 1056768
                              i32.store offset=1056780
                              i32.const 0
                              i32.const 1056760
                              i32.store offset=1056768
                              i32.const 0
                              local.get 6
                              i32.const 15
                              i32.add
                              i32.const -8
                              i32.and
                              local.tee 0
                              i32.const -8
                              i32.add
                              local.tee 2
                              i32.store offset=1056804
                              i32.const 0
                              i32.const 1056768
                              i32.store offset=1056776
                              i32.const 0
                              local.get 6
                              local.get 0
                              i32.sub
                              local.get 9
                              i32.const -40
                              i32.add
                              local.tee 0
                              i32.add
                              i32.const 8
                              i32.add
                              local.tee 7
                              i32.store offset=1056796
                              local.get 2
                              local.get 7
                              i32.const 1
                              i32.or
                              i32.store offset=4
                              local.get 6
                              local.get 0
                              i32.add
                              i32.const 40
                              i32.store offset=4
                              i32.const 0
                              i32.const 2097152
                              i32.store offset=1056816
                              br 8 (;@4;)
                            end
                            local.get 2
                            local.get 6
                            i32.ge_u
                            br_if 0 (;@11;)
                            local.get 7
                            local.get 2
                            i32.gt_u
                            br_if 0 (;@11;)
                            local.get 0
                            i32.load offset=12
                            local.tee 7
                            i32.const 1
                            i32.and
                            br_if 0 (;@11;)
                            local.get 7
                            i32.const 1
                            i32.shr_u
                            local.get 5
                            i32.eq
                            br_if 3 (;@8;)
                          end
                          i32.const 0
                          i32.const 0
                          i32.load offset=1056820
                          local.tee 0
                          local.get 6
                          local.get 6
                          local.get 0
                          i32.gt_u
                          select
                          i32.store offset=1056820
                          local.get 6
                          local.get 9
                          i32.add
                          local.set 7
                          i32.const 1056504
                          local.set 0
                          block ;; label = @11
                            block ;; label = @12
                              block ;; label = @13
                                loop ;; label = @14
                                  local.get 0
                                  i32.load
                                  local.get 7
                                  i32.eq
                                  br_if 1 (;@13;)
                                  local.get 0
                                  i32.load offset=8
                                  local.tee 0
                                  br_if 0 (;@14;)
                                  br 2 (;@12;)
                                end
                              end
                              local.get 0
                              i32.load offset=12
                              local.tee 8
                              i32.const 1
                              i32.and
                              br_if 0 (;@12;)
                              local.get 8
                              i32.const 1
                              i32.shr_u
                              local.get 5
                              i32.eq
                              br_if 1 (;@11;)
                            end
                            i32.const 1056504
                            local.set 0
                            block ;; label = @12
                              loop ;; label = @13
                                block ;; label = @14
                                  local.get 0
                                  i32.load
                                  local.tee 7
                                  local.get 2
                                  i32.gt_u
                                  br_if 0 (;@14;)
                                  local.get 2
                                  local.get 7
                                  local.get 0
                                  i32.load offset=4
                                  i32.add
                                  local.tee 7
                                  i32.lt_u
                                  br_if 2 (;@12;)
                                end
                                local.get 0
                                i32.load offset=8
                                local.set 0
                                br 0 (;@13;)
                              end
                            end
                            i32.const 0
                            local.get 6
                            i32.const 15
                            i32.add
                            i32.const -8
                            i32.and
                            local.tee 0
                            i32.const -8
                            i32.add
                            local.tee 8
                            i32.store offset=1056804
                            i32.const 0
                            local.get 6
                            local.get 0
                            i32.sub
                            local.get 9
                            i32.const -40
                            i32.add
                            local.tee 0
                            i32.add
                            i32.const 8
                            i32.add
                            local.tee 4
                            i32.store offset=1056796
                            local.get 8
                            local.get 4
                            i32.const 1
                            i32.or
                            i32.store offset=4
                            local.get 6
                            local.get 0
                            i32.add
                            i32.const 40
                            i32.store offset=4
                            i32.const 0
                            i32.const 2097152
                            i32.store offset=1056816
                            local.get 2
                            local.get 7
                            i32.const -32
                            i32.add
                            i32.const -8
                            i32.and
                            i32.const -8
                            i32.add
                            local.tee 0
                            local.get 0
                            local.get 2
                            i32.const 16
                            i32.add
                            i32.lt_u
                            select
                            local.tee 8
                            i32.const 27
                            i32.store offset=4
                            i32.const 0
                            i64.load offset=1056504 align=4
                            local.set 10
                            local.get 8
                            i32.const 16
                            i32.add
                            i32.const 0
                            i64.load offset=1056512 align=4
                            i64.store align=4
                            local.get 8
                            local.get 10
                            i64.store offset=8 align=4
                            i32.const 0
                            local.get 5
                            i32.store offset=1056516
                            i32.const 0
                            local.get 9
                            i32.store offset=1056508
                            i32.const 0
                            local.get 6
                            i32.store offset=1056504
                            i32.const 0
                            local.get 8
                            i32.const 8
                            i32.add
                            i32.store offset=1056512
                            local.get 8
                            i32.const 28
                            i32.add
                            local.set 0
                            loop ;; label = @12
                              local.get 0
                              i32.const 7
                              i32.store
                              local.get 0
                              i32.const 4
                              i32.add
                              local.tee 0
                              local.get 7
                              i32.lt_u
                              br_if 0 (;@12;)
                            end
                            local.get 8
                            local.get 2
                            i32.eq
                            br_if 7 (;@4;)
                            local.get 8
                            local.get 8
                            i32.load offset=4
                            i32.const -2
                            i32.and
                            i32.store offset=4
                            local.get 2
                            local.get 8
                            local.get 2
                            i32.sub
                            local.tee 0
                            i32.const 1
                            i32.or
                            i32.store offset=4
                            local.get 8
                            local.get 0
                            i32.store
                            block ;; label = @12
                              local.get 0
                              i32.const 256
                              i32.lt_u
                              br_if 0 (;@12;)
                              local.get 2
                              local.get 0
                              call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$18insert_large_chunk17h6d1973057372779eE
                              br 8 (;@4;)
                            end
                            local.get 0
                            i32.const -8
                            i32.and
                            i32.const 1056520
                            i32.add
                            local.set 7
                            block ;; label = @12
                              block ;; label = @13
                                i32.const 0
                                i32.load offset=1056784
                                local.tee 6
                                i32.const 1
                                local.get 0
                                i32.const 3
                                i32.shr_u
                                i32.shl
                                local.tee 0
                                i32.and
                                br_if 0 (;@13;)
                                i32.const 0
                                local.get 6
                                local.get 0
                                i32.or
                                i32.store offset=1056784
                                local.get 7
                                local.set 0
                                br 1 (;@12;)
                              end
                              local.get 7
                              i32.load offset=8
                              local.set 0
                            end
                            local.get 7
                            local.get 2
                            i32.store offset=8
                            local.get 0
                            local.get 2
                            i32.store offset=12
                            local.get 2
                            local.get 7
                            i32.store offset=12
                            local.get 2
                            local.get 0
                            i32.store offset=8
                            br 7 (;@4;)
                          end
                          local.get 0
                          local.get 6
                          i32.store
                          local.get 0
                          local.get 0
                          i32.load offset=4
                          local.get 9
                          i32.add
                          i32.store offset=4
                          local.get 6
                          i32.const 15
                          i32.add
                          i32.const -8
                          i32.and
                          i32.const -8
                          i32.add
                          local.tee 6
                          local.get 3
                          i32.const 3
                          i32.or
                          i32.store offset=4
                          local.get 7
                          i32.const 15
                          i32.add
                          i32.const -8
                          i32.and
                          i32.const -8
                          i32.add
                          local.tee 2
                          local.get 6
                          local.get 3
                          i32.add
                          local.tee 0
                          i32.sub
                          local.set 3
                          local.get 2
                          i32.const 0
                          i32.load offset=1056804
                          i32.eq
                          br_if 3 (;@7;)
                          local.get 2
                          i32.const 0
                          i32.load offset=1056800
                          i32.eq
                          br_if 4 (;@6;)
                          block ;; label = @11
                            local.get 2
                            i32.load offset=4
                            local.tee 7
                            i32.const 3
                            i32.and
                            i32.const 1
                            i32.ne
                            br_if 0 (;@11;)
                            local.get 2
                            local.get 7
                            i32.const -8
                            i32.and
                            local.tee 7
                            call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$12unlink_chunk17hc085d75ef8348387E
                            local.get 7
                            local.get 3
                            i32.add
                            local.set 3
                            local.get 2
                            local.get 7
                            i32.add
                            local.tee 2
                            i32.load offset=4
                            local.set 7
                          end
                          local.get 2
                          local.get 7
                          i32.const -2
                          i32.and
                          i32.store offset=4
                          local.get 0
                          local.get 3
                          i32.const 1
                          i32.or
                          i32.store offset=4
                          local.get 0
                          local.get 3
                          i32.add
                          local.get 3
                          i32.store
                          block ;; label = @11
                            local.get 3
                            i32.const 256
                            i32.lt_u
                            br_if 0 (;@11;)
                            local.get 0
                            local.get 3
                            call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$18insert_large_chunk17h6d1973057372779eE
                            br 6 (;@5;)
                          end
                          local.get 3
                          i32.const -8
                          i32.and
                          i32.const 1056520
                          i32.add
                          local.set 2
                          block ;; label = @11
                            block ;; label = @12
                              i32.const 0
                              i32.load offset=1056784
                              local.tee 7
                              i32.const 1
                              local.get 3
                              i32.const 3
                              i32.shr_u
                              i32.shl
                              local.tee 3
                              i32.and
                              br_if 0 (;@12;)
                              i32.const 0
                              local.get 7
                              local.get 3
                              i32.or
                              i32.store offset=1056784
                              local.get 2
                              local.set 3
                              br 1 (;@11;)
                            end
                            local.get 2
                            i32.load offset=8
                            local.set 3
                          end
                          local.get 2
                          local.get 0
                          i32.store offset=8
                          local.get 3
                          local.get 0
                          i32.store offset=12
                          local.get 0
                          local.get 2
                          i32.store offset=12
                          local.get 0
                          local.get 3
                          i32.store offset=8
                          br 5 (;@5;)
                        end
                        i32.const 0
                        local.get 0
                        local.get 3
                        i32.sub
                        local.tee 2
                        i32.store offset=1056796
                        i32.const 0
                        i32.const 0
                        i32.load offset=1056804
                        local.tee 0
                        local.get 3
                        i32.add
                        local.tee 7
                        i32.store offset=1056804
                        local.get 7
                        local.get 2
                        i32.const 1
                        i32.or
                        i32.store offset=4
                        local.get 0
                        local.get 3
                        i32.const 3
                        i32.or
                        i32.store offset=4
                        local.get 0
                        i32.const 8
                        i32.add
                        local.set 2
                        br 8 (;@1;)
                      end
                      i32.const 0
                      i32.load offset=1056800
                      local.set 2
                      block ;; label = @9
                        block ;; label = @10
                          local.get 0
                          local.get 3
                          i32.sub
                          local.tee 7
                          i32.const 15
                          i32.gt_u
                          br_if 0 (;@10;)
                          i32.const 0
                          i32.const 0
                          i32.store offset=1056800
                          i32.const 0
                          i32.const 0
                          i32.store offset=1056792
                          local.get 2
                          local.get 0
                          i32.const 3
                          i32.or
                          i32.store offset=4
                          local.get 2
                          local.get 0
                          i32.add
                          local.tee 0
                          local.get 0
                          i32.load offset=4
                          i32.const 1
                          i32.or
                          i32.store offset=4
                          br 1 (;@9;)
                        end
                        i32.const 0
                        local.get 7
                        i32.store offset=1056792
                        i32.const 0
                        local.get 2
                        local.get 3
                        i32.add
                        local.tee 6
                        i32.store offset=1056800
                        local.get 6
                        local.get 7
                        i32.const 1
                        i32.or
                        i32.store offset=4
                        local.get 2
                        local.get 0
                        i32.add
                        local.get 7
                        i32.store
                        local.get 2
                        local.get 3
                        i32.const 3
                        i32.or
                        i32.store offset=4
                      end
                      local.get 2
                      i32.const 8
                      i32.add
                      local.set 2
                      br 7 (;@1;)
                    end
                    local.get 0
                    local.get 8
                    local.get 9
                    i32.add
                    i32.store offset=4
                    i32.const 0
                    i32.const 0
                    i32.load offset=1056804
                    local.tee 0
                    i32.const 15
                    i32.add
                    i32.const -8
                    i32.and
                    local.tee 2
                    i32.const -8
                    i32.add
                    local.tee 7
                    i32.store offset=1056804
                    i32.const 0
                    local.get 0
                    local.get 2
                    i32.sub
                    i32.const 0
                    i32.load offset=1056796
                    local.get 9
                    i32.add
                    local.tee 2
                    i32.add
                    i32.const 8
                    i32.add
                    local.tee 6
                    i32.store offset=1056796
                    local.get 7
                    local.get 6
                    i32.const 1
                    i32.or
                    i32.store offset=4
                    local.get 0
                    local.get 2
                    i32.add
                    i32.const 40
                    i32.store offset=4
                    i32.const 0
                    i32.const 2097152
                    i32.store offset=1056816
                    br 3 (;@4;)
                  end
                  i32.const 0
                  local.get 0
                  i32.store offset=1056804
                  i32.const 0
                  i32.const 0
                  i32.load offset=1056796
                  local.get 3
                  i32.add
                  local.tee 3
                  i32.store offset=1056796
                  local.get 0
                  local.get 3
                  i32.const 1
                  i32.or
                  i32.store offset=4
                  br 1 (;@5;)
                end
                i32.const 0
                local.get 0
                i32.store offset=1056800
                i32.const 0
                i32.const 0
                i32.load offset=1056792
                local.get 3
                i32.add
                local.tee 3
                i32.store offset=1056792
                local.get 0
                local.get 3
                i32.const 1
                i32.or
                i32.store offset=4
                local.get 0
                local.get 3
                i32.add
                local.get 3
                i32.store
              end
              local.get 6
              i32.const 8
              i32.add
              local.set 2
              br 3 (;@1;)
            end
            i32.const 0
            local.set 2
            i32.const 0
            i32.load offset=1056796
            local.tee 0
            local.get 3
            i32.le_u
            br_if 2 (;@1;)
            i32.const 0
            local.get 0
            local.get 3
            i32.sub
            local.tee 2
            i32.store offset=1056796
            i32.const 0
            i32.const 0
            i32.load offset=1056804
            local.tee 0
            local.get 3
            i32.add
            local.tee 7
            i32.store offset=1056804
            local.get 7
            local.get 2
            i32.const 1
            i32.or
            i32.store offset=4
            local.get 0
            local.get 3
            i32.const 3
            i32.or
            i32.store offset=4
            local.get 0
            i32.const 8
            i32.add
            local.set 2
            br 2 (;@1;)
          end
          local.get 0
          local.get 5
          i32.store offset=24
          block ;; label = @3
            local.get 7
            i32.load offset=16
            local.tee 6
            i32.eqz
            br_if 0 (;@3;)
            local.get 0
            local.get 6
            i32.store offset=16
            local.get 6
            local.get 0
            i32.store offset=24
          end
          local.get 7
          i32.load offset=20
          local.tee 6
          i32.eqz
          br_if 0 (;@2;)
          local.get 0
          local.get 6
          i32.store offset=20
          local.get 6
          local.get 0
          i32.store offset=24
        end
        block ;; label = @2
          block ;; label = @3
            local.get 2
            i32.const 16
            i32.lt_u
            br_if 0 (;@3;)
            local.get 7
            local.get 3
            i32.const 3
            i32.or
            i32.store offset=4
            local.get 7
            local.get 3
            i32.add
            local.tee 0
            local.get 2
            i32.const 1
            i32.or
            i32.store offset=4
            local.get 0
            local.get 2
            i32.add
            local.get 2
            i32.store
            block ;; label = @4
              local.get 2
              i32.const 256
              i32.lt_u
              br_if 0 (;@4;)
              local.get 0
              local.get 2
              call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$18insert_large_chunk17h6d1973057372779eE
              br 2 (;@2;)
            end
            local.get 2
            i32.const -8
            i32.and
            i32.const 1056520
            i32.add
            local.set 3
            block ;; label = @4
              block ;; label = @5
                i32.const 0
                i32.load offset=1056784
                local.tee 6
                i32.const 1
                local.get 2
                i32.const 3
                i32.shr_u
                i32.shl
                local.tee 2
                i32.and
                br_if 0 (;@5;)
                i32.const 0
                local.get 6
                local.get 2
                i32.or
                i32.store offset=1056784
                local.get 3
                local.set 2
                br 1 (;@4;)
              end
              local.get 3
              i32.load offset=8
              local.set 2
            end
            local.get 3
            local.get 0
            i32.store offset=8
            local.get 2
            local.get 0
            i32.store offset=12
            local.get 0
            local.get 3
            i32.store offset=12
            local.get 0
            local.get 2
            i32.store offset=8
            br 1 (;@2;)
          end
          local.get 7
          local.get 2
          local.get 3
          i32.add
          local.tee 0
          i32.const 3
          i32.or
          i32.store offset=4
          local.get 7
          local.get 0
          i32.add
          local.tee 0
          local.get 0
          i32.load offset=4
          i32.const 1
          i32.or
          i32.store offset=4
        end
        local.get 7
        i32.const 8
        i32.add
        local.set 2
      end
      local.get 1
      i32.const 16
      i32.add
      global.set $__stack_pointer
      local.get 2
    )
    (func $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$8memalign17h03306062bd0c5f07E (;110;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32)
      i32.const 0
      local.set 2
      block ;; label = @1
        i32.const -65587
        local.get 0
        i32.const 16
        local.get 0
        i32.const 16
        i32.gt_u
        select
        local.tee 0
        i32.sub
        local.get 1
        i32.le_u
        br_if 0 (;@1;)
        local.get 0
        i32.const 16
        local.get 1
        i32.const 11
        i32.add
        i32.const -8
        i32.and
        local.get 1
        i32.const 11
        i32.lt_u
        select
        local.tee 3
        i32.add
        i32.const 12
        i32.add
        call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$6malloc17h78827ecb73929b04E
        local.tee 1
        i32.eqz
        br_if 0 (;@1;)
        local.get 1
        i32.const -8
        i32.add
        local.set 2
        block ;; label = @2
          block ;; label = @3
            local.get 0
            i32.const -1
            i32.add
            local.tee 4
            local.get 1
            i32.and
            br_if 0 (;@3;)
            local.get 2
            local.set 0
            br 1 (;@2;)
          end
          local.get 1
          i32.const -4
          i32.add
          local.tee 5
          i32.load
          local.tee 6
          i32.const -8
          i32.and
          local.get 4
          local.get 1
          i32.add
          i32.const 0
          local.get 0
          i32.sub
          i32.and
          i32.const -8
          i32.add
          local.tee 1
          i32.const 0
          local.get 0
          local.get 1
          local.get 2
          i32.sub
          i32.const 16
          i32.gt_u
          select
          i32.add
          local.tee 0
          local.get 2
          i32.sub
          local.tee 1
          i32.sub
          local.set 4
          block ;; label = @3
            local.get 6
            i32.const 3
            i32.and
            i32.eqz
            br_if 0 (;@3;)
            local.get 0
            local.get 4
            local.get 0
            i32.load offset=4
            i32.const 1
            i32.and
            i32.or
            i32.const 2
            i32.or
            i32.store offset=4
            local.get 0
            local.get 4
            i32.add
            local.tee 4
            local.get 4
            i32.load offset=4
            i32.const 1
            i32.or
            i32.store offset=4
            local.get 5
            local.get 1
            local.get 5
            i32.load
            i32.const 1
            i32.and
            i32.or
            i32.const 2
            i32.or
            i32.store
            local.get 2
            local.get 1
            i32.add
            local.tee 4
            local.get 4
            i32.load offset=4
            i32.const 1
            i32.or
            i32.store offset=4
            local.get 2
            local.get 1
            call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$13dispose_chunk17h812266503d4b53e1E
            br 1 (;@2;)
          end
          local.get 2
          i32.load
          local.set 2
          local.get 0
          local.get 4
          i32.store offset=4
          local.get 0
          local.get 2
          local.get 1
          i32.add
          i32.store
        end
        block ;; label = @2
          local.get 0
          i32.load offset=4
          local.tee 1
          i32.const 3
          i32.and
          i32.eqz
          br_if 0 (;@2;)
          local.get 1
          i32.const -8
          i32.and
          local.tee 2
          local.get 3
          i32.const 16
          i32.add
          i32.le_u
          br_if 0 (;@2;)
          local.get 0
          local.get 3
          local.get 1
          i32.const 1
          i32.and
          i32.or
          i32.const 2
          i32.or
          i32.store offset=4
          local.get 0
          local.get 3
          i32.add
          local.tee 1
          local.get 2
          local.get 3
          i32.sub
          local.tee 3
          i32.const 3
          i32.or
          i32.store offset=4
          local.get 0
          local.get 2
          i32.add
          local.tee 2
          local.get 2
          i32.load offset=4
          i32.const 1
          i32.or
          i32.store offset=4
          local.get 1
          local.get 3
          call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$13dispose_chunk17h812266503d4b53e1E
        end
        local.get 0
        i32.const 8
        i32.add
        local.set 2
      end
      local.get 2
    )
    (func $_ZN3std3sys9backtrace26__rust_end_short_backtrace17hf71b7423153b2aa7E (;111;) (type 9) (param i32)
      local.get 0
      call $_ZN3std9panicking19begin_panic_handler28_$u7b$$u7b$closure$u7d$$u7d$17h23c143fd98d85da8E
      unreachable
    )
    (func $_ZN3std9panicking19begin_panic_handler28_$u7b$$u7b$closure$u7d$$u7d$17h23c143fd98d85da8E (;112;) (type 9) (param i32)
      (local i32 i32 i32)
      global.get $__stack_pointer
      i32.const 16
      i32.sub
      local.tee 1
      global.set $__stack_pointer
      local.get 0
      i32.load offset=12
      local.set 2
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              local.get 0
              i32.load offset=4
              br_table 0 (;@4;) 1 (;@3;) 2 (;@2;)
            end
            local.get 2
            br_if 1 (;@2;)
            i32.const 1
            local.set 2
            i32.const 0
            local.set 3
            br 2 (;@1;)
          end
          local.get 2
          br_if 0 (;@2;)
          local.get 0
          i32.load
          local.tee 2
          i32.load offset=4
          local.set 3
          local.get 2
          i32.load
          local.set 2
          br 1 (;@1;)
        end
        local.get 1
        i32.const -2147483648
        i32.store
        local.get 1
        local.get 0
        i32.store offset=12
        local.get 1
        i32.const 1052092
        local.get 0
        i32.load offset=24
        local.get 0
        i32.load offset=28
        local.tee 0
        i32.load8_u offset=28
        local.get 0
        i32.load8_u offset=29
        call $_ZN3std9panicking20rust_panic_with_hook17h47540a6ab2275386E
        unreachable
      end
      local.get 1
      local.get 3
      i32.store offset=4
      local.get 1
      local.get 2
      i32.store
      local.get 1
      i32.const 1052064
      local.get 0
      i32.load offset=24
      local.get 0
      i32.load offset=28
      local.tee 0
      i32.load8_u offset=28
      local.get 0
      i32.load8_u offset=29
      call $_ZN3std9panicking20rust_panic_with_hook17h47540a6ab2275386E
      unreachable
    )
    (func $_ZN3std5alloc24default_alloc_error_hook17h6f1c3591a9b459d3E (;113;) (type 0) (param i32 i32)
      (local i32)
      global.get $__stack_pointer
      i32.const 48
      i32.sub
      local.tee 2
      global.set $__stack_pointer
      block ;; label = @1
        i32.const 0
        i32.load8_u offset=1056352
        i32.eqz
        br_if 0 (;@1;)
        local.get 2
        i32.const 2
        i32.store offset=12
        local.get 2
        i32.const 1051976
        i32.store offset=8
        local.get 2
        i64.const 1
        i64.store offset=20 align=4
        local.get 2
        local.get 1
        i32.store offset=44
        local.get 2
        i32.const 17
        i64.extend_i32_u
        i64.const 32
        i64.shl
        local.get 2
        i32.const 44
        i32.add
        i64.extend_i32_u
        i64.or
        i64.store offset=32
        local.get 2
        local.get 2
        i32.const 32
        i32.add
        i32.store offset=16
        local.get 2
        i32.const 8
        i32.add
        i32.const 1052016
        call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
        unreachable
      end
      local.get 2
      i32.const 48
      i32.add
      global.set $__stack_pointer
    )
    (func $__rdl_alloc (;114;) (type 2) (param i32 i32) (result i32)
      block ;; label = @1
        local.get 1
        i32.const 9
        i32.lt_u
        br_if 0 (;@1;)
        local.get 1
        local.get 0
        call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$8memalign17h03306062bd0c5f07E
        return
      end
      local.get 0
      call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$6malloc17h78827ecb73929b04E
    )
    (func $__rdl_dealloc (;115;) (type 5) (param i32 i32 i32)
      (local i32 i32)
      block ;; label = @1
        block ;; label = @2
          local.get 0
          i32.const -4
          i32.add
          i32.load
          local.tee 3
          i32.const -8
          i32.and
          local.tee 4
          i32.const 4
          i32.const 8
          local.get 3
          i32.const 3
          i32.and
          local.tee 3
          select
          local.get 1
          i32.add
          i32.lt_u
          br_if 0 (;@2;)
          block ;; label = @3
            local.get 3
            i32.eqz
            br_if 0 (;@3;)
            local.get 4
            local.get 1
            i32.const 39
            i32.add
            i32.gt_u
            br_if 2 (;@1;)
          end
          local.get 0
          call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$4free17h092132f18ab141f8E
          return
        end
        i32.const 1051813
        i32.const 46
        i32.const 1051860
        call $_ZN4core9panicking5panic17h7a23dec82192b807E
        unreachable
      end
      i32.const 1051876
      i32.const 46
      i32.const 1051924
      call $_ZN4core9panicking5panic17h7a23dec82192b807E
      unreachable
    )
    (func $__rdl_realloc (;116;) (type 7) (param i32 i32 i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32)
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                local.get 0
                i32.const -4
                i32.add
                local.tee 4
                i32.load
                local.tee 5
                i32.const -8
                i32.and
                local.tee 6
                i32.const 4
                i32.const 8
                local.get 5
                i32.const 3
                i32.and
                local.tee 7
                select
                local.get 1
                i32.add
                i32.lt_u
                br_if 0 (;@5;)
                local.get 1
                i32.const 39
                i32.add
                local.set 8
                block ;; label = @6
                  local.get 7
                  i32.eqz
                  br_if 0 (;@6;)
                  local.get 6
                  local.get 8
                  i32.gt_u
                  br_if 2 (;@4;)
                end
                block ;; label = @6
                  block ;; label = @7
                    block ;; label = @8
                      local.get 2
                      i32.const 9
                      i32.lt_u
                      br_if 0 (;@8;)
                      local.get 2
                      local.get 3
                      call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$8memalign17h03306062bd0c5f07E
                      local.tee 2
                      br_if 1 (;@7;)
                      i32.const 0
                      return
                    end
                    i32.const 0
                    local.set 2
                    local.get 3
                    i32.const -65588
                    i32.gt_u
                    br_if 1 (;@6;)
                    i32.const 16
                    local.get 3
                    i32.const 11
                    i32.add
                    i32.const -8
                    i32.and
                    local.get 3
                    i32.const 11
                    i32.lt_u
                    select
                    local.set 1
                    block ;; label = @8
                      block ;; label = @9
                        local.get 7
                        br_if 0 (;@9;)
                        local.get 1
                        i32.const 256
                        i32.lt_u
                        br_if 1 (;@8;)
                        local.get 6
                        local.get 1
                        i32.const 4
                        i32.or
                        i32.lt_u
                        br_if 1 (;@8;)
                        local.get 6
                        local.get 1
                        i32.sub
                        i32.const 131073
                        i32.ge_u
                        br_if 1 (;@8;)
                        local.get 0
                        return
                      end
                      local.get 0
                      i32.const -8
                      i32.add
                      local.tee 8
                      local.get 6
                      i32.add
                      local.set 7
                      block ;; label = @9
                        block ;; label = @10
                          block ;; label = @11
                            block ;; label = @12
                              block ;; label = @13
                                local.get 6
                                local.get 1
                                i32.ge_u
                                br_if 0 (;@13;)
                                local.get 7
                                i32.const 0
                                i32.load offset=1056804
                                i32.eq
                                br_if 4 (;@9;)
                                local.get 7
                                i32.const 0
                                i32.load offset=1056800
                                i32.eq
                                br_if 2 (;@11;)
                                local.get 7
                                i32.load offset=4
                                local.tee 5
                                i32.const 2
                                i32.and
                                br_if 5 (;@8;)
                                local.get 5
                                i32.const -8
                                i32.and
                                local.tee 9
                                local.get 6
                                i32.add
                                local.tee 5
                                local.get 1
                                i32.lt_u
                                br_if 5 (;@8;)
                                local.get 7
                                local.get 9
                                call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$12unlink_chunk17hc085d75ef8348387E
                                local.get 5
                                local.get 1
                                i32.sub
                                local.tee 3
                                i32.const 16
                                i32.lt_u
                                br_if 1 (;@12;)
                                local.get 4
                                local.get 1
                                local.get 4
                                i32.load
                                i32.const 1
                                i32.and
                                i32.or
                                i32.const 2
                                i32.or
                                i32.store
                                local.get 8
                                local.get 1
                                i32.add
                                local.tee 1
                                local.get 3
                                i32.const 3
                                i32.or
                                i32.store offset=4
                                local.get 8
                                local.get 5
                                i32.add
                                local.tee 2
                                local.get 2
                                i32.load offset=4
                                i32.const 1
                                i32.or
                                i32.store offset=4
                                local.get 1
                                local.get 3
                                call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$13dispose_chunk17h812266503d4b53e1E
                                local.get 0
                                return
                              end
                              local.get 6
                              local.get 1
                              i32.sub
                              local.tee 3
                              i32.const 15
                              i32.gt_u
                              br_if 2 (;@10;)
                              local.get 0
                              return
                            end
                            local.get 4
                            local.get 5
                            local.get 4
                            i32.load
                            i32.const 1
                            i32.and
                            i32.or
                            i32.const 2
                            i32.or
                            i32.store
                            local.get 8
                            local.get 5
                            i32.add
                            local.tee 1
                            local.get 1
                            i32.load offset=4
                            i32.const 1
                            i32.or
                            i32.store offset=4
                            local.get 0
                            return
                          end
                          i32.const 0
                          i32.load offset=1056792
                          local.get 6
                          i32.add
                          local.tee 7
                          local.get 1
                          i32.lt_u
                          br_if 2 (;@8;)
                          block ;; label = @11
                            block ;; label = @12
                              local.get 7
                              local.get 1
                              i32.sub
                              local.tee 3
                              i32.const 15
                              i32.gt_u
                              br_if 0 (;@12;)
                              local.get 4
                              local.get 5
                              i32.const 1
                              i32.and
                              local.get 7
                              i32.or
                              i32.const 2
                              i32.or
                              i32.store
                              local.get 8
                              local.get 7
                              i32.add
                              local.tee 1
                              local.get 1
                              i32.load offset=4
                              i32.const 1
                              i32.or
                              i32.store offset=4
                              i32.const 0
                              local.set 3
                              i32.const 0
                              local.set 1
                              br 1 (;@11;)
                            end
                            local.get 4
                            local.get 1
                            local.get 5
                            i32.const 1
                            i32.and
                            i32.or
                            i32.const 2
                            i32.or
                            i32.store
                            local.get 8
                            local.get 1
                            i32.add
                            local.tee 1
                            local.get 3
                            i32.const 1
                            i32.or
                            i32.store offset=4
                            local.get 8
                            local.get 7
                            i32.add
                            local.tee 2
                            local.get 3
                            i32.store
                            local.get 2
                            local.get 2
                            i32.load offset=4
                            i32.const -2
                            i32.and
                            i32.store offset=4
                          end
                          i32.const 0
                          local.get 1
                          i32.store offset=1056800
                          i32.const 0
                          local.get 3
                          i32.store offset=1056792
                          local.get 0
                          return
                        end
                        local.get 4
                        local.get 1
                        local.get 5
                        i32.const 1
                        i32.and
                        i32.or
                        i32.const 2
                        i32.or
                        i32.store
                        local.get 8
                        local.get 1
                        i32.add
                        local.tee 1
                        local.get 3
                        i32.const 3
                        i32.or
                        i32.store offset=4
                        local.get 7
                        local.get 7
                        i32.load offset=4
                        i32.const 1
                        i32.or
                        i32.store offset=4
                        local.get 1
                        local.get 3
                        call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$13dispose_chunk17h812266503d4b53e1E
                        local.get 0
                        return
                      end
                      i32.const 0
                      i32.load offset=1056796
                      local.get 6
                      i32.add
                      local.tee 7
                      local.get 1
                      i32.gt_u
                      br_if 7 (;@1;)
                    end
                    local.get 3
                    call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$6malloc17h78827ecb73929b04E
                    local.tee 1
                    i32.eqz
                    br_if 1 (;@6;)
                    local.get 1
                    local.get 0
                    i32.const -4
                    i32.const -8
                    local.get 4
                    i32.load
                    local.tee 2
                    i32.const 3
                    i32.and
                    select
                    local.get 2
                    i32.const -8
                    i32.and
                    i32.add
                    local.tee 2
                    local.get 3
                    local.get 2
                    local.get 3
                    i32.lt_u
                    select
                    call $memcpy
                    local.set 1
                    local.get 0
                    call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$4free17h092132f18ab141f8E
                    local.get 1
                    return
                  end
                  local.get 2
                  local.get 0
                  local.get 1
                  local.get 3
                  local.get 1
                  local.get 3
                  i32.lt_u
                  select
                  call $memcpy
                  drop
                  local.get 4
                  i32.load
                  local.tee 3
                  i32.const -8
                  i32.and
                  local.tee 7
                  i32.const 4
                  i32.const 8
                  local.get 3
                  i32.const 3
                  i32.and
                  local.tee 3
                  select
                  local.get 1
                  i32.add
                  i32.lt_u
                  br_if 3 (;@3;)
                  block ;; label = @7
                    local.get 3
                    i32.eqz
                    br_if 0 (;@7;)
                    local.get 7
                    local.get 8
                    i32.gt_u
                    br_if 5 (;@2;)
                  end
                  local.get 0
                  call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$4free17h092132f18ab141f8E
                end
                local.get 2
                return
              end
              i32.const 1051813
              i32.const 46
              i32.const 1051860
              call $_ZN4core9panicking5panic17h7a23dec82192b807E
              unreachable
            end
            i32.const 1051876
            i32.const 46
            i32.const 1051924
            call $_ZN4core9panicking5panic17h7a23dec82192b807E
            unreachable
          end
          i32.const 1051813
          i32.const 46
          i32.const 1051860
          call $_ZN4core9panicking5panic17h7a23dec82192b807E
          unreachable
        end
        i32.const 1051876
        i32.const 46
        i32.const 1051924
        call $_ZN4core9panicking5panic17h7a23dec82192b807E
        unreachable
      end
      local.get 4
      local.get 1
      local.get 5
      i32.const 1
      i32.and
      i32.or
      i32.const 2
      i32.or
      i32.store
      local.get 8
      local.get 1
      i32.add
      local.tee 3
      local.get 7
      local.get 1
      i32.sub
      local.tee 1
      i32.const 1
      i32.or
      i32.store offset=4
      i32.const 0
      local.get 1
      i32.store offset=1056796
      i32.const 0
      local.get 3
      i32.store offset=1056804
      local.get 0
    )
    (func $__rdl_alloc_zeroed (;117;) (type 2) (param i32 i32) (result i32)
      block ;; label = @1
        block ;; label = @2
          local.get 1
          i32.const 9
          i32.lt_u
          br_if 0 (;@2;)
          local.get 1
          local.get 0
          call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$8memalign17h03306062bd0c5f07E
          local.set 1
          br 1 (;@1;)
        end
        local.get 0
        call $_ZN8dlmalloc8dlmalloc17Dlmalloc$LT$A$GT$6malloc17h78827ecb73929b04E
        local.set 1
      end
      block ;; label = @1
        local.get 1
        i32.eqz
        br_if 0 (;@1;)
        local.get 1
        i32.const -4
        i32.add
        i32.load8_u
        i32.const 3
        i32.and
        i32.eqz
        br_if 0 (;@1;)
        local.get 1
        i32.const 0
        local.get 0
        call $memset
        drop
      end
      local.get 1
    )
    (func $rust_begin_unwind (;118;) (type 9) (param i32)
      (local i32 i32)
      global.get $__stack_pointer
      i32.const 32
      i32.sub
      local.tee 1
      global.set $__stack_pointer
      local.get 0
      i32.load offset=24
      local.set 2
      local.get 1
      i32.const 16
      i32.add
      local.get 0
      i32.const 16
      i32.add
      i64.load align=4
      i64.store
      local.get 1
      i32.const 8
      i32.add
      local.get 0
      i32.const 8
      i32.add
      i64.load align=4
      i64.store
      local.get 1
      local.get 0
      i32.store offset=28
      local.get 1
      local.get 2
      i32.store offset=24
      local.get 1
      local.get 0
      i64.load align=4
      i64.store
      local.get 1
      call $_ZN3std3sys9backtrace26__rust_end_short_backtrace17hf71b7423153b2aa7E
      unreachable
    )
    (func $_ZN102_$LT$std..panicking..begin_panic_handler..FormatStringPayload$u20$as$u20$core..panic..PanicPayload$GT$8take_box17hb2610ff035054299E (;119;) (type 0) (param i32 i32)
      (local i32 i32 i32 i64)
      global.get $__stack_pointer
      i32.const 64
      i32.sub
      local.tee 2
      global.set $__stack_pointer
      block ;; label = @1
        local.get 1
        i32.load
        i32.const -2147483648
        i32.ne
        br_if 0 (;@1;)
        local.get 1
        i32.load offset=12
        local.set 3
        local.get 2
        i32.const 28
        i32.add
        i32.const 8
        i32.add
        local.tee 4
        i32.const 0
        i32.store
        local.get 2
        i64.const 4294967296
        i64.store offset=28 align=4
        local.get 2
        i32.const 40
        i32.add
        i32.const 16
        i32.add
        local.get 3
        i32.const 16
        i32.add
        i64.load align=4
        i64.store
        local.get 2
        i32.const 40
        i32.add
        i32.const 8
        i32.add
        local.get 3
        i32.const 8
        i32.add
        i64.load align=4
        i64.store
        local.get 2
        local.get 3
        i64.load align=4
        i64.store offset=40
        local.get 2
        i32.const 28
        i32.add
        i32.const 1051748
        local.get 2
        i32.const 40
        i32.add
        call $_ZN4core3fmt5write17h43164ada91fcaaeeE
        drop
        local.get 2
        i32.const 16
        i32.add
        i32.const 8
        i32.add
        local.get 4
        i32.load
        local.tee 3
        i32.store
        local.get 2
        local.get 2
        i64.load offset=28 align=4
        local.tee 5
        i64.store offset=16
        local.get 1
        i32.const 8
        i32.add
        local.get 3
        i32.store
        local.get 1
        local.get 5
        i64.store align=4
      end
      local.get 1
      i64.load align=4
      local.set 5
      local.get 1
      i64.const 4294967296
      i64.store align=4
      local.get 2
      i32.const 8
      i32.add
      local.tee 3
      local.get 1
      i32.const 8
      i32.add
      local.tee 1
      i32.load
      i32.store
      local.get 1
      i32.const 0
      i32.store
      i32.const 0
      i32.load8_u offset=1056353
      drop
      local.get 2
      local.get 5
      i64.store
      block ;; label = @1
        i32.const 12
        i32.const 4
        call $__rust_alloc
        local.tee 1
        i32.eqz
        br_if 0 (;@1;)
        local.get 1
        local.get 2
        i64.load
        i64.store align=4
        local.get 1
        i32.const 8
        i32.add
        local.get 3
        i32.load
        i32.store
        local.get 0
        i32.const 1052032
        i32.store offset=4
        local.get 0
        local.get 1
        i32.store
        local.get 2
        i32.const 64
        i32.add
        global.set $__stack_pointer
        return
      end
      i32.const 4
      i32.const 12
      call $_ZN5alloc5alloc18handle_alloc_error17h0ba28a7c65be46c8E
      unreachable
    )
    (func $_ZN102_$LT$std..panicking..begin_panic_handler..FormatStringPayload$u20$as$u20$core..panic..PanicPayload$GT$3get17h8f1a16407c949c6bE (;120;) (type 0) (param i32 i32)
      (local i32 i32 i32 i64)
      global.get $__stack_pointer
      i32.const 48
      i32.sub
      local.tee 2
      global.set $__stack_pointer
      block ;; label = @1
        local.get 1
        i32.load
        i32.const -2147483648
        i32.ne
        br_if 0 (;@1;)
        local.get 1
        i32.load offset=12
        local.set 3
        local.get 2
        i32.const 12
        i32.add
        i32.const 8
        i32.add
        local.tee 4
        i32.const 0
        i32.store
        local.get 2
        i64.const 4294967296
        i64.store offset=12 align=4
        local.get 2
        i32.const 24
        i32.add
        i32.const 16
        i32.add
        local.get 3
        i32.const 16
        i32.add
        i64.load align=4
        i64.store
        local.get 2
        i32.const 24
        i32.add
        i32.const 8
        i32.add
        local.get 3
        i32.const 8
        i32.add
        i64.load align=4
        i64.store
        local.get 2
        local.get 3
        i64.load align=4
        i64.store offset=24
        local.get 2
        i32.const 12
        i32.add
        i32.const 1051748
        local.get 2
        i32.const 24
        i32.add
        call $_ZN4core3fmt5write17h43164ada91fcaaeeE
        drop
        local.get 2
        i32.const 8
        i32.add
        local.get 4
        i32.load
        local.tee 3
        i32.store
        local.get 2
        local.get 2
        i64.load offset=12 align=4
        local.tee 5
        i64.store
        local.get 1
        i32.const 8
        i32.add
        local.get 3
        i32.store
        local.get 1
        local.get 5
        i64.store align=4
      end
      local.get 0
      i32.const 1052032
      i32.store offset=4
      local.get 0
      local.get 1
      i32.store
      local.get 2
      i32.const 48
      i32.add
      global.set $__stack_pointer
    )
    (func $_ZN95_$LT$std..panicking..begin_panic_handler..FormatStringPayload$u20$as$u20$core..fmt..Display$GT$3fmt17h3a0d742924f1b3adE (;121;) (type 2) (param i32 i32) (result i32)
      (local i32)
      global.get $__stack_pointer
      i32.const 32
      i32.sub
      local.tee 2
      global.set $__stack_pointer
      block ;; label = @1
        block ;; label = @2
          local.get 0
          i32.load
          i32.const -2147483648
          i32.eq
          br_if 0 (;@2;)
          local.get 1
          local.get 0
          i32.load offset=4
          local.get 0
          i32.load offset=8
          call $_ZN4core3fmt9Formatter9write_str17h8905e7a9d9d32ff1E
          local.set 0
          br 1 (;@1;)
        end
        local.get 2
        i32.const 8
        i32.add
        i32.const 16
        i32.add
        local.get 0
        i32.load offset=12
        local.tee 0
        i32.const 16
        i32.add
        i64.load align=4
        i64.store
        local.get 2
        i32.const 8
        i32.add
        i32.const 8
        i32.add
        local.get 0
        i32.const 8
        i32.add
        i64.load align=4
        i64.store
        local.get 2
        local.get 0
        i64.load align=4
        i64.store offset=8
        local.get 1
        i32.load offset=20
        local.get 1
        i32.load offset=24
        local.get 2
        i32.const 8
        i32.add
        call $_ZN4core3fmt5write17h43164ada91fcaaeeE
        local.set 0
      end
      local.get 2
      i32.const 32
      i32.add
      global.set $__stack_pointer
      local.get 0
    )
    (func $_ZN99_$LT$std..panicking..begin_panic_handler..StaticStrPayload$u20$as$u20$core..panic..PanicPayload$GT$8take_box17hde89bca33496dd2eE (;122;) (type 0) (param i32 i32)
      (local i32 i32)
      i32.const 0
      i32.load8_u offset=1056353
      drop
      local.get 1
      i32.load offset=4
      local.set 2
      local.get 1
      i32.load
      local.set 3
      block ;; label = @1
        i32.const 8
        i32.const 4
        call $__rust_alloc
        local.tee 1
        i32.eqz
        br_if 0 (;@1;)
        local.get 1
        local.get 2
        i32.store offset=4
        local.get 1
        local.get 3
        i32.store
        local.get 0
        i32.const 1052048
        i32.store offset=4
        local.get 0
        local.get 1
        i32.store
        return
      end
      i32.const 4
      i32.const 8
      call $_ZN5alloc5alloc18handle_alloc_error17h0ba28a7c65be46c8E
      unreachable
    )
    (func $_ZN99_$LT$std..panicking..begin_panic_handler..StaticStrPayload$u20$as$u20$core..panic..PanicPayload$GT$3get17hf54ac27f28f24a8bE (;123;) (type 0) (param i32 i32)
      local.get 0
      i32.const 1052048
      i32.store offset=4
      local.get 0
      local.get 1
      i32.store
    )
    (func $_ZN99_$LT$std..panicking..begin_panic_handler..StaticStrPayload$u20$as$u20$core..panic..PanicPayload$GT$6as_str17hf4f8f0e82151ed5dE (;124;) (type 0) (param i32 i32)
      local.get 0
      local.get 1
      i64.load align=4
      i64.store
    )
    (func $_ZN92_$LT$std..panicking..begin_panic_handler..StaticStrPayload$u20$as$u20$core..fmt..Display$GT$3fmt17h38570d007b206003E (;125;) (type 2) (param i32 i32) (result i32)
      local.get 1
      local.get 0
      i32.load
      local.get 0
      i32.load offset=4
      call $_ZN4core3fmt9Formatter9write_str17h8905e7a9d9d32ff1E
    )
    (func $_ZN3std9panicking20rust_panic_with_hook17h47540a6ab2275386E (;126;) (type 8) (param i32 i32 i32 i32 i32)
      (local i32 i32)
      global.get $__stack_pointer
      i32.const 32
      i32.sub
      local.tee 5
      global.set $__stack_pointer
      i32.const 0
      i32.const 0
      i32.load offset=1056372
      local.tee 6
      i32.const 1
      i32.add
      i32.store offset=1056372
      block ;; label = @1
        local.get 6
        i32.const 0
        i32.lt_s
        br_if 0 (;@1;)
        block ;; label = @2
          i32.const 0
          i32.load8_u offset=1056832
          br_if 0 (;@2;)
          i32.const 0
          i32.const 1
          i32.store8 offset=1056832
          i32.const 0
          i32.const 0
          i32.load offset=1056828
          i32.const 1
          i32.add
          i32.store offset=1056828
          i32.const 0
          i32.load offset=1056360
          local.tee 6
          i32.const -1
          i32.le_s
          br_if 1 (;@1;)
          i32.const 0
          local.get 6
          i32.const 1
          i32.add
          i32.store offset=1056360
          block ;; label = @3
            i32.const 0
            i32.load offset=1056364
            i32.eqz
            br_if 0 (;@3;)
            local.get 5
            local.get 0
            local.get 1
            i32.load offset=20
            call_indirect (type 0)
            local.get 5
            local.get 4
            i32.store8 offset=29
            local.get 5
            local.get 3
            i32.store8 offset=28
            local.get 5
            local.get 2
            i32.store offset=24
            local.get 5
            local.get 5
            i64.load
            i64.store offset=16 align=4
            i32.const 0
            i32.load offset=1056364
            local.get 5
            i32.const 16
            i32.add
            i32.const 0
            i32.load offset=1056368
            i32.load offset=20
            call_indirect (type 0)
            i32.const 0
            i32.load offset=1056360
            i32.const -1
            i32.add
            local.set 6
          end
          i32.const 0
          local.get 6
          i32.store offset=1056360
          i32.const 0
          i32.const 0
          i32.store8 offset=1056832
          local.get 3
          i32.eqz
          br_if 1 (;@1;)
          local.get 0
          local.get 1
          call $rust_panic
          unreachable
        end
        local.get 5
        i32.const 8
        i32.add
        local.get 0
        local.get 1
        i32.load offset=24
        call_indirect (type 0)
      end
      unreachable
      unreachable
    )
    (func $rust_panic (;127;) (type 0) (param i32 i32)
      local.get 0
      local.get 1
      call $__rust_start_panic
      drop
      unreachable
      unreachable
    )
    (func $__rg_oom (;128;) (type 0) (param i32 i32)
      (local i32)
      local.get 1
      local.get 0
      i32.const 0
      i32.load offset=1056356
      local.tee 2
      i32.const 18
      local.get 2
      select
      call_indirect (type 0)
      unreachable
      unreachable
    )
    (func $__rust_start_panic (;129;) (type 2) (param i32 i32) (result i32)
      unreachable
      unreachable
    )
    (func $_ZN61_$LT$dlmalloc..sys..System$u20$as$u20$dlmalloc..Allocator$GT$5alloc17h282328c9a7a8a484E (;130;) (type 5) (param i32 i32 i32)
      (local i32)
      local.get 2
      i32.const 16
      i32.shr_u
      memory.grow
      local.set 3
      local.get 0
      i32.const 0
      i32.store offset=8
      local.get 0
      i32.const 0
      local.get 2
      i32.const -65536
      i32.and
      local.get 3
      i32.const -1
      i32.eq
      local.tee 2
      select
      i32.store offset=4
      local.get 0
      i32.const 0
      local.get 3
      i32.const 16
      i32.shl
      local.get 2
      select
      i32.store
    )
    (func $_ZN4core3fmt5Write9write_fmt17h39cd1de01decbf37E (;131;) (type 2) (param i32 i32) (result i32)
      local.get 0
      i32.const 1052128
      local.get 1
      call $_ZN4core3fmt5write17h43164ada91fcaaeeE
    )
    (func $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17hafd3ba595a153b9dE (;132;) (type 9) (param i32)
      (local i32)
      block ;; label = @1
        local.get 0
        i32.load
        local.tee 1
        i32.eqz
        br_if 0 (;@1;)
        local.get 0
        i32.load offset=4
        local.get 1
        i32.const 1
        call $__rust_dealloc
      end
    )
    (func $_ZN53_$LT$core..fmt..Error$u20$as$u20$core..fmt..Debug$GT$3fmt17hd605511bb8a4d61cE (;133;) (type 2) (param i32 i32) (result i32)
      local.get 1
      i32.const 1052120
      i32.const 5
      call $_ZN4core3fmt9Formatter9write_str17h8905e7a9d9d32ff1E
    )
    (func $_ZN5alloc7raw_vec17capacity_overflow17h9ca1dc3bbc03a54cE (;134;) (type 4)
      (local i32)
      global.get $__stack_pointer
      i32.const 32
      i32.sub
      local.tee 0
      global.set $__stack_pointer
      local.get 0
      i32.const 0
      i32.store offset=24
      local.get 0
      i32.const 1
      i32.store offset=12
      local.get 0
      i32.const 1052172
      i32.store offset=8
      local.get 0
      i64.const 4
      i64.store offset=16 align=4
      local.get 0
      i32.const 8
      i32.add
      i32.const 1052208
      call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
      unreachable
    )
    (func $_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$7reserve21do_reserve_and_handle17h9339f3816104bc62E (;135;) (type 5) (param i32 i32 i32)
      (local i32 i32 i32)
      global.get $__stack_pointer
      i32.const 32
      i32.sub
      local.tee 3
      global.set $__stack_pointer
      block ;; label = @1
        local.get 1
        local.get 2
        i32.add
        local.tee 2
        local.get 1
        i32.ge_u
        br_if 0 (;@1;)
        i32.const 0
        i32.const 0
        call $_ZN5alloc7raw_vec12handle_error17h5204850add322ec3E
        unreachable
      end
      i32.const 1
      local.set 4
      local.get 0
      i32.load
      local.tee 5
      i32.const 1
      i32.shl
      local.tee 1
      local.get 2
      local.get 1
      local.get 2
      i32.gt_u
      select
      local.tee 1
      i32.const 8
      local.get 1
      i32.const 8
      i32.gt_u
      select
      local.tee 1
      i32.const -1
      i32.xor
      i32.const 31
      i32.shr_u
      local.set 2
      block ;; label = @1
        block ;; label = @2
          local.get 5
          br_if 0 (;@2;)
          i32.const 0
          local.set 4
          br 1 (;@1;)
        end
        local.get 3
        local.get 5
        i32.store offset=28
        local.get 3
        local.get 0
        i32.load offset=4
        i32.store offset=20
      end
      local.get 3
      local.get 4
      i32.store offset=24
      local.get 3
      i32.const 8
      i32.add
      local.get 2
      local.get 1
      local.get 3
      i32.const 20
      i32.add
      call $_ZN5alloc7raw_vec11finish_grow17h9bc8bbdea15e5b16E
      block ;; label = @1
        local.get 3
        i32.load offset=8
        i32.eqz
        br_if 0 (;@1;)
        local.get 3
        i32.load offset=12
        local.get 3
        i32.load offset=16
        call $_ZN5alloc7raw_vec12handle_error17h5204850add322ec3E
        unreachable
      end
      local.get 3
      i32.load offset=12
      local.set 2
      local.get 0
      local.get 1
      i32.store
      local.get 0
      local.get 2
      i32.store offset=4
      local.get 3
      i32.const 32
      i32.add
      global.set $__stack_pointer
    )
    (func $_ZN5alloc7raw_vec12handle_error17h5204850add322ec3E (;136;) (type 0) (param i32 i32)
      block ;; label = @1
        local.get 0
        br_if 0 (;@1;)
        call $_ZN5alloc7raw_vec17capacity_overflow17h9ca1dc3bbc03a54cE
        unreachable
      end
      local.get 0
      local.get 1
      call $_ZN5alloc5alloc18handle_alloc_error17h0ba28a7c65be46c8E
      unreachable
    )
    (func $_ZN5alloc7raw_vec11finish_grow17h9bc8bbdea15e5b16E (;137;) (type 6) (param i32 i32 i32 i32)
      (local i32 i32 i32)
      i32.const 1
      local.set 4
      i32.const 0
      local.set 5
      i32.const 4
      local.set 6
      block ;; label = @1
        local.get 1
        i32.eqz
        br_if 0 (;@1;)
        local.get 2
        i32.const 0
        i32.lt_s
        br_if 0 (;@1;)
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  local.get 3
                  i32.load offset=4
                  i32.eqz
                  br_if 0 (;@6;)
                  block ;; label = @7
                    local.get 3
                    i32.load offset=8
                    local.tee 4
                    br_if 0 (;@7;)
                    block ;; label = @8
                      local.get 2
                      br_if 0 (;@8;)
                      i32.const 1
                      local.set 4
                      br 4 (;@4;)
                    end
                    i32.const 0
                    i32.load8_u offset=1056353
                    drop
                    local.get 2
                    i32.const 1
                    call $__rust_alloc
                    local.set 4
                    br 2 (;@5;)
                  end
                  local.get 3
                  i32.load
                  local.get 4
                  i32.const 1
                  local.get 2
                  call $__rust_realloc
                  local.set 4
                  br 1 (;@5;)
                end
                block ;; label = @6
                  local.get 2
                  br_if 0 (;@6;)
                  i32.const 1
                  local.set 4
                  br 2 (;@4;)
                end
                i32.const 0
                i32.load8_u offset=1056353
                drop
                local.get 2
                i32.const 1
                call $__rust_alloc
                local.set 4
              end
              local.get 4
              i32.eqz
              br_if 1 (;@3;)
            end
            local.get 0
            local.get 4
            i32.store offset=4
            i32.const 0
            local.set 4
            br 1 (;@2;)
          end
          i32.const 1
          local.set 4
          local.get 0
          i32.const 1
          i32.store offset=4
        end
        i32.const 8
        local.set 6
        local.get 2
        local.set 5
      end
      local.get 0
      local.get 6
      i32.add
      local.get 5
      i32.store
      local.get 0
      local.get 4
      i32.store
    )
    (func $_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h2aee45c76796d4e2E (;138;) (type 9) (param i32)
      (local i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      i32.const 32
      i32.sub
      local.tee 1
      global.set $__stack_pointer
      block ;; label = @1
        local.get 0
        i32.load
        local.tee 2
        i32.const -1
        i32.ne
        br_if 0 (;@1;)
        i32.const 0
        i32.const 0
        call $_ZN5alloc7raw_vec12handle_error17h5204850add322ec3E
        unreachable
      end
      i32.const 1
      local.set 3
      local.get 2
      i32.const 1
      i32.shl
      local.tee 4
      local.get 2
      i32.const 1
      i32.add
      local.tee 5
      local.get 4
      local.get 5
      i32.gt_u
      select
      local.tee 4
      i32.const 8
      local.get 4
      i32.const 8
      i32.gt_u
      select
      local.tee 4
      i32.const -1
      i32.xor
      i32.const 31
      i32.shr_u
      local.set 5
      block ;; label = @1
        block ;; label = @2
          local.get 2
          br_if 0 (;@2;)
          i32.const 0
          local.set 3
          br 1 (;@1;)
        end
        local.get 1
        local.get 2
        i32.store offset=28
        local.get 1
        local.get 0
        i32.load offset=4
        i32.store offset=20
      end
      local.get 1
      local.get 3
      i32.store offset=24
      local.get 1
      i32.const 8
      i32.add
      local.get 5
      local.get 4
      local.get 1
      i32.const 20
      i32.add
      call $_ZN5alloc7raw_vec11finish_grow17h9bc8bbdea15e5b16E
      block ;; label = @1
        local.get 1
        i32.load offset=8
        i32.eqz
        br_if 0 (;@1;)
        local.get 1
        i32.load offset=12
        local.get 1
        i32.load offset=16
        call $_ZN5alloc7raw_vec12handle_error17h5204850add322ec3E
        unreachable
      end
      local.get 1
      i32.load offset=12
      local.set 2
      local.get 0
      local.get 4
      i32.store
      local.get 0
      local.get 2
      i32.store offset=4
      local.get 1
      i32.const 32
      i32.add
      global.set $__stack_pointer
    )
    (func $_ZN5alloc5alloc18handle_alloc_error17h0ba28a7c65be46c8E (;139;) (type 0) (param i32 i32)
      local.get 1
      local.get 0
      call $__rust_alloc_error_handler
      unreachable
    )
    (func $_ZN5alloc3fmt6format12format_inner17hddf591775ec9b0ebE (;140;) (type 0) (param i32 i32)
      (local i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      i32.const 16
      i32.sub
      local.tee 2
      global.set $__stack_pointer
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                local.get 1
                i32.load offset=4
                local.tee 3
                i32.eqz
                br_if 0 (;@5;)
                local.get 1
                i32.load
                local.set 4
                local.get 3
                i32.const 3
                i32.and
                local.set 5
                block ;; label = @6
                  block ;; label = @7
                    local.get 3
                    i32.const 4
                    i32.ge_u
                    br_if 0 (;@7;)
                    i32.const 0
                    local.set 3
                    i32.const 0
                    local.set 6
                    br 1 (;@6;)
                  end
                  local.get 4
                  i32.const 28
                  i32.add
                  local.set 7
                  local.get 3
                  i32.const -4
                  i32.and
                  local.set 8
                  i32.const 0
                  local.set 3
                  i32.const 0
                  local.set 6
                  loop ;; label = @7
                    local.get 7
                    i32.load
                    local.get 7
                    i32.const -8
                    i32.add
                    i32.load
                    local.get 7
                    i32.const -16
                    i32.add
                    i32.load
                    local.get 7
                    i32.const -24
                    i32.add
                    i32.load
                    local.get 3
                    i32.add
                    i32.add
                    i32.add
                    i32.add
                    local.set 3
                    local.get 7
                    i32.const 32
                    i32.add
                    local.set 7
                    local.get 8
                    local.get 6
                    i32.const 4
                    i32.add
                    local.tee 6
                    i32.ne
                    br_if 0 (;@7;)
                  end
                end
                block ;; label = @6
                  local.get 5
                  i32.eqz
                  br_if 0 (;@6;)
                  local.get 6
                  i32.const 3
                  i32.shl
                  local.get 4
                  i32.add
                  i32.const 4
                  i32.add
                  local.set 7
                  loop ;; label = @7
                    local.get 7
                    i32.load
                    local.get 3
                    i32.add
                    local.set 3
                    local.get 7
                    i32.const 8
                    i32.add
                    local.set 7
                    local.get 5
                    i32.const -1
                    i32.add
                    local.tee 5
                    br_if 0 (;@7;)
                  end
                end
                block ;; label = @6
                  local.get 1
                  i32.load offset=12
                  i32.eqz
                  br_if 0 (;@6;)
                  local.get 3
                  i32.const 0
                  i32.lt_s
                  br_if 1 (;@5;)
                  local.get 3
                  i32.const 16
                  i32.lt_u
                  local.get 4
                  i32.load offset=4
                  i32.eqz
                  i32.and
                  br_if 1 (;@5;)
                  local.get 3
                  i32.const 1
                  i32.shl
                  local.set 3
                end
                local.get 3
                br_if 1 (;@4;)
              end
              i32.const 1
              local.set 7
              i32.const 0
              local.set 3
              br 1 (;@3;)
            end
            i32.const 0
            local.set 5
            local.get 3
            i32.const 0
            i32.lt_s
            br_if 1 (;@2;)
            i32.const 0
            i32.load8_u offset=1056353
            drop
            i32.const 1
            local.set 5
            local.get 3
            i32.const 1
            call $__rust_alloc
            local.tee 7
            i32.eqz
            br_if 1 (;@2;)
          end
          local.get 2
          i32.const 0
          i32.store offset=8
          local.get 2
          local.get 7
          i32.store offset=4
          local.get 2
          local.get 3
          i32.store
          local.get 2
          i32.const 1052128
          local.get 1
          call $_ZN4core3fmt5write17h43164ada91fcaaeeE
          i32.eqz
          br_if 1 (;@1;)
          i32.const 1052240
          i32.const 86
          local.get 2
          i32.const 15
          i32.add
          i32.const 1052224
          i32.const 1052352
          call $_ZN4core6result13unwrap_failed17h3e6036b583f82d93E
          unreachable
        end
        local.get 5
        local.get 3
        call $_ZN5alloc7raw_vec12handle_error17h5204850add322ec3E
        unreachable
      end
      local.get 0
      local.get 2
      i64.load align=4
      i64.store align=4
      local.get 0
      i32.const 8
      i32.add
      local.get 2
      i32.const 8
      i32.add
      i32.load
      i32.store
      local.get 2
      i32.const 16
      i32.add
      global.set $__stack_pointer
    )
    (func $_ZN5alloc6string6String4push17hff85efbfe939184eE (;141;) (type 0) (param i32 i32)
      (local i32 i32)
      global.get $__stack_pointer
      i32.const 16
      i32.sub
      local.tee 2
      global.set $__stack_pointer
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              local.get 1
              i32.const 128
              i32.lt_u
              br_if 0 (;@4;)
              local.get 2
              i32.const 0
              i32.store offset=12
              local.get 1
              i32.const 2048
              i32.lt_u
              br_if 1 (;@3;)
              block ;; label = @5
                local.get 1
                i32.const 65536
                i32.ge_u
                br_if 0 (;@5;)
                local.get 2
                local.get 1
                i32.const 63
                i32.and
                i32.const 128
                i32.or
                i32.store8 offset=14
                local.get 2
                local.get 1
                i32.const 12
                i32.shr_u
                i32.const 224
                i32.or
                i32.store8 offset=12
                local.get 2
                local.get 1
                i32.const 6
                i32.shr_u
                i32.const 63
                i32.and
                i32.const 128
                i32.or
                i32.store8 offset=13
                i32.const 3
                local.set 1
                br 3 (;@2;)
              end
              local.get 2
              local.get 1
              i32.const 63
              i32.and
              i32.const 128
              i32.or
              i32.store8 offset=15
              local.get 2
              local.get 1
              i32.const 6
              i32.shr_u
              i32.const 63
              i32.and
              i32.const 128
              i32.or
              i32.store8 offset=14
              local.get 2
              local.get 1
              i32.const 12
              i32.shr_u
              i32.const 63
              i32.and
              i32.const 128
              i32.or
              i32.store8 offset=13
              local.get 2
              local.get 1
              i32.const 18
              i32.shr_u
              i32.const 7
              i32.and
              i32.const 240
              i32.or
              i32.store8 offset=12
              i32.const 4
              local.set 1
              br 2 (;@2;)
            end
            block ;; label = @4
              local.get 0
              i32.load offset=8
              local.tee 3
              local.get 0
              i32.load
              i32.ne
              br_if 0 (;@4;)
              local.get 0
              call $_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h2aee45c76796d4e2E
            end
            local.get 0
            local.get 3
            i32.const 1
            i32.add
            i32.store offset=8
            local.get 0
            i32.load offset=4
            local.get 3
            i32.add
            local.get 1
            i32.store8
            br 2 (;@1;)
          end
          local.get 2
          local.get 1
          i32.const 63
          i32.and
          i32.const 128
          i32.or
          i32.store8 offset=13
          local.get 2
          local.get 1
          i32.const 6
          i32.shr_u
          i32.const 192
          i32.or
          i32.store8 offset=12
          i32.const 2
          local.set 1
        end
        block ;; label = @2
          local.get 0
          i32.load
          local.get 0
          i32.load offset=8
          local.tee 3
          i32.sub
          local.get 1
          i32.ge_u
          br_if 0 (;@2;)
          local.get 0
          local.get 3
          local.get 1
          call $_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$7reserve21do_reserve_and_handle17h9339f3816104bc62E
          local.get 0
          i32.load offset=8
          local.set 3
        end
        local.get 0
        i32.load offset=4
        local.get 3
        i32.add
        local.get 2
        i32.const 12
        i32.add
        local.get 1
        call $memcpy
        drop
        local.get 0
        local.get 3
        local.get 1
        i32.add
        i32.store offset=8
      end
      local.get 2
      i32.const 16
      i32.add
      global.set $__stack_pointer
    )
    (func $#func142<_ZN58_$LT$alloc..string..String$u20$as$u20$core..fmt..Write$GT$9write_str17h8c93e1ada73d40b2E> (@name "_ZN58_$LT$alloc..string..String$u20$as$u20$core..fmt..Write$GT$9write_str17h8c93e1ada73d40b2E") (;142;) (type 1) (param i32 i32 i32) (result i32)
      (local i32)
      block ;; label = @1
        local.get 0
        i32.load
        local.get 0
        i32.load offset=8
        local.tee 3
        i32.sub
        local.get 2
        i32.ge_u
        br_if 0 (;@1;)
        local.get 0
        local.get 3
        local.get 2
        call $_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$7reserve21do_reserve_and_handle17h9339f3816104bc62E
        local.get 0
        i32.load offset=8
        local.set 3
      end
      local.get 0
      i32.load offset=4
      local.get 3
      i32.add
      local.get 1
      local.get 2
      call $memcpy
      drop
      local.get 0
      local.get 3
      local.get 2
      i32.add
      i32.store offset=8
      i32.const 0
    )
    (func $#func143<_ZN58_$LT$alloc..string..String$u20$as$u20$core..fmt..Write$GT$10write_char17h4c0947de865746f3E> (@name "_ZN58_$LT$alloc..string..String$u20$as$u20$core..fmt..Write$GT$10write_char17h4c0947de865746f3E") (;143;) (type 2) (param i32 i32) (result i32)
      local.get 0
      local.get 1
      call $_ZN5alloc6string6String4push17hff85efbfe939184eE
      i32.const 0
    )
    (func $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E (;144;) (type 0) (param i32 i32)
      (local i32)
      global.get $__stack_pointer
      i32.const 32
      i32.sub
      local.tee 2
      global.set $__stack_pointer
      local.get 2
      i32.const 16
      i32.add
      local.get 0
      i32.const 16
      i32.add
      i64.load align=4
      i64.store
      local.get 2
      i32.const 8
      i32.add
      local.get 0
      i32.const 8
      i32.add
      i64.load align=4
      i64.store
      local.get 2
      i32.const 1
      i32.store16 offset=28
      local.get 2
      local.get 1
      i32.store offset=24
      local.get 2
      local.get 0
      i64.load align=4
      i64.store
      local.get 2
      call $rust_begin_unwind
      unreachable
    )
    (func $_ZN4core5slice5index26slice_start_index_len_fail17h0fcaa929c46d2711E (;145;) (type 5) (param i32 i32 i32)
      (local i32 i64)
      global.get $__stack_pointer
      i32.const 48
      i32.sub
      local.tee 3
      global.set $__stack_pointer
      local.get 3
      local.get 0
      i32.store
      local.get 3
      local.get 1
      i32.store offset=4
      local.get 3
      i32.const 2
      i32.store offset=12
      local.get 3
      i32.const 1053240
      i32.store offset=8
      local.get 3
      i64.const 2
      i64.store offset=20 align=4
      local.get 3
      i32.const 17
      i64.extend_i32_u
      i64.const 32
      i64.shl
      local.tee 4
      local.get 3
      i32.const 4
      i32.add
      i64.extend_i32_u
      i64.or
      i64.store offset=40
      local.get 3
      local.get 4
      local.get 3
      i64.extend_i32_u
      i64.or
      i64.store offset=32
      local.get 3
      local.get 3
      i32.const 32
      i32.add
      i32.store offset=16
      local.get 3
      i32.const 8
      i32.add
      local.get 2
      call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
      unreachable
    )
    (func $_ZN4core9panicking18panic_bounds_check17h066dcb66622af6b9E (;146;) (type 5) (param i32 i32 i32)
      (local i32 i64)
      global.get $__stack_pointer
      i32.const 48
      i32.sub
      local.tee 3
      global.set $__stack_pointer
      local.get 3
      local.get 1
      i32.store offset=4
      local.get 3
      local.get 0
      i32.store
      local.get 3
      i32.const 2
      i32.store offset=12
      local.get 3
      i32.const 1052564
      i32.store offset=8
      local.get 3
      i64.const 2
      i64.store offset=20 align=4
      local.get 3
      i32.const 17
      i64.extend_i32_u
      i64.const 32
      i64.shl
      local.tee 4
      local.get 3
      i64.extend_i32_u
      i64.or
      i64.store offset=40
      local.get 3
      local.get 4
      local.get 3
      i32.const 4
      i32.add
      i64.extend_i32_u
      i64.or
      i64.store offset=32
      local.get 3
      local.get 3
      i32.const 32
      i32.add
      i32.store offset=16
      local.get 3
      i32.const 8
      i32.add
      local.get 2
      call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
      unreachable
    )
    (func $_ZN4core5slice5index24slice_end_index_len_fail17h828a72b1d1bef5dcE (;147;) (type 5) (param i32 i32 i32)
      (local i32 i64)
      global.get $__stack_pointer
      i32.const 48
      i32.sub
      local.tee 3
      global.set $__stack_pointer
      local.get 3
      local.get 0
      i32.store
      local.get 3
      local.get 1
      i32.store offset=4
      local.get 3
      i32.const 2
      i32.store offset=12
      local.get 3
      i32.const 1053272
      i32.store offset=8
      local.get 3
      i64.const 2
      i64.store offset=20 align=4
      local.get 3
      i32.const 17
      i64.extend_i32_u
      i64.const 32
      i64.shl
      local.tee 4
      local.get 3
      i32.const 4
      i32.add
      i64.extend_i32_u
      i64.or
      i64.store offset=40
      local.get 3
      local.get 4
      local.get 3
      i64.extend_i32_u
      i64.or
      i64.store offset=32
      local.get 3
      local.get 3
      i32.const 32
      i32.add
      i32.store offset=16
      local.get 3
      i32.const 8
      i32.add
      local.get 2
      call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
      unreachable
    )
    (func $_ZN4core3fmt9Formatter3pad17hce9cc0d410ecbe47E (;148;) (type 1) (param i32 i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32)
      block ;; label = @1
        local.get 0
        i32.load
        local.tee 3
        local.get 0
        i32.load offset=8
        local.tee 4
        i32.or
        i32.eqz
        br_if 0 (;@1;)
        block ;; label = @2
          local.get 4
          i32.eqz
          br_if 0 (;@2;)
          local.get 1
          local.get 2
          i32.add
          local.set 5
          block ;; label = @3
            block ;; label = @4
              local.get 0
              i32.load offset=12
              local.tee 6
              br_if 0 (;@4;)
              i32.const 0
              local.set 7
              local.get 1
              local.set 8
              br 1 (;@3;)
            end
            i32.const 0
            local.set 7
            local.get 1
            local.set 8
            loop ;; label = @4
              local.get 8
              local.tee 4
              local.get 5
              i32.eq
              br_if 2 (;@2;)
              block ;; label = @5
                block ;; label = @6
                  local.get 4
                  i32.load8_s
                  local.tee 8
                  i32.const -1
                  i32.le_s
                  br_if 0 (;@6;)
                  local.get 4
                  i32.const 1
                  i32.add
                  local.set 8
                  br 1 (;@5;)
                end
                block ;; label = @6
                  local.get 8
                  i32.const -32
                  i32.ge_u
                  br_if 0 (;@6;)
                  local.get 4
                  i32.const 2
                  i32.add
                  local.set 8
                  br 1 (;@5;)
                end
                block ;; label = @6
                  local.get 8
                  i32.const -16
                  i32.ge_u
                  br_if 0 (;@6;)
                  local.get 4
                  i32.const 3
                  i32.add
                  local.set 8
                  br 1 (;@5;)
                end
                local.get 4
                i32.const 4
                i32.add
                local.set 8
              end
              local.get 8
              local.get 4
              i32.sub
              local.get 7
              i32.add
              local.set 7
              local.get 6
              i32.const -1
              i32.add
              local.tee 6
              br_if 0 (;@4;)
            end
          end
          local.get 8
          local.get 5
          i32.eq
          br_if 0 (;@2;)
          block ;; label = @3
            local.get 8
            i32.load8_s
            local.tee 4
            i32.const -1
            i32.gt_s
            br_if 0 (;@3;)
            local.get 4
            i32.const -32
            i32.lt_u
            drop
          end
          block ;; label = @3
            block ;; label = @4
              local.get 7
              i32.eqz
              br_if 0 (;@4;)
              block ;; label = @5
                local.get 7
                local.get 2
                i32.ge_u
                br_if 0 (;@5;)
                i32.const 0
                local.set 4
                local.get 1
                local.get 7
                i32.add
                i32.load8_s
                i32.const -65
                i32.gt_s
                br_if 1 (;@4;)
                br 2 (;@3;)
              end
              i32.const 0
              local.set 4
              local.get 7
              local.get 2
              i32.ne
              br_if 1 (;@3;)
            end
            local.get 1
            local.set 4
          end
          local.get 7
          local.get 2
          local.get 4
          select
          local.set 2
          local.get 4
          local.get 1
          local.get 4
          select
          local.set 1
        end
        block ;; label = @2
          local.get 3
          br_if 0 (;@2;)
          local.get 0
          i32.load offset=20
          local.get 1
          local.get 2
          local.get 0
          i32.load offset=24
          i32.load offset=12
          call_indirect (type 1)
          return
        end
        local.get 0
        i32.load offset=4
        local.set 3
        block ;; label = @2
          block ;; label = @3
            local.get 2
            i32.const 16
            i32.lt_u
            br_if 0 (;@3;)
            local.get 1
            local.get 2
            call $_ZN4core3str5count14do_count_chars17h5443f20d90e1a41cE
            local.set 4
            br 1 (;@2;)
          end
          block ;; label = @3
            local.get 2
            br_if 0 (;@3;)
            i32.const 0
            local.set 4
            br 1 (;@2;)
          end
          local.get 2
          i32.const 3
          i32.and
          local.set 6
          block ;; label = @3
            block ;; label = @4
              local.get 2
              i32.const 4
              i32.ge_u
              br_if 0 (;@4;)
              i32.const 0
              local.set 4
              i32.const 0
              local.set 7
              br 1 (;@3;)
            end
            local.get 2
            i32.const 12
            i32.and
            local.set 5
            i32.const 0
            local.set 4
            i32.const 0
            local.set 7
            loop ;; label = @4
              local.get 4
              local.get 1
              local.get 7
              i32.add
              local.tee 8
              i32.load8_s
              i32.const -65
              i32.gt_s
              i32.add
              local.get 8
              i32.const 1
              i32.add
              i32.load8_s
              i32.const -65
              i32.gt_s
              i32.add
              local.get 8
              i32.const 2
              i32.add
              i32.load8_s
              i32.const -65
              i32.gt_s
              i32.add
              local.get 8
              i32.const 3
              i32.add
              i32.load8_s
              i32.const -65
              i32.gt_s
              i32.add
              local.set 4
              local.get 5
              local.get 7
              i32.const 4
              i32.add
              local.tee 7
              i32.ne
              br_if 0 (;@4;)
            end
          end
          local.get 6
          i32.eqz
          br_if 0 (;@2;)
          local.get 1
          local.get 7
          i32.add
          local.set 8
          loop ;; label = @3
            local.get 4
            local.get 8
            i32.load8_s
            i32.const -65
            i32.gt_s
            i32.add
            local.set 4
            local.get 8
            i32.const 1
            i32.add
            local.set 8
            local.get 6
            i32.const -1
            i32.add
            local.tee 6
            br_if 0 (;@3;)
          end
        end
        block ;; label = @2
          block ;; label = @3
            local.get 3
            local.get 4
            i32.le_u
            br_if 0 (;@3;)
            local.get 3
            local.get 4
            i32.sub
            local.set 5
            i32.const 0
            local.set 4
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  local.get 0
                  i32.load8_u offset=32
                  br_table 2 (;@4;) 0 (;@6;) 1 (;@5;) 2 (;@4;) 2 (;@4;)
                end
                local.get 5
                local.set 4
                i32.const 0
                local.set 5
                br 1 (;@4;)
              end
              local.get 5
              i32.const 1
              i32.shr_u
              local.set 4
              local.get 5
              i32.const 1
              i32.add
              i32.const 1
              i32.shr_u
              local.set 5
            end
            local.get 4
            i32.const 1
            i32.add
            local.set 4
            local.get 0
            i32.load offset=16
            local.set 6
            local.get 0
            i32.load offset=24
            local.set 8
            local.get 0
            i32.load offset=20
            local.set 7
            loop ;; label = @4
              local.get 4
              i32.const -1
              i32.add
              local.tee 4
              i32.eqz
              br_if 2 (;@2;)
              local.get 7
              local.get 6
              local.get 8
              i32.load offset=16
              call_indirect (type 2)
              i32.eqz
              br_if 0 (;@4;)
            end
            i32.const 1
            return
          end
          local.get 0
          i32.load offset=20
          local.get 1
          local.get 2
          local.get 0
          i32.load offset=24
          i32.load offset=12
          call_indirect (type 1)
          return
        end
        i32.const 1
        local.set 4
        block ;; label = @2
          local.get 7
          local.get 1
          local.get 2
          local.get 8
          i32.load offset=12
          call_indirect (type 1)
          br_if 0 (;@2;)
          i32.const 0
          local.set 4
          block ;; label = @3
            loop ;; label = @4
              block ;; label = @5
                local.get 5
                local.get 4
                i32.ne
                br_if 0 (;@5;)
                local.get 5
                local.set 4
                br 2 (;@3;)
              end
              local.get 4
              i32.const 1
              i32.add
              local.set 4
              local.get 7
              local.get 6
              local.get 8
              i32.load offset=16
              call_indirect (type 2)
              i32.eqz
              br_if 0 (;@4;)
            end
            local.get 4
            i32.const -1
            i32.add
            local.set 4
          end
          local.get 4
          local.get 5
          i32.lt_u
          local.set 4
        end
        local.get 4
        return
      end
      local.get 0
      i32.load offset=20
      local.get 1
      local.get 2
      local.get 0
      i32.load offset=24
      i32.load offset=12
      call_indirect (type 1)
    )
    (func $_ZN4core9panicking5panic17h7a23dec82192b807E (;149;) (type 5) (param i32 i32 i32)
      (local i32)
      global.get $__stack_pointer
      i32.const 32
      i32.sub
      local.tee 3
      global.set $__stack_pointer
      local.get 3
      i32.const 0
      i32.store offset=16
      local.get 3
      i32.const 1
      i32.store offset=4
      local.get 3
      i64.const 4
      i64.store offset=8 align=4
      local.get 3
      local.get 1
      i32.store offset=28
      local.get 3
      local.get 0
      i32.store offset=24
      local.get 3
      local.get 3
      i32.const 24
      i32.add
      i32.store
      local.get 3
      local.get 2
      call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
      unreachable
    )
    (func $_ZN4core9panicking11panic_const23panic_const_div_by_zero17hed37a86622bbbb5bE (;150;) (type 9) (param i32)
      (local i32)
      global.get $__stack_pointer
      i32.const 32
      i32.sub
      local.tee 1
      global.set $__stack_pointer
      local.get 1
      i32.const 0
      i32.store offset=24
      local.get 1
      i32.const 1
      i32.store offset=12
      local.get 1
      i32.const 1055484
      i32.store offset=8
      local.get 1
      i64.const 4
      i64.store offset=16 align=4
      local.get 1
      i32.const 8
      i32.add
      local.get 0
      call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
      unreachable
    )
    (func $_ZN4core3fmt3num3imp52_$LT$impl$u20$core..fmt..Display$u20$for$u20$u32$GT$3fmt17h0baf20a96941cbedE (;151;) (type 2) (param i32 i32) (result i32)
      local.get 0
      i64.load32_u
      i32.const 1
      local.get 1
      call $_ZN4core3fmt3num3imp7fmt_u6417h79c8ebe903dabc4fE
    )
    (func $_ZN4core3fmt3num50_$LT$impl$u20$core..fmt..Debug$u20$for$u20$u32$GT$3fmt17hae878ffaf355f9dfE (;152;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32)
      global.get $__stack_pointer
      i32.const 128
      i32.sub
      local.tee 2
      global.set $__stack_pointer
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              local.get 1
              i32.load offset=28
              local.tee 3
              i32.const 16
              i32.and
              br_if 0 (;@4;)
              local.get 3
              i32.const 32
              i32.and
              br_if 1 (;@3;)
              local.get 0
              i64.load32_u
              i32.const 1
              local.get 1
              call $_ZN4core3fmt3num3imp7fmt_u6417h79c8ebe903dabc4fE
              local.set 0
              br 3 (;@1;)
            end
            local.get 0
            i32.load
            local.set 0
            i32.const 0
            local.set 3
            loop ;; label = @4
              local.get 2
              local.get 3
              i32.add
              i32.const 127
              i32.add
              local.get 0
              i32.const 15
              i32.and
              local.tee 4
              i32.const 48
              i32.or
              local.get 4
              i32.const 87
              i32.add
              local.get 4
              i32.const 10
              i32.lt_u
              select
              i32.store8
              local.get 3
              i32.const -1
              i32.add
              local.set 3
              local.get 0
              i32.const 16
              i32.lt_u
              local.set 4
              local.get 0
              i32.const 4
              i32.shr_u
              local.set 0
              local.get 4
              i32.eqz
              br_if 0 (;@4;)
              br 2 (;@2;)
            end
          end
          local.get 0
          i32.load
          local.set 0
          i32.const 0
          local.set 3
          loop ;; label = @3
            local.get 2
            local.get 3
            i32.add
            i32.const 127
            i32.add
            local.get 0
            i32.const 15
            i32.and
            local.tee 4
            i32.const 48
            i32.or
            local.get 4
            i32.const 55
            i32.add
            local.get 4
            i32.const 10
            i32.lt_u
            select
            i32.store8
            local.get 3
            i32.const -1
            i32.add
            local.set 3
            local.get 0
            i32.const 16
            i32.lt_u
            local.set 4
            local.get 0
            i32.const 4
            i32.shr_u
            local.set 0
            local.get 4
            i32.eqz
            br_if 0 (;@3;)
          end
          block ;; label = @3
            local.get 3
            i32.const 128
            i32.add
            local.tee 0
            i32.const 129
            i32.lt_u
            br_if 0 (;@3;)
            local.get 0
            i32.const 128
            i32.const 1052908
            call $_ZN4core5slice5index26slice_start_index_len_fail17h0fcaa929c46d2711E
            unreachable
          end
          local.get 1
          i32.const 1
          i32.const 1052924
          i32.const 2
          local.get 2
          local.get 3
          i32.add
          i32.const 128
          i32.add
          i32.const 0
          local.get 3
          i32.sub
          call $_ZN4core3fmt9Formatter12pad_integral17h2d614e17aad082d7E
          local.set 0
          br 1 (;@1;)
        end
        block ;; label = @2
          local.get 3
          i32.const 128
          i32.add
          local.tee 0
          i32.const 129
          i32.lt_u
          br_if 0 (;@2;)
          local.get 0
          i32.const 128
          i32.const 1052908
          call $_ZN4core5slice5index26slice_start_index_len_fail17h0fcaa929c46d2711E
          unreachable
        end
        local.get 1
        i32.const 1
        i32.const 1052924
        i32.const 2
        local.get 2
        local.get 3
        i32.add
        i32.const 128
        i32.add
        i32.const 0
        local.get 3
        i32.sub
        call $_ZN4core3fmt9Formatter12pad_integral17h2d614e17aad082d7E
        local.set 0
      end
      local.get 2
      i32.const 128
      i32.add
      global.set $__stack_pointer
      local.get 0
    )
    (func $_ZN4core3fmt5write17h43164ada91fcaaeeE (;153;) (type 1) (param i32 i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      i32.const 48
      i32.sub
      local.tee 3
      global.set $__stack_pointer
      local.get 3
      i32.const 3
      i32.store8 offset=44
      local.get 3
      i32.const 32
      i32.store offset=28
      i32.const 0
      local.set 4
      local.get 3
      i32.const 0
      i32.store offset=40
      local.get 3
      local.get 1
      i32.store offset=36
      local.get 3
      local.get 0
      i32.store offset=32
      local.get 3
      i32.const 0
      i32.store offset=20
      local.get 3
      i32.const 0
      i32.store offset=12
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                local.get 2
                i32.load offset=16
                local.tee 5
                br_if 0 (;@5;)
                local.get 2
                i32.load offset=12
                local.tee 0
                i32.eqz
                br_if 1 (;@4;)
                local.get 2
                i32.load offset=8
                local.set 1
                local.get 0
                i32.const 3
                i32.shl
                local.set 6
                local.get 0
                i32.const -1
                i32.add
                i32.const 536870911
                i32.and
                i32.const 1
                i32.add
                local.set 4
                local.get 2
                i32.load
                local.set 0
                loop ;; label = @6
                  block ;; label = @7
                    local.get 0
                    i32.const 4
                    i32.add
                    i32.load
                    local.tee 7
                    i32.eqz
                    br_if 0 (;@7;)
                    local.get 3
                    i32.load offset=32
                    local.get 0
                    i32.load
                    local.get 7
                    local.get 3
                    i32.load offset=36
                    i32.load offset=12
                    call_indirect (type 1)
                    br_if 4 (;@3;)
                  end
                  local.get 1
                  i32.load
                  local.get 3
                  i32.const 12
                  i32.add
                  local.get 1
                  i32.load offset=4
                  call_indirect (type 2)
                  br_if 3 (;@3;)
                  local.get 1
                  i32.const 8
                  i32.add
                  local.set 1
                  local.get 0
                  i32.const 8
                  i32.add
                  local.set 0
                  local.get 6
                  i32.const -8
                  i32.add
                  local.tee 6
                  br_if 0 (;@6;)
                  br 2 (;@4;)
                end
              end
              local.get 2
              i32.load offset=20
              local.tee 1
              i32.eqz
              br_if 0 (;@4;)
              local.get 1
              i32.const 5
              i32.shl
              local.set 8
              local.get 1
              i32.const -1
              i32.add
              i32.const 134217727
              i32.and
              i32.const 1
              i32.add
              local.set 4
              local.get 2
              i32.load offset=8
              local.set 9
              local.get 2
              i32.load
              local.set 0
              i32.const 0
              local.set 6
              loop ;; label = @5
                block ;; label = @6
                  local.get 0
                  i32.const 4
                  i32.add
                  i32.load
                  local.tee 1
                  i32.eqz
                  br_if 0 (;@6;)
                  local.get 3
                  i32.load offset=32
                  local.get 0
                  i32.load
                  local.get 1
                  local.get 3
                  i32.load offset=36
                  i32.load offset=12
                  call_indirect (type 1)
                  br_if 3 (;@3;)
                end
                local.get 3
                local.get 5
                local.get 6
                i32.add
                local.tee 1
                i32.const 16
                i32.add
                i32.load
                i32.store offset=28
                local.get 3
                local.get 1
                i32.const 28
                i32.add
                i32.load8_u
                i32.store8 offset=44
                local.get 3
                local.get 1
                i32.const 24
                i32.add
                i32.load
                i32.store offset=40
                local.get 1
                i32.const 12
                i32.add
                i32.load
                local.set 7
                i32.const 0
                local.set 10
                i32.const 0
                local.set 11
                block ;; label = @6
                  block ;; label = @7
                    block ;; label = @8
                      local.get 1
                      i32.const 8
                      i32.add
                      i32.load
                      br_table 1 (;@7;) 0 (;@8;) 2 (;@6;) 1 (;@7;)
                    end
                    local.get 7
                    i32.const 3
                    i32.shl
                    local.set 12
                    i32.const 0
                    local.set 11
                    local.get 9
                    local.get 12
                    i32.add
                    local.tee 12
                    i32.load offset=4
                    br_if 1 (;@6;)
                    local.get 12
                    i32.load
                    local.set 7
                  end
                  i32.const 1
                  local.set 11
                end
                local.get 3
                local.get 7
                i32.store offset=16
                local.get 3
                local.get 11
                i32.store offset=12
                local.get 1
                i32.const 4
                i32.add
                i32.load
                local.set 7
                block ;; label = @6
                  block ;; label = @7
                    block ;; label = @8
                      local.get 1
                      i32.load
                      br_table 1 (;@7;) 0 (;@8;) 2 (;@6;) 1 (;@7;)
                    end
                    local.get 7
                    i32.const 3
                    i32.shl
                    local.set 11
                    local.get 9
                    local.get 11
                    i32.add
                    local.tee 11
                    i32.load offset=4
                    br_if 1 (;@6;)
                    local.get 11
                    i32.load
                    local.set 7
                  end
                  i32.const 1
                  local.set 10
                end
                local.get 3
                local.get 7
                i32.store offset=24
                local.get 3
                local.get 10
                i32.store offset=20
                local.get 9
                local.get 1
                i32.const 20
                i32.add
                i32.load
                i32.const 3
                i32.shl
                i32.add
                local.tee 1
                i32.load
                local.get 3
                i32.const 12
                i32.add
                local.get 1
                i32.load offset=4
                call_indirect (type 2)
                br_if 2 (;@3;)
                local.get 0
                i32.const 8
                i32.add
                local.set 0
                local.get 8
                local.get 6
                i32.const 32
                i32.add
                local.tee 6
                i32.ne
                br_if 0 (;@5;)
              end
            end
            local.get 4
            local.get 2
            i32.load offset=4
            i32.ge_u
            br_if 1 (;@2;)
            local.get 3
            i32.load offset=32
            local.get 2
            i32.load
            local.get 4
            i32.const 3
            i32.shl
            i32.add
            local.tee 1
            i32.load
            local.get 1
            i32.load offset=4
            local.get 3
            i32.load offset=36
            i32.load offset=12
            call_indirect (type 1)
            i32.eqz
            br_if 1 (;@2;)
          end
          i32.const 1
          local.set 1
          br 1 (;@1;)
        end
        i32.const 0
        local.set 1
      end
      local.get 3
      i32.const 48
      i32.add
      global.set $__stack_pointer
      local.get 1
    )
    (func $_ZN71_$LT$core..ops..range..Range$LT$Idx$GT$$u20$as$u20$core..fmt..Debug$GT$3fmt17h02d6bc52556291bcE (;154;) (type 2) (param i32 i32) (result i32)
      (local i32)
      i32.const 1
      local.set 2
      block ;; label = @1
        local.get 0
        local.get 1
        call $_ZN4core3fmt3num50_$LT$impl$u20$core..fmt..Debug$u20$for$u20$u32$GT$3fmt17hae878ffaf355f9dfE
        br_if 0 (;@1;)
        local.get 1
        i32.load offset=20
        i32.const 1052405
        i32.const 2
        local.get 1
        i32.load offset=24
        i32.load offset=12
        call_indirect (type 1)
        br_if 0 (;@1;)
        local.get 0
        i32.const 4
        i32.add
        local.get 1
        call $_ZN4core3fmt3num50_$LT$impl$u20$core..fmt..Debug$u20$for$u20$u32$GT$3fmt17hae878ffaf355f9dfE
        local.set 2
      end
      local.get 2
    )
    (func $_ZN4core4char7methods22_$LT$impl$u20$char$GT$16escape_debug_ext17h0025cf56996c3cbcE (;155;) (type 5) (param i32 i32 i32)
      (local i32)
      global.get $__stack_pointer
      i32.const 16
      i32.sub
      local.tee 3
      global.set $__stack_pointer
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  block ;; label = @7
                    block ;; label = @8
                      block ;; label = @9
                        block ;; label = @10
                          block ;; label = @11
                            block ;; label = @12
                              block ;; label = @13
                                block ;; label = @14
                                  local.get 1
                                  br_table 6 (;@8;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 2 (;@12;) 4 (;@10;) 1 (;@13;) 1 (;@13;) 3 (;@11;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 8 (;@6;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 1 (;@13;) 7 (;@7;) 0 (;@14;)
                                end
                                local.get 1
                                i32.const 92
                                i32.eq
                                br_if 4 (;@9;)
                              end
                              local.get 1
                              i32.const 768
                              i32.lt_u
                              br_if 7 (;@5;)
                              local.get 2
                              i32.const 1
                              i32.and
                              i32.eqz
                              br_if 7 (;@5;)
                              local.get 1
                              call $_ZN4core7unicode12unicode_data15grapheme_extend11lookup_slow17h42be77a2e33a04f6E
                              i32.eqz
                              br_if 7 (;@5;)
                              local.get 3
                              i32.const 6
                              i32.add
                              i32.const 2
                              i32.add
                              i32.const 0
                              i32.store8
                              local.get 3
                              i32.const 0
                              i32.store16 offset=6
                              local.get 3
                              i32.const 125
                              i32.store8 offset=15
                              local.get 3
                              local.get 1
                              i32.const 15
                              i32.and
                              i32.const 1052407
                              i32.add
                              i32.load8_u
                              i32.store8 offset=14
                              local.get 3
                              local.get 1
                              i32.const 4
                              i32.shr_u
                              i32.const 15
                              i32.and
                              i32.const 1052407
                              i32.add
                              i32.load8_u
                              i32.store8 offset=13
                              local.get 3
                              local.get 1
                              i32.const 8
                              i32.shr_u
                              i32.const 15
                              i32.and
                              i32.const 1052407
                              i32.add
                              i32.load8_u
                              i32.store8 offset=12
                              local.get 3
                              local.get 1
                              i32.const 12
                              i32.shr_u
                              i32.const 15
                              i32.and
                              i32.const 1052407
                              i32.add
                              i32.load8_u
                              i32.store8 offset=11
                              local.get 3
                              local.get 1
                              i32.const 16
                              i32.shr_u
                              i32.const 15
                              i32.and
                              i32.const 1052407
                              i32.add
                              i32.load8_u
                              i32.store8 offset=10
                              local.get 3
                              local.get 1
                              i32.const 20
                              i32.shr_u
                              i32.const 15
                              i32.and
                              i32.const 1052407
                              i32.add
                              i32.load8_u
                              i32.store8 offset=9
                              local.get 1
                              i32.const 1
                              i32.or
                              i32.clz
                              i32.const 2
                              i32.shr_u
                              local.tee 2
                              i32.const -2
                              i32.add
                              local.tee 1
                              i32.const 10
                              i32.ge_u
                              br_if 8 (;@4;)
                              local.get 3
                              i32.const 6
                              i32.add
                              local.get 1
                              i32.add
                              i32.const 92
                              i32.store8
                              local.get 2
                              local.get 3
                              i32.const 6
                              i32.add
                              i32.add
                              i32.const -1
                              i32.add
                              i32.const 31605
                              i32.store16 align=1
                              local.get 0
                              local.get 3
                              i64.load offset=6 align=2
                              i64.store align=1
                              local.get 0
                              i32.const 8
                              i32.add
                              local.get 3
                              i32.const 6
                              i32.add
                              i32.const 8
                              i32.add
                              i32.load16_u
                              i32.store16 align=1
                              local.get 0
                              i32.const 10
                              i32.store8 offset=11
                              local.get 0
                              local.get 1
                              i32.store8 offset=10
                              br 11 (;@1;)
                            end
                            local.get 0
                            i32.const 512
                            i32.store16 offset=10
                            local.get 0
                            i64.const 0
                            i64.store offset=2 align=2
                            local.get 0
                            i32.const 29788
                            i32.store16
                            br 10 (;@1;)
                          end
                          local.get 0
                          i32.const 512
                          i32.store16 offset=10
                          local.get 0
                          i64.const 0
                          i64.store offset=2 align=2
                          local.get 0
                          i32.const 29276
                          i32.store16
                          br 9 (;@1;)
                        end
                        local.get 0
                        i32.const 512
                        i32.store16 offset=10
                        local.get 0
                        i64.const 0
                        i64.store offset=2 align=2
                        local.get 0
                        i32.const 28252
                        i32.store16
                        br 8 (;@1;)
                      end
                      local.get 0
                      i32.const 512
                      i32.store16 offset=10
                      local.get 0
                      i64.const 0
                      i64.store offset=2 align=2
                      local.get 0
                      i32.const 23644
                      i32.store16
                      br 7 (;@1;)
                    end
                    local.get 0
                    i32.const 512
                    i32.store16 offset=10
                    local.get 0
                    i64.const 0
                    i64.store offset=2 align=2
                    local.get 0
                    i32.const 12380
                    i32.store16
                    br 6 (;@1;)
                  end
                  local.get 2
                  i32.const 256
                  i32.and
                  i32.eqz
                  br_if 1 (;@5;)
                  local.get 0
                  i32.const 512
                  i32.store16 offset=10
                  local.get 0
                  i64.const 0
                  i64.store offset=2 align=2
                  local.get 0
                  i32.const 10076
                  i32.store16
                  br 5 (;@1;)
                end
                local.get 2
                i32.const 65536
                i32.and
                br_if 3 (;@2;)
              end
              block ;; label = @5
                local.get 1
                call $_ZN4core7unicode9printable12is_printable17hb95b95dafc7e99c1E
                i32.eqz
                br_if 0 (;@5;)
                local.get 0
                local.get 1
                i32.store offset=4
                local.get 0
                i32.const 128
                i32.store8
                br 4 (;@1;)
              end
              local.get 3
              i32.const 6
              i32.add
              i32.const 2
              i32.add
              i32.const 0
              i32.store8
              local.get 3
              i32.const 0
              i32.store16 offset=6
              local.get 3
              i32.const 125
              i32.store8 offset=15
              local.get 3
              local.get 1
              i32.const 15
              i32.and
              i32.const 1052407
              i32.add
              i32.load8_u
              i32.store8 offset=14
              local.get 3
              local.get 1
              i32.const 4
              i32.shr_u
              i32.const 15
              i32.and
              i32.const 1052407
              i32.add
              i32.load8_u
              i32.store8 offset=13
              local.get 3
              local.get 1
              i32.const 8
              i32.shr_u
              i32.const 15
              i32.and
              i32.const 1052407
              i32.add
              i32.load8_u
              i32.store8 offset=12
              local.get 3
              local.get 1
              i32.const 12
              i32.shr_u
              i32.const 15
              i32.and
              i32.const 1052407
              i32.add
              i32.load8_u
              i32.store8 offset=11
              local.get 3
              local.get 1
              i32.const 16
              i32.shr_u
              i32.const 15
              i32.and
              i32.const 1052407
              i32.add
              i32.load8_u
              i32.store8 offset=10
              local.get 3
              local.get 1
              i32.const 20
              i32.shr_u
              i32.const 15
              i32.and
              i32.const 1052407
              i32.add
              i32.load8_u
              i32.store8 offset=9
              local.get 1
              i32.const 1
              i32.or
              i32.clz
              i32.const 2
              i32.shr_u
              local.tee 2
              i32.const -2
              i32.add
              local.tee 1
              i32.const 10
              i32.ge_u
              br_if 1 (;@3;)
              local.get 3
              i32.const 6
              i32.add
              local.get 1
              i32.add
              i32.const 92
              i32.store8
              local.get 2
              local.get 3
              i32.const 6
              i32.add
              i32.add
              i32.const -1
              i32.add
              i32.const 31605
              i32.store16 align=1
              local.get 0
              local.get 3
              i64.load offset=6 align=2
              i64.store align=1
              local.get 0
              i32.const 8
              i32.add
              local.get 3
              i32.const 6
              i32.add
              i32.const 8
              i32.add
              i32.load16_u
              i32.store16 align=1
              local.get 0
              i32.const 10
              i32.store8 offset=11
              local.get 0
              local.get 1
              i32.store8 offset=10
              br 3 (;@1;)
            end
            local.get 1
            i32.const 10
            i32.const 1055440
            call $_ZN4core9panicking18panic_bounds_check17h066dcb66622af6b9E
            unreachable
          end
          local.get 1
          i32.const 10
          i32.const 1055440
          call $_ZN4core9panicking18panic_bounds_check17h066dcb66622af6b9E
          unreachable
        end
        local.get 0
        i32.const 512
        i32.store16 offset=10
        local.get 0
        i64.const 0
        i64.store offset=2 align=2
        local.get 0
        i32.const 8796
        i32.store16
      end
      local.get 3
      i32.const 16
      i32.add
      global.set $__stack_pointer
    )
    (func $_ZN4core7unicode12unicode_data15grapheme_extend11lookup_slow17h42be77a2e33a04f6E (;156;) (type 3) (param i32) (result i32)
      (local i32 i32 i32 i32 i32)
      local.get 0
      i32.const 11
      i32.shl
      local.set 1
      i32.const 0
      local.set 2
      i32.const 33
      local.set 3
      i32.const 33
      local.set 4
      block ;; label = @1
        block ;; label = @2
          loop ;; label = @3
            local.get 3
            i32.const 1
            i32.shr_u
            local.get 2
            i32.add
            local.tee 3
            i32.const 2
            i32.shl
            i32.const 1055492
            i32.add
            i32.load
            i32.const 11
            i32.shl
            local.tee 5
            local.get 1
            i32.eq
            br_if 1 (;@2;)
            local.get 3
            local.get 4
            local.get 5
            local.get 1
            i32.gt_u
            select
            local.tee 4
            local.get 3
            i32.const 1
            i32.add
            local.get 2
            local.get 5
            local.get 1
            i32.lt_u
            select
            local.tee 2
            i32.sub
            local.set 3
            local.get 4
            local.get 2
            i32.gt_u
            br_if 0 (;@3;)
            br 2 (;@1;)
          end
        end
        local.get 3
        i32.const 1
        i32.add
        local.set 2
      end
      block ;; label = @1
        block ;; label = @2
          local.get 2
          i32.const 32
          i32.gt_u
          br_if 0 (;@2;)
          local.get 2
          i32.const 2
          i32.shl
          local.tee 3
          i32.const 1055492
          i32.add
          local.tee 4
          i32.load
          i32.const 21
          i32.shr_u
          local.set 1
          i32.const 727
          local.set 5
          block ;; label = @3
            block ;; label = @4
              local.get 2
              i32.const 32
              i32.eq
              br_if 0 (;@4;)
              local.get 4
              i32.const 4
              i32.add
              i32.load
              i32.const 21
              i32.shr_u
              local.set 5
              local.get 2
              br_if 0 (;@4;)
              i32.const 0
              local.set 2
              br 1 (;@3;)
            end
            local.get 3
            i32.const 1055488
            i32.add
            i32.load
            i32.const 2097151
            i32.and
            local.set 2
          end
          block ;; label = @3
            local.get 5
            local.get 1
            i32.const -1
            i32.xor
            i32.add
            i32.eqz
            br_if 0 (;@3;)
            local.get 0
            local.get 2
            i32.sub
            local.set 4
            local.get 1
            i32.const 727
            local.get 1
            i32.const 727
            i32.gt_u
            select
            local.set 3
            local.get 5
            i32.const -1
            i32.add
            local.set 5
            i32.const 0
            local.set 2
            loop ;; label = @4
              local.get 3
              local.get 1
              i32.eq
              br_if 3 (;@1;)
              local.get 2
              local.get 1
              i32.const 1055624
              i32.add
              i32.load8_u
              i32.add
              local.tee 2
              local.get 4
              i32.gt_u
              br_if 1 (;@3;)
              local.get 5
              local.get 1
              i32.const 1
              i32.add
              local.tee 1
              i32.ne
              br_if 0 (;@4;)
            end
            local.get 5
            local.set 1
          end
          local.get 1
          i32.const 1
          i32.and
          return
        end
        local.get 2
        i32.const 33
        i32.const 1055380
        call $_ZN4core9panicking18panic_bounds_check17h066dcb66622af6b9E
        unreachable
      end
      local.get 3
      i32.const 727
      i32.const 1055396
      call $_ZN4core9panicking18panic_bounds_check17h066dcb66622af6b9E
      unreachable
    )
    (func $_ZN4core7unicode9printable12is_printable17hb95b95dafc7e99c1E (;157;) (type 3) (param i32) (result i32)
      (local i32)
      block ;; label = @1
        local.get 0
        i32.const 32
        i32.ge_u
        br_if 0 (;@1;)
        i32.const 0
        return
      end
      i32.const 1
      local.set 1
      block ;; label = @1
        block ;; label = @2
          local.get 0
          i32.const 127
          i32.lt_u
          br_if 0 (;@2;)
          local.get 0
          i32.const 65536
          i32.lt_u
          br_if 1 (;@1;)
          block ;; label = @3
            block ;; label = @4
              local.get 0
              i32.const 131072
              i32.lt_u
              br_if 0 (;@4;)
              block ;; label = @5
                local.get 0
                i32.const -205744
                i32.add
                i32.const 712016
                i32.ge_u
                br_if 0 (;@5;)
                i32.const 0
                return
              end
              block ;; label = @5
                local.get 0
                i32.const -201547
                i32.add
                i32.const 5
                i32.ge_u
                br_if 0 (;@5;)
                i32.const 0
                return
              end
              block ;; label = @5
                local.get 0
                i32.const -195102
                i32.add
                i32.const 1506
                i32.ge_u
                br_if 0 (;@5;)
                i32.const 0
                return
              end
              block ;; label = @5
                local.get 0
                i32.const -192094
                i32.add
                i32.const 2466
                i32.ge_u
                br_if 0 (;@5;)
                i32.const 0
                return
              end
              block ;; label = @5
                local.get 0
                i32.const -191457
                i32.add
                i32.const 15
                i32.ge_u
                br_if 0 (;@5;)
                i32.const 0
                return
              end
              block ;; label = @5
                local.get 0
                i32.const -183970
                i32.add
                i32.const 14
                i32.ge_u
                br_if 0 (;@5;)
                i32.const 0
                return
              end
              block ;; label = @5
                local.get 0
                i32.const -2
                i32.and
                i32.const 178206
                i32.ne
                br_if 0 (;@5;)
                i32.const 0
                return
              end
              local.get 0
              i32.const -32
              i32.and
              i32.const 173792
              i32.ne
              br_if 1 (;@3;)
              i32.const 0
              return
            end
            local.get 0
            i32.const 1053936
            i32.const 44
            i32.const 1054024
            i32.const 196
            i32.const 1054220
            i32.const 450
            call $_ZN4core7unicode9printable5check17ha84a2bb0c8b2a936E
            return
          end
          i32.const 0
          local.set 1
          local.get 0
          i32.const -177978
          i32.add
          i32.const 6
          i32.lt_u
          br_if 0 (;@2;)
          local.get 0
          i32.const -1114112
          i32.add
          i32.const -196112
          i32.lt_u
          local.set 1
        end
        local.get 1
        return
      end
      local.get 0
      i32.const 1054670
      i32.const 40
      i32.const 1054750
      i32.const 288
      i32.const 1055038
      i32.const 301
      call $_ZN4core7unicode9printable5check17ha84a2bb0c8b2a936E
    )
    (func $_ZN4core3str8converts9from_utf817he2399b8738172384E (;158;) (type 5) (param i32 i32 i32)
      (local i32 i32 i32 i32 i32 i64 i64 i32)
      block ;; label = @1
        local.get 2
        i32.eqz
        br_if 0 (;@1;)
        i32.const 0
        local.get 2
        i32.const -7
        i32.add
        local.tee 3
        local.get 3
        local.get 2
        i32.gt_u
        select
        local.set 4
        local.get 1
        i32.const 3
        i32.add
        i32.const -4
        i32.and
        local.get 1
        i32.sub
        local.set 5
        i32.const 0
        local.set 3
        loop ;; label = @2
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  local.get 1
                  local.get 3
                  i32.add
                  i32.load8_u
                  local.tee 6
                  i32.extend8_s
                  local.tee 7
                  i32.const 0
                  i32.lt_s
                  br_if 0 (;@6;)
                  local.get 5
                  local.get 3
                  i32.sub
                  i32.const 3
                  i32.and
                  br_if 1 (;@5;)
                  local.get 3
                  local.get 4
                  i32.ge_u
                  br_if 2 (;@4;)
                  loop ;; label = @7
                    local.get 1
                    local.get 3
                    i32.add
                    local.tee 6
                    i32.const 4
                    i32.add
                    i32.load
                    local.get 6
                    i32.load
                    i32.or
                    i32.const -2139062144
                    i32.and
                    br_if 3 (;@4;)
                    local.get 3
                    i32.const 8
                    i32.add
                    local.tee 3
                    local.get 4
                    i32.lt_u
                    br_if 0 (;@7;)
                    br 3 (;@4;)
                  end
                end
                i64.const 1099511627776
                local.set 8
                i64.const 4294967296
                local.set 9
                block ;; label = @6
                  block ;; label = @7
                    block ;; label = @8
                      block ;; label = @9
                        block ;; label = @10
                          block ;; label = @11
                            block ;; label = @12
                              block ;; label = @13
                                block ;; label = @14
                                  block ;; label = @15
                                    block ;; label = @16
                                      block ;; label = @17
                                        local.get 6
                                        i32.const 1053340
                                        i32.add
                                        i32.load8_u
                                        i32.const -2
                                        i32.add
                                        br_table 0 (;@17;) 1 (;@16;) 2 (;@15;) 10 (;@7;)
                                      end
                                      local.get 3
                                      i32.const 1
                                      i32.add
                                      local.tee 6
                                      local.get 2
                                      i32.lt_u
                                      br_if 2 (;@14;)
                                      i64.const 0
                                      local.set 8
                                      i64.const 0
                                      local.set 9
                                      br 9 (;@7;)
                                    end
                                    i64.const 0
                                    local.set 8
                                    local.get 3
                                    i32.const 1
                                    i32.add
                                    local.tee 10
                                    local.get 2
                                    i32.lt_u
                                    br_if 2 (;@13;)
                                    i64.const 0
                                    local.set 9
                                    br 8 (;@7;)
                                  end
                                  i64.const 0
                                  local.set 8
                                  local.get 3
                                  i32.const 1
                                  i32.add
                                  local.tee 10
                                  local.get 2
                                  i32.lt_u
                                  br_if 2 (;@12;)
                                  i64.const 0
                                  local.set 9
                                  br 7 (;@7;)
                                end
                                i64.const 1099511627776
                                local.set 8
                                i64.const 4294967296
                                local.set 9
                                local.get 1
                                local.get 6
                                i32.add
                                i32.load8_s
                                i32.const -65
                                i32.gt_s
                                br_if 6 (;@7;)
                                br 7 (;@6;)
                              end
                              local.get 1
                              local.get 10
                              i32.add
                              i32.load8_s
                              local.set 10
                              block ;; label = @13
                                block ;; label = @14
                                  block ;; label = @15
                                    local.get 6
                                    i32.const -224
                                    i32.add
                                    br_table 0 (;@15;) 2 (;@13;) 2 (;@13;) 2 (;@13;) 2 (;@13;) 2 (;@13;) 2 (;@13;) 2 (;@13;) 2 (;@13;) 2 (;@13;) 2 (;@13;) 2 (;@13;) 2 (;@13;) 1 (;@14;) 2 (;@13;)
                                  end
                                  local.get 10
                                  i32.const -32
                                  i32.and
                                  i32.const -96
                                  i32.eq
                                  br_if 4 (;@10;)
                                  br 3 (;@11;)
                                end
                                local.get 10
                                i32.const -97
                                i32.gt_s
                                br_if 2 (;@11;)
                                br 3 (;@10;)
                              end
                              block ;; label = @13
                                local.get 7
                                i32.const 31
                                i32.add
                                i32.const 255
                                i32.and
                                i32.const 12
                                i32.lt_u
                                br_if 0 (;@13;)
                                local.get 7
                                i32.const -2
                                i32.and
                                i32.const -18
                                i32.ne
                                br_if 2 (;@11;)
                                local.get 10
                                i32.const -64
                                i32.lt_s
                                br_if 3 (;@10;)
                                br 2 (;@11;)
                              end
                              local.get 10
                              i32.const -64
                              i32.lt_s
                              br_if 2 (;@10;)
                              br 1 (;@11;)
                            end
                            local.get 1
                            local.get 10
                            i32.add
                            i32.load8_s
                            local.set 10
                            block ;; label = @12
                              block ;; label = @13
                                block ;; label = @14
                                  block ;; label = @15
                                    local.get 6
                                    i32.const -240
                                    i32.add
                                    br_table 1 (;@14;) 0 (;@15;) 0 (;@15;) 0 (;@15;) 2 (;@13;) 0 (;@15;)
                                  end
                                  local.get 7
                                  i32.const 15
                                  i32.add
                                  i32.const 255
                                  i32.and
                                  i32.const 2
                                  i32.gt_u
                                  br_if 3 (;@11;)
                                  local.get 10
                                  i32.const -64
                                  i32.ge_s
                                  br_if 3 (;@11;)
                                  br 2 (;@12;)
                                end
                                local.get 10
                                i32.const 112
                                i32.add
                                i32.const 255
                                i32.and
                                i32.const 48
                                i32.ge_u
                                br_if 2 (;@11;)
                                br 1 (;@12;)
                              end
                              local.get 10
                              i32.const -113
                              i32.gt_s
                              br_if 1 (;@11;)
                            end
                            block ;; label = @12
                              local.get 3
                              i32.const 2
                              i32.add
                              local.tee 6
                              local.get 2
                              i32.lt_u
                              br_if 0 (;@12;)
                              i64.const 0
                              local.set 9
                              br 5 (;@7;)
                            end
                            local.get 1
                            local.get 6
                            i32.add
                            i32.load8_s
                            i32.const -65
                            i32.gt_s
                            br_if 2 (;@9;)
                            i64.const 0
                            local.set 9
                            local.get 3
                            i32.const 3
                            i32.add
                            local.tee 6
                            local.get 2
                            i32.ge_u
                            br_if 4 (;@7;)
                            local.get 1
                            local.get 6
                            i32.add
                            i32.load8_s
                            i32.const -65
                            i32.le_s
                            br_if 5 (;@6;)
                            i64.const 3298534883328
                            local.set 8
                            br 3 (;@8;)
                          end
                          i64.const 1099511627776
                          local.set 8
                          br 2 (;@8;)
                        end
                        i64.const 0
                        local.set 9
                        local.get 3
                        i32.const 2
                        i32.add
                        local.tee 6
                        local.get 2
                        i32.ge_u
                        br_if 2 (;@7;)
                        local.get 1
                        local.get 6
                        i32.add
                        i32.load8_s
                        i32.const -65
                        i32.le_s
                        br_if 3 (;@6;)
                      end
                      i64.const 2199023255552
                      local.set 8
                    end
                    i64.const 4294967296
                    local.set 9
                  end
                  local.get 0
                  local.get 8
                  local.get 3
                  i64.extend_i32_u
                  i64.or
                  local.get 9
                  i64.or
                  i64.store offset=4 align=4
                  local.get 0
                  i32.const 1
                  i32.store
                  return
                end
                local.get 6
                i32.const 1
                i32.add
                local.set 3
                br 2 (;@3;)
              end
              local.get 3
              i32.const 1
              i32.add
              local.set 3
              br 1 (;@3;)
            end
            local.get 3
            local.get 2
            i32.ge_u
            br_if 0 (;@3;)
            loop ;; label = @4
              local.get 1
              local.get 3
              i32.add
              i32.load8_s
              i32.const 0
              i32.lt_s
              br_if 1 (;@3;)
              local.get 2
              local.get 3
              i32.const 1
              i32.add
              local.tee 3
              i32.ne
              br_if 0 (;@4;)
              br 3 (;@1;)
            end
          end
          local.get 3
          local.get 2
          i32.lt_u
          br_if 0 (;@2;)
        end
      end
      local.get 0
      local.get 2
      i32.store offset=8
      local.get 0
      local.get 1
      i32.store offset=4
      local.get 0
      i32.const 0
      i32.store
    )
    (func $_ZN4core3fmt8builders11DebugStruct5field17h1f6a8e88b3d0a888E (;159;) (type 11) (param i32 i32 i32 i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i64)
      global.get $__stack_pointer
      i32.const 64
      i32.sub
      local.tee 5
      global.set $__stack_pointer
      i32.const 1
      local.set 6
      block ;; label = @1
        local.get 0
        i32.load8_u offset=4
        br_if 0 (;@1;)
        local.get 0
        i32.load8_u offset=5
        local.set 7
        block ;; label = @2
          local.get 0
          i32.load
          local.tee 8
          i32.load offset=28
          local.tee 9
          i32.const 4
          i32.and
          br_if 0 (;@2;)
          i32.const 1
          local.set 6
          local.get 8
          i32.load offset=20
          i32.const 1052863
          i32.const 1052860
          local.get 7
          i32.const 255
          i32.and
          local.tee 7
          select
          i32.const 2
          i32.const 3
          local.get 7
          select
          local.get 8
          i32.load offset=24
          i32.load offset=12
          call_indirect (type 1)
          br_if 1 (;@1;)
          i32.const 1
          local.set 6
          local.get 8
          i32.load offset=20
          local.get 1
          local.get 2
          local.get 8
          i32.load offset=24
          i32.load offset=12
          call_indirect (type 1)
          br_if 1 (;@1;)
          i32.const 1
          local.set 6
          local.get 8
          i32.load offset=20
          i32.const 1052812
          i32.const 2
          local.get 8
          i32.load offset=24
          i32.load offset=12
          call_indirect (type 1)
          br_if 1 (;@1;)
          local.get 3
          local.get 8
          local.get 4
          i32.load offset=12
          call_indirect (type 2)
          local.set 6
          br 1 (;@1;)
        end
        block ;; label = @2
          local.get 7
          i32.const 255
          i32.and
          br_if 0 (;@2;)
          i32.const 1
          local.set 6
          local.get 8
          i32.load offset=20
          i32.const 1052865
          i32.const 3
          local.get 8
          i32.load offset=24
          i32.load offset=12
          call_indirect (type 1)
          br_if 1 (;@1;)
          local.get 8
          i32.load offset=28
          local.set 9
        end
        i32.const 1
        local.set 6
        local.get 5
        i32.const 1
        i32.store8 offset=27
        local.get 5
        local.get 8
        i64.load offset=20 align=4
        i64.store offset=12 align=4
        local.get 5
        i32.const 1052832
        i32.store offset=52
        local.get 5
        local.get 5
        i32.const 27
        i32.add
        i32.store offset=20
        local.get 5
        local.get 8
        i64.load offset=8 align=4
        i64.store offset=36 align=4
        local.get 8
        i64.load align=4
        local.set 10
        local.get 5
        local.get 9
        i32.store offset=56
        local.get 5
        local.get 8
        i32.load offset=16
        i32.store offset=44
        local.get 5
        local.get 8
        i32.load8_u offset=32
        i32.store8 offset=60
        local.get 5
        local.get 10
        i64.store offset=28 align=4
        local.get 5
        local.get 5
        i32.const 12
        i32.add
        i32.store offset=48
        local.get 5
        i32.const 12
        i32.add
        local.get 1
        local.get 2
        call $_ZN68_$LT$core..fmt..builders..PadAdapter$u20$as$u20$core..fmt..Write$GT$9write_str17h41f43023c6a54529E
        br_if 0 (;@1;)
        local.get 5
        i32.const 12
        i32.add
        i32.const 1052812
        i32.const 2
        call $_ZN68_$LT$core..fmt..builders..PadAdapter$u20$as$u20$core..fmt..Write$GT$9write_str17h41f43023c6a54529E
        br_if 0 (;@1;)
        local.get 3
        local.get 5
        i32.const 28
        i32.add
        local.get 4
        i32.load offset=12
        call_indirect (type 2)
        br_if 0 (;@1;)
        local.get 5
        i32.load offset=48
        i32.const 1052868
        i32.const 2
        local.get 5
        i32.load offset=52
        i32.load offset=12
        call_indirect (type 1)
        local.set 6
      end
      local.get 0
      i32.const 1
      i32.store8 offset=5
      local.get 0
      local.get 6
      i32.store8 offset=4
      local.get 5
      i32.const 64
      i32.add
      global.set $__stack_pointer
      local.get 0
    )
    (func $_ZN4core3fmt3num3imp51_$LT$impl$u20$core..fmt..Display$u20$for$u20$u8$GT$3fmt17h2c58f1fef76e6ba7E (;160;) (type 2) (param i32 i32) (result i32)
      local.get 0
      i64.load8_u
      i32.const 1
      local.get 1
      call $_ZN4core3fmt3num3imp7fmt_u6417h79c8ebe903dabc4fE
    )
    (func $_ZN4core6result13unwrap_failed17h3e6036b583f82d93E (;161;) (type 8) (param i32 i32 i32 i32 i32)
      (local i32)
      global.get $__stack_pointer
      i32.const 64
      i32.sub
      local.tee 5
      global.set $__stack_pointer
      local.get 5
      local.get 1
      i32.store offset=12
      local.get 5
      local.get 0
      i32.store offset=8
      local.get 5
      local.get 3
      i32.store offset=20
      local.get 5
      local.get 2
      i32.store offset=16
      local.get 5
      i32.const 2
      i32.store offset=28
      local.get 5
      i32.const 1052816
      i32.store offset=24
      local.get 5
      i64.const 2
      i64.store offset=36 align=4
      local.get 5
      i32.const 39
      i64.extend_i32_u
      i64.const 32
      i64.shl
      local.get 5
      i32.const 16
      i32.add
      i64.extend_i32_u
      i64.or
      i64.store offset=56
      local.get 5
      i32.const 40
      i64.extend_i32_u
      i64.const 32
      i64.shl
      local.get 5
      i32.const 8
      i32.add
      i64.extend_i32_u
      i64.or
      i64.store offset=48
      local.get 5
      local.get 5
      i32.const 48
      i32.add
      i32.store offset=32
      local.get 5
      i32.const 24
      i32.add
      local.get 4
      call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
      unreachable
    )
    (func $_ZN4core5slice5index22slice_index_order_fail17hf860bb9cafe28110E (;162;) (type 5) (param i32 i32 i32)
      (local i32 i64)
      global.get $__stack_pointer
      i32.const 48
      i32.sub
      local.tee 3
      global.set $__stack_pointer
      local.get 3
      local.get 0
      i32.store
      local.get 3
      local.get 1
      i32.store offset=4
      local.get 3
      i32.const 2
      i32.store offset=12
      local.get 3
      i32.const 1053324
      i32.store offset=8
      local.get 3
      i64.const 2
      i64.store offset=20 align=4
      local.get 3
      i32.const 17
      i64.extend_i32_u
      i64.const 32
      i64.shl
      local.tee 4
      local.get 3
      i32.const 4
      i32.add
      i64.extend_i32_u
      i64.or
      i64.store offset=40
      local.get 3
      local.get 4
      local.get 3
      i64.extend_i32_u
      i64.or
      i64.store offset=32
      local.get 3
      local.get 3
      i32.const 32
      i32.add
      i32.store offset=16
      local.get 3
      i32.const 8
      i32.add
      local.get 2
      call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
      unreachable
    )
    (func $_ZN4core6option13unwrap_failed17hd879e969cd9bde53E (;163;) (type 9) (param i32)
      i32.const 1052424
      i32.const 43
      local.get 0
      call $_ZN4core9panicking5panic17h7a23dec82192b807E
      unreachable
    )
    (func $_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17hc6b06257c279c844E (;164;) (type 2) (param i32 i32) (result i32)
      local.get 1
      local.get 0
      i32.load
      local.get 0
      i32.load offset=4
      call $_ZN4core3fmt9Formatter3pad17hce9cc0d410ecbe47E
    )
    (func $_ZN4core9panicking18panic_nounwind_fmt17hcd4da79cab72d568E (;165;) (type 5) (param i32 i32 i32)
      (local i32)
      global.get $__stack_pointer
      i32.const 32
      i32.sub
      local.tee 3
      global.set $__stack_pointer
      local.get 3
      i32.const 16
      i32.add
      local.get 0
      i32.const 16
      i32.add
      i64.load align=4
      i64.store
      local.get 3
      i32.const 8
      i32.add
      local.get 0
      i32.const 8
      i32.add
      i64.load align=4
      i64.store
      local.get 3
      local.get 1
      i32.store8 offset=29
      local.get 3
      i32.const 0
      i32.store8 offset=28
      local.get 3
      local.get 2
      i32.store offset=24
      local.get 3
      local.get 0
      i64.load align=4
      i64.store
      local.get 3
      call $rust_begin_unwind
      unreachable
    )
    (func $_ZN4core9panicking14panic_nounwind17he2774a18f33c590cE (;166;) (type 0) (param i32 i32)
      (local i32)
      global.get $__stack_pointer
      i32.const 32
      i32.sub
      local.tee 2
      global.set $__stack_pointer
      local.get 2
      i32.const 0
      i32.store offset=16
      local.get 2
      i32.const 1
      i32.store offset=4
      local.get 2
      i64.const 4
      i64.store offset=8 align=4
      local.get 2
      local.get 1
      i32.store offset=28
      local.get 2
      local.get 0
      i32.store offset=24
      local.get 2
      local.get 2
      i32.const 24
      i32.add
      i32.store
      local.get 2
      i32.const 0
      i32.const 1052496
      call $_ZN4core9panicking18panic_nounwind_fmt17hcd4da79cab72d568E
      unreachable
    )
    (func $_ZN4core9panicking36panic_misaligned_pointer_dereference17hb39f65062076de38E (;167;) (type 5) (param i32 i32 i32)
      (local i32 i64)
      global.get $__stack_pointer
      i32.const 112
      i32.sub
      local.tee 3
      global.set $__stack_pointer
      local.get 3
      local.get 1
      i32.store offset=4
      local.get 3
      local.get 0
      i32.store
      local.get 3
      i32.const 41
      i64.extend_i32_u
      i64.const 32
      i64.shl
      local.tee 4
      local.get 3
      i32.const 4
      i32.add
      i64.extend_i32_u
      i64.or
      i64.store offset=40
      local.get 3
      local.get 4
      local.get 3
      i64.extend_i32_u
      i64.or
      i64.store offset=32
      local.get 3
      i32.const 108
      i32.add
      i32.const 3
      i32.store8
      local.get 3
      i32.const 104
      i32.add
      i32.const 4
      i32.store
      local.get 3
      i32.const 96
      i32.add
      i64.const 4294967328
      i64.store align=4
      local.get 3
      i32.const 88
      i32.add
      i32.const 2
      i32.store
      local.get 3
      i32.const 2
      i32.store offset=28
      local.get 3
      i32.const 2
      i32.store offset=12
      local.get 3
      i32.const 1052652
      i32.store offset=8
      local.get 3
      i32.const 2
      i32.store offset=20
      local.get 3
      i32.const 2
      i32.store offset=80
      local.get 3
      i32.const 3
      i32.store8 offset=76
      local.get 3
      i32.const 4
      i32.store offset=72
      local.get 3
      i64.const 32
      i64.store offset=64 align=4
      local.get 3
      i32.const 2
      i32.store offset=56
      local.get 3
      i32.const 2
      i32.store offset=48
      local.get 3
      local.get 3
      i32.const 48
      i32.add
      i32.store offset=24
      local.get 3
      local.get 3
      i32.const 32
      i32.add
      i32.store offset=16
      local.get 3
      i32.const 8
      i32.add
      i32.const 0
      local.get 2
      call $_ZN4core9panicking18panic_nounwind_fmt17hcd4da79cab72d568E
      unreachable
    )
    (func $_ZN4core3fmt3num53_$LT$impl$u20$core..fmt..LowerHex$u20$for$u20$i32$GT$3fmt17hc7fe287f99d5f9c4E (;168;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32)
      global.get $__stack_pointer
      i32.const 128
      i32.sub
      local.tee 2
      global.set $__stack_pointer
      local.get 0
      i32.load
      local.set 0
      i32.const 0
      local.set 3
      loop ;; label = @1
        local.get 2
        local.get 3
        i32.add
        i32.const 127
        i32.add
        local.get 0
        i32.const 15
        i32.and
        local.tee 4
        i32.const 48
        i32.or
        local.get 4
        i32.const 87
        i32.add
        local.get 4
        i32.const 10
        i32.lt_u
        select
        i32.store8
        local.get 3
        i32.const -1
        i32.add
        local.set 3
        local.get 0
        i32.const 16
        i32.lt_u
        local.set 4
        local.get 0
        i32.const 4
        i32.shr_u
        local.set 0
        local.get 4
        i32.eqz
        br_if 0 (;@1;)
      end
      block ;; label = @1
        local.get 3
        i32.const 128
        i32.add
        local.tee 0
        i32.const 129
        i32.lt_u
        br_if 0 (;@1;)
        local.get 0
        i32.const 128
        i32.const 1052908
        call $_ZN4core5slice5index26slice_start_index_len_fail17h0fcaa929c46d2711E
        unreachable
      end
      local.get 1
      i32.const 1
      i32.const 1052924
      i32.const 2
      local.get 2
      local.get 3
      i32.add
      i32.const 128
      i32.add
      i32.const 0
      local.get 3
      i32.sub
      call $_ZN4core3fmt9Formatter12pad_integral17h2d614e17aad082d7E
      local.set 0
      local.get 2
      i32.const 128
      i32.add
      global.set $__stack_pointer
      local.get 0
    )
    (func $_ZN4core9panicking19assert_failed_inner17h162ff0d740a1cd68E (;169;) (type 12) (param i32 i32 i32 i32 i32 i32 i32)
      (local i32 i64)
      global.get $__stack_pointer
      i32.const 112
      i32.sub
      local.tee 7
      global.set $__stack_pointer
      local.get 7
      local.get 2
      i32.store offset=12
      local.get 7
      local.get 1
      i32.store offset=8
      local.get 7
      local.get 4
      i32.store offset=20
      local.get 7
      local.get 3
      i32.store offset=16
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              local.get 0
              i32.const 255
              i32.and
              br_table 0 (;@4;) 1 (;@3;) 2 (;@2;) 0 (;@4;)
            end
            local.get 7
            i32.const 1052668
            i32.store offset=24
            i32.const 2
            local.set 2
            br 2 (;@1;)
          end
          local.get 7
          i32.const 1052670
          i32.store offset=24
          i32.const 2
          local.set 2
          br 1 (;@1;)
        end
        local.get 7
        i32.const 1052672
        i32.store offset=24
        i32.const 7
        local.set 2
      end
      local.get 7
      local.get 2
      i32.store offset=28
      block ;; label = @1
        local.get 5
        i32.load
        br_if 0 (;@1;)
        local.get 7
        i32.const 3
        i32.store offset=92
        local.get 7
        i32.const 1052728
        i32.store offset=88
        local.get 7
        i64.const 3
        i64.store offset=100 align=4
        local.get 7
        i32.const 39
        i64.extend_i32_u
        i64.const 32
        i64.shl
        local.tee 8
        local.get 7
        i32.const 16
        i32.add
        i64.extend_i32_u
        i64.or
        i64.store offset=72
        local.get 7
        local.get 8
        local.get 7
        i32.const 8
        i32.add
        i64.extend_i32_u
        i64.or
        i64.store offset=64
        local.get 7
        i32.const 40
        i64.extend_i32_u
        i64.const 32
        i64.shl
        local.get 7
        i32.const 24
        i32.add
        i64.extend_i32_u
        i64.or
        i64.store offset=56
        local.get 7
        local.get 7
        i32.const 56
        i32.add
        i32.store offset=96
        local.get 7
        i32.const 88
        i32.add
        local.get 6
        call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
        unreachable
      end
      local.get 7
      i32.const 32
      i32.add
      i32.const 16
      i32.add
      local.get 5
      i32.const 16
      i32.add
      i64.load align=4
      i64.store
      local.get 7
      i32.const 32
      i32.add
      i32.const 8
      i32.add
      local.get 5
      i32.const 8
      i32.add
      i64.load align=4
      i64.store
      local.get 7
      local.get 5
      i64.load align=4
      i64.store offset=32
      local.get 7
      i32.const 4
      i32.store offset=92
      local.get 7
      i32.const 1052780
      i32.store offset=88
      local.get 7
      i64.const 4
      i64.store offset=100 align=4
      local.get 7
      i32.const 39
      i64.extend_i32_u
      i64.const 32
      i64.shl
      local.tee 8
      local.get 7
      i32.const 16
      i32.add
      i64.extend_i32_u
      i64.or
      i64.store offset=80
      local.get 7
      local.get 8
      local.get 7
      i32.const 8
      i32.add
      i64.extend_i32_u
      i64.or
      i64.store offset=72
      local.get 7
      i32.const 42
      i64.extend_i32_u
      i64.const 32
      i64.shl
      local.get 7
      i32.const 32
      i32.add
      i64.extend_i32_u
      i64.or
      i64.store offset=64
      local.get 7
      i32.const 40
      i64.extend_i32_u
      i64.const 32
      i64.shl
      local.get 7
      i32.const 24
      i32.add
      i64.extend_i32_u
      i64.or
      i64.store offset=56
      local.get 7
      local.get 7
      i32.const 56
      i32.add
      i32.store offset=96
      local.get 7
      i32.const 88
      i32.add
      local.get 6
      call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
      unreachable
    )
    (func $_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h2edac6506478e24dE (;170;) (type 2) (param i32 i32) (result i32)
      local.get 0
      i32.load
      local.get 1
      local.get 0
      i32.load offset=4
      i32.load offset=12
      call_indirect (type 2)
    )
    (func $_ZN59_$LT$core..fmt..Arguments$u20$as$u20$core..fmt..Display$GT$3fmt17h7bc36dab8af3ab23E (;171;) (type 2) (param i32 i32) (result i32)
      local.get 1
      i32.load offset=20
      local.get 1
      i32.load offset=24
      local.get 0
      call $_ZN4core3fmt5write17h43164ada91fcaaeeE
    )
    (func $_ZN68_$LT$core..fmt..builders..PadAdapter$u20$as$u20$core..fmt..Write$GT$9write_str17h41f43023c6a54529E (;172;) (type 1) (param i32 i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      local.get 1
      i32.const -1
      i32.add
      local.set 3
      local.get 0
      i32.load offset=4
      local.set 4
      local.get 0
      i32.load
      local.set 5
      local.get 0
      i32.load offset=8
      local.set 6
      i32.const 0
      local.set 7
      i32.const 0
      local.set 8
      loop ;; label = @1
        block ;; label = @2
          block ;; label = @3
            local.get 7
            local.get 2
            i32.gt_u
            br_if 0 (;@3;)
            loop ;; label = @4
              local.get 1
              local.get 7
              i32.add
              local.set 9
              block ;; label = @5
                block ;; label = @6
                  block ;; label = @7
                    block ;; label = @8
                      local.get 2
                      local.get 7
                      i32.sub
                      local.tee 10
                      i32.const 7
                      i32.gt_u
                      br_if 0 (;@8;)
                      local.get 2
                      local.get 7
                      i32.ne
                      br_if 1 (;@7;)
                      local.get 2
                      local.set 7
                      br 5 (;@3;)
                    end
                    block ;; label = @8
                      block ;; label = @9
                        local.get 9
                        i32.const 3
                        i32.add
                        i32.const -4
                        i32.and
                        local.tee 11
                        local.get 9
                        i32.sub
                        local.tee 12
                        i32.eqz
                        br_if 0 (;@9;)
                        i32.const 0
                        local.set 0
                        loop ;; label = @10
                          local.get 9
                          local.get 0
                          i32.add
                          i32.load8_u
                          i32.const 10
                          i32.eq
                          br_if 5 (;@5;)
                          local.get 12
                          local.get 0
                          i32.const 1
                          i32.add
                          local.tee 0
                          i32.ne
                          br_if 0 (;@10;)
                        end
                        local.get 12
                        local.get 10
                        i32.const -8
                        i32.add
                        local.tee 13
                        i32.le_u
                        br_if 1 (;@8;)
                        br 3 (;@6;)
                      end
                      local.get 10
                      i32.const -8
                      i32.add
                      local.set 13
                    end
                    loop ;; label = @8
                      local.get 11
                      i32.const 4
                      i32.add
                      i32.load
                      local.tee 0
                      i32.const 168430090
                      i32.xor
                      i32.const -16843009
                      i32.add
                      local.get 0
                      i32.const -1
                      i32.xor
                      i32.and
                      local.get 11
                      i32.load
                      local.tee 0
                      i32.const 168430090
                      i32.xor
                      i32.const -16843009
                      i32.add
                      local.get 0
                      i32.const -1
                      i32.xor
                      i32.and
                      i32.or
                      i32.const -2139062144
                      i32.and
                      br_if 2 (;@6;)
                      local.get 11
                      i32.const 8
                      i32.add
                      local.set 11
                      local.get 12
                      i32.const 8
                      i32.add
                      local.tee 12
                      local.get 13
                      i32.le_u
                      br_if 0 (;@8;)
                      br 2 (;@6;)
                    end
                  end
                  i32.const 0
                  local.set 0
                  loop ;; label = @7
                    local.get 9
                    local.get 0
                    i32.add
                    i32.load8_u
                    i32.const 10
                    i32.eq
                    br_if 2 (;@5;)
                    local.get 10
                    local.get 0
                    i32.const 1
                    i32.add
                    local.tee 0
                    i32.ne
                    br_if 0 (;@7;)
                  end
                  local.get 2
                  local.set 7
                  br 3 (;@3;)
                end
                block ;; label = @6
                  local.get 12
                  local.get 10
                  i32.ne
                  br_if 0 (;@6;)
                  local.get 2
                  local.set 7
                  br 3 (;@3;)
                end
                loop ;; label = @6
                  block ;; label = @7
                    local.get 9
                    local.get 12
                    i32.add
                    i32.load8_u
                    i32.const 10
                    i32.ne
                    br_if 0 (;@7;)
                    local.get 12
                    local.set 0
                    br 2 (;@5;)
                  end
                  local.get 10
                  local.get 12
                  i32.const 1
                  i32.add
                  local.tee 12
                  i32.ne
                  br_if 0 (;@6;)
                end
                local.get 2
                local.set 7
                br 2 (;@3;)
              end
              local.get 0
              local.get 7
              i32.add
              local.tee 12
              i32.const 1
              i32.add
              local.set 7
              block ;; label = @5
                local.get 12
                local.get 2
                i32.ge_u
                br_if 0 (;@5;)
                local.get 9
                local.get 0
                i32.add
                i32.load8_u
                i32.const 10
                i32.ne
                br_if 0 (;@5;)
                i32.const 0
                local.set 9
                local.get 7
                local.set 11
                local.get 7
                local.set 0
                br 3 (;@2;)
              end
              local.get 7
              local.get 2
              i32.le_u
              br_if 0 (;@4;)
            end
          end
          i32.const 1
          local.set 9
          local.get 8
          local.set 11
          local.get 2
          local.set 0
          local.get 8
          local.get 2
          i32.ne
          br_if 0 (;@2;)
          i32.const 0
          return
        end
        block ;; label = @2
          local.get 6
          i32.load8_u
          i32.eqz
          br_if 0 (;@2;)
          local.get 5
          i32.const 1052856
          i32.const 4
          local.get 4
          i32.load offset=12
          call_indirect (type 1)
          i32.eqz
          br_if 0 (;@2;)
          i32.const 1
          return
        end
        local.get 0
        local.get 8
        i32.sub
        local.set 10
        i32.const 0
        local.set 12
        block ;; label = @2
          local.get 0
          local.get 8
          i32.eq
          br_if 0 (;@2;)
          local.get 3
          local.get 0
          i32.add
          i32.load8_u
          i32.const 10
          i32.eq
          local.set 12
        end
        local.get 1
        local.get 8
        i32.add
        local.set 0
        local.get 6
        local.get 12
        i32.store8
        local.get 11
        local.set 8
        local.get 5
        local.get 0
        local.get 10
        local.get 4
        i32.load offset=12
        call_indirect (type 1)
        local.tee 0
        local.get 9
        i32.or
        i32.eqz
        br_if 0 (;@1;)
      end
      local.get 0
    )
    (func $_ZN68_$LT$core..fmt..builders..PadAdapter$u20$as$u20$core..fmt..Write$GT$10write_char17hd93b2338104cb451E (;173;) (type 2) (param i32 i32) (result i32)
      (local i32 i32)
      local.get 0
      i32.load offset=4
      local.set 2
      local.get 0
      i32.load
      local.set 3
      block ;; label = @1
        local.get 0
        i32.load offset=8
        local.tee 0
        i32.load8_u
        i32.eqz
        br_if 0 (;@1;)
        local.get 3
        i32.const 1052856
        i32.const 4
        local.get 2
        i32.load offset=12
        call_indirect (type 1)
        i32.eqz
        br_if 0 (;@1;)
        i32.const 1
        return
      end
      local.get 0
      local.get 1
      i32.const 10
      i32.eq
      i32.store8
      local.get 3
      local.get 1
      local.get 2
      i32.load offset=16
      call_indirect (type 2)
    )
    (func $_ZN4core3fmt8builders10DebugTuple5field17h16e17751cd068088E (;174;) (type 1) (param i32 i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i64)
      global.get $__stack_pointer
      i32.const 64
      i32.sub
      local.tee 3
      global.set $__stack_pointer
      local.get 0
      i32.load
      local.set 4
      i32.const 1
      local.set 5
      block ;; label = @1
        local.get 0
        i32.load8_u offset=8
        br_if 0 (;@1;)
        block ;; label = @2
          local.get 0
          i32.load offset=4
          local.tee 6
          i32.load offset=28
          local.tee 7
          i32.const 4
          i32.and
          br_if 0 (;@2;)
          i32.const 1
          local.set 5
          local.get 6
          i32.load offset=20
          i32.const 1052863
          i32.const 1052873
          local.get 4
          select
          i32.const 2
          i32.const 1
          local.get 4
          select
          local.get 6
          i32.load offset=24
          i32.load offset=12
          call_indirect (type 1)
          br_if 1 (;@1;)
          local.get 1
          local.get 6
          local.get 2
          i32.load offset=12
          call_indirect (type 2)
          local.set 5
          br 1 (;@1;)
        end
        block ;; label = @2
          local.get 4
          br_if 0 (;@2;)
          i32.const 1
          local.set 5
          local.get 6
          i32.load offset=20
          i32.const 1052874
          i32.const 2
          local.get 6
          i32.load offset=24
          i32.load offset=12
          call_indirect (type 1)
          br_if 1 (;@1;)
          local.get 6
          i32.load offset=28
          local.set 7
        end
        i32.const 1
        local.set 5
        local.get 3
        i32.const 1
        i32.store8 offset=27
        local.get 3
        local.get 6
        i64.load offset=20 align=4
        i64.store offset=12 align=4
        local.get 3
        i32.const 1052832
        i32.store offset=52
        local.get 3
        local.get 3
        i32.const 27
        i32.add
        i32.store offset=20
        local.get 3
        local.get 6
        i64.load offset=8 align=4
        i64.store offset=36 align=4
        local.get 6
        i64.load align=4
        local.set 8
        local.get 3
        local.get 7
        i32.store offset=56
        local.get 3
        local.get 6
        i32.load offset=16
        i32.store offset=44
        local.get 3
        local.get 6
        i32.load8_u offset=32
        i32.store8 offset=60
        local.get 3
        local.get 8
        i64.store offset=28 align=4
        local.get 3
        local.get 3
        i32.const 12
        i32.add
        i32.store offset=48
        local.get 1
        local.get 3
        i32.const 28
        i32.add
        local.get 2
        i32.load offset=12
        call_indirect (type 2)
        br_if 0 (;@1;)
        local.get 3
        i32.load offset=48
        i32.const 1052868
        i32.const 2
        local.get 3
        i32.load offset=52
        i32.load offset=12
        call_indirect (type 1)
        local.set 5
      end
      local.get 0
      local.get 5
      i32.store8 offset=8
      local.get 0
      local.get 4
      i32.const 1
      i32.add
      i32.store
      local.get 3
      i32.const 64
      i32.add
      global.set $__stack_pointer
      local.get 0
    )
    (func $_ZN4core3fmt8builders8DebugSet5entry17hb6091ce1c7746e19E (;175;) (type 1) (param i32 i32 i32) (result i32)
      (local i32 i32 i32 i32 i64)
      global.get $__stack_pointer
      i32.const 64
      i32.sub
      local.tee 3
      global.set $__stack_pointer
      i32.const 1
      local.set 4
      block ;; label = @1
        local.get 0
        i32.load8_u offset=4
        br_if 0 (;@1;)
        local.get 0
        i32.load8_u offset=5
        local.set 4
        block ;; label = @2
          block ;; label = @3
            local.get 0
            i32.load
            local.tee 5
            i32.load offset=28
            local.tee 6
            i32.const 4
            i32.and
            br_if 0 (;@3;)
            local.get 4
            i32.const 255
            i32.and
            i32.eqz
            br_if 1 (;@2;)
            i32.const 1
            local.set 4
            local.get 5
            i32.load offset=20
            i32.const 1052863
            i32.const 2
            local.get 5
            i32.load offset=24
            i32.load offset=12
            call_indirect (type 1)
            i32.eqz
            br_if 1 (;@2;)
            br 2 (;@1;)
          end
          block ;; label = @3
            local.get 4
            i32.const 255
            i32.and
            br_if 0 (;@3;)
            i32.const 1
            local.set 4
            local.get 5
            i32.load offset=20
            i32.const 1052877
            i32.const 1
            local.get 5
            i32.load offset=24
            i32.load offset=12
            call_indirect (type 1)
            br_if 2 (;@1;)
            local.get 5
            i32.load offset=28
            local.set 6
          end
          i32.const 1
          local.set 4
          local.get 3
          i32.const 1
          i32.store8 offset=27
          local.get 3
          local.get 5
          i64.load offset=20 align=4
          i64.store offset=12 align=4
          local.get 3
          i32.const 1052832
          i32.store offset=52
          local.get 3
          local.get 3
          i32.const 27
          i32.add
          i32.store offset=20
          local.get 3
          local.get 5
          i64.load offset=8 align=4
          i64.store offset=36 align=4
          local.get 5
          i64.load align=4
          local.set 7
          local.get 3
          local.get 6
          i32.store offset=56
          local.get 3
          local.get 5
          i32.load offset=16
          i32.store offset=44
          local.get 3
          local.get 5
          i32.load8_u offset=32
          i32.store8 offset=60
          local.get 3
          local.get 7
          i64.store offset=28 align=4
          local.get 3
          local.get 3
          i32.const 12
          i32.add
          i32.store offset=48
          local.get 1
          local.get 3
          i32.const 28
          i32.add
          local.get 2
          i32.load offset=12
          call_indirect (type 2)
          br_if 1 (;@1;)
          local.get 3
          i32.load offset=48
          i32.const 1052868
          i32.const 2
          local.get 3
          i32.load offset=52
          i32.load offset=12
          call_indirect (type 1)
          local.set 4
          br 1 (;@1;)
        end
        local.get 1
        local.get 5
        local.get 2
        i32.load offset=12
        call_indirect (type 2)
        local.set 4
      end
      local.get 0
      i32.const 1
      i32.store8 offset=5
      local.get 0
      local.get 4
      i32.store8 offset=4
      local.get 3
      i32.const 64
      i32.add
      global.set $__stack_pointer
      local.get 0
    )
    (func $_ZN4core3fmt8builders9DebugList6finish17h184ffc6ee21bdc8bE (;176;) (type 3) (param i32) (result i32)
      (local i32)
      i32.const 1
      local.set 1
      block ;; label = @1
        local.get 0
        i32.load8_u offset=4
        br_if 0 (;@1;)
        local.get 0
        i32.load
        local.tee 0
        i32.load offset=20
        i32.const 1052878
        i32.const 1
        local.get 0
        i32.load offset=24
        i32.load offset=12
        call_indirect (type 1)
        local.set 1
      end
      local.get 1
    )
    (func $_ZN4core3fmt9Formatter12pad_integral17h2d614e17aad082d7E (;177;) (type 13) (param i32 i32 i32 i32 i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32)
      block ;; label = @1
        block ;; label = @2
          local.get 1
          br_if 0 (;@2;)
          local.get 5
          i32.const 1
          i32.add
          local.set 6
          local.get 0
          i32.load offset=28
          local.set 7
          i32.const 45
          local.set 8
          br 1 (;@1;)
        end
        i32.const 43
        i32.const 1114112
        local.get 0
        i32.load offset=28
        local.tee 7
        i32.const 1
        i32.and
        local.tee 1
        select
        local.set 8
        local.get 1
        local.get 5
        i32.add
        local.set 6
      end
      block ;; label = @1
        block ;; label = @2
          local.get 7
          i32.const 4
          i32.and
          br_if 0 (;@2;)
          i32.const 0
          local.set 2
          br 1 (;@1;)
        end
        block ;; label = @2
          block ;; label = @3
            local.get 3
            i32.const 16
            i32.lt_u
            br_if 0 (;@3;)
            local.get 2
            local.get 3
            call $_ZN4core3str5count14do_count_chars17h5443f20d90e1a41cE
            local.set 1
            br 1 (;@2;)
          end
          block ;; label = @3
            local.get 3
            br_if 0 (;@3;)
            i32.const 0
            local.set 1
            br 1 (;@2;)
          end
          local.get 3
          i32.const 3
          i32.and
          local.set 9
          block ;; label = @3
            block ;; label = @4
              local.get 3
              i32.const 4
              i32.ge_u
              br_if 0 (;@4;)
              i32.const 0
              local.set 1
              i32.const 0
              local.set 10
              br 1 (;@3;)
            end
            local.get 3
            i32.const 12
            i32.and
            local.set 11
            i32.const 0
            local.set 1
            i32.const 0
            local.set 10
            loop ;; label = @4
              local.get 1
              local.get 2
              local.get 10
              i32.add
              local.tee 12
              i32.load8_s
              i32.const -65
              i32.gt_s
              i32.add
              local.get 12
              i32.const 1
              i32.add
              i32.load8_s
              i32.const -65
              i32.gt_s
              i32.add
              local.get 12
              i32.const 2
              i32.add
              i32.load8_s
              i32.const -65
              i32.gt_s
              i32.add
              local.get 12
              i32.const 3
              i32.add
              i32.load8_s
              i32.const -65
              i32.gt_s
              i32.add
              local.set 1
              local.get 11
              local.get 10
              i32.const 4
              i32.add
              local.tee 10
              i32.ne
              br_if 0 (;@4;)
            end
          end
          local.get 9
          i32.eqz
          br_if 0 (;@2;)
          local.get 2
          local.get 10
          i32.add
          local.set 12
          loop ;; label = @3
            local.get 1
            local.get 12
            i32.load8_s
            i32.const -65
            i32.gt_s
            i32.add
            local.set 1
            local.get 12
            i32.const 1
            i32.add
            local.set 12
            local.get 9
            i32.const -1
            i32.add
            local.tee 9
            br_if 0 (;@3;)
          end
        end
        local.get 1
        local.get 6
        i32.add
        local.set 6
      end
      block ;; label = @1
        block ;; label = @2
          local.get 0
          i32.load
          br_if 0 (;@2;)
          i32.const 1
          local.set 1
          local.get 0
          i32.load offset=20
          local.tee 12
          local.get 0
          i32.load offset=24
          local.tee 10
          local.get 8
          local.get 2
          local.get 3
          call $_ZN4core3fmt9Formatter12pad_integral12write_prefix17h87435251ed4e8074E
          br_if 1 (;@1;)
          local.get 12
          local.get 4
          local.get 5
          local.get 10
          i32.load offset=12
          call_indirect (type 1)
          return
        end
        block ;; label = @2
          local.get 0
          i32.load offset=4
          local.tee 9
          local.get 6
          i32.gt_u
          br_if 0 (;@2;)
          i32.const 1
          local.set 1
          local.get 0
          i32.load offset=20
          local.tee 12
          local.get 0
          i32.load offset=24
          local.tee 10
          local.get 8
          local.get 2
          local.get 3
          call $_ZN4core3fmt9Formatter12pad_integral12write_prefix17h87435251ed4e8074E
          br_if 1 (;@1;)
          local.get 12
          local.get 4
          local.get 5
          local.get 10
          i32.load offset=12
          call_indirect (type 1)
          return
        end
        block ;; label = @2
          local.get 7
          i32.const 8
          i32.and
          i32.eqz
          br_if 0 (;@2;)
          local.get 0
          i32.load offset=16
          local.set 11
          local.get 0
          i32.const 48
          i32.store offset=16
          local.get 0
          i32.load8_u offset=32
          local.set 7
          i32.const 1
          local.set 1
          local.get 0
          i32.const 1
          i32.store8 offset=32
          local.get 0
          i32.load offset=20
          local.tee 12
          local.get 0
          i32.load offset=24
          local.tee 10
          local.get 8
          local.get 2
          local.get 3
          call $_ZN4core3fmt9Formatter12pad_integral12write_prefix17h87435251ed4e8074E
          br_if 1 (;@1;)
          local.get 9
          local.get 6
          i32.sub
          i32.const 1
          i32.add
          local.set 1
          block ;; label = @3
            loop ;; label = @4
              local.get 1
              i32.const -1
              i32.add
              local.tee 1
              i32.eqz
              br_if 1 (;@3;)
              local.get 12
              i32.const 48
              local.get 10
              i32.load offset=16
              call_indirect (type 2)
              i32.eqz
              br_if 0 (;@4;)
            end
            i32.const 1
            return
          end
          i32.const 1
          local.set 1
          local.get 12
          local.get 4
          local.get 5
          local.get 10
          i32.load offset=12
          call_indirect (type 1)
          br_if 1 (;@1;)
          local.get 0
          local.get 7
          i32.store8 offset=32
          local.get 0
          local.get 11
          i32.store offset=16
          i32.const 0
          local.set 1
          br 1 (;@1;)
        end
        local.get 9
        local.get 6
        i32.sub
        local.set 6
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              local.get 0
              i32.load8_u offset=32
              local.tee 1
              br_table 2 (;@2;) 0 (;@4;) 1 (;@3;) 0 (;@4;) 2 (;@2;)
            end
            local.get 6
            local.set 1
            i32.const 0
            local.set 6
            br 1 (;@2;)
          end
          local.get 6
          i32.const 1
          i32.shr_u
          local.set 1
          local.get 6
          i32.const 1
          i32.add
          i32.const 1
          i32.shr_u
          local.set 6
        end
        local.get 1
        i32.const 1
        i32.add
        local.set 1
        local.get 0
        i32.load offset=16
        local.set 9
        local.get 0
        i32.load offset=24
        local.set 12
        local.get 0
        i32.load offset=20
        local.set 10
        block ;; label = @2
          loop ;; label = @3
            local.get 1
            i32.const -1
            i32.add
            local.tee 1
            i32.eqz
            br_if 1 (;@2;)
            local.get 10
            local.get 9
            local.get 12
            i32.load offset=16
            call_indirect (type 2)
            i32.eqz
            br_if 0 (;@3;)
          end
          i32.const 1
          return
        end
        i32.const 1
        local.set 1
        local.get 10
        local.get 12
        local.get 8
        local.get 2
        local.get 3
        call $_ZN4core3fmt9Formatter12pad_integral12write_prefix17h87435251ed4e8074E
        br_if 0 (;@1;)
        local.get 10
        local.get 4
        local.get 5
        local.get 12
        i32.load offset=12
        call_indirect (type 1)
        br_if 0 (;@1;)
        i32.const 0
        local.set 1
        loop ;; label = @2
          block ;; label = @3
            local.get 6
            local.get 1
            i32.ne
            br_if 0 (;@3;)
            local.get 6
            local.get 6
            i32.lt_u
            return
          end
          local.get 1
          i32.const 1
          i32.add
          local.set 1
          local.get 10
          local.get 9
          local.get 12
          i32.load offset=16
          call_indirect (type 2)
          i32.eqz
          br_if 0 (;@2;)
        end
        local.get 1
        i32.const -1
        i32.add
        local.get 6
        i32.lt_u
        return
      end
      local.get 1
    )
    (func $_ZN4core3fmt5Write9write_fmt17h41a97b915efb10b4E (;178;) (type 2) (param i32 i32) (result i32)
      local.get 0
      i32.const 1052832
      local.get 1
      call $_ZN4core3fmt5write17h43164ada91fcaaeeE
    )
    (func $_ZN4core3str5count14do_count_chars17h5443f20d90e1a41cE (;179;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32)
      block ;; label = @1
        block ;; label = @2
          local.get 1
          local.get 0
          i32.const 3
          i32.add
          i32.const -4
          i32.and
          local.tee 2
          local.get 0
          i32.sub
          local.tee 3
          i32.lt_u
          br_if 0 (;@2;)
          local.get 1
          local.get 3
          i32.sub
          local.tee 4
          i32.const 4
          i32.lt_u
          br_if 0 (;@2;)
          local.get 4
          i32.const 3
          i32.and
          local.set 5
          i32.const 0
          local.set 6
          i32.const 0
          local.set 1
          block ;; label = @3
            local.get 2
            local.get 0
            i32.eq
            local.tee 7
            br_if 0 (;@3;)
            i32.const 0
            local.set 1
            block ;; label = @4
              block ;; label = @5
                local.get 0
                local.get 2
                i32.sub
                local.tee 8
                i32.const -4
                i32.le_u
                br_if 0 (;@5;)
                i32.const 0
                local.set 9
                br 1 (;@4;)
              end
              i32.const 0
              local.set 9
              loop ;; label = @5
                local.get 1
                local.get 0
                local.get 9
                i32.add
                local.tee 2
                i32.load8_s
                i32.const -65
                i32.gt_s
                i32.add
                local.get 2
                i32.const 1
                i32.add
                i32.load8_s
                i32.const -65
                i32.gt_s
                i32.add
                local.get 2
                i32.const 2
                i32.add
                i32.load8_s
                i32.const -65
                i32.gt_s
                i32.add
                local.get 2
                i32.const 3
                i32.add
                i32.load8_s
                i32.const -65
                i32.gt_s
                i32.add
                local.set 1
                local.get 9
                i32.const 4
                i32.add
                local.tee 9
                br_if 0 (;@5;)
              end
            end
            local.get 7
            br_if 0 (;@3;)
            local.get 0
            local.get 9
            i32.add
            local.set 2
            loop ;; label = @4
              local.get 1
              local.get 2
              i32.load8_s
              i32.const -65
              i32.gt_s
              i32.add
              local.set 1
              local.get 2
              i32.const 1
              i32.add
              local.set 2
              local.get 8
              i32.const 1
              i32.add
              local.tee 8
              br_if 0 (;@4;)
            end
          end
          local.get 0
          local.get 3
          i32.add
          local.set 9
          block ;; label = @3
            local.get 5
            i32.eqz
            br_if 0 (;@3;)
            local.get 9
            local.get 4
            i32.const -4
            i32.and
            i32.add
            local.tee 2
            i32.load8_s
            i32.const -65
            i32.gt_s
            local.set 6
            local.get 5
            i32.const 1
            i32.eq
            br_if 0 (;@3;)
            local.get 6
            local.get 2
            i32.load8_s offset=1
            i32.const -65
            i32.gt_s
            i32.add
            local.set 6
            local.get 5
            i32.const 2
            i32.eq
            br_if 0 (;@3;)
            local.get 6
            local.get 2
            i32.load8_s offset=2
            i32.const -65
            i32.gt_s
            i32.add
            local.set 6
          end
          local.get 4
          i32.const 2
          i32.shr_u
          local.set 3
          local.get 6
          local.get 1
          i32.add
          local.set 8
          loop ;; label = @3
            local.get 9
            local.set 4
            local.get 3
            i32.eqz
            br_if 2 (;@1;)
            local.get 3
            i32.const 192
            local.get 3
            i32.const 192
            i32.lt_u
            select
            local.tee 6
            i32.const 3
            i32.and
            local.set 7
            local.get 6
            i32.const 2
            i32.shl
            local.set 5
            i32.const 0
            local.set 2
            block ;; label = @4
              local.get 3
              i32.const 4
              i32.lt_u
              br_if 0 (;@4;)
              local.get 4
              local.get 5
              i32.const 1008
              i32.and
              i32.add
              local.set 0
              i32.const 0
              local.set 2
              local.get 4
              local.set 1
              loop ;; label = @5
                local.get 1
                i32.load offset=12
                local.tee 9
                i32.const -1
                i32.xor
                i32.const 7
                i32.shr_u
                local.get 9
                i32.const 6
                i32.shr_u
                i32.or
                i32.const 16843009
                i32.and
                local.get 1
                i32.load offset=8
                local.tee 9
                i32.const -1
                i32.xor
                i32.const 7
                i32.shr_u
                local.get 9
                i32.const 6
                i32.shr_u
                i32.or
                i32.const 16843009
                i32.and
                local.get 1
                i32.load offset=4
                local.tee 9
                i32.const -1
                i32.xor
                i32.const 7
                i32.shr_u
                local.get 9
                i32.const 6
                i32.shr_u
                i32.or
                i32.const 16843009
                i32.and
                local.get 1
                i32.load
                local.tee 9
                i32.const -1
                i32.xor
                i32.const 7
                i32.shr_u
                local.get 9
                i32.const 6
                i32.shr_u
                i32.or
                i32.const 16843009
                i32.and
                local.get 2
                i32.add
                i32.add
                i32.add
                i32.add
                local.set 2
                local.get 1
                i32.const 16
                i32.add
                local.tee 1
                local.get 0
                i32.ne
                br_if 0 (;@5;)
              end
            end
            local.get 3
            local.get 6
            i32.sub
            local.set 3
            local.get 4
            local.get 5
            i32.add
            local.set 9
            local.get 2
            i32.const 8
            i32.shr_u
            i32.const 16711935
            i32.and
            local.get 2
            i32.const 16711935
            i32.and
            i32.add
            i32.const 65537
            i32.mul
            i32.const 16
            i32.shr_u
            local.get 8
            i32.add
            local.set 8
            local.get 7
            i32.eqz
            br_if 0 (;@3;)
          end
          local.get 4
          local.get 6
          i32.const 252
          i32.and
          i32.const 2
          i32.shl
          i32.add
          local.tee 2
          i32.load
          local.tee 1
          i32.const -1
          i32.xor
          i32.const 7
          i32.shr_u
          local.get 1
          i32.const 6
          i32.shr_u
          i32.or
          i32.const 16843009
          i32.and
          local.set 1
          block ;; label = @3
            local.get 7
            i32.const 1
            i32.eq
            br_if 0 (;@3;)
            local.get 2
            i32.load offset=4
            local.tee 9
            i32.const -1
            i32.xor
            i32.const 7
            i32.shr_u
            local.get 9
            i32.const 6
            i32.shr_u
            i32.or
            i32.const 16843009
            i32.and
            local.get 1
            i32.add
            local.set 1
            local.get 7
            i32.const 2
            i32.eq
            br_if 0 (;@3;)
            local.get 2
            i32.load offset=8
            local.tee 2
            i32.const -1
            i32.xor
            i32.const 7
            i32.shr_u
            local.get 2
            i32.const 6
            i32.shr_u
            i32.or
            i32.const 16843009
            i32.and
            local.get 1
            i32.add
            local.set 1
          end
          local.get 1
          i32.const 8
          i32.shr_u
          i32.const 459007
          i32.and
          local.get 1
          i32.const 16711935
          i32.and
          i32.add
          i32.const 65537
          i32.mul
          i32.const 16
          i32.shr_u
          local.get 8
          i32.add
          return
        end
        block ;; label = @2
          local.get 1
          br_if 0 (;@2;)
          i32.const 0
          return
        end
        local.get 1
        i32.const 3
        i32.and
        local.set 9
        block ;; label = @2
          block ;; label = @3
            local.get 1
            i32.const 4
            i32.ge_u
            br_if 0 (;@3;)
            i32.const 0
            local.set 8
            i32.const 0
            local.set 2
            br 1 (;@2;)
          end
          local.get 1
          i32.const -4
          i32.and
          local.set 3
          i32.const 0
          local.set 8
          i32.const 0
          local.set 2
          loop ;; label = @3
            local.get 8
            local.get 0
            local.get 2
            i32.add
            local.tee 1
            i32.load8_s
            i32.const -65
            i32.gt_s
            i32.add
            local.get 1
            i32.const 1
            i32.add
            i32.load8_s
            i32.const -65
            i32.gt_s
            i32.add
            local.get 1
            i32.const 2
            i32.add
            i32.load8_s
            i32.const -65
            i32.gt_s
            i32.add
            local.get 1
            i32.const 3
            i32.add
            i32.load8_s
            i32.const -65
            i32.gt_s
            i32.add
            local.set 8
            local.get 3
            local.get 2
            i32.const 4
            i32.add
            local.tee 2
            i32.ne
            br_if 0 (;@3;)
          end
        end
        local.get 9
        i32.eqz
        br_if 0 (;@1;)
        local.get 0
        local.get 2
        i32.add
        local.set 1
        loop ;; label = @2
          local.get 8
          local.get 1
          i32.load8_s
          i32.const -65
          i32.gt_s
          i32.add
          local.set 8
          local.get 1
          i32.const 1
          i32.add
          local.set 1
          local.get 9
          i32.const -1
          i32.add
          local.tee 9
          br_if 0 (;@2;)
        end
      end
      local.get 8
    )
    (func $_ZN4core3fmt9Formatter12pad_integral12write_prefix17h87435251ed4e8074E (;180;) (type 11) (param i32 i32 i32 i32 i32) (result i32)
      (local i32)
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            local.get 2
            i32.const 1114112
            i32.eq
            br_if 0 (;@3;)
            i32.const 1
            local.set 5
            local.get 0
            local.get 2
            local.get 1
            i32.load offset=16
            call_indirect (type 2)
            br_if 1 (;@2;)
          end
          local.get 3
          br_if 1 (;@1;)
          i32.const 0
          local.set 5
        end
        local.get 5
        return
      end
      local.get 0
      local.get 3
      local.get 4
      local.get 1
      i32.load offset=12
      call_indirect (type 1)
    )
    (func $_ZN4core3fmt9Formatter9write_str17h8905e7a9d9d32ff1E (;181;) (type 1) (param i32 i32 i32) (result i32)
      local.get 0
      i32.load offset=20
      local.get 1
      local.get 2
      local.get 0
      i32.load offset=24
      i32.load offset=12
      call_indirect (type 1)
    )
    (func $_ZN4core3fmt9Formatter26debug_struct_field2_finish17h73c9201d8434494dE (;182;) (type 14) (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)
      (local i32)
      global.get $__stack_pointer
      i32.const 16
      i32.sub
      local.tee 11
      global.set $__stack_pointer
      local.get 0
      i32.load offset=20
      local.get 1
      local.get 2
      local.get 0
      i32.load offset=24
      i32.load offset=12
      call_indirect (type 1)
      local.set 2
      local.get 11
      i32.const 0
      i32.store8 offset=13
      local.get 11
      local.get 2
      i32.store8 offset=12
      local.get 11
      local.get 0
      i32.store offset=8
      local.get 11
      i32.const 8
      i32.add
      local.get 3
      local.get 4
      local.get 5
      local.get 6
      call $_ZN4core3fmt8builders11DebugStruct5field17h1f6a8e88b3d0a888E
      local.get 7
      local.get 8
      local.get 9
      local.get 10
      call $_ZN4core3fmt8builders11DebugStruct5field17h1f6a8e88b3d0a888E
      local.set 1
      local.get 11
      i32.load8_u offset=12
      local.set 2
      block ;; label = @1
        block ;; label = @2
          local.get 11
          i32.load8_u offset=13
          br_if 0 (;@2;)
          local.get 2
          i32.const 255
          i32.and
          i32.const 0
          i32.ne
          local.set 0
          br 1 (;@1;)
        end
        i32.const 1
        local.set 0
        local.get 2
        i32.const 255
        i32.and
        br_if 0 (;@1;)
        block ;; label = @2
          local.get 1
          i32.load
          local.tee 0
          i32.load8_u offset=28
          i32.const 4
          i32.and
          br_if 0 (;@2;)
          local.get 0
          i32.load offset=20
          i32.const 1052871
          i32.const 2
          local.get 0
          i32.load offset=24
          i32.load offset=12
          call_indirect (type 1)
          local.set 0
          br 1 (;@1;)
        end
        local.get 0
        i32.load offset=20
        i32.const 1052870
        i32.const 1
        local.get 0
        i32.load offset=24
        i32.load offset=12
        call_indirect (type 1)
        local.set 0
      end
      local.get 11
      i32.const 16
      i32.add
      global.set $__stack_pointer
      local.get 0
    )
    (func $_ZN4core3fmt9Formatter25debug_tuple_field1_finish17h60a7f9de909b7316E (;183;) (type 11) (param i32 i32 i32 i32 i32) (result i32)
      (local i32)
      global.get $__stack_pointer
      i32.const 16
      i32.sub
      local.tee 5
      global.set $__stack_pointer
      local.get 5
      local.get 0
      i32.load offset=20
      local.get 1
      local.get 2
      local.get 0
      i32.load offset=24
      i32.load offset=12
      call_indirect (type 1)
      i32.store8 offset=12
      local.get 5
      local.get 0
      i32.store offset=8
      local.get 5
      local.get 2
      i32.eqz
      i32.store8 offset=13
      local.get 5
      i32.const 0
      i32.store offset=4
      local.get 5
      i32.const 4
      i32.add
      local.get 3
      local.get 4
      call $_ZN4core3fmt8builders10DebugTuple5field17h16e17751cd068088E
      local.set 0
      local.get 5
      i32.load8_u offset=12
      local.set 2
      block ;; label = @1
        block ;; label = @2
          local.get 0
          i32.load
          local.tee 1
          br_if 0 (;@2;)
          local.get 2
          i32.const 255
          i32.and
          i32.const 0
          i32.ne
          local.set 0
          br 1 (;@1;)
        end
        i32.const 1
        local.set 0
        local.get 2
        i32.const 255
        i32.and
        br_if 0 (;@1;)
        local.get 5
        i32.load offset=8
        local.set 2
        block ;; label = @2
          local.get 1
          i32.const 1
          i32.ne
          br_if 0 (;@2;)
          local.get 5
          i32.load8_u offset=13
          i32.const 255
          i32.and
          i32.eqz
          br_if 0 (;@2;)
          local.get 2
          i32.load8_u offset=28
          i32.const 4
          i32.and
          br_if 0 (;@2;)
          i32.const 1
          local.set 0
          local.get 2
          i32.load offset=20
          i32.const 1052876
          i32.const 1
          local.get 2
          i32.load offset=24
          i32.load offset=12
          call_indirect (type 1)
          br_if 1 (;@1;)
        end
        local.get 2
        i32.load offset=20
        i32.const 1052404
        i32.const 1
        local.get 2
        i32.load offset=24
        i32.load offset=12
        call_indirect (type 1)
        local.set 0
      end
      local.get 5
      i32.const 16
      i32.add
      global.set $__stack_pointer
      local.get 0
    )
    (func $_ZN4core3fmt9Formatter10debug_list17h53cbf25952dc9840E (;184;) (type 0) (param i32 i32)
      (local i32)
      local.get 1
      i32.load offset=20
      i32.const 1052423
      i32.const 1
      local.get 1
      i32.load offset=24
      i32.load offset=12
      call_indirect (type 1)
      local.set 2
      local.get 0
      i32.const 0
      i32.store8 offset=5
      local.get 0
      local.get 2
      i32.store8 offset=4
      local.get 0
      local.get 1
      i32.store
    )
    (func $_ZN40_$LT$str$u20$as$u20$core..fmt..Debug$GT$3fmt17h9c38ad0737c78d42E (;185;) (type 1) (param i32 i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
      global.get $__stack_pointer
      i32.const 16
      i32.sub
      local.tee 3
      global.set $__stack_pointer
      i32.const 1
      local.set 4
      block ;; label = @1
        local.get 2
        i32.load offset=20
        local.tee 5
        i32.const 34
        local.get 2
        i32.load offset=24
        local.tee 6
        i32.load offset=16
        local.tee 7
        call_indirect (type 2)
        br_if 0 (;@1;)
        block ;; label = @2
          block ;; label = @3
            local.get 1
            br_if 0 (;@3;)
            i32.const 0
            local.set 2
            i32.const 0
            local.set 8
            br 1 (;@2;)
          end
          i32.const 0
          local.set 9
          i32.const 0
          local.get 1
          i32.sub
          local.set 10
          i32.const 0
          local.set 11
          local.get 0
          local.set 12
          local.get 1
          local.set 13
          block ;; label = @3
            block ;; label = @4
              block ;; label = @5
                loop ;; label = @6
                  local.get 12
                  local.get 13
                  i32.add
                  local.set 14
                  i32.const 0
                  local.set 2
                  block ;; label = @7
                    loop ;; label = @8
                      local.get 12
                      local.get 2
                      i32.add
                      local.tee 15
                      i32.load8_u
                      local.tee 8
                      i32.const -127
                      i32.add
                      i32.const 255
                      i32.and
                      i32.const 161
                      i32.lt_u
                      br_if 1 (;@7;)
                      local.get 8
                      i32.const 34
                      i32.eq
                      br_if 1 (;@7;)
                      local.get 8
                      i32.const 92
                      i32.eq
                      br_if 1 (;@7;)
                      local.get 13
                      local.get 2
                      i32.const 1
                      i32.add
                      local.tee 2
                      i32.ne
                      br_if 0 (;@8;)
                    end
                    local.get 11
                    local.get 13
                    i32.add
                    local.set 2
                    br 4 (;@3;)
                  end
                  local.get 15
                  i32.const 1
                  i32.add
                  local.set 12
                  block ;; label = @7
                    block ;; label = @8
                      local.get 15
                      i32.load8_s
                      local.tee 8
                      i32.const -1
                      i32.le_s
                      br_if 0 (;@8;)
                      local.get 8
                      i32.const 255
                      i32.and
                      local.set 8
                      br 1 (;@7;)
                    end
                    local.get 12
                    i32.load8_u
                    i32.const 63
                    i32.and
                    local.set 13
                    local.get 8
                    i32.const 31
                    i32.and
                    local.set 16
                    local.get 15
                    i32.const 2
                    i32.add
                    local.set 12
                    block ;; label = @8
                      local.get 8
                      i32.const -33
                      i32.gt_u
                      br_if 0 (;@8;)
                      local.get 16
                      i32.const 6
                      i32.shl
                      local.get 13
                      i32.or
                      local.set 8
                      br 1 (;@7;)
                    end
                    local.get 13
                    i32.const 6
                    i32.shl
                    local.get 12
                    i32.load8_u
                    i32.const 63
                    i32.and
                    i32.or
                    local.set 13
                    local.get 15
                    i32.const 3
                    i32.add
                    local.set 12
                    block ;; label = @8
                      local.get 8
                      i32.const -16
                      i32.ge_u
                      br_if 0 (;@8;)
                      local.get 13
                      local.get 16
                      i32.const 12
                      i32.shl
                      i32.or
                      local.set 8
                      br 1 (;@7;)
                    end
                    local.get 13
                    i32.const 6
                    i32.shl
                    local.get 12
                    i32.load8_u
                    i32.const 63
                    i32.and
                    i32.or
                    local.get 16
                    i32.const 18
                    i32.shl
                    i32.const 1835008
                    i32.and
                    i32.or
                    local.set 8
                    local.get 15
                    i32.const 4
                    i32.add
                    local.set 12
                  end
                  local.get 3
                  i32.const 4
                  i32.add
                  local.get 8
                  i32.const 65537
                  call $_ZN4core4char7methods22_$LT$impl$u20$char$GT$16escape_debug_ext17h0025cf56996c3cbcE
                  block ;; label = @7
                    block ;; label = @8
                      local.get 3
                      i32.load8_u offset=4
                      i32.const 128
                      i32.eq
                      br_if 0 (;@8;)
                      local.get 3
                      i32.load8_u offset=15
                      local.get 3
                      i32.load8_u offset=14
                      i32.sub
                      i32.const 255
                      i32.and
                      i32.const 1
                      i32.eq
                      br_if 0 (;@8;)
                      local.get 9
                      local.get 11
                      local.get 2
                      i32.add
                      local.tee 15
                      i32.gt_u
                      br_if 3 (;@5;)
                      block ;; label = @9
                        local.get 9
                        i32.eqz
                        br_if 0 (;@9;)
                        block ;; label = @10
                          local.get 9
                          local.get 1
                          i32.ge_u
                          br_if 0 (;@10;)
                          local.get 0
                          local.get 9
                          i32.add
                          i32.load8_s
                          i32.const -65
                          i32.gt_s
                          br_if 1 (;@9;)
                          br 5 (;@5;)
                        end
                        local.get 9
                        local.get 1
                        i32.ne
                        br_if 4 (;@5;)
                      end
                      block ;; label = @9
                        local.get 15
                        i32.eqz
                        br_if 0 (;@9;)
                        block ;; label = @10
                          local.get 15
                          local.get 1
                          i32.ge_u
                          br_if 0 (;@10;)
                          local.get 0
                          local.get 11
                          i32.add
                          local.get 2
                          i32.add
                          i32.load8_s
                          i32.const -65
                          i32.le_s
                          br_if 5 (;@5;)
                          br 1 (;@9;)
                        end
                        local.get 15
                        local.get 10
                        i32.add
                        br_if 4 (;@5;)
                      end
                      local.get 5
                      local.get 0
                      local.get 9
                      i32.add
                      local.get 11
                      local.get 9
                      i32.sub
                      local.get 2
                      i32.add
                      local.get 6
                      i32.load offset=12
                      local.tee 15
                      call_indirect (type 1)
                      br_if 1 (;@7;)
                      block ;; label = @9
                        block ;; label = @10
                          local.get 3
                          i32.load8_u offset=4
                          i32.const 128
                          i32.ne
                          br_if 0 (;@10;)
                          local.get 5
                          local.get 3
                          i32.load offset=8
                          local.get 7
                          call_indirect (type 2)
                          i32.eqz
                          br_if 1 (;@9;)
                          br 3 (;@7;)
                        end
                        local.get 5
                        local.get 3
                        i32.const 4
                        i32.add
                        local.get 3
                        i32.load8_u offset=14
                        local.tee 13
                        i32.add
                        local.get 3
                        i32.load8_u offset=15
                        local.get 13
                        i32.sub
                        local.get 15
                        call_indirect (type 1)
                        br_if 2 (;@7;)
                      end
                      i32.const 1
                      local.set 15
                      block ;; label = @9
                        local.get 8
                        i32.const 128
                        i32.lt_u
                        br_if 0 (;@9;)
                        i32.const 2
                        local.set 15
                        local.get 8
                        i32.const 2048
                        i32.lt_u
                        br_if 0 (;@9;)
                        i32.const 3
                        i32.const 4
                        local.get 8
                        i32.const 65536
                        i32.lt_u
                        select
                        local.set 15
                      end
                      local.get 15
                      local.get 11
                      i32.add
                      local.get 2
                      i32.add
                      local.set 9
                    end
                    i32.const 1
                    local.set 15
                    block ;; label = @8
                      local.get 8
                      i32.const 128
                      i32.lt_u
                      br_if 0 (;@8;)
                      i32.const 2
                      local.set 15
                      local.get 8
                      i32.const 2048
                      i32.lt_u
                      br_if 0 (;@8;)
                      i32.const 3
                      i32.const 4
                      local.get 8
                      i32.const 65536
                      i32.lt_u
                      select
                      local.set 15
                    end
                    local.get 15
                    local.get 11
                    i32.add
                    local.tee 8
                    local.get 2
                    i32.add
                    local.set 11
                    local.get 14
                    local.get 12
                    i32.sub
                    local.tee 13
                    i32.eqz
                    br_if 3 (;@4;)
                    br 1 (;@6;)
                  end
                end
                i32.const 1
                local.set 4
                br 4 (;@1;)
              end
              local.get 0
              local.get 1
              local.get 9
              local.get 15
              i32.const 1053156
              call $_ZN4core3str16slice_error_fail17h8a0d598ae9311b97E
              unreachable
            end
            local.get 8
            local.get 2
            i32.add
            local.set 2
          end
          block ;; label = @3
            local.get 9
            local.get 2
            i32.gt_u
            br_if 0 (;@3;)
            i32.const 0
            local.set 8
            block ;; label = @4
              local.get 9
              i32.eqz
              br_if 0 (;@4;)
              block ;; label = @5
                local.get 9
                local.get 1
                i32.ge_u
                br_if 0 (;@5;)
                local.get 9
                local.set 8
                local.get 0
                local.get 9
                i32.add
                i32.load8_s
                i32.const -65
                i32.le_s
                br_if 2 (;@3;)
                br 1 (;@4;)
              end
              local.get 9
              local.set 8
              local.get 9
              local.get 1
              i32.ne
              br_if 1 (;@3;)
            end
            block ;; label = @4
              local.get 2
              br_if 0 (;@4;)
              i32.const 0
              local.set 2
              br 2 (;@2;)
            end
            block ;; label = @4
              local.get 2
              local.get 1
              i32.ge_u
              br_if 0 (;@4;)
              local.get 8
              local.set 9
              local.get 0
              local.get 2
              i32.add
              i32.load8_s
              i32.const -65
              i32.gt_s
              br_if 2 (;@2;)
              br 1 (;@3;)
            end
            local.get 8
            local.set 9
            local.get 2
            local.get 1
            i32.eq
            br_if 1 (;@2;)
          end
          local.get 0
          local.get 1
          local.get 9
          local.get 2
          i32.const 1053172
          call $_ZN4core3str16slice_error_fail17h8a0d598ae9311b97E
          unreachable
        end
        local.get 5
        local.get 0
        local.get 8
        i32.add
        local.get 2
        local.get 8
        i32.sub
        local.get 6
        i32.load offset=12
        call_indirect (type 1)
        br_if 0 (;@1;)
        local.get 5
        i32.const 34
        local.get 7
        call_indirect (type 2)
        local.set 4
      end
      local.get 3
      i32.const 16
      i32.add
      global.set $__stack_pointer
      local.get 4
    )
    (func $_ZN4core3str16slice_error_fail17h8a0d598ae9311b97E (;186;) (type 8) (param i32 i32 i32 i32 i32)
      local.get 0
      local.get 1
      local.get 2
      local.get 3
      local.get 4
      call $_ZN4core3str19slice_error_fail_rt17hdcab032072352c3fE
      unreachable
    )
    (func $_ZN41_$LT$char$u20$as$u20$core..fmt..Debug$GT$3fmt17hc694b486c95ba763E (;187;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32 i32)
      global.get $__stack_pointer
      i32.const 16
      i32.sub
      local.tee 2
      global.set $__stack_pointer
      i32.const 1
      local.set 3
      block ;; label = @1
        local.get 1
        i32.load offset=20
        local.tee 4
        i32.const 39
        local.get 1
        i32.load offset=24
        local.tee 5
        i32.load offset=16
        local.tee 1
        call_indirect (type 2)
        br_if 0 (;@1;)
        local.get 2
        i32.const 4
        i32.add
        local.get 0
        i32.load
        i32.const 257
        call $_ZN4core4char7methods22_$LT$impl$u20$char$GT$16escape_debug_ext17h0025cf56996c3cbcE
        block ;; label = @2
          block ;; label = @3
            local.get 2
            i32.load8_u offset=4
            i32.const 128
            i32.ne
            br_if 0 (;@3;)
            local.get 4
            local.get 2
            i32.load offset=8
            local.get 1
            call_indirect (type 2)
            i32.eqz
            br_if 1 (;@2;)
            br 2 (;@1;)
          end
          local.get 4
          local.get 2
          i32.const 4
          i32.add
          local.get 2
          i32.load8_u offset=14
          local.tee 0
          i32.add
          local.get 2
          i32.load8_u offset=15
          local.get 0
          i32.sub
          local.get 5
          i32.load offset=12
          call_indirect (type 1)
          br_if 1 (;@1;)
        end
        local.get 4
        i32.const 39
        local.get 1
        call_indirect (type 2)
        local.set 3
      end
      local.get 2
      i32.const 16
      i32.add
      global.set $__stack_pointer
      local.get 3
    )
    (func $_ZN4core3fmt3num52_$LT$impl$u20$core..fmt..UpperHex$u20$for$u20$i8$GT$3fmt17h2f283451cba24807E (;188;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32)
      global.get $__stack_pointer
      i32.const 128
      i32.sub
      local.tee 2
      global.set $__stack_pointer
      local.get 0
      i32.load8_u
      local.set 3
      i32.const 0
      local.set 0
      loop ;; label = @1
        local.get 2
        local.get 0
        i32.add
        i32.const 127
        i32.add
        local.get 3
        i32.const 15
        i32.and
        local.tee 4
        i32.const 48
        i32.or
        local.get 4
        i32.const 55
        i32.add
        local.get 4
        i32.const 10
        i32.lt_u
        select
        i32.store8
        local.get 0
        i32.const -1
        i32.add
        local.set 0
        local.get 3
        i32.const 255
        i32.and
        local.tee 4
        i32.const 4
        i32.shr_u
        local.set 3
        local.get 4
        i32.const 16
        i32.ge_u
        br_if 0 (;@1;)
      end
      block ;; label = @1
        local.get 0
        i32.const 128
        i32.add
        local.tee 3
        i32.const 129
        i32.lt_u
        br_if 0 (;@1;)
        local.get 3
        i32.const 128
        i32.const 1052908
        call $_ZN4core5slice5index26slice_start_index_len_fail17h0fcaa929c46d2711E
        unreachable
      end
      local.get 1
      i32.const 1
      i32.const 1052924
      i32.const 2
      local.get 2
      local.get 0
      i32.add
      i32.const 128
      i32.add
      i32.const 0
      local.get 0
      i32.sub
      call $_ZN4core3fmt9Formatter12pad_integral17h2d614e17aad082d7E
      local.set 0
      local.get 2
      i32.const 128
      i32.add
      global.set $__stack_pointer
      local.get 0
    )
    (func $_ZN4core3str19slice_error_fail_rt17hdcab032072352c3fE (;189;) (type 8) (param i32 i32 i32 i32 i32)
      (local i32 i32 i32 i32 i32 i64)
      global.get $__stack_pointer
      i32.const 112
      i32.sub
      local.tee 5
      global.set $__stack_pointer
      local.get 5
      local.get 3
      i32.store offset=12
      local.get 5
      local.get 2
      i32.store offset=8
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            local.get 1
            i32.const 257
            i32.lt_u
            br_if 0 (;@3;)
            i32.const 3
            local.set 6
            block ;; label = @4
              local.get 0
              i32.load8_s offset=256
              i32.const -65
              i32.gt_s
              br_if 0 (;@4;)
              i32.const 2
              local.set 6
              local.get 0
              i32.load8_s offset=255
              i32.const -65
              i32.gt_s
              br_if 0 (;@4;)
              local.get 0
              i32.load8_s offset=254
              i32.const -65
              i32.gt_s
              local.set 6
            end
            local.get 0
            local.get 6
            i32.const 253
            i32.add
            local.tee 6
            i32.add
            i32.load8_s
            i32.const -65
            i32.le_s
            br_if 1 (;@2;)
            local.get 5
            local.get 6
            i32.store offset=20
            local.get 5
            local.get 0
            i32.store offset=16
            i32.const 5
            local.set 6
            i32.const 1053596
            local.set 7
            br 2 (;@1;)
          end
          local.get 5
          local.get 1
          i32.store offset=20
          local.get 5
          local.get 0
          i32.store offset=16
          i32.const 0
          local.set 6
          i32.const 1
          local.set 7
          br 1 (;@1;)
        end
        local.get 0
        local.get 1
        i32.const 0
        local.get 6
        local.get 4
        call $_ZN4core3str16slice_error_fail17h8a0d598ae9311b97E
        unreachable
      end
      local.get 5
      local.get 6
      i32.store offset=28
      local.get 5
      local.get 7
      i32.store offset=24
      block ;; label = @1
        block ;; label = @2
          block ;; label = @3
            block ;; label = @4
              local.get 2
              local.get 1
              i32.gt_u
              local.tee 6
              br_if 0 (;@4;)
              local.get 3
              local.get 1
              i32.gt_u
              br_if 0 (;@4;)
              local.get 2
              local.get 3
              i32.gt_u
              br_if 1 (;@3;)
              block ;; label = @5
                local.get 2
                i32.eqz
                br_if 0 (;@5;)
                local.get 2
                local.get 1
                i32.ge_u
                br_if 0 (;@5;)
                local.get 5
                i32.const 12
                i32.add
                local.get 5
                i32.const 8
                i32.add
                local.get 0
                local.get 2
                i32.add
                i32.load8_s
                i32.const -65
                i32.gt_s
                select
                i32.load
                local.set 3
              end
              local.get 5
              local.get 3
              i32.store offset=32
              local.get 1
              local.set 2
              block ;; label = @5
                local.get 3
                local.get 1
                i32.ge_u
                br_if 0 (;@5;)
                local.get 3
                i32.const 1
                i32.add
                local.tee 6
                i32.const 0
                local.get 3
                i32.const -3
                i32.add
                local.tee 2
                local.get 2
                local.get 3
                i32.gt_u
                select
                local.tee 2
                i32.lt_u
                br_if 3 (;@2;)
                block ;; label = @6
                  local.get 2
                  local.get 6
                  i32.eq
                  br_if 0 (;@6;)
                  local.get 0
                  local.get 6
                  i32.add
                  local.get 0
                  local.get 2
                  i32.add
                  local.tee 8
                  i32.sub
                  local.set 6
                  block ;; label = @7
                    local.get 0
                    local.get 3
                    i32.add
                    local.tee 9
                    i32.load8_s
                    i32.const -65
                    i32.le_s
                    br_if 0 (;@7;)
                    local.get 6
                    i32.const -1
                    i32.add
                    local.set 7
                    br 1 (;@6;)
                  end
                  local.get 2
                  local.get 3
                  i32.eq
                  br_if 0 (;@6;)
                  block ;; label = @7
                    local.get 9
                    i32.const -1
                    i32.add
                    local.tee 3
                    i32.load8_s
                    i32.const -65
                    i32.le_s
                    br_if 0 (;@7;)
                    local.get 6
                    i32.const -2
                    i32.add
                    local.set 7
                    br 1 (;@6;)
                  end
                  local.get 8
                  local.get 3
                  i32.eq
                  br_if 0 (;@6;)
                  block ;; label = @7
                    local.get 9
                    i32.const -2
                    i32.add
                    local.tee 3
                    i32.load8_s
                    i32.const -65
                    i32.le_s
                    br_if 0 (;@7;)
                    local.get 6
                    i32.const -3
                    i32.add
                    local.set 7
                    br 1 (;@6;)
                  end
                  local.get 8
                  local.get 3
                  i32.eq
                  br_if 0 (;@6;)
                  block ;; label = @7
                    local.get 9
                    i32.const -3
                    i32.add
                    local.tee 3
                    i32.load8_s
                    i32.const -65
                    i32.le_s
                    br_if 0 (;@7;)
                    local.get 6
                    i32.const -4
                    i32.add
                    local.set 7
                    br 1 (;@6;)
                  end
                  local.get 8
                  local.get 3
                  i32.eq
                  br_if 0 (;@6;)
                  local.get 6
                  i32.const -5
                  i32.add
                  local.set 7
                end
                local.get 7
                local.get 2
                i32.add
                local.set 2
              end
              block ;; label = @5
                local.get 2
                i32.eqz
                br_if 0 (;@5;)
                block ;; label = @6
                  local.get 2
                  local.get 1
                  i32.ge_u
                  br_if 0 (;@6;)
                  local.get 0
                  local.get 2
                  i32.add
                  i32.load8_s
                  i32.const -65
                  i32.gt_s
                  br_if 1 (;@5;)
                  br 5 (;@1;)
                end
                local.get 2
                local.get 1
                i32.ne
                br_if 4 (;@1;)
              end
              block ;; label = @5
                block ;; label = @6
                  block ;; label = @7
                    block ;; label = @8
                      local.get 2
                      local.get 1
                      i32.eq
                      br_if 0 (;@8;)
                      block ;; label = @9
                        block ;; label = @10
                          local.get 0
                          local.get 2
                          i32.add
                          local.tee 3
                          i32.load8_s
                          local.tee 1
                          i32.const -1
                          i32.gt_s
                          br_if 0 (;@10;)
                          local.get 3
                          i32.load8_u offset=1
                          i32.const 63
                          i32.and
                          local.set 0
                          local.get 1
                          i32.const 31
                          i32.and
                          local.set 6
                          local.get 1
                          i32.const -33
                          i32.gt_u
                          br_if 1 (;@9;)
                          local.get 6
                          i32.const 6
                          i32.shl
                          local.get 0
                          i32.or
                          local.set 3
                          br 4 (;@6;)
                        end
                        local.get 5
                        local.get 1
                        i32.const 255
                        i32.and
                        i32.store offset=36
                        i32.const 1
                        local.set 1
                        br 4 (;@5;)
                      end
                      local.get 0
                      i32.const 6
                      i32.shl
                      local.get 3
                      i32.load8_u offset=2
                      i32.const 63
                      i32.and
                      i32.or
                      local.set 0
                      local.get 1
                      i32.const -16
                      i32.ge_u
                      br_if 1 (;@7;)
                      local.get 0
                      local.get 6
                      i32.const 12
                      i32.shl
                      i32.or
                      local.set 3
                      br 2 (;@6;)
                    end
                    local.get 4
                    call $_ZN4core6option13unwrap_failed17hd879e969cd9bde53E
                    unreachable
                  end
                  local.get 0
                  i32.const 6
                  i32.shl
                  local.get 3
                  i32.load8_u offset=3
                  i32.const 63
                  i32.and
                  i32.or
                  local.get 6
                  i32.const 18
                  i32.shl
                  i32.const 1835008
                  i32.and
                  i32.or
                  local.set 3
                end
                local.get 5
                local.get 3
                i32.store offset=36
                i32.const 1
                local.set 1
                local.get 3
                i32.const 128
                i32.lt_u
                br_if 0 (;@5;)
                i32.const 2
                local.set 1
                local.get 3
                i32.const 2048
                i32.lt_u
                br_if 0 (;@5;)
                i32.const 3
                i32.const 4
                local.get 3
                i32.const 65536
                i32.lt_u
                select
                local.set 1
              end
              local.get 5
              local.get 2
              i32.store offset=40
              local.get 5
              local.get 1
              local.get 2
              i32.add
              i32.store offset=44
              local.get 5
              i32.const 5
              i32.store offset=52
              local.get 5
              i32.const 1053732
              i32.store offset=48
              local.get 5
              i64.const 5
              i64.store offset=60 align=4
              local.get 5
              i32.const 40
              i64.extend_i32_u
              i64.const 32
              i64.shl
              local.tee 10
              local.get 5
              i32.const 24
              i32.add
              i64.extend_i32_u
              i64.or
              i64.store offset=104
              local.get 5
              local.get 10
              local.get 5
              i32.const 16
              i32.add
              i64.extend_i32_u
              i64.or
              i64.store offset=96
              local.get 5
              i32.const 43
              i64.extend_i32_u
              i64.const 32
              i64.shl
              local.get 5
              i32.const 40
              i32.add
              i64.extend_i32_u
              i64.or
              i64.store offset=88
              local.get 5
              i32.const 44
              i64.extend_i32_u
              i64.const 32
              i64.shl
              local.get 5
              i32.const 36
              i32.add
              i64.extend_i32_u
              i64.or
              i64.store offset=80
              local.get 5
              i32.const 17
              i64.extend_i32_u
              i64.const 32
              i64.shl
              local.get 5
              i32.const 32
              i32.add
              i64.extend_i32_u
              i64.or
              i64.store offset=72
              local.get 5
              local.get 5
              i32.const 72
              i32.add
              i32.store offset=56
              local.get 5
              i32.const 48
              i32.add
              local.get 4
              call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
              unreachable
            end
            local.get 5
            local.get 2
            local.get 3
            local.get 6
            select
            i32.store offset=40
            local.get 5
            i32.const 3
            i32.store offset=52
            local.get 5
            i32.const 1053796
            i32.store offset=48
            local.get 5
            i64.const 3
            i64.store offset=60 align=4
            local.get 5
            i32.const 40
            i64.extend_i32_u
            i64.const 32
            i64.shl
            local.tee 10
            local.get 5
            i32.const 24
            i32.add
            i64.extend_i32_u
            i64.or
            i64.store offset=88
            local.get 5
            local.get 10
            local.get 5
            i32.const 16
            i32.add
            i64.extend_i32_u
            i64.or
            i64.store offset=80
            local.get 5
            i32.const 17
            i64.extend_i32_u
            i64.const 32
            i64.shl
            local.get 5
            i32.const 40
            i32.add
            i64.extend_i32_u
            i64.or
            i64.store offset=72
            local.get 5
            local.get 5
            i32.const 72
            i32.add
            i32.store offset=56
            local.get 5
            i32.const 48
            i32.add
            local.get 4
            call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
            unreachable
          end
          local.get 5
          i32.const 4
          i32.store offset=52
          local.get 5
          i32.const 1053636
          i32.store offset=48
          local.get 5
          i64.const 4
          i64.store offset=60 align=4
          local.get 5
          i32.const 40
          i64.extend_i32_u
          i64.const 32
          i64.shl
          local.tee 10
          local.get 5
          i32.const 24
          i32.add
          i64.extend_i32_u
          i64.or
          i64.store offset=96
          local.get 5
          local.get 10
          local.get 5
          i32.const 16
          i32.add
          i64.extend_i32_u
          i64.or
          i64.store offset=88
          local.get 5
          i32.const 17
          i64.extend_i32_u
          i64.const 32
          i64.shl
          local.tee 10
          local.get 5
          i32.const 12
          i32.add
          i64.extend_i32_u
          i64.or
          i64.store offset=80
          local.get 5
          local.get 10
          local.get 5
          i32.const 8
          i32.add
          i64.extend_i32_u
          i64.or
          i64.store offset=72
          local.get 5
          local.get 5
          i32.const 72
          i32.add
          i32.store offset=56
          local.get 5
          i32.const 48
          i32.add
          local.get 4
          call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
          unreachable
        end
        local.get 2
        local.get 6
        i32.const 1053848
        call $_ZN4core5slice5index22slice_index_order_fail17hf860bb9cafe28110E
        unreachable
      end
      local.get 0
      local.get 1
      local.get 2
      local.get 1
      local.get 4
      call $_ZN4core3str16slice_error_fail17h8a0d598ae9311b97E
      unreachable
    )
    (func $_ZN4core7unicode9printable5check17ha84a2bb0c8b2a936E (;190;) (type 15) (param i32 i32 i32 i32 i32 i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32)
      i32.const 1
      local.set 7
      block ;; label = @1
        block ;; label = @2
          local.get 2
          i32.eqz
          br_if 0 (;@2;)
          local.get 1
          local.get 2
          i32.const 1
          i32.shl
          i32.add
          local.set 8
          local.get 0
          i32.const 65280
          i32.and
          i32.const 8
          i32.shr_u
          local.set 9
          i32.const 0
          local.set 10
          local.get 0
          i32.const 255
          i32.and
          local.set 11
          loop ;; label = @3
            local.get 1
            i32.const 2
            i32.add
            local.set 12
            local.get 10
            local.get 1
            i32.load8_u offset=1
            local.tee 2
            i32.add
            local.set 13
            block ;; label = @4
              local.get 1
              i32.load8_u
              local.tee 1
              local.get 9
              i32.eq
              br_if 0 (;@4;)
              local.get 1
              local.get 9
              i32.gt_u
              br_if 2 (;@2;)
              local.get 13
              local.set 10
              local.get 12
              local.set 1
              local.get 12
              local.get 8
              i32.eq
              br_if 2 (;@2;)
              br 1 (;@3;)
            end
            block ;; label = @4
              block ;; label = @5
                block ;; label = @6
                  local.get 13
                  local.get 10
                  i32.lt_u
                  br_if 0 (;@6;)
                  local.get 13
                  local.get 4
                  i32.gt_u
                  br_if 1 (;@5;)
                  local.get 3
                  local.get 10
                  i32.add
                  local.set 1
                  loop ;; label = @7
                    local.get 2
                    i32.eqz
                    br_if 3 (;@4;)
                    local.get 2
                    i32.const -1
                    i32.add
                    local.set 2
                    local.get 1
                    i32.load8_u
                    local.set 10
                    local.get 1
                    i32.const 1
                    i32.add
                    local.set 1
                    local.get 10
                    local.get 11
                    i32.ne
                    br_if 0 (;@7;)
                  end
                  i32.const 0
                  local.set 7
                  br 5 (;@1;)
                end
                local.get 10
                local.get 13
                i32.const 1053920
                call $_ZN4core5slice5index22slice_index_order_fail17hf860bb9cafe28110E
                unreachable
              end
              local.get 13
              local.get 4
              i32.const 1053920
              call $_ZN4core5slice5index24slice_end_index_len_fail17h828a72b1d1bef5dcE
              unreachable
            end
            local.get 13
            local.set 10
            local.get 12
            local.set 1
            local.get 12
            local.get 8
            i32.ne
            br_if 0 (;@3;)
          end
        end
        local.get 6
        i32.eqz
        br_if 0 (;@1;)
        local.get 5
        local.get 6
        i32.add
        local.set 11
        local.get 0
        i32.const 65535
        i32.and
        local.set 1
        i32.const 1
        local.set 7
        loop ;; label = @2
          local.get 5
          i32.const 1
          i32.add
          local.set 10
          block ;; label = @3
            block ;; label = @4
              local.get 5
              i32.load8_u
              local.tee 2
              i32.extend8_s
              local.tee 13
              i32.const 0
              i32.lt_s
              br_if 0 (;@4;)
              local.get 10
              local.set 5
              br 1 (;@3;)
            end
            block ;; label = @4
              local.get 10
              local.get 11
              i32.eq
              br_if 0 (;@4;)
              local.get 13
              i32.const 127
              i32.and
              i32.const 8
              i32.shl
              local.get 5
              i32.load8_u offset=1
              i32.or
              local.set 2
              local.get 5
              i32.const 2
              i32.add
              local.set 5
              br 1 (;@3;)
            end
            i32.const 1053904
            call $_ZN4core6option13unwrap_failed17hd879e969cd9bde53E
            unreachable
          end
          local.get 1
          local.get 2
          i32.sub
          local.tee 1
          i32.const 0
          i32.lt_s
          br_if 1 (;@1;)
          local.get 7
          i32.const 1
          i32.xor
          local.set 7
          local.get 5
          local.get 11
          i32.ne
          br_if 0 (;@2;)
        end
      end
      local.get 7
      i32.const 1
      i32.and
    )
    (func $_ZN4core3fmt3num52_$LT$impl$u20$core..fmt..LowerHex$u20$for$u20$i8$GT$3fmt17he6877f4a3b3ac905E (;191;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32)
      global.get $__stack_pointer
      i32.const 128
      i32.sub
      local.tee 2
      global.set $__stack_pointer
      local.get 0
      i32.load8_u
      local.set 3
      i32.const 0
      local.set 0
      loop ;; label = @1
        local.get 2
        local.get 0
        i32.add
        i32.const 127
        i32.add
        local.get 3
        i32.const 15
        i32.and
        local.tee 4
        i32.const 48
        i32.or
        local.get 4
        i32.const 87
        i32.add
        local.get 4
        i32.const 10
        i32.lt_u
        select
        i32.store8
        local.get 0
        i32.const -1
        i32.add
        local.set 0
        local.get 3
        i32.const 255
        i32.and
        local.tee 4
        i32.const 4
        i32.shr_u
        local.set 3
        local.get 4
        i32.const 16
        i32.ge_u
        br_if 0 (;@1;)
      end
      block ;; label = @1
        local.get 0
        i32.const 128
        i32.add
        local.tee 3
        i32.const 129
        i32.lt_u
        br_if 0 (;@1;)
        local.get 3
        i32.const 128
        i32.const 1052908
        call $_ZN4core5slice5index26slice_start_index_len_fail17h0fcaa929c46d2711E
        unreachable
      end
      local.get 1
      i32.const 1
      i32.const 1052924
      i32.const 2
      local.get 2
      local.get 0
      i32.add
      i32.const 128
      i32.add
      i32.const 0
      local.get 0
      i32.sub
      call $_ZN4core3fmt9Formatter12pad_integral17h2d614e17aad082d7E
      local.set 0
      local.get 2
      i32.const 128
      i32.add
      global.set $__stack_pointer
      local.get 0
    )
    (func $_ZN4core9panicking11panic_const24panic_const_add_overflow17h3b4e4f99c6b52bd0E (;192;) (type 9) (param i32)
      (local i32)
      global.get $__stack_pointer
      i32.const 32
      i32.sub
      local.tee 1
      global.set $__stack_pointer
      local.get 1
      i32.const 0
      i32.store offset=24
      local.get 1
      i32.const 1
      i32.store offset=12
      local.get 1
      i32.const 1052396
      i32.store offset=8
      local.get 1
      i64.const 4
      i64.store offset=16 align=4
      local.get 1
      i32.const 8
      i32.add
      local.get 0
      call $_ZN4core9panicking9panic_fmt17h3f9bc37e14968102E
      unreachable
    )
    (func $_ZN4core3fmt3num3imp7fmt_u6417h79c8ebe903dabc4fE (;193;) (type 16) (param i64 i32 i32) (result i32)
      (local i32 i32 i64 i32 i32 i32)
      global.get $__stack_pointer
      i32.const 48
      i32.sub
      local.tee 3
      global.set $__stack_pointer
      i32.const 39
      local.set 4
      block ;; label = @1
        block ;; label = @2
          local.get 0
          i64.const 10000
          i64.ge_u
          br_if 0 (;@2;)
          local.get 0
          local.set 5
          br 1 (;@1;)
        end
        i32.const 39
        local.set 4
        loop ;; label = @2
          local.get 3
          i32.const 9
          i32.add
          local.get 4
          i32.add
          local.tee 6
          i32.const -4
          i32.add
          local.get 0
          local.get 0
          i64.const 10000
          i64.div_u
          local.tee 5
          i64.const 10000
          i64.mul
          i64.sub
          i32.wrap_i64
          local.tee 7
          i32.const 65535
          i32.and
          i32.const 100
          i32.div_u
          local.tee 8
          i32.const 1
          i32.shl
          i32.const 1052926
          i32.add
          i32.load16_u align=1
          i32.store16 align=1
          local.get 6
          i32.const -2
          i32.add
          local.get 7
          local.get 8
          i32.const 100
          i32.mul
          i32.sub
          i32.const 65535
          i32.and
          i32.const 1
          i32.shl
          i32.const 1052926
          i32.add
          i32.load16_u align=1
          i32.store16 align=1
          local.get 4
          i32.const -4
          i32.add
          local.set 4
          local.get 0
          i64.const 99999999
          i64.gt_u
          local.set 6
          local.get 5
          local.set 0
          local.get 6
          br_if 0 (;@2;)
        end
      end
      block ;; label = @1
        local.get 5
        i32.wrap_i64
        local.tee 6
        i32.const 99
        i32.le_u
        br_if 0 (;@1;)
        local.get 3
        i32.const 9
        i32.add
        local.get 4
        i32.const -2
        i32.add
        local.tee 4
        i32.add
        local.get 5
        i32.wrap_i64
        local.tee 6
        local.get 6
        i32.const 65535
        i32.and
        i32.const 100
        i32.div_u
        local.tee 6
        i32.const 100
        i32.mul
        i32.sub
        i32.const 65535
        i32.and
        i32.const 1
        i32.shl
        i32.const 1052926
        i32.add
        i32.load16_u align=1
        i32.store16 align=1
      end
      block ;; label = @1
        block ;; label = @2
          local.get 6
          i32.const 10
          i32.lt_u
          br_if 0 (;@2;)
          local.get 3
          i32.const 9
          i32.add
          local.get 4
          i32.const -2
          i32.add
          local.tee 4
          i32.add
          local.get 6
          i32.const 1
          i32.shl
          i32.const 1052926
          i32.add
          i32.load16_u align=1
          i32.store16 align=1
          br 1 (;@1;)
        end
        local.get 3
        i32.const 9
        i32.add
        local.get 4
        i32.const -1
        i32.add
        local.tee 4
        i32.add
        local.get 6
        i32.const 48
        i32.or
        i32.store8
      end
      local.get 2
      local.get 1
      i32.const 1
      i32.const 0
      local.get 3
      i32.const 9
      i32.add
      local.get 4
      i32.add
      i32.const 39
      local.get 4
      i32.sub
      call $_ZN4core3fmt9Formatter12pad_integral17h2d614e17aad082d7E
      local.set 4
      local.get 3
      i32.const 48
      i32.add
      global.set $__stack_pointer
      local.get 4
    )
    (func $_ZN4core3fmt3num53_$LT$impl$u20$core..fmt..UpperHex$u20$for$u20$i32$GT$3fmt17hde5ba4f379c8f81cE (;194;) (type 2) (param i32 i32) (result i32)
      (local i32 i32 i32)
      global.get $__stack_pointer
      i32.const 128
      i32.sub
      local.tee 2
      global.set $__stack_pointer
      local.get 0
      i32.load
      local.set 0
      i32.const 0
      local.set 3
      loop ;; label = @1
        local.get 2
        local.get 3
        i32.add
        i32.const 127
        i32.add
        local.get 0
        i32.const 15
        i32.and
        local.tee 4
        i32.const 48
        i32.or
        local.get 4
        i32.const 55
        i32.add
        local.get 4
        i32.const 10
        i32.lt_u
        select
        i32.store8
        local.get 3
        i32.const -1
        i32.add
        local.set 3
        local.get 0
        i32.const 16
        i32.lt_u
        local.set 4
        local.get 0
        i32.const 4
        i32.shr_u
        local.set 0
        local.get 4
        i32.eqz
        br_if 0 (;@1;)
      end
      block ;; label = @1
        local.get 3
        i32.const 128
        i32.add
        local.tee 0
        i32.const 129
        i32.lt_u
        br_if 0 (;@1;)
        local.get 0
        i32.const 128
        i32.const 1052908
        call $_ZN4core5slice5index26slice_start_index_len_fail17h0fcaa929c46d2711E
        unreachable
      end
      local.get 1
      i32.const 1
      i32.const 1052924
      i32.const 2
      local.get 2
      local.get 3
      i32.add
      i32.const 128
      i32.add
      i32.const 0
      local.get 3
      i32.sub
      call $_ZN4core3fmt9Formatter12pad_integral17h2d614e17aad082d7E
      local.set 0
      local.get 2
      i32.const 128
      i32.add
      global.set $__stack_pointer
      local.get 0
    )
    (func $_ZN17compiler_builtins3mem6memcpy17hd701c6b4e0a5ee15E (;195;) (type 1) (param i32 i32 i32) (result i32)
      (local i32 i32 i32 i32 i32 i32 i32 i32)
      block ;; label = @1
        block ;; label = @2
          local.get 2
          i32.const 16
          i32.ge_u
          br_if 0 (;@2;)
          local.get 0
          local.set 3
          br 1 (;@1;)
        end
        local.get 0
        i32.const 0
        local.get 0
        i32.sub
        i32.const 3
        i32.and
        local.tee 4
        i32.add
        local.set 5
        block ;; label = @2
          local.get 4
          i32.eqz
          br_if 0 (;@2;)
          local.get 0
          local.set 3
          local.get 1
          local.set 6
          loop ;; label = @3
            local.get 3
            local.get 6
            i32.load8_u
            i32.store8
            local.get 6
            i32.const 1
            i32.add
            local.set 6
            local.get 3
            i32.const 1
            i32.add
            local.tee 3
            local.get 5
            i32.lt_u
            br_if 0 (;@3;)
          end
        end
        local.get 5
        local.get 2
        local.get 4
        i32.sub
        local.tee 7
        i32.const -4
        i32.and
        local.tee 8
        i32.add
        local.set 3
        block ;; label = @2
          block ;; label = @3
            local.get 1
            local.get 4
            i32.add
            local.tee 9
            i32.const 3
            i32.and
            i32.eqz
            br_if 0 (;@3;)
            local.get 8
            i32.const 1
            i32.lt_s
            br_if 1 (;@2;)
            local.get 9
            i32.const 3
            i32.shl
            local.tee 6
            i32.const 24
            i32.and
            local.set 2
            local.get 9
            i32.const -4
            i32.and
            local.tee 10
            i32.const 4
            i32.add
            local.set 1
            i32.const 0
            local.get 6
            i32.sub
            i32.const 24
            i32.and
            local.set 4
            local.get 10
            i32.load
            local.set 6
            loop ;; label = @4
              local.get 5
              local.get 6
              local.get 2
              i32.shr_u
              local.get 1
              i32.load
              local.tee 6
              local.get 4
              i32.shl
              i32.or
              i32.store
              local.get 1
              i32.const 4
              i32.add
              local.set 1
              local.get 5
              i32.const 4
              i32.add
              local.tee 5
              local.get 3
              i32.lt_u
              br_if 0 (;@4;)
              br 2 (;@2;)
            end
          end
          local.get 8
          i32.const 1
          i32.lt_s
          br_if 0 (;@2;)
          local.get 9
          local.set 1
          loop ;; label = @3
            local.get 5
            local.get 1
            i32.load
            i32.store
            local.get 1
            i32.const 4
            i32.add
            local.set 1
            local.get 5
            i32.const 4
            i32.add
            local.tee 5
            local.get 3
            i32.lt_u
            br_if 0 (;@3;)
          end
        end
        local.get 7
        i32.const 3
        i32.and
        local.set 2
        local.get 9
        local.get 8
        i32.add
        local.set 1
      end
      block ;; label = @1
        local.get 2
        i32.eqz
        br_if 0 (;@1;)
        local.get 3
        local.get 2
        i32.add
        local.set 5
        loop ;; label = @2
          local.get 3
          local.get 1
          i32.load8_u
          i32.store8
          local.get 1
          i32.const 1
          i32.add
          local.set 1
          local.get 3
          i32.const 1
          i32.add
          local.tee 3
          local.get 5
          i32.lt_u
          br_if 0 (;@2;)
        end
      end
      local.get 0
    )
    (func $_ZN17compiler_builtins3mem6memset17hfd222261076a5e29E (;196;) (type 1) (param i32 i32 i32) (result i32)
      (local i32 i32 i32)
      block ;; label = @1
        block ;; label = @2
          local.get 2
          i32.const 16
          i32.ge_u
          br_if 0 (;@2;)
          local.get 0
          local.set 3
          br 1 (;@1;)
        end
        local.get 0
        i32.const 0
        local.get 0
        i32.sub
        i32.const 3
        i32.and
        local.tee 4
        i32.add
        local.set 5
        block ;; label = @2
          local.get 4
          i32.eqz
          br_if 0 (;@2;)
          local.get 0
          local.set 3
          loop ;; label = @3
            local.get 3
            local.get 1
            i32.store8
            local.get 3
            i32.const 1
            i32.add
            local.tee 3
            local.get 5
            i32.lt_u
            br_if 0 (;@3;)
          end
        end
        local.get 5
        local.get 2
        local.get 4
        i32.sub
        local.tee 4
        i32.const -4
        i32.and
        local.tee 2
        i32.add
        local.set 3
        block ;; label = @2
          local.get 2
          i32.const 1
          i32.lt_s
          br_if 0 (;@2;)
          local.get 1
          i32.const 255
          i32.and
          i32.const 16843009
          i32.mul
          local.set 2
          loop ;; label = @3
            local.get 5
            local.get 2
            i32.store
            local.get 5
            i32.const 4
            i32.add
            local.tee 5
            local.get 3
            i32.lt_u
            br_if 0 (;@3;)
          end
        end
        local.get 4
        i32.const 3
        i32.and
        local.set 2
      end
      block ;; label = @1
        local.get 2
        i32.eqz
        br_if 0 (;@1;)
        local.get 3
        local.get 2
        i32.add
        local.set 5
        loop ;; label = @2
          local.get 3
          local.get 1
          i32.store8
          local.get 3
          i32.const 1
          i32.add
          local.tee 3
          local.get 5
          i32.lt_u
          br_if 0 (;@2;)
        end
      end
      local.get 0
    )
    (func $memset (;197;) (type 1) (param i32 i32 i32) (result i32)
      local.get 0
      local.get 1
      local.get 2
      call $_ZN17compiler_builtins3mem6memset17hfd222261076a5e29E
    )
    (func $memcpy (;198;) (type 1) (param i32 i32 i32) (result i32)
      local.get 0
      local.get 1
      local.get 2
      call $_ZN17compiler_builtins3mem6memcpy17hd701c6b4e0a5ee15E
    )
    (table (;0;) 48 48 funcref)
    (memory (;0;) 17)
    (global $__stack_pointer (;0;) (mut i32) i32.const 1048576)
    (global (;1;) i32 i32.const 1056833)
    (global (;2;) i32 i32.const 1056848)
    (export "memory" (memory 0))
    (export "test:guest/run#start" (func $test:guest/run#start))
    (export "cabi_realloc" (func $cabi_realloc))
    (export "__data_end" (global 1))
    (export "__heap_base" (global 2))
    (elem (;0;) (i32.const 1) func $_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17hce44cecf0f119939E $_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h4c1b6f1d0f27a9faE $_ZN66_$LT$core..option..Option$LT$T$GT$$u20$as$u20$core..fmt..Debug$GT$3fmt17hd92efa95cfc864f3E $_ZN70_$LT$core..result..Result$LT$T$C$E$GT$$u20$as$u20$core..fmt..Debug$GT$3fmt17h2148cc03c39dba01E $_ZN70_$LT$core..result..Result$LT$T$C$E$GT$$u20$as$u20$core..fmt..Debug$GT$3fmt17h65c3f310f2d696a0E $_ZN70_$LT$core..result..Result$LT$T$C$E$GT$$u20$as$u20$core..fmt..Debug$GT$3fmt17h691212d9090277eaE $_ZN70_$LT$core..result..Result$LT$T$C$E$GT$$u20$as$u20$core..fmt..Debug$GT$3fmt17he46203b2fce9b184E $_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h6ca084259debba81E $_ZN4core3ptr49drop_in_place$LT$alloc..string..FromUtf8Error$GT$17hec3a267b9bd19dd8E $_ZN65_$LT$alloc..string..FromUtf8Error$u20$as$u20$core..fmt..Debug$GT$3fmt17hc6b77939b21326f3E $_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17ha6c7b58c4bb7816eE $_ZN4core3fmt3num52_$LT$impl$u20$core..fmt..Debug$u20$for$u20$usize$GT$3fmt17h6d706772c335a50cE $_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17he31bc5b485e387bbE $_ZN4core3ptr46drop_in_place$LT$alloc..vec..Vec$LT$u8$GT$$GT$17h57b6af354035ef20E $_ZN65_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$core..fmt..Debug$GT$3fmt17h940d068692ae5b09E $_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h9c2833f5366bf8eaE $_ZN4core3fmt3num3imp52_$LT$impl$u20$core..fmt..Display$u20$for$u20$u32$GT$3fmt17h0baf20a96941cbedE $_ZN3std5alloc24default_alloc_error_hook17h6f1c3591a9b459d3E $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17ha5fb9ff401ff50deE $_ZN58_$LT$alloc..string..String$u20$as$u20$core..fmt..Write$GT$9write_str17h8c93e1ada73d40b2E $_ZN58_$LT$alloc..string..String$u20$as$u20$core..fmt..Write$GT$10write_char17h4c0947de865746f3E $_ZN4core3fmt5Write9write_fmt17h8e7667e000b2e00bE $_ZN36_$LT$T$u20$as$u20$core..any..Any$GT$7type_id17hf8c8bae692844a7bE $_ZN36_$LT$T$u20$as$u20$core..any..Any$GT$7type_id17hd1cbeb6a70b6771bE $_ZN92_$LT$std..panicking..begin_panic_handler..StaticStrPayload$u20$as$u20$core..fmt..Display$GT$3fmt17h38570d007b206003E $_ZN99_$LT$std..panicking..begin_panic_handler..StaticStrPayload$u20$as$u20$core..panic..PanicPayload$GT$8take_box17hde89bca33496dd2eE $_ZN99_$LT$std..panicking..begin_panic_handler..StaticStrPayload$u20$as$u20$core..panic..PanicPayload$GT$3get17hf54ac27f28f24a8bE $_ZN99_$LT$std..panicking..begin_panic_handler..StaticStrPayload$u20$as$u20$core..panic..PanicPayload$GT$6as_str17hf4f8f0e82151ed5dE $_ZN4core3ptr77drop_in_place$LT$std..panicking..begin_panic_handler..FormatStringPayload$GT$17h32febad92427b942E $_ZN95_$LT$std..panicking..begin_panic_handler..FormatStringPayload$u20$as$u20$core..fmt..Display$GT$3fmt17h3a0d742924f1b3adE $_ZN102_$LT$std..panicking..begin_panic_handler..FormatStringPayload$u20$as$u20$core..panic..PanicPayload$GT$8take_box17hb2610ff035054299E $_ZN102_$LT$std..panicking..begin_panic_handler..FormatStringPayload$u20$as$u20$core..panic..PanicPayload$GT$3get17h8f1a16407c949c6bE $_ZN4core5panic12PanicPayload6as_str17haccd18006a9ab86bE $_ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17hafd3ba595a153b9dE $#func142<_ZN58_$LT$alloc..string..String$u20$as$u20$core..fmt..Write$GT$9write_str17h8c93e1ada73d40b2E> $#func143<_ZN58_$LT$alloc..string..String$u20$as$u20$core..fmt..Write$GT$10write_char17h4c0947de865746f3E> $_ZN4core3fmt5Write9write_fmt17h39cd1de01decbf37E $_ZN53_$LT$core..fmt..Error$u20$as$u20$core..fmt..Debug$GT$3fmt17hd605511bb8a4d61cE $_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h2edac6506478e24dE $_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17hc6b06257c279c844E $_ZN4core3fmt3num53_$LT$impl$u20$core..fmt..LowerHex$u20$for$u20$i32$GT$3fmt17hc7fe287f99d5f9c4E $_ZN59_$LT$core..fmt..Arguments$u20$as$u20$core..fmt..Display$GT$3fmt17h7bc36dab8af3ab23E $_ZN71_$LT$core..ops..range..Range$LT$Idx$GT$$u20$as$u20$core..fmt..Debug$GT$3fmt17h02d6bc52556291bcE $_ZN41_$LT$char$u20$as$u20$core..fmt..Debug$GT$3fmt17hc694b486c95ba763E $_ZN68_$LT$core..fmt..builders..PadAdapter$u20$as$u20$core..fmt..Write$GT$9write_str17h41f43023c6a54529E $_ZN68_$LT$core..fmt..builders..PadAdapter$u20$as$u20$core..fmt..Write$GT$10write_char17hd93b2338104cb451E $_ZN4core3fmt5Write9write_fmt17h41a97b915efb10b4E)
    (data $.rodata (;0;) (i32.const 1048576) "\00\00\00\00\00\00\00\00is_aligned_to: align is not a power-of-two\00\00\08\00\10\00*\00\00\00/rustc/6be96e3865c4e59028fd50396f7a46c3498ce91d/library/core/src/ub_checks.rs\00\00\00<\00\10\00M\00\00\00|\00\00\006\00\00\00unsafe precondition(s) violated: slice::from_raw_parts requires the pointer to be aligned and non-null, and the total size of the slice not to exceed `isize::MAX`\00\00\00\00\00\00\00\00\00\00/rustc/6be96e3865c4e59028fd50396f7a46c3498ce91d/library/core/src/ptr/const_ptr.rs\00\00\00H\01\10\00Q\00\00\00\19\06\00\00\0d\00\00\00is_nonoverlapping: `size_of::<T>() * count` overflows a usize\00\00\00\00\00\00\00\00\00\00\00/rustc/6be96e3865c4e59028fd50396f7a46c3498ce91d/library/core/src/alloc/layout.rs\f4\01\10\00P\00\00\00\c3\01\00\00)\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00invalid enum discriminant\00\00\00d\02\10\00\19\00\00\00/Users/clunt/.cargo/registry/src/index.crates.io-6f17d22bba15001f/wit-bindgen-0.16.0/src/lib.rs\00\88\02\10\00_\00\00\00\94\00\00\00\0d\00\00\00is_aligned_to: align is not a power-of-two\00\00\f8\02\10\00*\00\00\00unsafe precondition(s) violated: ptr::read_volatile requires that the pointer argument is aligned and non-null\00\00\00\00\00\00\00\00\00\00/rustc/6be96e3865c4e59028fd50396f7a46c3498ce91d/library/core/src/ptr/const_ptr.rs\00\00\00\a4\03\10\00Q\00\00\00\19\06\00\00\0d\00\00\00\00\00\00\00\00\00\00\00unsafe precondition(s) violated: NonNull::new_unchecked requires that the pointer is non-null\00\00\00\00\00\00\00\04\00\00\00\04\00\00\00\01\00\00\00OkErr\00\00\00\00\00\00\00\04\00\00\00\04\00\00\00\02\00\00\00None\00\00\00\00\04\00\00\00\04\00\00\00\01\00\00\00Some`result-option` result should Some(\22OK\22) is \b0\04\10\00,\00\00\00`result-option` result should None is \00\00\e4\04\10\00&\00\00\00`result-result` result should Ok(\22OK\22) is \00\00\14\05\10\00*\00\00\00`result-result` result should Err(\22Err\22) is H\05\10\00,\00\00\00`result-result-ok` result should Ok(\22OK\22) is \00\00\00|\05\10\00-\00\00\00`result-result-ok` result should Err(()) is \b4\05\10\00,\00\00\00`result-result-err` result should Ok(()) is \e8\05\10\00,\00\00\00`result-result-err` result should Err(\22Err\22) is \1c\06\10\000\00\00\00`result-result-none` result should Ok(()) is \00\00\00T\06\10\00-\00\00\00`result-result-none` result should Err(()) is \00\00\8c\06\10\00.\00\00\00is_aligned_to: align is not a power-of-two\00\00\c4\06\10\00*\00\00\00unsafe precondition(s) violated: ptr::copy_nonoverlapping requires that both pointer arguments are aligned and non-null and the specified memory ranges do not overlap\00\00\00\00\00\00\00\00\00\00/rustc/6be96e3865c4e59028fd50396f7a46c3498ce91d/library/core/src/ptr/const_ptr.rs\00\00\00\a8\07\10\00Q\00\00\00\19\06\00\00\0d\00\00\00()unsafe precondition(s) violated: usize::unchecked_mul cannot overflowsrc/lib.rs\00\00\00S\08\10\00\0a\00\00\00\01\00\00\00\01\00\00\00\00\00\00\00\04\00\00\00\04\00\00\00\08\00\00\00unsafe precondition(s) violated: usize::unchecked_mul cannot overflowis_aligned_to: align is not a power-of-two\00\c5\08\10\00*\00\00\00unsafe precondition(s) violated: ptr::read_volatile requires that the pointer argument is aligned and non-null\00\00\00\00\00\00\00\00\00\00/rustc/6be96e3865c4e59028fd50396f7a46c3498ce91d/library/core/src/ptr/const_ptr.rs\00\00\00p\09\10\00Q\00\00\00\19\06\00\00\0d\00\00\00/rustc/6be96e3865c4e59028fd50396f7a46c3498ce91d/library/core/src/ub_checks.rs\00\00\00\d4\09\10\00M\00\00\00|\00\00\006\00\00\00unsafe precondition(s) violated: slice::from_raw_parts requires the pointer to be aligned and non-null, and the total size of the slice not to exceed `isize::MAX`\00\00\09\00\00\00\14\00\00\00\04\00\00\00\0a\00\00\00called `Result::unwrap()` on an `Err` value\00\00\00\00\00\04\00\00\00\04\00\00\00\0b\00\00\00\00\00\00\00\04\00\00\00\04\00\00\00\0c\00\00\00\00\00\00\00\04\00\00\00\04\00\00\00\0d\00\00\00Utf8Errorvalid_up_toerror_len\00\00\00\0e\00\00\00\0c\00\00\00\04\00\00\00\0f\00\00\00\00\00\00\00\04\00\00\00\04\00\00\00\10\00\00\00FromUtf8ErrorbyteserrorNoneSome\00\00\00\00\00non-zero old_len requires non-zero new_len!\00\a8\0b\10\00+\00\00\00/Users/clunt/.cargo/registry/src/index.crates.io-6f17d22bba15001f/wit-bindgen-0.16.0/src/lib.rs\00\dc\0b\10\00_\00\00\00K\00\00\00\0d\00\00\00\dc\0b\10\00_\00\00\00\8c\00\00\00&\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\0c\00\00\00\04\00\00\00\14\00\00\00\15\00\00\00\16\00\00\00/rust/deps/dlmalloc-0.2.6/src/dlmalloc.rsassertion failed: psize >= size + min_overhead\00|\0c\10\00)\00\00\00\a8\04\00\00\09\00\00\00assertion failed: psize <= size + max_overhead\00\00|\0c\10\00)\00\00\00\ae\04\00\00\0d\00\00\00memory allocation of  bytes failed\00\00$\0d\10\00\15\00\00\009\0d\10\00\0d\00\00\00library/std/src/alloc.rsX\0d\10\00\18\00\00\00d\01\00\00\09\00\00\00\13\00\00\00\0c\00\00\00\04\00\00\00\17\00\00\00\00\00\00\00\08\00\00\00\04\00\00\00\18\00\00\00\00\00\00\00\08\00\00\00\04\00\00\00\19\00\00\00\1a\00\00\00\1b\00\00\00\1c\00\00\00\1d\00\00\00\10\00\00\00\04\00\00\00\1e\00\00\00\1f\00\00\00 \00\00\00!\00\00\00Error\00\00\00\22\00\00\00\0c\00\00\00\04\00\00\00#\00\00\00$\00\00\00%\00\00\00capacity overflow\00\00\00\f8\0d\10\00\11\00\00\00library/alloc/src/raw_vec.rs\14\0e\10\00\1c\00\00\00\19\00\00\00\05\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00&\00\00\00a formatting trait implementation returned an error when the underlying stream did notlibrary/alloc/src/fmt.rs\00\00\a6\0e\10\00\18\00\00\00\7f\02\00\00\0e\00\00\00attempt to add with overflow\d0\0e\10\00\1c\00\00\00)..0123456789abcdef[called `Option::unwrap()` on a `None` valuelibrary/core/src/panicking.rs3\0f\10\00\1d\00\00\00\dd\00\00\00\05\00\00\00index out of bounds: the len is  but the index is \00\00`\0f\10\00 \00\00\00\80\0f\10\00\12\00\00\00misaligned pointer dereference: address must be a multiple of  but is \00\00\a4\0f\10\00>\00\00\00\e2\0f\10\00\08\00\00\00==!=matchesassertion `left  right` failed\0a  left: \0a right: \00\07\10\10\00\10\00\00\00\17\10\10\00\17\00\00\00.\10\10\00\09\00\00\00 right` failed: \0a  left: \00\00\00\07\10\10\00\10\00\00\00P\10\10\00\10\00\00\00`\10\10\00\09\00\00\00.\10\10\00\09\00\00\00: \00\00\01\00\00\00\00\00\00\00\8c\10\10\00\02\00\00\00\00\00\00\00\0c\00\00\00\04\00\00\00-\00\00\00.\00\00\00/\00\00\00     { ,  {\0a,\0a} }((\0a,\0a]library/core/src/fmt/num.rs\00\00\cf\10\10\00\1b\00\00\00i\00\00\00\17\00\00\000x00010203040506070809101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899library/core/src/fmt/mod.rs\00\00\00\c6\11\10\00\1b\00\00\00\8d\09\00\00&\00\00\00\c6\11\10\00\1b\00\00\00\96\09\00\00\1a\00\00\00range start index  out of range for slice of length \04\12\10\00\12\00\00\00\16\12\10\00\22\00\00\00range end index H\12\10\00\10\00\00\00\16\12\10\00\22\00\00\00slice index starts at  but ends at \00h\12\10\00\16\00\00\00~\12\10\00\0d\00\00\00\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\03\03\03\03\03\03\03\03\03\03\03\03\03\03\03\03\04\04\04\04\04\00\00\00\00\00\00\00\00\00\00\00[...]begin <= end ( <= ) when slicing ``\a1\13\10\00\0e\00\00\00\af\13\10\00\04\00\00\00\b3\13\10\00\10\00\00\00\c3\13\10\00\01\00\00\00byte index  is not a char boundary; it is inside  (bytes ) of `\00\e4\13\10\00\0b\00\00\00\ef\13\10\00&\00\00\00\15\14\10\00\08\00\00\00\1d\14\10\00\06\00\00\00\c3\13\10\00\01\00\00\00 is out of bounds of `\00\00\e4\13\10\00\0b\00\00\00L\14\10\00\16\00\00\00\c3\13\10\00\01\00\00\00library/core/src/str/mod.rs\00|\14\10\00\1b\00\00\00\05\01\00\00,\00\00\00library/core/src/unicode/printable.rs\00\00\00\a8\14\10\00%\00\00\00\1a\00\00\006\00\00\00\a8\14\10\00%\00\00\00\0a\00\00\00+\00\00\00\00\06\01\01\03\01\04\02\05\07\07\02\08\08\09\02\0a\05\0b\02\0e\04\10\01\11\02\12\05\13\11\14\01\15\02\17\02\19\0d\1c\05\1d\08\1f\01$\01j\04k\02\af\03\b1\02\bc\02\cf\02\d1\02\d4\0c\d5\09\d6\02\d7\02\da\01\e0\05\e1\02\e7\04\e8\02\ee \f0\04\f8\02\fa\03\fb\01\0c';>NO\8f\9e\9e\9f{\8b\93\96\a2\b2\ba\86\b1\06\07\096=>V\f3\d0\d1\04\14\1867VW\7f\aa\ae\af\bd5\e0\12\87\89\8e\9e\04\0d\0e\11\12)14:EFIJNOde\5c\b6\b7\1b\1c\07\08\0a\0b\14\1769:\a8\a9\d8\d9\097\90\91\a8\07\0a;>fi\8f\92\11o_\bf\ee\efZb\f4\fc\ffST\9a\9b./'(U\9d\a0\a1\a3\a4\a7\a8\ad\ba\bc\c4\06\0b\0c\15\1d:?EQ\a6\a7\cc\cd\a0\07\19\1a\22%>?\e7\ec\ef\ff\c5\c6\04 #%&(38:HJLPSUVXZ\5c^`cefksx}\7f\8a\a4\aa\af\b0\c0\d0\ae\afno\be\93^\22{\05\03\04-\03f\03\01/.\80\82\1d\031\0f\1c\04$\09\1e\05+\05D\04\0e*\80\aa\06$\04$\04(\084\0bNC\817\09\16\0a\08\18;E9\03c\08\090\16\05!\03\1b\05\01@8\04K\05/\04\0a\07\09\07@ '\04\0c\096\03:\05\1a\07\04\0c\07PI73\0d3\07.\08\0a\81&RK+\08*\16\1a&\1c\14\17\09N\04$\09D\0d\19\07\0a\06H\08'\09u\0bB>*\06;\05\0a\06Q\06\01\05\10\03\05\80\8bb\1eH\08\0a\80\a6^\22E\0b\0a\06\0d\13:\06\0a6,\04\17\80\b9<dS\0cH\09\0aFE\1bH\08S\0dI\07\0a\80\f6F\0a\1d\03GI7\03\0e\08\0a\069\07\0a\816\19\07;\03\1cV\01\0f2\0d\83\9bfu\0b\80\c4\8aLc\0d\840\10\16\8f\aa\82G\a1\b9\829\07*\04\5c\06&\0aF\0a(\05\13\82\b0[eK\049\07\11@\05\0b\02\0e\97\f8\08\84\d6*\09\a2\e7\813\0f\01\1d\06\0e\04\08\81\8c\89\04k\05\0d\03\09\07\10\92`G\09t<\80\f6\0as\08p\15Fz\14\0c\14\0cW\09\19\80\87\81G\03\85B\0f\15\84P\1f\06\06\80\d5+\05>!\01p-\03\1a\04\02\81@\1f\11:\05\01\81\d0*\82\e6\80\f7)L\04\0a\04\02\83\11DL=\80\c2<\06\01\04U\05\1b4\02\81\0e,\04d\0cV\0a\80\ae8\1d\0d,\04\09\07\02\0e\06\80\9a\83\d8\04\11\03\0d\03w\04_\06\0c\04\01\0f\0c\048\08\0a\06(\08\22N\81T\0c\1d\03\09\076\08\0e\04\09\07\09\07\80\cb%\0a\84\06\00\01\03\05\05\06\06\02\07\06\08\07\09\11\0a\1c\0b\19\0c\1a\0d\10\0e\0c\0f\04\10\03\12\12\13\09\16\01\17\04\18\01\19\03\1a\07\1b\01\1c\02\1f\16 \03+\03-\0b.\010\041\022\01\a7\02\a9\02\aa\04\ab\08\fa\02\fb\05\fd\02\fe\03\ff\09\adxy\8b\8d\a20WX\8b\8c\90\1c\dd\0e\0fKL\fb\fc./?\5c]_\e2\84\8d\8e\91\92\a9\b1\ba\bb\c5\c6\c9\ca\de\e4\e5\ff\00\04\11\12)147:;=IJ]\84\8e\92\a9\b1\b4\ba\bb\c6\ca\ce\cf\e4\e5\00\04\0d\0e\11\12)14:;EFIJ^de\84\91\9b\9d\c9\ce\cf\0d\11):;EIW[\5c^_de\8d\91\a9\b4\ba\bb\c5\c9\df\e4\e5\f0\0d\11EIde\80\84\b2\bc\be\bf\d5\d7\f0\f1\83\85\8b\a4\a6\be\bf\c5\c7\cf\da\dbH\98\bd\cd\c6\ce\cfINOWY^_\89\8e\8f\b1\b6\b7\bf\c1\c6\c7\d7\11\16\17[\5c\f6\f7\fe\ff\80mq\de\df\0e\1fno\1c\1d_}~\ae\af\7f\bb\bc\16\17\1e\1fFGNOXZ\5c^~\7f\b5\c5\d4\d5\dc\f0\f1\f5rs\8ftu\96&./\a7\af\b7\bf\c7\cf\d7\df\9a\00@\97\980\8f\1f\d2\d4\ce\ffNOZ[\07\08\0f\10'/\ee\efno7=?BE\90\91Sgu\c8\c9\d0\d1\d8\d9\e7\fe\ff\00 _\22\82\df\04\82D\08\1b\04\06\11\81\ac\0e\80\ab\05\1f\09\81\1b\03\19\08\01\04/\044\04\07\03\01\07\06\07\11\0aP\0f\12\07U\07\03\04\1c\0a\09\03\08\03\07\03\02\03\03\03\0c\04\05\03\0b\06\01\0e\15\05N\07\1b\07W\07\02\06\17\0cP\04C\03-\03\01\04\11\06\0f\0c:\04\1d%_ m\04j%\80\c8\05\82\b0\03\1a\06\82\fd\03Y\07\16\09\18\09\14\0c\14\0cj\06\0a\06\1a\06Y\07+\05F\0a,\04\0c\04\01\031\0b,\04\1a\06\0b\03\80\ac\06\0a\06/1M\03\80\a4\08<\03\0f\03<\078\08+\05\82\ff\11\18\08/\11-\03!\0f!\0f\80\8c\04\82\97\19\0b\15\88\94\05/\05;\07\02\0e\18\09\80\be\22t\0c\80\d6\1a\81\10\05\80\df\0b\f2\9e\037\09\81\5c\14\80\b8\08\80\cb\05\0a\18;\03\0a\068\08F\08\0c\06t\0b\1e\03Z\04Y\09\80\83\18\1c\0a\16\09L\04\80\8a\06\ab\a4\0c\17\041\a1\04\81\da&\07\0c\05\05\80\a6\10\81\f5\07\01 *\06L\04\80\8d\04\80\be\03\1b\03\0f\0dlibrary/core/src/unicode/unicode_data.rs\00k\1a\10\00(\00\00\00P\00\00\00(\00\00\00k\1a\10\00(\00\00\00\5c\00\00\00\16\00\00\00library/core/src/escape.rs\00\00\b4\1a\10\00\1a\00\00\00M\00\00\00\05\00\00\00attempt to divide by zero\00\00\00\e0\1a\10\00\19\00\00\00\00\03\00\00\83\04 \00\91\05`\00]\13\a0\00\12\17 \1f\0c `\1f\ef,\a0+*0 ,o\a6\e0,\02\a8`-\1e\fb`.\00\fe 6\9e\ff`6\fd\01\e16\01\0a!7$\0d\e17\ab\0ea9/\18\a190\1caH\f3\1e\a1L@4aP\f0j\a1QOo!R\9d\bc\a1R\00\cfaSe\d1\a1S\00\da!T\00\e0\e1U\ae\e2aW\ec\e4!Y\d0\e8\a1Y \00\eeY\f0\01\7fZ\00p\00\07\00-\01\01\01\02\01\02\01\01H\0b0\15\10\01e\07\02\06\02\02\01\04#\01\1e\1b[\0b:\09\09\01\18\04\01\09\01\03\01\05+\03<\08*\18\01 7\01\01\01\04\08\04\01\03\07\0a\02\1d\01:\01\01\01\02\04\08\01\09\01\0a\02\1a\01\02\029\01\04\02\04\02\02\03\03\01\1e\02\03\01\0b\029\01\04\05\01\02\04\01\14\02\16\06\01\01:\01\01\02\01\04\08\01\07\03\0a\02\1e\01;\01\01\01\0c\01\09\01(\01\03\017\01\01\03\05\03\01\04\07\02\0b\02\1d\01:\01\02\01\02\01\03\01\05\02\07\02\0b\02\1c\029\02\01\01\02\04\08\01\09\01\0a\02\1d\01H\01\04\01\02\03\01\01\08\01Q\01\02\07\0c\08b\01\02\09\0b\07I\02\1b\01\01\01\01\017\0e\01\05\01\02\05\0b\01$\09\01f\04\01\06\01\02\02\02\19\02\04\03\10\04\0d\01\02\02\06\01\0f\01\00\03\00\03\1d\02\1e\02\1e\02@\02\01\07\08\01\02\0b\09\01-\03\01\01u\02\22\01v\03\04\02\09\01\06\03\db\02\02\01:\01\01\07\01\01\01\01\02\08\06\0a\02\010\1f1\040\07\01\01\05\01(\09\0c\02 \04\02\02\01\038\01\01\02\03\01\01\03:\08\02\02\98\03\01\0d\01\07\04\01\06\01\03\02\c6@\00\01\c3!\00\03\8d\01` \00\06i\02\00\04\01\0a \02P\02\00\01\03\01\04\01\19\02\05\01\97\02\1a\12\0d\01&\08\19\0b.\030\01\02\04\02\02'\01C\06\02\02\02\02\0c\01\08\01/\013\01\01\03\02\02\05\02\01\01*\02\08\01\ee\01\02\01\04\01\00\01\00\10\10\10\00\02\00\01\e2\01\95\05\00\03\01\02\05\04(\03\04\01\a5\02\00\04\00\02P\03F\0b1\04{\016\0f)\01\02\02\0a\031\04\02\02\07\01=\03$\05\01\08>\01\0c\024\09\0a\04\02\01_\03\02\01\01\02\06\01\02\01\9d\01\03\08\15\029\02\01\01\01\01\16\01\0e\07\03\05\c3\08\02\03\01\01\17\01Q\01\02\06\01\01\02\01\01\02\01\02\eb\01\02\04\06\02\01\02\1b\02U\08\02\01\01\02j\01\01\01\02\06\01\01e\03\02\04\01\05\00\09\01\02\f5\01\0a\02\01\01\04\01\90\04\02\02\04\01 \0a(\06\02\04\08\01\09\06\02\03.\0d\01\02\00\07\01\06\01\01R\16\02\07\01\02\01\02z\06\03\01\01\02\01\07\01\01H\02\03\01\01\01\00\02\0b\024\05\05\01\01\01\00\01\06\0f\00\05;\07\00\01?\04Q\01\00\02\00.\02\17\00\01\01\03\04\05\08\08\02\07\1e\04\94\03\007\042\08\01\0e\01\16\05\01\0f\00\07\01\11\02\07\01\02\01\05d\01\a0\07\00\01=\04\00\04\00\07m\07\00`\80\f0\00")
    (@custom ".debug_abbrev" (after data) "\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\029\01\03\0e\00\00\03\04\01I\13m\19\03\0e\0b\0b\88\01\0f\00\00\04(\00\03\0e\1c\0f\00\00\05\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\06\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\073\01\15\13\00\00\08\0d\00I\13\88\01\0f8\0b4\19\00\00\09\19\01\16\0b\00\00\0a\0d\00\03\0eI\13\88\01\0f8\0b\00\00\0b\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\0c\19\01\00\00\0d.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05I\13\00\00\0e\05\00\02\18\03\0e:\0b;\05I\13\00\00\0f/\00I\13\03\0e\00\00\10.\01n\0e\03\0e:\0b;\05I\13<\19\00\00\11\05\00I\13\00\00\12$\00\03\0e>\0b\0b\0b\00\00\13\13\01\03\0e\0b\0b\88\01\0f\00\00\14\0f\00I\133\06\00\00\15\0f\00I\13\03\0e3\06\00\00\16\15\01I\13\00\00\17\13\00\03\0e\0b\0b\88\01\0f\00\00\18\01\01I\13\00\00\19!\00I\13\22\0d7\0b\00\00\1a$\00\03\0e\0b\0b>\0b\00\00\1b.\01\11\01\12\06@\18G\13\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03\04\01I\13m\19\03\0e\0b\0b\88\01\0f\00\00\04(\00\03\0e\1c\0f\00\00\05\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\06\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\07/\00I\13\03\0e\00\00\08.\01n\0e\03\0e:\0b;\0bI\13<\19\00\00\09\05\00I\13\00\00\0a.\01n\0e\03\0e:\0b;\05I\13<\19\00\00\0b.\01n\0e\03\0e:\0b;\05 \0b\00\00\0c\0b\01\00\00\0d\05\00\03\0e:\0b;\05I\13\00\00\0e.\01n\0e\03\0e:\0b;\05I\13 \0b\00\00\0f$\00\03\0e>\0b\0b\0b\00\00\10\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\113\01\15\13\00\00\12\0d\00I\13\88\01\0f8\0b4\19\00\00\13\19\01\16\0b\00\00\14\0d\00\03\0eI\13\88\01\0f8\0b\00\00\15\19\01\00\00\16.\01n\0e\03\0e:\0b;\05<\19\00\00\17.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0bI\13\00\00\18\05\00\02\18\03\0e:\0b;\0bI\13\00\00\19\1d\011\13U\17X\0bY\0bW\0b\00\00\1a\0b\01U\17\00\00\1b\05\00\02\181\13\00\00\1c\1d\011\13U\17X\0bY\05W\0b\00\00\1d\0b\01\11\01\12\06\00\00\1e4\00\02\181\13\00\00\1f4\00\02\18\03\0e:\0b;\0bI\13\00\00 \1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00!\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00\22\0f\00I\13\03\0e3\06\00\00#.\01G\13 \0b\00\00$\05\00\03\0e:\0b;\0bI\13\00\00%4\00\03\0e:\0b;\0bI\13\00\00&\13\01\03\0e\0b\0b\88\01\0f\00\00'\0f\00I\133\06\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03\04\01I\13m\19\03\0e\0b\0b\88\01\0f\00\00\04(\00\03\0e\1c\0f\00\00\05\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\06\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\073\01\15\13\00\00\08\0d\00I\13\88\01\0f8\0b4\19\00\00\09\19\01\16\0b\00\00\0a\0d\00\03\0eI\13\88\01\0f8\0b\00\00\0b\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\0c\19\01\00\00\0d.\01n\0e\03\0e:\0b;\05I\13<\19\00\00\0e\05\00I\13\00\00\0f/\00I\13\03\0e\00\00\10.\01n\0e\03\0e:\0b;\05I\13 \0b\00\00\11\0b\01\00\00\12\05\00\03\0e:\0b;\05I\13\00\00\13.\01n\0e\03\0e:\0b;\0bI\13 \0b\00\00\14\05\00\03\0e:\0b;\0bI\13\00\00\154\00\03\0e:\0b;\0bI\13\00\00\16.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b\00\00\17\05\00\02\18\03\0e:\0b;\0bI\13\00\00\18\1d\011\13U\17X\0bY\0bW\0b\00\00\19\0b\01U\17\00\00\1a\05\00\02\181\13\00\00\1b4\00\02\181\13\00\00\1c\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00\1d\0b\01\11\01\12\06\00\00\1e\1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00\1f$\00\03\0e>\0b\0b\0b\00\00 \13\01\03\0e\0b\0b\88\01\0f\00\00!\0f\00I\133\06\00\00\22\0f\00I\13\03\0e3\06\00\00#\15\01I\13\00\00$\13\00\03\0e\0b\0b\88\01\0f\00\00%\01\01I\13\00\00&!\00I\13\22\0d7\0b\00\00'$\00\03\0e\0b\0b>\0b\00\00(.\01G\13 \0b\00\00)4\00\03\0e:\0b;\05I\13\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03.\01n\0e\03\0e:\0b;\0bI\13 \0b\00\00\04/\00I\13\03\0e\00\00\05\0b\01\00\00\06\05\00\03\0e:\0b;\0bI\13\00\00\07.\01n\0e\03\0e:\0b;\05I\13 \0b\00\00\08\05\00\03\0e:\0b;\05I\13\00\00\094\00\03\0e:\0b;\05I\13\00\00\0a\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\0b3\01\15\13\00\00\0c\0d\00I\13\88\01\0f8\0b4\19\00\00\0d\19\01\16\0b\00\00\0e\0d\00\03\0eI\13\88\01\0f8\0b\00\00\0f\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\10.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0bI\13\00\00\11\05\00\02\18\03\0e:\0b;\0bI\13\00\00\12\1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00\13\0b\01\11\01\12\06\00\00\14\05\00\02\181\13\00\00\154\00\02\18\03\0e:\0b;\0bI\13\00\00\16\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00\174\00\02\181\13\00\00\18\0b\01U\17\00\00\19\1d\011\13U\17X\0bY\0bW\0b\00\00\1a$\00\03\0e>\0b\0b\0b\00\00\1b\0f\00I\13\03\0e3\06\00\00\1c\13\01\03\0e\0b\0b\88\01\0f\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03\04\01I\13m\19\03\0e\0b\0b\88\01\0f\00\00\04(\00\03\0e\1c\0f\00\00\05\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\06\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\073\01\15\13\00\00\08\0d\00I\13\88\01\0f8\0b4\19\00\00\09\19\01\16\0b\00\00\0a\0d\00\03\0eI\13\88\01\0f8\0b\00\00\0b\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\0c\19\01\00\00\0d.\01n\0e\03\0e:\0b;\05I\13<\19\00\00\0e\05\00I\13\00\00\0f/\00I\13\03\0e\00\00\10$\00\03\0e>\0b\0b\0b\00\00\11\13\01\03\0e\0b\0b\88\01\0f\00\00\12\0f\00I\133\06\00\00\13\0f\00I\13\03\0e3\06\00\00\14\15\01I\13\00\00\15\13\00\03\0e\0b\0b\88\01\0f\00\00\16\01\01I\13\00\00\17!\00I\13\22\0d7\0b\00\00\18$\00\03\0e\0b\0b>\0b\00\00\19.\01G\13 \0b\00\00\1a\0b\01\00\00\1b\05\00\03\0e:\0b;\05I\13\00\00\1c4\00\03\0e:\0b;\05I\13\00\00\1d.\01n\0e\03\0e:\0b;\05I\13 \0b\00\00\1e.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05I\13\00\00\1f\05\00\02\18\03\0e:\0b;\05I\13\00\00 \1d\011\13U\17X\0bY\05W\0b\00\00!\0b\01U\17\00\00\22\05\00\02\181\13\00\00#\0b\01\11\01\12\06\00\00$4\00\02\181\13\00\00%\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03.\01n\0e\03\0e:\0b;\0bI\13 \0b\00\00\04/\00I\13\03\0e\00\00\05\0b\01\00\00\06\05\00\03\0e:\0b;\0bI\13\00\00\074\00\03\0e:\0b;\0bI\13\00\00\08.\01n\0e\03\0e:\0b;\05I\13 \0b\00\00\09\05\00\03\0e:\0b;\05I\13\00\00\0a4\00\03\0e:\0b;\05I\13\00\00\0b\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\0c\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\0d\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\0e.\01n\0e\03\0e:\0b;\05I\13<\19\00\00\0f\05\00I\13\00\00\10.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0bI\13\00\00\11\05\00\02\18\03\0e:\0b;\0bI\13\00\00\12\1d\011\13U\17X\0bY\0bW\0b\00\00\13\0b\01U\17\00\00\14\05\00\02\181\13\00\00\15\1d\011\13U\17X\0bY\05W\0b\00\00\164\00\02\181\13\00\00\17\1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00\18\0b\01\11\01\12\06\00\00\19$\00\03\0e>\0b\0b\0b\00\00\1a\0f\00I\13\03\0e3\06\00\00\1b\13\01\03\0e\0b\0b\88\01\0f\00\00\1c\0d\00\03\0eI\13\88\01\0f8\0b\00\00\1d\0f\00I\133\06\00\00\1e.\01G\13 \0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\029\01\03\0e\00\00\03\04\01I\13m\19\03\0e\0b\0b\88\01\0f\00\00\04(\00\03\0e\1c\0f\00\00\05\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\06\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\07\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\08/\00I\13\03\0e\00\00\09.\01n\0e\03\0e:\0b;\0bI\13<\19\00\00\0a\05\00I\13\00\00\0b.\01n\0e\03\0e:\0b;\05I\13<\19\00\00\0c.\01n\0e\03\0e:\0b;\0bI\13 \0b\00\00\0d\0b\01\00\00\0e\05\00\03\0e:\0b;\0bI\13\00\00\0f.\01n\0e\03\0e:\0b;\05I\13 \0b\00\00\10\05\00\03\0e:\0b;\05I\13\00\00\113\01\15\13\00\00\12\0d\00I\13\88\01\0f8\0b4\19\00\00\13\19\01\16\0b\00\00\14\0d\00\03\0eI\13\88\01\0f8\0b\00\00\15$\00\03\0e>\0b\0b\0b\00\00\16.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05I\13\00\00\17\05\00\02\18\03\0e:\0b;\05I\13\00\00\18\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00\19\0b\01\11\01\12\06\00\00\1a\05\00\02\181\13\00\00\1b4\00\02\181\13\00\00\1c\1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00\1d\13\01\03\0e\0b\0b\88\01\0f\00\00\1e\0f\00I\133\06\00\00\1f\0f\00I\13\03\0e3\06\00\00 .\01G\13 \0b\00\00!4\00\03\0e:\0b;\0bI\13\00\00\22\13\00\03\0e\0b\0b\88\01\0f\00\00#\01\01I\13\00\00$!\00I\13\22\0d7\0b\00\00%$\00\03\0e\0b\0b>\0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03\04\01I\13m\19\03\0e\0b\0b\88\01\0f\00\00\04(\00\03\0e\1c\0f\00\00\05\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\06\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\07.\01n\0e\03\0e:\0b;\0bI\13<\19\00\00\08\05\00I\13\00\00\09.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05I\13\00\00\0a\05\00\02\18\03\0e:\0b;\05I\13\00\00\0b\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00\0c\0b\01\11\01\12\06\00\00\0d\05\00\02\181\13\00\00\0e\1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00\0f4\00\02\18\03\0e:\0b;\05I\13\00\00\10\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\11.\01n\0e\03\0e:\0b;\05I\13 \0b\00\00\12\0b\01\00\00\13\05\00\03\0e:\0b;\05I\13\00\00\143\01\15\13\00\00\15\0d\00I\13\88\01\0f8\0b4\19\00\00\16\19\01\00\00\17\0d\00\03\0eI\13\88\01\0f8\0b\00\00\18\19\01\16\0b\00\00\19/\00I\13\03\0e\00\00\1a$\00\03\0e>\0b\0b\0b\00\00\1b.\01G\13 \0b\00\00\1c\05\00\03\0e:\0b;\0bI\13\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\029\01\03\0e\00\00\03\04\01I\13m\19\03\0e\0b\0b\88\01\0f\00\00\04(\00\03\0e\1c\0f\00\00\05\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\06\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\07.\01n\0e\03\0e:\0b;\0bI\13<\19\00\00\08\05\00I\13\00\00\09.\01n\0e\03\0e:\0b;\05I\13 \0b\00\00\0a/\00I\13\03\0e\00\00\0b\0b\01\00\00\0c4\00\03\0e:\0b;\05I\13\00\00\0d.\01n\0e\03\0e:\0b;\05I\13<\19\00\00\0e\05\00\03\0e:\0b;\05I\13\00\00\0f.\01n\0e\03\0e:\0b;\0bI\13 \0b\00\00\10\05\00\03\0e:\0b;\0bI\13\00\00\114\00\03\0e:\0b;\0bI\13\00\00\12\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\13.\01n\0e\03\0e:\0b;\05 \0b\00\00\143\00\00\00\153\01\15\13\00\00\16\0d\00I\13\88\01\0f8\0b4\19\00\00\17\19\01\00\00\18\0d\00\03\0eI\13\88\01\0f8\0b\00\00\19\19\01\16\0b\00\00\1a3\01\00\00\1b$\00\03\0e>\0b\0b\0b\00\00\1c\0f\00I\13\03\0e3\06\00\00\1d.\01G\13 \0b\00\00\1e.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0bI\13\00\00\1f\05\00\02\18\03\0e:\0b;\0bI\13\00\00 \1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00!\0b\01\11\01\12\06\00\00\224\00\02\181\13\00\00#\05\00\02\181\13\00\00$\13\01\03\0e\0b\0b\88\01\0f\00\00%\0f\00I\133\06\00\00&.\01\11\01\12\06@\18G\13\00\00'\1d\011\13U\17X\0bY\0bW\0b\00\00(\0b\01U\17\00\00)\1d\011\13U\17X\0bY\05W\0b\00\00*\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00+4\00\02\18\03\0e:\0b;\0bI\13\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\029\01\03\0e\00\00\03.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0bI\13\00\00\04/\00I\13\03\0e\00\00\05\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\063\01\15\13\00\00\07\0d\00I\13\88\01\0f8\0b4\19\00\00\08\19\01\00\00\09\0d\00\03\0eI\13\88\01\0f8\0b\00\00\0a\19\01\16\06\00\00\0b\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\0c\19\01\16\0b\00\00\0d$\00\03\0e>\0b\0b\0b\00\00\0e\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\0f\0f\00I\13\03\0e3\06\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03\04\01I\13m\19\03\0e\0b\0b\88\01\0f\00\00\04(\00\03\0e\1c\0f\00\00\05\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\06\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\073\01\15\13\00\00\08\0d\00I\13\88\01\0f8\0b4\19\00\00\09\19\01\16\0b\00\00\0a\0d\00\03\0eI\13\88\01\0f8\0b\00\00\0b\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\0c\19\01\00\00\0d.\01n\0e\03\0e:\0b;\05I\13<\19\00\00\0e\05\00I\13\00\00\0f/\00I\13\03\0e\00\00\10.\01n\0e\03\0e:\0b;\05I\13 \0b\00\00\11\0b\01\00\00\12\05\00\03\0e:\0b;\05I\13\00\00\13.\01n\0e\03\0e:\0b;\0bI\13 \0b\00\00\14\05\00\03\0e:\0b;\0bI\13\00\00\15.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b\00\00\16\05\00\02\18\03\0e:\0b;\0bI\13\00\00\17\1d\011\13U\17X\0bY\05W\0b\00\00\18\0b\01U\17\00\00\19\05\00\02\181\13\00\00\1a\1d\011\13U\17X\0bY\0bW\0b\00\00\1b4\00\02\181\13\00\00\1c\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00\1d\0b\01\11\01\12\06\00\00\1e\1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00\1f$\00\03\0e>\0b\0b\0b\00\00 \13\01\03\0e\0b\0b\88\01\0f\00\00!\0f\00I\133\06\00\00\22\0f\00I\13\03\0e3\06\00\00#\15\01I\13\00\00$\13\00\03\0e\0b\0b\88\01\0f\00\00%\01\01I\13\00\00&!\00I\13\22\0d7\0b\00\00'$\00\03\0e\0b\0b>\0b\00\00(.\01G\13 \0b\00\00)4\00\03\0e:\0b;\05I\13\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\029\01\03\0e\00\00\03\04\01I\13m\19\03\0e\0b\0b\88\01\0f\00\00\04(\00\03\0e\1c\0f\00\00\05\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\06\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\07.\01n\0e\03\0e:\0b;\0bI\13<\19\00\00\08/\00I\13\03\0e\00\00\09.\01n\0e\03\0e:\0b;\05I\13 \0b\00\00\0a\05\00I\13\00\00\0b.\01n\0e\03\0e:\0b;\05I\13<\19\00\00\0c\0b\01\00\00\0d\05\00\03\0e:\0b;\05I\13\00\00\0e4\00\03\0e:\0b;\05I\13\00\00\0f\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\103\01\15\13\00\00\11\0d\00I\13\88\01\0f8\0b4\19\00\00\12\19\01\16\06\00\00\13\0d\00\03\0eI\13\88\01\0f8\0b\00\00\14\19\01\00\00\15\19\01\16\0b\00\00\16$\00\03\0e>\0b\0b\0b\00\00\17.\01n\0e\03\0e:\0b;\0bI\13 \0b\00\00\184\00\03\0e:\0b;\0bI\13\00\00\19\05\00\03\0e:\0b;\0bI\13\00\00\1a\0f\00I\13\03\0e3\06\00\00\1b.\01G\13 \0b\00\00\1c\13\01\03\0e\0b\0b\88\01\0f\00\00\1d\0f\00I\133\06\00\00\1e.\01\11\01\12\06@\18G\13\00\00\1f\05\00\02\18\03\0e:\0b;\0bI\13\00\00 \1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00!\0b\01\11\01\12\06\00\00\224\00\02\181\13\00\00#\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00$\0b\01U\17\00\00%4\00\02\18\03\0e:\0b;\0bI\13\00\00&\1d\011\13U\17X\0bY\0bW\0b\00\00'\05\00\02\181\13\00\00(\1d\001\13\11\01\12\06X\0bY\05W\0b\00\00)\1d\001\13\11\01\12\06X\0bY\0bW\0b\00\00*\05\00\02\18\03\0e:\0b;\05I\13\00\00+4\00\02\18\03\0e:\0b;\05I\13\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03\04\01I\13m\19\03\0e\0b\0b\88\01\0f\00\00\04(\00\03\0e\1c\0f\00\00\05.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05I\13\00\00\06\05\00\02\18\03\0e:\0b;\05I\13\00\00\07/\00I\13\03\0e\00\00\08\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\09\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\0a\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\0b3\01\15\13\00\00\0c\0d\00I\13\88\01\0f8\0b4\19\00\00\0d\19\01\16\0b\00\00\0e\0d\00\03\0eI\13\88\01\0f8\0b\00\00\0f$\00\03\0e>\0b\0b\0b\00\00\10\0f\00I\13\03\0e3\06\00\00\11\13\01\03\0e\0b\0b\88\01\0f\00\00\12\0f\00I\133\06\00\00\13\13\00\03\0e\0b\0b\88\01\0f\00\00\14\01\01I\13\00\00\15!\00I\13\22\0d7\0b\00\00\16$\00\03\0e\0b\0b>\0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\029\01\03\0e\00\00\03.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0bI\13\00\00\04\05\00\02\18:\0b;\0bI\13\00\00\05/\00I\13\03\0e\00\00\06.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05\00\00\07\05\00\02\18:\0b;\05I\13\00\00\08\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\09\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\0a3\01\15\13\00\00\0b\0d\00I\13\88\01\0f8\0b4\19\00\00\0c\19\01\16\06\00\00\0d\0d\00\03\0eI\13\88\01\0f8\0b\00\00\0e\19\01\00\00\0f\19\01\16\0b\00\00\10\0f\00I\13\03\0e3\06\00\00\11\15\01I\13\00\00\12\05\00I\13\00\00\13\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\14$\00\03\0e>\0b\0b\0b\00\00\15\13\01\03\0e\0b\0b\88\01\0f\00\00\16\0f\00I\133\06\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03.\01n\0e\03\0e:\0b;\0bI\13 \0b\00\00\04/\00I\13\03\0e\00\00\05\0b\01\00\00\06\05\00\03\0e:\0b;\0bI\13\00\00\07.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b\00\00\08\05\00\02\18\03\0e:\0b;\0bI\13\00\00\09\1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00\0a\0b\01\11\01\12\06\00\00\0b\05\00\02\181\13\00\00\0c$\00\03\0e>\0b\0b\0b\00\00\0d\0f\00I\13\03\0e3\06\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\024\00\03\0eI\13\02\18\00\00\03\13\01\1d\13\03\0e\0b\0b\88\01\0f\00\00\04\0d\00\03\0eI\13\88\01\0f8\0b\00\00\05\0f\00I\13\03\0e3\06\00\00\06$\00\03\0e>\0b\0b\0b\00\00\079\01\03\0e\00\00\08\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\09\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\0a/\00I\13\03\0e\00\00\0b\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\0c\04\01I\13m\19\03\0e\0b\0b\88\01\0f\00\00\0d(\00\03\0e\1c\0f\00\00\0e.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05I\13\00\00\0f\05\00\02\18\03\0e:\0b;\05I\13\00\00\10\0b\01\11\01\12\06\00\00\114\00\02\18\03\0e:\0b;\05I\13\00\00\123\01\15\13\00\00\13\0d\00I\13\88\01\0f8\0b4\19\00\00\14\19\01\16\0b\00\00\15\19\01\00\00\16\19\01\16\06\00\00\17\13\01\03\0e\0b\0b\88\01\0f\00\00\18\0f\00I\133\06\00\00\19\13\00\03\0e\0b\0b\88\01\0f\00\00\1a\01\01I\13\00\00\1b!\00I\13\22\0d7\0b\00\00\1c$\00\03\0e\0b\0b>\0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\04/\00I\13\03\0e\00\00\05\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\06.\01n\0e\03\0e:\0b;\05I\13<\19\00\00\07\05\00I\13\00\00\08\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\09$\00\03\0e>\0b\0b\0b\00\00\0a\0f\00I\13\03\0e3\06\00\00\0b.\01G\13 \0b\00\00\0c\0b\01\00\00\0d\05\00\03\0e:\0b;\05I\13\00\00\0e4\00\03\0e:\0b;\05I\13\00\00\0f.\01\11\01\12\06@\18G\13\00\00\10\05\00\02\18\03\0e:\0b;\05I\13\00\00\11\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00\12\0b\01\11\01\12\06\00\00\13\05\00\02\181\13\00\00\144\00\02\181\13\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\024\00\03\0eI\13\02\18\00\00\03\13\01\1d\13\03\0e\0b\0b\88\01\0f\00\00\04\0d\00\03\0eI\13\88\01\0f8\0b\00\00\05\0f\00I\13\03\0e3\06\00\00\06$\00\03\0e>\0b\0b\0b\00\00\079\01\03\0e\00\00\08\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\09\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\0a/\00I\13\03\0e\00\00\0b\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\0c\04\01I\13m\19\03\0e\0b\0b\88\01\0f\00\00\0d(\00\03\0e\1c\0f\00\00\0e.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05I\13\00\00\0f\05\00\02\18\03\0e:\0b;\05I\13\00\00\10\0b\01\11\01\12\06\00\00\114\00\02\18\03\0e:\0b;\05I\13\00\00\123\01\15\13\00\00\13\0d\00I\13\88\01\0f8\0b4\19\00\00\14\19\01\16\06\00\00\15\19\01\00\00\16\19\01\16\0b\00\00\17\13\01\03\0e\0b\0b\88\01\0f\00\00\18\0f\00I\133\06\00\00\19\13\00\03\0e\0b\0b\88\01\0f\00\00\1a\01\01I\13\00\00\1b!\00I\13\22\0d7\0b\00\00\1c$\00\03\0e\0b\0b>\0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0bI\13\00\00\04\05\00\02\18\03\0e:\0b;\0bI\13\00\00\05$\00\03\0e>\0b\0b\0b\00\00\06\13\01\03\0e\0b\0b\88\01\0f\00\00\07\0d\00\03\0eI\13\88\01\0f8\0b\00\00\08\0f\00I\133\06\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\029\01\03\0e\00\00\034\00\03\0eI\13:\0b;\0b\88\01\0f\02\18n\0e\00\00\04.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b\00\00\05\1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00\06\05\00\02\181\13\00\00\07\0b\01\11\01\12\06\00\00\08.\00\11\01\12\06@\18n\0e\03\0e:\0b;\0b\00\00\09\01\01I\13\00\00\0a!\00I\13\22\0d7\05\00\00\0b$\00\03\0e>\0b\0b\0b\00\00\0c$\00\03\0e\0b\0b>\0b\00\00\0d\04\01I\13m\19\03\0e\0b\0b\88\01\0f\00\00\0e(\00\03\0e\1c\0f\00\00\0f\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\10\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\11.\01n\0e\03\0e:\0b;\0bI\13<\19\00\00\12/\00I\13\03\0e\00\00\13\05\00I\13\00\00\143\01\15\13\00\00\15\0d\00I\13\88\01\0f8\0b4\19\00\00\16\19\01\00\00\17\0d\00\03\0eI\13\88\01\0f8\0b\00\00\18\19\01\16\0b\00\00\19\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\1a.\01n\0e\03\0e:\0b;\05I\13 \0b\00\00\1b\0b\01\00\00\1c\05\00\03\0e:\0b;\05I\13\00\00\1d\19\01\16\06\00\00\1e\0f\00I\13\03\0e3\06\00\00\1f\15\01I\13\00\00 \13\01\03\0e\0b\0b\88\01\0f\00\00!\0f\00I\133\06\00\00\22\13\00\03\0e\0b\0b\88\01\0f\00\00#!\00I\13\22\0d7\0b\00\00$.\01G\13 \0b\00\00%\05\00\03\0e:\0b;\0bI\13\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\029\01\03\0e\00\00\03\04\01I\13m\19\03\0e\0b\0b\88\01\0f\00\00\04(\00\03\0e\1c\0f\00\00\05\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\06\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\073\01\15\13\00\00\08\0d\00I\13\88\01\0f8\0b4\19\00\00\09\19\01\16\0b\00\00\0a\0d\00\03\0eI\13\88\01\0f8\0b\00\00\0b\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\0c\19\01\00\00\0d.\01n\0e\03\0e:\0b;\05I\13<\19\00\00\0e\05\00I\13\00\00\0f/\00I\13\03\0e\00\00\10.\01n\0e\03\0e:\0b;\05I\13 \0b\00\00\11\0b\01\00\00\12\05\00\03\0e:\0b;\05I\13\00\00\13.\01n\0e\03\0e:\0b;\0bI\13 \0b\00\00\14\05\00\03\0e:\0b;\0bI\13\00\00\15.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b\00\00\16\05\00\02\18\03\0e:\0b;\0bI\13\00\00\17\1d\011\13U\17X\0bY\05W\0b\00\00\18\0b\01U\17\00\00\19\05\00\02\181\13\00\00\1a\1d\011\13U\17X\0bY\0bW\0b\00\00\1b4\00\02\181\13\00\00\1c\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00\1d\0b\01\11\01\12\06\00\00\1e\1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00\1f.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05I\13\00\00 \05\00\02\18\03\0e:\0b;\05I\13\00\00!$\00\03\0e>\0b\0b\0b\00\00\22\13\01\03\0e\0b\0b\88\01\0f\00\00#\0f\00I\133\06\00\00$\0f\00I\13\03\0e3\06\00\00%\15\01I\13\00\00&\13\00\03\0e\0b\0b\88\01\0f\00\00'\01\01I\13\00\00(!\00I\13\22\0d7\0b\00\00)$\00\03\0e\0b\0b>\0b\00\00*.\01G\13 \0b\00\00+4\00\03\0e:\0b;\05I\13\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03\04\01I\13m\19\03\0e\0b\0b\88\01\0f\00\00\04(\00\03\0e\1c\0f\00\00\05.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05I\13\00\00\06\05\00\02\18\03\0e:\0b;\05I\13\00\00\07\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\08\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\09\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\0a3\01\15\13\00\00\0b\0d\00I\13\88\01\0f8\0b4\19\00\00\0c\19\01\16\0b\00\00\0d\0d\00\03\0eI\13\88\01\0f8\0b\00\00\0e/\00I\13\03\0e\00\00\0f$\00\03\0e>\0b\0b\0b\00\00\10\0f\00I\13\03\0e3\06\00\00\11\13\01\03\0e\0b\0b\88\01\0f\00\00\12\0f\00I\133\06\00\00\13\13\00\03\0e\0b\0b\88\01\0f\00\00\14\01\01I\13\00\00\15!\00I\13\22\0d7\0b\00\00\16$\00\03\0e\0b\0b>\0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03.\01n\0e\03\0e:\0b;\05I\13 \0b\00\00\04\0b\01\00\00\05\05\00\03\0e:\0b;\05I\13\00\00\064\00\03\0e:\0b;\05I\13\00\00\07.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b\00\00\08\05\00\02\18\03\0e:\0b;\0bI\13\00\00\09\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00\0a\0b\01\11\01\12\06\00\00\0b\05\00\02\181\13\00\00\0c\13\01\03\0e\0b\0b\88\01\0f\00\00\0d\0d\00\03\0eI\13\88\01\0f8\0b\00\00\0e$\00\03\0e>\0b\0b\0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\034\00\03\0eI\13:\0b;\0b\88\01\0f\02\18n\0e\00\00\04.\00\11\01\12\06@\18n\0e\03\0e:\0b;\0b?\19\00\00\05\0f\00I\13\03\0e3\06\00\00\06\15\00\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\029\01\03\0e\00\00\034\00\03\0eI\13:\0b;\0b\88\01\0f\02\18n\0e\00\00\04.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b\00\00\05\05\00\02\18\03\0e:\0b;\0bI\13\00\00\06\0b\01\11\01\12\06\00\00\074\00\02\18\03\0e:\0b;\0bI\13\00\00\08.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0bI\13\00\00\09\0b\01U\17\00\00\0a\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\0b\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\0c\0f\00I\13\03\0e3\06\00\00\0d\15\00\00\00\0e3\01\15\13\00\00\0f\0d\00I\13\88\01\0f8\0b4\19\00\00\10\19\01\16\06\00\00\11\0d\00\03\0eI\13\88\01\0f8\0b\00\00\12\19\01\00\00\13/\00I\13\03\0e\00\00\14\19\01\16\0b\00\00\15\17\01\03\0e\0b\0b\88\01\0f\00\00\16$\00\03\0e>\0b\0b\0b\00\00\17\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\18\13\01\03\0e\0b\0b\88\01\0f\00\00\19\0f\00I\133\06\00\00\1a\01\01I\13\00\00\1b!\00I\13\22\0d7\0b\00\00\1c$\00\03\0e\0b\0b>\0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\024\00\03\0eI\13\02\18\00\00\03\13\01\1d\13\03\0e\0b\0b\88\01\0f\00\00\04\0d\00\03\0eI\13\88\01\0f8\0b\00\00\05\0f\00I\13\03\0e3\06\00\00\06$\00\03\0e>\0b\0b\0b\00\00\079\01\03\0e\00\00\08\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\09\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\0a.\01n\0e\03\0e:\0b;\05I\13<\19\00\00\0b\05\00I\13\00\00\0c.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05I\13\00\00\0d\05\00\02\18\03\0e:\0b;\05I\13\00\00\0e/\00I\13\03\0e\00\00\0f.\01n\0e\03\0e:\0b;\05I\13 \0b\00\00\10\0b\01\00\00\11\05\00\03\0e:\0b;\05I\13\00\00\12\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00\13\0b\01\11\01\12\06\00\00\14\05\00\02\181\13\00\00\154\00\02\181\13\00\00\16\1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00\17.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05\00\00\18\1d\001\13\11\01\12\06X\0bY\05W\0b\00\00\19\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\1a.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0bI\13\00\00\1b\05\00\02\18\03\0e:\0b;\0bI\13\00\00\1c.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b\00\00\1d.\01n\0e\03\0e:\0b;\0b \0b\00\00\1e\05\00\03\0e:\0b;\0bI\13\00\00\1f4\00\02\18\03\0e:\0b;\05I\13\00\00 .\01n\0e\03\0e:\0b;\0bI\13<\19\00\00!\04\01I\13m\19\03\0e\0b\0b\88\01\0f\00\00\22(\00\03\0e\1c\0f\00\00#.\01n\0e\03\0e:\0b;\0bI\13 \0b\00\00$\1d\011\13U\17X\0bY\05W\0b\00\00%\0b\01U\17\00\00&\1d\011\13U\17X\0bY\0bW\0b\00\00'\05\00\02\18:\0b;\05I\13\00\00(4\00\03\0e:\0b;\05I\13\00\00).\01n\0e\03\0e:\0b;\05 \0b\00\00*3\01\15\13\00\00+\0d\00I\13\88\01\0f8\0b4\19\00\00,\19\01\16\0b\00\00-\19\01\00\00.\19\01\16\06\00\00/.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05\87\01\19\00\0004\00\02\18\03\0e:\0b;\0bI\13\00\0014\00\03\0e:\0b;\0bI\13\00\0024\00\03\0eI\13:\0b;\0b\88\01\0f\02\18n\0e\00\003.\00\11\01\12\06@\18n\0e\03\0e:\0b;\0b?\19\00\004.\01\11\01\12\06@\18\03\0e:\0b;\0bI\13?\19\00\005.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b?\19\00\006.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0bI\13?\19\00\007\13\01\03\0e\0b\0b\88\01\0f\00\008\0f\00I\133\06\00\009.\01G\13 \0b\00\00:\13\00\03\0e\0b\0b\88\01\0f\00\00;\01\01I\13\00\00<!\00I\13\22\0d7\0b\00\00=$\00\03\0e\0b\0b>\0b\00\00>.\01\11\01\12\06@\18G\13\00\00?\15\01I\13\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\029\01\03\0e\00\00\03\04\01I\13m\19\03\0e\0b\0b\88\01\0f\00\00\04(\00\03\0e\1c\0f\00\00\05\13\01\03\0e\0b\0b2\0b\88\01\0f\00\00\06\0d\00\03\0eI\13\88\01\0f8\0b2\0b\00\00\073\01\15\13\00\00\08\0d\00I\13\88\01\0f8\0b4\19\00\00\09\19\01\16\0b\00\00\0a\0d\00\03\0eI\13\88\01\0f8\0b\00\00\0b\13\00\03\0e\0b\0b2\0b\88\01\0f\00\00\0c\19\01\00\00\0d.\01n\0e\03\0e:\0b;\05I\13<\19\00\00\0e\05\00I\13\00\00\0f.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05I\13\00\00\10\05\00\02\18\03\0e:\0b;\05I\13\00\00\11\0b\01\11\01\12\06\00\00\124\00\02\18\03\0e:\0b;\05I\13\00\00\13/\00I\13\03\0e\00\00\143\01\00\00\153\00\00\00\16$\00\03\0e>\0b\0b\0b\00\00\17\13\01\03\0e\0b\0b\88\01\0f\00\00\18\0f\00I\133\06\00\00\19\0f\00I\13\03\0e3\06\00\00\1a\15\01I\13\00\00\1b\13\00\03\0e\0b\0b\88\01\0f\00\00\1c\01\01I\13\00\00\1d!\00I\13\22\0d7\0b\00\00\1e$\00\03\0e\0b\0b>\0b\00\00\1f.\01\11\01\12\06@\18G\13\00\00 .\01G\13 \0b\00\00!\0b\01\00\00\22\05\00\03\0e:\0b;\05I\13\00\00#4\00\03\0e:\0b;\05I\13\00\00$\0b\01U\17\00\00%\1d\011\13U\17X\0bY\05W\0b\00\00&\05\00\02\181\13\00\00'\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00(4\00\02\181\13\00\00).\00n\0e\03\0e:\0b;\05I\13<\19?\19\00\00*.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05I\13?\19\00\00+\05\00\02\18:\0b;\05I\13\00\00,4\00\02\18\03\0e\88\01\0f:\0b;\05I\13\00\00-.\00\11\01\12\06@\18G\13\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\029\01\03\0e\00\00\03.\00\11\01\12\06@\18n\0e\03\0e:\0b;\0b\00\00\04.\00n\0e\03\0e:\0b;\0b \0b\00\00\05.\00n\0e\03\0e:\0b;\05 \0b\00\00\06.\00\11\01\12\06@\181\13\00\00\07.\00\11\01\12\06@\18n\0e\03\0e:\0b;\05\00\00\08.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05\00\00\09\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00\0a\1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00\0b\1d\001\13\11\01\12\06X\0bY\05W\0b\00\00\0c\1d\001\13\11\01\12\06X\0bY\0bW\0b\00\00\0d\1d\011\13U\17X\0bY\05W\0b\00\00\0e\1d\011\13U\17X\0bY\0bW\0b\00\00\0f\1d\001\13U\17X\0bY\05W\0b\00\00\10.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b\00\00\11.\01\11\01\12\06@\181\13\00\00\12\1d\001\13U\17X\0bY\0bW\0b\00\00\13.\00\11\01\12\06@\18n\0e\03\0e:\0b;\056\0b\87\01\19\00\00\14\1d\001\13\11\01\12\06X\0bY\0b\00\00\15\1d\001\13U\17X\0bY\0b\00\00\16\1d\011\13U\17X\0bY\0b\00\00\17\1d\011\13\11\01\12\06X\0bY\0b\00\00\18.\00n\0e\03\0e:\0b;\0b?\19 \0b\00\00\19.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b?\19\00\00\1a.\00n\0e\03\0e:\0b;\05?\19 \0b\00\00\1b.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05?\19\00\00\1c.\00\11\01\12\06@\18n\0e\03\0e:\0b;\05?\19\00\00\1d.\00n\0e\03\0e:\0b;\0b?\19\87\01\19 \0b\00\00\1e.\00n\0e\03\0e:\0b;\0b\87\01\19 \0b\00\00\1f.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b?\19\87\01\19\00\00 .\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b\87\01\19\00\00!.\00\11\01\12\06@\18n\0e\03\0e:\0b;\0b?\19\00\00\22.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b6\0b\00\00#.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05\87\01\19\00\00$.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05?\19\87\01\19\00\00%.\00\11\01\12\06@\18n\0e\03\0e:\0b;\05\87\01\19\00\00&.\01\11\01\12\06@\18\03\0e:\0b;\05?\19\87\01\19\00\00'.\00\11\01\12\06@\18n\0e\03\0e:\0b;\05?\19\87\01\19\00\00(.\00\11\01\12\06@\18n\0e\03\0e:\0b;\0b?\19\87\01\19\00\00).\00n\0e\03\0e:\0b;\05\87\01\19 \0b\00\00*.\00n\0e\03\0e:\0b;\05?\19\87\01\19 \0b\00\00+.\01\11\01\12\06@\18\03\0e:\0b;\05?\19\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\029\01\03\0e\00\00\03.\00\11\01\12\06@\18\03\0e:\0b;\0b?\19\00\00\04.\00n\0e\03\0e:\0b;\0b\87\01\19 \0b\00\00\05.\01\11\01\12\06@\18\03\0e:\0b;\0b?\19\00\00\06\1d\001\13\11\01\12\06X\0bY\0bW\0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03.\00n\0e\03\0e:\0b;\0b \0b\00\00\04.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b?\19\00\00\05\1d\001\13\11\01\12\06X\0bY\0bW\0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\029\01\03\0e\00\00\03.\00\11\01\12\06@\18n\0e\03\0e:\0b;\05\00\00\04.\00n\0e\03\0e:\0b;\0b \0b\00\00\05.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b\00\00\06\1d\001\13\11\01\12\06X\0bY\0bW\0b\00\00\07.\00\11\01\12\06@\18n\0e\03\0e:\0b;\0b\00\00\08.\00n\0e\03\0e:\0b;\05 \0b\00\00\09.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05\00\00\0a\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00\0b\1d\011\13U\17X\0bY\05W\0b\00\00\0c\1d\001\13\11\01\12\06X\0bY\05W\0b\00\00\0d.\01\11\01\12\06@\181\13\00\00\0e.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05?\19\00\00\0f.\00\11\01\12\06@\18n\0e\03\0e:\0b;\05?\19\00\00\10\1d\001\13U\17X\0bY\05W\0b\00\00\11.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b\87\01\19\00\00\12\1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00\13.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05?\19\87\01\19\00\00\14.\00n\0e\03\0e:\0b;\05\87\01\19 \0b\00\00\15.\01\11\01\12\06@\18\03\0e:\0b;\05?\19\87\01\19\00\00\16\1d\001\13U\17X\0bY\0bW\0b\00\00\17.\00\11\01\12\06@\18n\0e\03\0e:\0b;\0b?\19\00\00\18.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b?\19\00\00\19.\00n\0e\03\0e:\0b;\05?\19 \0b\00\00\1a\1d\011\13U\17X\0bY\0bW\0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\029\01\03\0e\00\00\03.\00n\0e\03\0e:\0b;\05 \0b\00\00\04.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05?\19\00\00\05\1d\011\13U\17X\0bY\05W\0b\00\00\06\1d\001\13\11\01\12\06X\0bY\05W\0b\00\00\07\1d\001\13U\17X\0bY\0bW\0b\00\00\08\1d\001\13U\17X\0bY\05W\0b\00\00\09.\00n\0e\03\0e:\0b;\05?\19 \0b\00\00\0a.\00\11\01\12\06@\181\13\00\00\0b\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00\0c\1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00\0d\1d\001\13\11\01\12\06X\0bY\0bW\0b\00\00\0e\1d\011\13U\17X\0bY\0bW\0b\00\00\0f.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05\00\00\10.\01\11\01\12\06@\181\13\00\00\11.\00n\0e\03\0e:\0b;\0b \0b\00\00\12.\00n\0e\03\0e:\0b;\0b?\19 \0b\00\00\13.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b?\19\00\00\14.\00\11\01\12\06@\18n\0e\03\0e:\0b;\05?\19\00\00\15.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b\00\00\16\1d\001\13\11\01\12\06X\0bY\0b\00\00\17.\00\11\01\12\06@\18n\0e\03\0e:\0b;\05\00\00\18\1d\011\13U\17X\0bY\0b\00\00\19.\00\11\01\12\06@\18n\0e\03\0e:\0b;\0b?\19\00\00\1a.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b?\19\87\01\19\00\00\1b.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05?\19\87\01\19\00\00\1c.\00n\0e\03\0e:\0b;\0b\87\01\19 \0b\00\00\1d.\00\11\01\12\06@\18n\0e\03\0e:\0b;\0b?\19\87\01\19\00\00\1e.\00\11\01\12\06@\18n\0e\03\0e:\0b;\05?\19\87\01\19\00\00\1f.\00n\0e\03\0e:\0b;\056\0b \0b\00\00 .\00n\0e\03\0e:\0b;\0b6\0b \0b\00\00!.\00n\0e\03\0e:\0b;\05\87\01\19 \0b\00\00\22.\00\11\01\12\06@\18n\0e\03\0e:\0b;\056\0b\87\01\19\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03.\01\11\01\12\06@\18\03\0e:\0b;\05?\19\00\00\04\1d\011\10\11\01\12\06X\0bY\05W\0b\00\00\05\1d\011\10\11\01\12\06X\0bY\0bW\0b\00\00\06\1d\001\10\11\01\12\06X\0bY\0bW\0b\00\00\07\1d\001\10U\17X\0bY\0bW\0b\00\00\08\1d\011\10U\17X\0bY\0bW\0b\00\00\09\1d\001\10U\17X\0bY\05W\0b\00\00\0a\1d\001\10\11\01\12\06X\0bY\05W\0b\00\00\0b\11\01%\0e\13\05\03\0e\10\17\1b\0e\00\00\0c.\00n\0e\03\0e:\0b;\0b \0b\00\00\0d.\00n\0e\03\0e:\0b;\05?\19 \0b\00\00\0e.\00n\0e\03\0e:\0b;\05 \0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\029\01\03\0e\00\00\03.\00n\0e\03\0e:\0b;\0b \0b\00\00\04.\00n\0e\03\0e:\0b;\05 \0b\00\00\05.\01\11\01\12\06@\18n\0e\03\0e:\0b;\0b\00\00\06\1d\001\13\11\01\12\06X\0bY\0bW\0b\00\00\07\1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00\08\1d\011\13\11\01\12\06X\0bY\0b\00\00\09\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00\0a\1d\001\13\11\01\12\06X\0bY\05W\0b\00\00\0b\1d\011\13U\17X\0bY\05W\0b\00\00\0c\1d\001\13U\17X\0bY\05W\0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03.\01\11\01\12\06@\18\03\0e:\0b;\05?\19\00\00\04\1d\001\10\11\01\12\06X\0bY\05W\0b\00\00\05\11\01%\0e\13\05\03\0e\10\17\1b\0e\00\00\06.\00n\0e\03\0e:\0b;\05?\19 \0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01U\17\00\00\029\01\03\0e\00\00\03.\00n\0e\03\0e:\0b;\0b \0b\00\00\04.\00n\0e\03\0e:\0b;\05 \0b\00\00\05.\01\11\01\12\06@\18n\0e\03\0e:\0b;\05?\19\00\00\06\1d\011\13\11\01\12\06X\0bY\0bW\0b\00\00\07\1d\001\13\11\01\12\06X\0bY\05W\0b\00\00\08\1d\001\13\11\01\12\06X\0bY\0bW\0b\00\00\09\1d\011\13U\17X\0bY\0bW\0b\00\00\0a\1d\011\13\11\01\12\06X\0bY\05W\0b\00\00\0b.\00\11\01\12\06@\18n\0e\03\0e:\0b;\05?\19\00\00\0c\1d\001\13U\17X\0bY\0bW\0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03.\00\11\01\12\06@\18\03\0e:\0b;\05?\19\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03.\01\11\01\12\06@\18\03\0e:\0b;\05?\19\00\00\04\1d\011\10U\17X\0bY\05W\0b\00\00\05\1d\011\10U\17X\0bY\0bW\0b\00\00\06\1d\011\10\11\01\12\06X\0bY\0bW\0b\00\00\07\1d\001\10\11\01\12\06X\0bY\05W\0b\00\00\08\1d\001\10\11\01\12\06X\0bY\0bW\0b\00\00\09\1d\001\10U\17X\0bY\0bW\0b\00\00\0a\11\01%\0e\13\05\03\0e\10\17\1b\0e\00\00\0b.\00n\0e\03\0e:\0b;\05 \0b\00\00\0c.\00n\0e\03\0e:\0b;\0b \0b\00\00\0d.\00n\0e\03\0e:\0b;\05?\19 \0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03.\00\11\01\12\06@\18\03\0e:\0b;\05?\19\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03.\01\11\01\12\06@\18\03\0e:\0b;\05?\19\00\00\04\1d\011\10\11\01\12\06X\0bY\05W\0b\00\00\05\1d\011\10\11\01\12\06X\0bY\0bW\0b\00\00\06\1d\011\10U\17X\0bY\0bW\0b\00\00\07\1d\001\10U\17X\0bY\05W\0b\00\00\08\1d\001\10\11\01\12\06X\0bY\05W\0b\00\00\09\1d\001\10\11\01\12\06X\0bY\0bW\0b\00\00\0a\11\01%\0e\13\05\03\0e\10\17\1b\0e\00\00\0b.\00n\0e\03\0e:\0b;\0b \0b\00\00\0c.\00n\0e\03\0e:\0b;\05?\19 \0b\00\00\0d.\00n\0e\03\0e:\0b;\05 \0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03.\00\11\01\12\06@\18\03\0e:\0b;\05?\19\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03.\01\11\01\12\06@\18\03\0e:\0b;\05?\19\00\00\04\1d\001\10\11\01\12\06X\0bY\05W\0b\00\00\05\11\01%\0e\13\05\03\0e\10\17\1b\0e\00\00\06.\00n\0e\03\0e:\0b;\05?\19 \0b\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03.\00\11\01\12\06@\18\03\0e:\0b;\05?\19\00\00\00\01\11\01%\0e\13\05\03\0e\10\17\1b\0e\11\01\12\06\00\00\029\01\03\0e\00\00\03.\00\11\01\12\06@\18\03\0e:\0b;\05?\19\00\00\00")
    (@producers
      (language "Rust" "")
      (processed-by "rustc" "1.81.0-nightly (6be96e386 2024-07-09)")
      (processed-by "wit-component" "0.18.2")
      (processed-by "wit-bindgen-rust" "0.16.0")
    )
    (@custom "target_features" (after data) "\02+\0fmutable-globals+\08sign-ext")
  )
  (core module (;1;)
    (type (;0;) (func (param i32 i32)))
    (type (;1;) (func (param i32 i32)))
    (func $indirect-test:guest/host-log (;0;) (type 0) (param i32 i32)
      local.get 0
      local.get 1
      i32.const 0
      call_indirect (type 0)
    )
    (func $indirect-test:guest/host-result-option (;1;) (type 1) (param i32 i32)
      local.get 0
      local.get 1
      i32.const 1
      call_indirect (type 1)
    )
    (func $indirect-test:guest/host-result-result (;2;) (type 1) (param i32 i32)
      local.get 0
      local.get 1
      i32.const 2
      call_indirect (type 1)
    )
    (func $indirect-test:guest/host-result-result-ok (;3;) (type 1) (param i32 i32)
      local.get 0
      local.get 1
      i32.const 3
      call_indirect (type 1)
    )
    (func $indirect-test:guest/host-result-result-err (;4;) (type 1) (param i32 i32)
      local.get 0
      local.get 1
      i32.const 4
      call_indirect (type 1)
    )
    (table (;0;) 5 5 funcref)
    (export "0" (func $indirect-test:guest/host-log))
    (export "1" (func $indirect-test:guest/host-result-option))
    (export "2" (func $indirect-test:guest/host-result-result))
    (export "3" (func $indirect-test:guest/host-result-result-ok))
    (export "4" (func $indirect-test:guest/host-result-result-err))
    (export "$imports" (table 0))
    (@producers
      (processed-by "wit-component" "0.208.1")
    )
  )
  (core module (;2;)
    (type (;0;) (func (param i32 i32)))
    (type (;1;) (func (param i32 i32)))
    (import "" "0" (func (;0;) (type 0)))
    (import "" "1" (func (;1;) (type 1)))
    (import "" "2" (func (;2;) (type 1)))
    (import "" "3" (func (;3;) (type 1)))
    (import "" "4" (func (;4;) (type 1)))
    (import "" "$imports" (table (;0;) 5 5 funcref))
    (elem (;0;) (i32.const 0) func 0 1 2 3 4)
    (@producers
      (processed-by "wit-component" "0.208.1")
    )
  )
  (core instance (;0;) (instantiate 1))
  (alias core export 0 "0" (core func (;0;)))
  (alias core export 0 "1" (core func (;1;)))
  (alias core export 0 "2" (core func (;2;)))
  (alias core export 0 "3" (core func (;3;)))
  (alias core export 0 "4" (core func (;4;)))
  (alias export 0 "result-result-none" (func (;0;)))
  (core func (;5;) (canon lower (func 0)))
  (core instance (;1;)
    (export "log" (func 0))
    (export "result-option" (func 1))
    (export "result-result" (func 2))
    (export "result-result-ok" (func 3))
    (export "result-result-err" (func 4))
    (export "result-result-none" (func 5))
  )
  (core instance (;2;) (instantiate 0
      (with "test:guest/host" (instance 1))
    )
  )
  (alias core export 2 "memory" (core memory (;0;)))
  (alias core export 2 "cabi_realloc" (core func (;6;)))
  (alias core export 0 "$imports" (core table (;0;)))
  (alias export 0 "log" (func (;1;)))
  (core func (;7;) (canon lower (func 1) (memory 0) string-encoding=utf8))
  (alias export 0 "result-option" (func (;2;)))
  (core func (;8;) (canon lower (func 2) (memory 0) (realloc 6) string-encoding=utf8))
  (alias export 0 "result-result" (func (;3;)))
  (core func (;9;) (canon lower (func 3) (memory 0) (realloc 6) string-encoding=utf8))
  (alias export 0 "result-result-ok" (func (;4;)))
  (core func (;10;) (canon lower (func 4) (memory 0) (realloc 6) string-encoding=utf8))
  (alias export 0 "result-result-err" (func (;5;)))
  (core func (;11;) (canon lower (func 5) (memory 0) (realloc 6) string-encoding=utf8))
  (core instance (;3;)
    (export "$imports" (table 0))
    (export "0" (func 7))
    (export "1" (func 8))
    (export "2" (func 9))
    (export "3" (func 10))
    (export "4" (func 11))
  )
  (core instance (;4;) (instantiate 2
      (with "" (instance 3))
    )
  )
  (type (;1;) (func))
  (alias core export 2 "test:guest/run#start" (core func (;12;)))
  (func (;6;) (type 1) (canon lift (core func 12)))
  (component (;0;)
    (type (;0;) (func))
    (import "import-func-start" (func (;0;) (type 0)))
    (type (;1;) (func))
    (export (;1;) "start" (func 0) (func (type 1)))
  )
  (instance (;1;) (instantiate 0
      (with "import-func-start" (func 6))
    )
  )
  (export (;2;) "test:guest/run" (instance 1))
  (@producers
    (processed-by "wit-component" "0.208.1")
  )
)