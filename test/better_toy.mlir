module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @_Z10better_toyiii(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c214748364 = arith.constant 214748364 : index
    %c0 = arith.constant 0 : index
    %c214748164_i32 = arith.constant 214748164 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    %1:2 = scf.for %arg3 = %c0 to %c214748364 step %0 iter_args(%arg4 = %arg1, %arg5 = %arg0) -> (i32, i32) {
      %3 = arith.addi %arg5, %c1_i32 : i32
      %4 = arith.cmpi sle, %3, %c214748164_i32 : i32
      %5 = arith.select %4, %3, %c0_i32 : i32
      %6 = scf.if %4 -> (i32) {
        %7 = arith.addi %arg4, %c1_i32 : i32
        scf.yield %7 : i32
      } else {
        scf.yield %arg4 : i32
      }
      scf.yield %6, %5 : i32, i32
    }
    %2 = arith.addi %1#1, %1#0 : i32
    return %2 : i32
  }
}
