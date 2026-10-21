source_filename = "crosscheck_shader"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"


declare float @llvm.fabs.f32(float)




define void @crosscheckKernel(ptr addrspace(1) %__0, ptr addrspace(1) %__1, ptr addrspace(1) %__2, ptr addrspace(2) %__3, i32 %__4) {
__5:
  %__6 = load i32, ptr addrspace(2) %__3, align 4
  %.not = icmp ult i32 %__4, %__6
  br i1 %.not, label %__7, label %common.ret

__7:                                                ; preds = %__5
  %__8 = zext i32 %__4 to i64
  %__9 = getelementptr inbounds [4 x i8], ptr addrspace(1) %__2, i64 %__8
  %__10 = getelementptr inbounds [4 x i8], ptr addrspace(1) %__0, i64 %__8
  %__11 = load i32, ptr addrspace(1) %__10, align 4
  %__12 = getelementptr inbounds [4 x i8], ptr addrspace(1) %__1, i64 %__8
  %__13 = load i32, ptr addrspace(1) %__12, align 4
  %__14 = bitcast i32 %__11 to float
  %__15 = tail call float @llvm.fabs.f32(float %__14)
  %__16 = fcmp olt float %__15, 0x3810000000000000
  %__17 = fcmp olt float %__14, 0.000000e+00
  %__18 = fcmp oeq float %__14, 0.000000e+00
  %__19 = icmp slt i32 %__11, 0
  %__20 = and i1 %__19, %__18
  %__21 = or i1 %__17, %__20
  %__22 = select i1 %__21, float -0.000000e+00, float 0.000000e+00
  %__23 = select i1 %__16, float %__22, float %__14
  %__24 = bitcast i32 %__13 to float
  %__25 = tail call float @llvm.fabs.f32(float %__24)
  %__26 = fcmp olt float %__25, 0x3810000000000000
  %__27 = fcmp olt float %__24, 0.000000e+00
  %__28 = fcmp oeq float %__24, 0.000000e+00
  %__29 = icmp slt i32 %__13, 0
  %__30 = and i1 %__29, %__28
  %__31 = or i1 %__27, %__30
  %__32 = select i1 %__31, float -0.000000e+00, float 0.000000e+00
  %__33 = select i1 %__26, float %__32, float %__24
  %__34 = frem float %__23, %__33
  %__35 = tail call float @llvm.fabs.f32(float %__34)
  %__36 = fcmp olt float %__35, 0x3810000000000000
  %__37 = fcmp olt float %__34, 0.000000e+00
  %__38 = fcmp oeq float %__34, 0.000000e+00
  %__39 = bitcast float %__34 to i32
  %__40 = icmp slt i32 %__39, 0
  %__41 = and i1 %__38, %__40
  %__42 = select i1 %__37, i1 true, i1 %__41
  %__43 = select i1 %__42, float -0.000000e+00, float 0.000000e+00
  %__44 = select i1 %__36, float %__43, float %__34
  %__45 = fcmp uno float %__44, 0.000000e+00
  %__46 = bitcast float %__44 to i32
  %__47 = select i1 %__45, i32 2143289344, i32 %__46
  store i32 %__47, ptr addrspace(1) %__9, align 4
  br label %common.ret
common.ret:                                       ; preds = %__5, %__7
  ret void

}


!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}
!air.version = !{!9}
!air.language_version = !{!10}
!air.compile_options = !{!11, !12, !13}
!air.kernel = !{!22}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 7, !"air.max_device_buffers", i32 31}
!3 = !{i32 7, !"air.max_constant_buffers", i32 31}
!4 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!5 = !{i32 7, !"air.max_textures", i32 128}
!6 = !{i32 7, !"air.max_read_write_textures", i32 8}
!7 = !{i32 7, !"air.max_samplers", i32 16}
!8 = !{!"zig air-splice"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
!11 = !{!"air.compile.denorms_disable"}
!12 = !{!"air.compile.fast_math_disable"}
!13 = !{!"air.compile.framebuffer_fetch_enable"}
!14 = !{}
!15 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"in0"}
!16 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"in1"}
!17 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"out"}
!18 = !{i32 0, i32 4, i32 0, !"uint", !"count"}
!19 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !18, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!20 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!21 = !{!15, !16, !17, !19, !20}
!22 = !{ptr @crosscheckKernel, !14, !21}
