//! The `shader` module of tools whose entry points arrive at runtime
//! (spirv-splice, tools/spirv_splice.zig): it lists no comptime entry
//! points, so tools/air/metadata.zig's `function_metadata` is empty, and it
//! re-exports the host-safe interface types (`air`, `gpu`) that
//! metadata.zig and the assembler are written against. The SPIR-V path
//! builds each kernel's `FunctionMetadata` from the module's interface
//! instead (tools/spirv/manifest.zig).

pub const air = @import("air.zig");
pub const gpu = @import("gpu.zig");

pub const functions = [_]air.Function{};
