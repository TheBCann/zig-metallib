// Runs `crosscheckKernel` from a .metallib on given (a, b) input pairs and
// prints every result as a 32-bit pattern. Standalone: no Zig, no project code.
//
//   xcrun metal -fno-fast-math round_negative_zero.metal -o round.metallib
//   swift run_kernel.swift round.metallib 0xBE99999A,0 0x80000000,0
//
// Values are 32-bit patterns in hex (0x...) or decimal. The kernel has the
// shape of src/engine/crosscheck_harness.zig: two int inputs, one int output,
// a count, the thread id (buffers 0..3).
import Foundation
import Metal

struct Params { var count: UInt32 }

func parse(_ s: String) -> UInt32 {
    if s.hasPrefix("0x") || s.hasPrefix("0X"), let v = UInt32(s.dropFirst(2), radix: 16) { return v }
    if let v = Int64(s), v >= Int64(Int32.min), v <= Int64(UInt32.max) { return UInt32(truncatingIfNeeded: v) }
    fputs("bad 32-bit value '\(s)'\n", stderr)
    exit(2)
}

let args = CommandLine.arguments
guard args.count >= 3 else {
    fputs("usage: swift run_kernel.swift <lib.metallib> <a>,<b> [<a>,<b> ...]\n", stderr)
    exit(2)
}
let pairs: [(UInt32, UInt32)] = args[2...].map { arg in
    let parts = arg.split(separator: ",").map(String.init)
    guard parts.count == 2 else { fputs("each input is a,b: '\(arg)'\n", stderr); exit(2) }
    return (parse(parts[0]), parse(parts[1]))
}

guard let device = MTLCreateSystemDefaultDevice() else { fputs("no Metal device\n", stderr); exit(1) }
print("device: \(device.name)")
let library: MTLLibrary
let pipeline: MTLComputePipelineState
do {
    library = try device.makeLibrary(URL: URL(fileURLWithPath: args[1]))
    guard let function = library.makeFunction(name: "crosscheckKernel") else {
        fputs("crosscheckKernel not found in \(args[1])\n", stderr); exit(1)
    }
    pipeline = try device.makeComputePipelineState(function: function)
} catch {
    fputs("FAIL \(error)\n", stderr); exit(1)
}

let n = pairs.count
let in0 = device.makeBuffer(length: n * 4, options: .storageModeShared)!
let in1 = device.makeBuffer(length: n * 4, options: .storageModeShared)!
let out = device.makeBuffer(length: n * 4, options: .storageModeShared)!
let params = device.makeBuffer(length: MemoryLayout<Params>.stride, options: .storageModeShared)!
let a = in0.contents().bindMemory(to: UInt32.self, capacity: n)
let b = in1.contents().bindMemory(to: UInt32.self, capacity: n)
let r = out.contents().bindMemory(to: UInt32.self, capacity: n)
for (i, p) in pairs.enumerated() { a[i] = p.0; b[i] = p.1; r[i] = 0xDEADBEEF }
params.contents().bindMemory(to: Params.self, capacity: 1).pointee = Params(count: UInt32(n))

let queue = device.makeCommandQueue()!
let commands = queue.makeCommandBuffer()!
let encoder = commands.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(in0, offset: 0, index: 0)
encoder.setBuffer(in1, offset: 0, index: 1)
encoder.setBuffer(out, offset: 0, index: 2)
encoder.setBuffer(params, offset: 0, index: 3)
let group = min(n, pipeline.maxTotalThreadsPerThreadgroup)
encoder.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: group, height: 1, depth: 1))
encoder.endEncoding()
commands.commit()
commands.waitUntilCompleted()
if let error = commands.error { fputs("command buffer failed: \(error)\n", stderr); exit(1) }
for (i, p) in pairs.enumerated() {
    print(String(format: "a=0x%08X b=0x%08X -> 0x%08X", p.0, p.1, r[i]))
}
