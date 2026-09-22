const std = @import("std");
const objc = @import("objc.zig");

// 1. Link the Metal framework C-function directly
extern "c" fn MTLCreateSystemDefaultDevice() ?objc.Object;

// 2. Define the memory layout for macOS Window frames (maps perfectly to CGRect)
const NSRect = extern struct { x: f64, y: f64, w: f64, h: f64 };

pub fn main(init: std.process.Init) !void {
    _ = init;

    // --- BOOTSTRAP MACOS ---
    const NSApp = objc.msgSend(?objc.Object, objc.class("NSApplication"), "sharedApplication", .{});
    _ = objc.msgSend(void, NSApp.?, "setActivationPolicy:", .{ @as(i64, 0) }); // Regular App

    const frame = NSRect{ .x = 0, .y = 0, .w = 800, .h = 600 };
    const window_alloc = objc.msgSend(?objc.Object, objc.class("NSWindow"), "alloc", .{});
    
    // Titled (1) | Closable (2) | Resizable (8) = 11
    const window = objc.msgSend(?objc.Object, window_alloc.?, "initWithContentRect:styleMask:backing:defer:", .{
        frame, @as(u64, 11), @as(u64, 2), false
    });

    _ = objc.msgSend(void, window.?, "setTitle:", .{ objc.createNSString("zig-metallib") });
    _ = objc.msgSend(void, window.?, "center", .{});

    // --- INITIALIZE METAL ---
    const device = MTLCreateSystemDefaultDevice() orelse {
        std.debug.print("Metal is not supported on this device\n", .{});
        return;
    };

    const view_alloc = objc.msgSend(?objc.Object, objc.class("MTKView"), "alloc", .{});
    const view = objc.msgSend(?objc.Object, view_alloc.?, "initWithFrame:device:", .{ frame, device });
    
    // CRITICAL: We are taking over the render loop. Disable MTKView's internal timer.
    _ = objc.msgSend(void, view.?, "setPaused:", .{ true });
    _ = objc.msgSend(void, view.?, "setEnableSetNeedsDisplay:", .{ false });

    _ = objc.msgSend(void, window.?, "setContentView:", .{ view.? });
    _ = objc.msgSend(void, window.?, "makeKeyAndOrderFront:", .{ @as(?objc.Object, null) });
    _ = objc.msgSend(void, NSApp.?, "activateIgnoringOtherApps:", .{ true });

    // --- LOAD THE LLVM PAYLOAD ---
    const command_queue = objc.msgSend(?objc.Object, device, "newCommandQueue", .{});
    
    // Compile the binary directly into the Zig executable to avoid file path tracking
    const metallib_payload = @embedFile("default_metallib"); 
    const lib_data = objc.dispatch_data_create(metallib_payload.ptr, metallib_payload.len, null, null) orelse
        return error.DispatchDataFailed;
    defer objc.dispatch_release(lib_data);

    var error_out: ?objc.Object = null;
    const library = objc.msgSend(?objc.Object, device, "newLibraryWithData:error:", .{ lib_data, &error_out }) orelse {
        logNSError(error_out);
        return error.MetalLibraryLoadFailed;
    };
    
    const vertex_fn = objc.msgSend(?objc.Object, library, "newFunctionWithName:", .{ objc.createNSString("vertexShader") });
    const frag_fn = objc.msgSend(?objc.Object, library, "newFunctionWithName:", .{ objc.createNSString("fragmentShader") });
    
    const pipeline_desc = objc.msgSend(?objc.Object, objc.class("MTLRenderPipelineDescriptor"), "alloc", .{});
    _ = objc.msgSend(?objc.Object, pipeline_desc.?, "init", .{});
    _ = objc.msgSend(void, pipeline_desc.?, "setVertexFunction:", .{ vertex_fn.? });
    _ = objc.msgSend(void, pipeline_desc.?, "setFragmentFunction:", .{ frag_fn.? });
    
    const color_attachments = objc.msgSend(?objc.Object, pipeline_desc.?, "colorAttachments", .{});
    const attach_0 = objc.msgSend(?objc.Object, color_attachments.?, "objectAtIndexedSubscript:", .{ @as(u64, 0) });
    
    const pixel_format = objc.msgSend(u64, view.?, "colorPixelFormat", .{});
    _ = objc.msgSend(void, attach_0.?, "setPixelFormat:", .{ pixel_format });

    const pipeline_state = objc.msgSend(?objc.Object, device, "newRenderPipelineStateWithDescriptor:error:", .{
        pipeline_desc.?, &error_out,
    }) orelse {
        logNSError(error_out);
        return error.MetalPipelineFailed;
    };

    // We drive the render loop ourselves, so take drawables straight from the
    // CAMetalLayer. MTKView's currentDrawable is only released by its own
    // draw cycle, which we never run, so it would hand back the same drawable
    // every frame.
    const layer = objc.msgSend(?objc.Object, view.?, "layer", .{}) orelse return error.NoMetalLayer;

    // --- THE GAME LOOP ---
    //

    // --- CREATE CHECKERBOARD TEXTURE ---
    // MTLPixelFormatRGBA8Unorm = 70
    const tex_desc = objc.msgSend(?objc.Object, objc.class("MTLTextureDescriptor"), "texture2DDescriptorWithPixelFormat:width:height:mipmapped:", .{
        @as(u64, 70), @as(u64, 8), @as(u64, 8), false
    }) orelse return error.TextureDescFailed;

    // MTLTextureUsageShaderRead = 1
    _ = objc.msgSend(void, tex_desc, "setUsage:", .{@as(u64, 1)}); 

    const checker_texture = objc.msgSend(?objc.Object, device, "newTextureWithDescriptor:", .{tex_desc}) orelse return error.TextureFailed;

    // Build an 8x8 RGBA8 checkerboard array
    var pixels: [8 * 8 * 4]u8 = undefined;
    for (0..8) |y| {
        for (0..8) |x| {
            // 4x4 pixel squares
            const is_white = ((x / 4) + (y / 4)) % 2 == 0;
            const c: u8 = if (is_white) 255 else 0;
            const idx = (y * 8 + x) * 4;
            pixels[idx + 0] = c;
            pixels[idx + 1] = c;
            pixels[idx + 2] = c;
            pixels[idx + 3] = 255;
        }
    }

    // Define the C-ABI packed struct for MTLRegion
    const MTLRegion = extern struct {
        origin_x: u64, origin_y: u64, origin_z: u64,
        width: u64, height: u64, depth: u64,
    };
    const region = MTLRegion{ .origin_x = 0, .origin_y = 0, .origin_z = 0, .width = 8, .height = 8, .depth = 1 };

    // Upload the bytes to the GPU texture
    _ = objc.msgSend(void, checker_texture, "replaceRegion:mipmapLevel:withBytes:bytesPerRow:", .{
        region, @as(u64, 0), &pixels, @as(u64, 8 * 4)
    });

    const distantPast = objc.msgSend(?objc.Object, objc.class("NSDate"), "distantPast", .{});
    const runLoopMode = objc.createNSString("kCFRunLoopDefaultMode");

    while (true) {
        // Every autoreleased object from this iteration (events, command
        // buffers, drawables, descriptors) is released here. Without a pool
        // drawables never return to the layer and it stops rendering.
        const pool = objc.objc_autoreleasePoolPush();
        defer objc.objc_autoreleasePoolPop(pool);

        // 1. Pump OS Events (Non-blocking)
        const event = objc.msgSend(?objc.Object, NSApp.?, "nextEventMatchingMask:untilDate:inMode:dequeue:", .{
            @as(u64, 0xffffffffffffffff),
            distantPast.?,
            runLoopMode,
            true
        });

        if (event != null) {
            _ = objc.msgSend(void, NSApp.?, "sendEvent:", .{ event.? });
            _ = objc.msgSend(void, NSApp.?, "updateWindows", .{});
        }

        // 2. Issue Draw Call
        // draw(command_queue.?, pipeline_state, layer);
        // 2. Issue Draw Call
        draw(command_queue.?, pipeline_state, layer, checker_texture); // Pass the texture here
    }
}

fn logNSError(err: ?objc.Object) void {
    const e = err orelse return;
    const desc = objc.msgSend(?objc.Object, e, "localizedDescription", .{}) orelse return;
    const cstr = objc.msgSend([*:0]const u8, desc, "UTF8String", .{});
    std.debug.print("Metal error: {s}\n", .{cstr});
}

// 3. The Pure Zig Translation of drawInMTKView
fn draw(command_queue: objc.Object, pipeline_state: objc.Object, layer: objc.Object, checker_texture: objc.Object) void {
    // nextDrawable blocks until the layer has a free drawable, which also
    // paces this loop to the display refresh rate.
    const drawable = objc.msgSend(?objc.Object, layer, "nextDrawable", .{}) orelse return;
    const texture = objc.msgSend(?objc.Object, drawable, "texture", .{}) orelse return;

    const cmd_buffer = objc.msgSend(?objc.Object, command_queue, "commandBuffer", .{}) orelse return;

    const rpd = objc.msgSend(?objc.Object, objc.class("MTLRenderPassDescriptor"), "renderPassDescriptor", .{}) orelse return;
    const color_attachments = objc.msgSend(?objc.Object, rpd, "colorAttachments", .{}) orelse return;
    const attach_0 = objc.msgSend(?objc.Object, color_attachments, "objectAtIndexedSubscript:", .{@as(u64, 0)}) orelse return;

    // Set clear color to Dark Blue
    const clear_color = extern struct { r: f64, g: f64, b: f64, a: f64 }{ .r = 0.1, .g = 0.1, .b = 0.2, .a = 1.0 };
    _ = objc.msgSend(void, attach_0, "setTexture:", .{texture});
    _ = objc.msgSend(void, attach_0, "setLoadAction:", .{@as(u64, 2)}); // MTLLoadActionClear
    _ = objc.msgSend(void, attach_0, "setStoreAction:", .{@as(u64, 1)}); // MTLStoreActionStore
    _ = objc.msgSend(void, attach_0, "setClearColor:", .{clear_color});

    const render_encoder = objc.msgSend(?objc.Object, cmd_buffer, "renderCommandEncoderWithDescriptor:", .{rpd}) orelse return;

    _ = objc.msgSend(void, render_encoder, "setRenderPipelineState:", .{pipeline_state});

    _ = objc.msgSend(void, render_encoder, "setFragmentTexture:atIndex:", .{ checker_texture, @as(u64, 0) });

    const triangleData = [_]f32{
         0.0,  0.5, 0.0, 1.0,   0.0, 0.0, 1.0, 0.0,   0.5, 0.0,   0.0, 0.0,
        -0.5, -0.5, 0.0, 1.0,   0.0, 0.0, 1.0, 0.0,   0.0, 1.0,   0.0, 0.0,
         0.5, -0.5, 0.0, 1.0,   0.0, 0.0, 1.0, 0.0,   1.0, 1.0,   0.0, 0.0,
    };

    _ = objc.msgSend(void, render_encoder, "setVertexBytes:length:atIndex:", .{
        &triangleData, @as(usize, @sizeOf(@TypeOf(triangleData))), @as(usize, 0),
    });

    _ = objc.msgSend(void, render_encoder, "drawPrimitives:vertexStart:vertexCount:", .{
        @as(usize, 3), @as(usize, 0), @as(usize, 3),
    });

    _ = objc.msgSend(void, render_encoder, "endEncoding", .{});
    _ = objc.msgSend(void, cmd_buffer, "presentDrawable:", .{drawable});
    _ = objc.msgSend(void, cmd_buffer, "commit", .{});
}

