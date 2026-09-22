const std = @import("std");
const math = @import("math.zig");

// 1. The UMA-optimized Vertex. 
// Uses Vec4 (16 bytes) for position and normal to ensure perfect alignment 
// between the CPU SIMD registers and the Apple Silicon GPU memory fetcher.
pub const Vertex = extern struct {
    position: math.Vec4,
    normal: math.Vec4,
    texCoords: [2]f32,
};

pub const Glyph = struct {
    x: f32, y: f32, width: f32, height: f32, y_offset: f32, advance: f32,
};

// 2. Updated text generation to output the 3D, 16-byte aligned Vertex
pub fn buildTextVertices(
    allocator: std.mem.Allocator, 
    text: []const u8, 
    atlas: std.AutoHashMap(u8, Glyph),
    start_x: f32, 
    start_y: f32,
) ![]Vertex {
    var vertices = std.ArrayList(Vertex).init(allocator);
    var cursor_x = start_x;
    
    // Default normal pointing straight back at the camera
    const default_normal = math.initVector(0.0, 0.0, -1.0);
    
    for (text) |char| {
        const glyph = atlas.get(char) orelse continue; 
        
        const x0 = cursor_x;
        const y0 = start_y + glyph.y_offset;
        const x1 = x0 + glyph.width;  
        const y1 = y0 + glyph.height;

        const @"u0" = glyph.x;
        const v0 = glyph.y;
        const @"u1" = glyph.x + glyph.width;
        const v1 = glyph.y + glyph.height;

        try vertices.appendSlice(&[_]Vertex{
            .{ .position = math.initPoint(x0, y0, 0.0), .normal = default_normal, .texCoords = .{ @"u0", v0 } },
            .{ .position = math.initPoint(x1, y0, 0.0), .normal = default_normal, .texCoords = .{ @"u1", v0 } },
            .{ .position = math.initPoint(x0, y1, 0.0), .normal = default_normal, .texCoords = .{ @"u0", v1 } },
            .{ .position = math.initPoint(x1, y0, 0.0), .normal = default_normal, .texCoords = .{ @"u1", v0 } },
            .{ .position = math.initPoint(x1, y1, 0.0), .normal = default_normal, .texCoords = .{ @"u1", v1 } },
            .{ .position = math.initPoint(x0, y1, 0.0), .normal = default_normal, .texCoords = .{ @"u0", v1 } },
        });
        
        cursor_x += glyph.advance;
    }
    
    return vertices.toOwnedSlice();
}
