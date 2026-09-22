// math.zig
pub const Vec4 = @Vector(4, f32);

pub fn initPoint(x: f32, y: f32, z: f32) Vec4 {
    return .{ x, y, z, 1.0 }; // w = 1.0 for positions/translations
}

pub fn initVector(x: f32, y: f32, z: f32) Vec4 {
    return .{ x, y, z, 0.0 }; // w = 0.0 for directions/normals
}
