struct VertexOutput {
    [[builtin(position)]] pos: vec4<f32>;
    [[location(0)]] p: vec2<f32>;
};

var<private> coords: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-0.0, -1.0),
    vec2<f32>(-0.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-0.0,  1.0),
    vec2<f32>( 1.0,  1.0),
);

var<private> ps: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0,  1.0),
);

struct Uniforms {
    looking: mat3x3<f32>;
    pos: vec3<f32>;
    size: vec3<i32>;
};

struct Space {
    voxels: array<u32>;
};

[[group(0), binding(0)]]
var<uniform> uniforms: Uniforms;

[[group(1), binding(0)]]
var<storage, read> space: Space;

[[stage(vertex)]]
fn vertex_main(
    [[builtin(vertex_index)]] vertex_index: u32
) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(coords[vertex_index], 0.0, 1.0);
    out.p = ps[vertex_index];
    return out;
}

struct RaycastResult {
    hit: bool;
    color: vec4<f32>;
};

fn raycast(from: vec3<f32>, d: vec3<f32>) -> RaycastResult {
    var result: RaycastResult;
    result.hit = false;
    result.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);

    var step_f = sign(d);
    var t_delta = step_f / d;
    var fudge = (1.0 + step_f) / 2.0;
    var t_max = t_delta * (fudge - fract(from) * step_f);
    if (t_max.x != t_max.x) {
        t_max.x = 1.0 / 0.0;
    }
    if (t_max.y != t_max.y) {
        t_max.y = 1.0 / 0.0;
    }
    if (t_max.z != t_max.z) {
        t_max.z = 1.0 / 0.0;
    }
    var p = vec3<i32>(floor(from));
    var step = vec3<i32>(step_f);
    for (var i = 0; i < 100; i = i + 1) {
        var t = min(t_max.x, min(t_max.y, t_max.z));
        var f = vec3<f32>(0.0, 0.0, 0.0);
        if (t_max.x < t_max.y) {
            if (t_max.x < t_max.z) {
                t_max.x = t_max.x + t_delta.x;
                p.x = p.x + step.x;
                f.x = f32(step.x);
            } else {
                t_max.z = t_max.z + t_delta.z;
                p.z = p.z + step.z;
                f.z = f32(step.z);
            }
        } else {
            if (t_max.y < t_max.z) {
                t_max.y = t_max.y + t_delta.y;
                p.y = p.y + step.y;
                f.y = f32(step.y);
            } else {
                t_max.z = t_max.z + t_delta.z;
                p.z = p.z + step.z;
                f.z = f32(step.z);
            }
        }

        if (all(p >= vec3<i32>(0, 0, 0)) && all(p < uniforms.size)) {
            var color = unpack4x8unorm(
                space.voxels[(p.x * uniforms.size.y + p.y) * uniforms.size.z + p.z]
            );
            if (color.a >= 0.5) {
                result.hit = true;
                result.color = color;
                break;
            }
        } else {
            result.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            break;
        }
    }

    return result;
}

[[stage(fragment)]]
fn fragment_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    var d: vec3<f32> = uniforms.looking * normalize(vec3<f32>(in.p.xy, 1.0));
    var result = raycast(uniforms.pos, d);
    return result.color;
}
