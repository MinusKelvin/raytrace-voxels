var<private> coords: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
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
    sun: vec3<f32>;
    vp_size: vec2<f32>;
    rng: vec3<u32>;
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
) -> [[builtin(position)]] vec4<f32> {
    return vec4<f32>(coords[vertex_index], 0.0, 1.0);
}

struct RaycastResult {
    hit: bool;
    color: vec4<f32>;
    distance: f32;
    normal: vec3<f32>;
};

fn raycast(from: vec3<f32>, d: vec3<f32>, limit: f32) -> RaycastResult {
    var result: RaycastResult;
    result.hit = false;
    result.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    result.distance = 0.0;
    result.normal = vec3<f32>(0.0, 0.0, 0.0);

    let step_f = sign(d);
    let t_delta = step_f / d;
    let fudge = (1.0 + step_f) / 2.0;
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
    let step = vec3<i32>(step_f);
    let iter_limit = uniforms.size.x + uniforms.size.y + uniforms.size.z;
    for (var i = 0; i < iter_limit; i = i + 1) {
        let t = min(t_max.x, min(t_max.y, t_max.z));
        if (t > limit) {
            break;
        }
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
                result.distance = t;
                result.normal = f;
                break;
            }
        } else {
            result.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            break;
        }
    }

    return result;
}

fn pcg3d(b: vec3<u32>) -> vec3<u32> {
    var v = b
        * vec3<u32>(1664525u, 1664525u, 1664525u)
        + 1013904223u
        ^ uniforms.rng
        ;
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.x * v.z;
    v.z = v.z + v.y * v.x;
    v = v ^ v >> vec3<u32>(16u, 16u, 16u);
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.x * v.z;
    v.z = v.z + v.y * v.x;
    return v;
}

fn random_direction(p: vec3<f32>, i: i32) -> vec3<f32> {
    return normalize(vec3<f32>(pcg3d(
        bitcast<vec3<u32>>(p + f32(i))
    ) % 65536u) / 32768.0 - 1.0);
}

fn raytrace(from: vec3<f32>, d: vec3<f32>) -> vec4<f32> {
    var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var pos = from;
    var dir = d;
    var multiplier = 1.0;
    for (var depth = 0; depth < 4; depth = depth + 1) {
        let ray = raycast(pos, dir, 1.0 / 0.0);
        if (ray.hit) {
            let p = pos + dir * ray.distance;
            var lighting = max(0.0, dot(uniforms.sun, -ray.normal));

            var shadow = 0.0;
            for (var i = 0; i < 1; i = i + 1) {
                let sun = normalize(100.0 * uniforms.sun + random_direction(p, i));
                let shadowcast = raycast(p - ray.normal * 0.001, sun, 1.0 / 0.0);
                if (!shadowcast.hit) {
                    shadow = shadow + 1.0;
                }
            }
            lighting = min(lighting, shadow / 1.0);

            var ao = 0.0;
            for (var i = 0; i < 2; i = i + 1) {
                let d = random_direction(p, i + 100);
                let aocast = raycast(
                    p - ray.normal * 0.001,
                    faceForward(d, reflect(d, ray.normal), -ray.normal),
                    4.0
                );
                if (aocast.hit) {
                    ao = ao + aocast.distance / 4.0;
                } else {
                    ao = ao + 1.0;
                }
            }
            ao = ao / 2.0;

            color = color + ray.color *
                (lighting / 2.0 + 0.5) *
                multiplier * ao;

            if (all(ray.color == vec4<f32>(1.0, 1.0, 1.0, 1.0))) {
                multiplier = multiplier / 2.0;
                color = color * (1.0 - multiplier);
                pos = p - ray.normal * 0.001;
                dir = reflect(dir, ray.normal);
                continue;
            }
        }
        break;
    }
    return color;
}

[[stage(fragment)]]
fn fragment_main([[builtin(position)]] pos: vec4<f32>) -> [[location(0)]] vec4<f32> {
    var ld = 2.0 * (pos.xy - uniforms.vp_size / 2.0) / uniforms.vp_size.y;
    var d = uniforms.looking * normalize(vec3<f32>(ld.x, -ld.y, 1.0));
    return raytrace(uniforms.pos, d);
}
