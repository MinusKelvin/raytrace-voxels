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

let NUM_AO_RAYS: i32 = 2;
let AO_RAY_DISTANCE: f32 = 4.0;
let NUM_SHADOW_RAYS: i32 = 1;
let P: f32 = 0.01;

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

let EPS: f32 = 0.0000001;

fn raycast(from: vec3<f32>, d: vec3<f32>, limit: f32) -> RaycastResult {
    var result: RaycastResult;
    result.hit = false;
    result.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    result.distance = 0.0;
    result.normal = vec3<f32>(0.0, 0.0, 0.0);

    var d = d;
    if (abs(d.x) < EPS) {
        if (d.x >= 0.0) {
            d.x = EPS;
        } else {
            d.x = -EPS;
        }
    }
    if (abs(d.y) < EPS) {
        if (d.y >= 0.0) {
            d.y = EPS;
        } else {
            d.y = -EPS;
        }
    }
    if (abs(d.z) < EPS) {
        if (d.z >= 0.0) {
            d.z = EPS;
        } else {
            d.z = -EPS;
        }
    }

    let bits_offset = u32(d.y < 0.0) * 12u;
    let step_f = sign(d);
    let t_delta = step_f / d;
    let fudge = (1.0 + step_f) / 2.0;
    var t_curr = t_delta * (fudge - fract(from) * step_f - 1.0);
    var p = vec3<i32>(floor(from));
    var empty_size = 1.0;
    let step = vec3<i32>(step_f);
    loop {
        let t_max = t_curr + t_delta * empty_size;
        let t = min(t_max.x, min(t_max.y, t_max.z));
        if (t > limit) {
            break;
        }
        var f = vec3<f32>(0.0, 0.0, 0.0);
        var step_size = (t - t_curr) / t_delta;
        if (t_max.x < t_max.y) {
            if (t_max.x < t_max.z) {
                step_size.x = empty_size;
                f.x = f32(step.x);
            } else {
                step_size.z = empty_size;
                f.z = f32(step.z);
            }
        } else {
            if (t_max.y < t_max.z) {
                step_size.y = empty_size;
                f.y = f32(step.y);
            } else {
                step_size.z = empty_size;
                f.z = f32(step.z);
            }
        }
        t_curr = t_curr + t_delta * floor(step_size);
        p = p + step * vec3<i32>(step_size);

        if (all(p >= vec3<i32>(0, 0, 0)) && all(p < uniforms.size)) {
            let voxel = space.voxels[(p.x * uniforms.size.y + p.y) * uniforms.size.z + p.z];
            if ((voxel & 0xFF000000u) != 0u) {
                result.hit = true;
                result.color = unpack4x8unorm(voxel);
                result.distance = t;
                result.normal = f;
                break;
            } else {
                empty_size = f32(extractBits(voxel, bits_offset, 12u));
            }
        } else {
            result.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            break;
        }
    }

    return result;
}

var<private> rng: u32 = 0;
fn pcg3d(b: vec3<u32>) -> vec3<u32> {
    rng = rng + 1u;
    var v = (b + rng)
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

fn random(p: vec3<f32>) -> f32 {
    return f32(pcg3d(bitcast<vec3<u32>>(p)).x >> 16u) / 65536.0;
}

fn random_cube(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(pcg3d(bitcast<vec3<u32>>(p)) >> vec3<u32>(16u)) / 32768.0 - 1.0;
}

fn random_direction(p: vec3<f32>) -> vec3<f32> {
    loop {
        let d = random_cube(p);
        if (dot(d, d) <= 1.0) {
            return normalize(d);
        }
    }
    return vec3<f32>(0.0);
}

fn random_disk(p: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    loop {
        let s = random_cube(p).xy;
        if (dot(s, s) <= 1.0) {
            let tan = normalize(cross(d, vec3<f32>(0.0, 0.0, 1.0)));
            let bitan = cross(d, tan);
            return s.x * tan + s.y * bitan;
        }
    }
    return vec3<f32>(0.0);
}

fn raytrace(from: vec3<f32>, d: vec3<f32>) -> vec4<f32> {
    var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var pos = from;
    var dir = d;
    var multiplier = 1.0;
    let dist = -log(1.0-random(d))/P;
    for (var depth = 0; depth < 4; depth = depth + 1) {
        var ray = raycast(pos, dir, dist);
        if (!ray.hit) {
            ray.distance = dist;
            ray.color = vec4<f32>(0.25, 0.4, 0.5, 1.0);
            let n = random_direction(pos + dir * dist);
            ray.normal = faceForward(n, reflect(n, d), -d);
        }
        let p = pos + dir * ray.distance;
        let in_bounds = all(p >= vec3<f32>(0.0, 0.0, 0.0)) && all(p < vec3<f32>(uniforms.size));
        var lighting = max(0.0, dot(uniforms.sun, -ray.normal));

        var shadow = 0.0;
        if (in_bounds) {
            for (var i = 0; i < NUM_SHADOW_RAYS; i = i + 1) {
                let sun = normalize(200.0 * uniforms.sun + random_disk(p, uniforms.sun));
                let shadowcast = raycast(p - ray.normal * 0.001, sun, 1.0 / 0.0);
                if (!shadowcast.hit) {
                    shadow = shadow + 1.0;
                }
            }
            shadow = shadow / f32(NUM_SHADOW_RAYS);
        } else {
            shadow = 1.0;
        }
        lighting = min(lighting, shadow);

        var ao = 0.0;
        if (in_bounds) {
            for (var i = 0; i < NUM_AO_RAYS; i = i + 1) {
                let d = random_direction(p);
                let aocast = raycast(
                    p - ray.normal * 0.001,
                    faceForward(d, reflect(d, ray.normal), -ray.normal),
                    AO_RAY_DISTANCE
                );
                if (aocast.hit) {
                    ao = ao + aocast.distance / AO_RAY_DISTANCE;
                } else {
                    ao = ao + 1.0;
                }
            }
            ao = ao / f32(NUM_AO_RAYS);
        } else {
            ao = 1.0;
        }

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
        break;
    }
    return color;
}

[[stage(fragment)]]
fn fragment_main([[builtin(position)]] pos: vec4<f32>) -> [[location(0)]] vec4<f32> {
    var ld = 2.0 * (pos.xy - uniforms.vp_size / 2.0) / uniforms.vp_size.y;
    var d = uniforms.looking * normalize(vec3<f32>(ld.x, -ld.y, 1.0));
    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var i = 0; i < 1; i = i + 1) {
        sum = sum + raytrace(uniforms.pos, d);
    }
    return sum / 1.0;
}
