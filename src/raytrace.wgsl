var<private> coords: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0,  1.0),
);

struct Uniforms {
    looking: mat3x3<f32>,
    pos: vec3<f32>,
    size: vec3<i32>,
    sun: vec3<f32>,
    vp_size: vec2<f32>,
    rng: vec3<u32>,
};

struct Space {
    voxels: array<u32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var<storage, read> space: Space;

@vertex
fn vertex_main(
    @builtin(vertex_index) vertex_index: u32
) -> @builtin(position) vec4<f32> {
    return vec4<f32>(coords[vertex_index], 0.0, 1.0);
}

struct RaycastResult {
    hit: bool,
    color: vec4<f32>,
    distance: f32,
    normal: vec3<f32>,
};

const EPS: f32 = 0.0000001;

fn raycast(start: vec3<f32>, d_: vec3<f32>, limit: f32) -> RaycastResult {
    var result: RaycastResult;
    result.hit = false;
    result.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    result.distance = 0.0;
    result.normal = vec3<f32>(0.0, 0.0, 0.0);

    var d = d_;
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
    var t_curr = t_delta * (fudge - fract(start) * step_f - 1.0);
    var p = vec3<i32>(floor(start));
    var empty_size = 1.0;
    let step = vec3<i32>(step_f);
    for (var i = 0; i < 512; i = i + 1) {
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

        if (any(p < vec3<i32>(0, 0, 0))) {
            empty_size = f32(-min(p.x, min(p.y, p.z)));
        } else if (any(p >= uniforms.size)) {
            let tmp = p - uniforms.size;
            empty_size = f32(max(tmp.x, max(tmp.y, tmp.z)) + 1);
        } else {
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
        }
    }

    return result;
}

var<private> rng: vec3<u32>;
fn pcg3d() -> vec3<u32> {
    var v = rng
        * vec3<u32>(1664525u, 1664525u, 1664525u)
        + 1013904223u;
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.x * v.z;
    v.z = v.z + v.y * v.x;
    v = v ^ v >> vec3<u32>(16u, 16u, 16u);
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.x * v.z;
    v.z = v.z + v.y * v.x;
    rng = v;
    return v;
}

fn random() -> vec3<f32> {
    return vec3<f32>(pcg3d() >> vec3<u32>(16u)) / 65536.0;
}

fn random_disk(n: vec3<f32>) -> vec3<f32> {
    let rand = random();
    let r = sqrt(rand.x);
    let angle = rand.y * 2.0 * 3.1415926535;
    let p = vec2<f32>(r * cos(angle), r * sin(angle));

    let bitangent = normalize(cross(random(), n));
    let tangent = cross(bitangent, n);
    return tangent * p.x + bitangent * p.y;
}

fn random_direction(n: vec3<f32>) -> vec3<f32> {
    let disk = random_disk(n);
    return disk + n * sqrt(1.0 - dot(disk, disk));
}

fn random_sphere() -> vec3<f32> {
    for (var i = 0; i < 100; i = i + 1) {
        let d = random() * 2.0 - 1.0;
        if (dot(d, d) <= 1.0) {
            return normalize(d);
        }
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

fn raytrace(start: vec3<f32>, d: vec3<f32>) -> vec4<f32> {
    var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var light_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    var pos = start;
    var dir = d;
    for (var depth = 0; depth < 5; depth = depth + 1) {
        let dist = -log(1.0-random().x)/0.01;

        var ray = raycast(pos, dir, dist);
        if (!ray.hit) {
            ray.normal = random_direction(-dir);
            ray.distance = dist;
            ray.color = vec4<f32>(1.0, 1.0, 1.0, 0.0);
        }

        pos = pos + dir * ray.distance;
        dir = random_direction(-ray.normal);
        light_color = light_color * ray.color;

        // let sundir = normalize(random_disk(uniforms.sun) + 20.0 * uniforms.sun);
        // let sundot = dot(-ray.normal, sundir);
        // if (sundot > 0.0) {
        //     let sunray = raycast(pos, sundir, 1.0 / 0.0);
        //     if (!sunray.hit) {
        //         color = color + light_color * sundot;
        //     }
        // }

        if (all(ray.color == vec4<f32>(1.0, 1.0, 1.0, 1.0))) {
            color = color + light_color * 100.0;
        }
    }
    return color;
}

@fragment
fn fragment_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    rng = uniforms.rng ^ bitcast<vec3<u32>>(pos.xyz);

    var ld = 2.0 * (pos.xy - uniforms.vp_size / 2.0) / uniforms.vp_size.y;
    let px_size = vec2<f32>(dpdx(ld.x), dpdy(ld.y));
    var result = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var i = 0; i < 1; i = i + 1) {
        let rnd = (random().xy - 0.5) * px_size + ld;
        var d = uniforms.looking * normalize(vec3<f32>(rnd.x, -rnd.y, 1.0));
        result = result + raytrace(uniforms.pos, d);
    }
    return result / 1.0;
}
