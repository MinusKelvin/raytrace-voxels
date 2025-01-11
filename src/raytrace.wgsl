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

@group(2) @binding(0)
var wl_to_color_tex: texture_1d<f32>;
@group(2) @binding(1)
var wl_to_color_samp: sampler;

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
const PI: f32 = 3.1415926535;

fn raycast(start: vec3<f32>, d_: vec3<f32>, limit: f32) -> RaycastResult {
    var result: RaycastResult;
    result.hit = false;
    result.color = vec4<f32>(0.0);
    result.distance = 0.0;
    result.normal = vec3<f32>(0.0);

    var d = d_;
    if abs(d.x) < EPS {
        if d.x >= 0.0 {
            d.x = EPS;
        } else {
            d.x = -EPS;
        }
    }
    if abs(d.y) < EPS {
        if d.y >= 0.0 {
            d.y = EPS;
        } else {
            d.y = -EPS;
        }
    }
    if abs(d.z) < EPS {
        if d.z >= 0.0 {
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
        if t > limit {
            break;
        }
        var f = vec3<f32>(0.0, 0.0, 0.0);
        var step_size = (t - t_curr) / t_delta;
        if t_max.x < t_max.y {
            if t_max.x < t_max.z {
                step_size.x = empty_size;
                f.x = f32(step.x);
            } else {
                step_size.z = empty_size;
                f.z = f32(step.z);
            }
        } else {
            if t_max.y < t_max.z {
                step_size.y = empty_size;
                f.y = f32(step.y);
            } else {
                step_size.z = empty_size;
                f.z = f32(step.z);
            }
        }
        t_curr = t_curr + t_delta * floor(step_size);
        p = p + step * vec3<i32>(step_size);

        if any(p < vec3<i32>(0, 0, 0)) {
            empty_size = f32(-min(p.x, min(p.y, p.z)));
        } else if any(p >= uniforms.size) {
            let tmp = p - uniforms.size;
            empty_size = f32(max(tmp.x, max(tmp.y, tmp.z)) + 1);
        } else {
            let voxel = space.voxels[(p.x * uniforms.size.y + p.y) * uniforms.size.z + p.z];
            if (voxel & 0xFF000000u) != 0u {
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
    let angle = rand.y * 2.0 * PI;
    let p = vec2<f32>(r * cos(angle), r * sin(angle));

    let bitangent = normalize(cross(random(), n));
    let tangent = cross(bitangent, n);
    return tangent * p.x + bitangent * p.y;
}

fn cos_hemisphere(n: vec3<f32>) -> vec3<f32> {
    let disk = random_disk(n);
    return disk + n * sqrt(1.0 - dot(disk, disk));
}

fn cos_hemisphere_pdf(n: vec3<f32>, d: vec3<f32>) -> f32 {
    return max(dot(n, d), 0.0);
}

fn uniform_hemisphere(n: vec3<f32>) -> vec3<f32> {
    let rand = random();
    let z = rand.x;
    let angle = rand.y * 2.0 * PI;
    let p = vec2<f32>(cos(angle), sin(angle)) * sqrt(1 - z*z);

    let bitangent = normalize(cross(random(), n));
    let tangent = cross(bitangent, n);
    return tangent * p.x + bitangent * p.y + n * z;
}

fn brdf(outgoing: vec3<f32>, incoming: vec3<f32>, normal: vec3<f32>) -> f32 {
    return 1.0 / PI;
}

const SUN_ANGULAR_RADIUS: f32 = 0.535 * PI / 180;
const COS_SUN_RADIUS: f32 = cos(SUN_ANGULAR_RADIUS);
const SUN_COLOR: vec3<f32> = vec3<f32>(10000.0);

fn sample_sun() -> vec3<f32> {
    let rand = random();
    let z = rand.x * (1 - COS_SUN_RADIUS) + COS_SUN_RADIUS;
    let angle = rand.y * 2.0 * PI;
    let p = vec2<f32>(cos(angle), sin(angle)) * sqrt(1 - z*z);

    let bitangent = normalize(cross(random(), uniforms.sun));
    let tangent = cross(bitangent, uniforms.sun);
    return tangent * p.x + bitangent * p.y + uniforms.sun * z;
}

const SUN_WEIGHT: f32 = (1 - COS_SUN_RADIUS);

fn sun_pdf(d: vec3<f32>) -> f32 {
    if dot(d, uniforms.sun) > COS_SUN_RADIUS {
        return 1 / (1 - COS_SUN_RADIUS);
    } else {
        return 0.0;
    }
}

const PLANET_RADIUS: f32 = 1000000.0;
const FOG_HALFLIFE: f32 = 1000.0;
const FOG_RADIUS: f32 = PLANET_RADIUS + 5 * FOG_HALFLIFE;
const FOG_FACTOR: f32 = log(2.0) / FOG_HALFLIFE;

fn raycast_planet(start: vec3<f32>, dir: vec3<f32>, sea_level_density: f32) -> RaycastResult {
    var result: RaycastResult;
    result.hit = false;
    result.color = vec4<f32>(0.0);
    result.distance = 0.0;
    result.normal = vec3<f32>(0.0);

    var p = start + vec3<f32>(0.0, PLANET_RADIUS, 0.0);

    let b = 2.0 * dot(p, dir);
    let c = dot(p, p) - FOG_RADIUS*FOG_RADIUS;

    let det = b*b - 4.0 * c;
    if det < 0.0 {
        return result;
    }

    let t0 = max((-b - sqrt(det)) / 2.0, 0.0);
    let t1 = (-b + sqrt(det)) / 2.0;

    if t1 < 0 {
        return result;
    }

    let density_scaled = (t1 - t0) / 10.0 * sea_level_density;
    var y = -log(1.0 - random().x);

    for (var i = 0; i < 10; i += 1) {
        let a = f32(i + 1) / 10.0;
        let t_s = a * t0 + (1 - a) * t1;
        let altitude = length(p + dir * t_s) - PLANET_RADIUS;
        let d = density_scaled * exp(-altitude * FOG_FACTOR);
        if y < d {
            result.hit = true;
            result.normal = cos_hemisphere(-dir);
            result.distance = t_s - (y / d) * (t1 - t0);
            result.color = vec4<f32>(0.8, 0.8, 0.8, 0.0);
            break;
        }
        y -= d;
    }

    return result;
}

fn raytrace(start: vec3<f32>, d: vec3<f32>, wavelength: f32) -> vec3<f32> {
    let wlp1_cubed = (wavelength + 1.0) * (wavelength + 1.0) * (wavelength + 1.0);
    let density = 1.0e-3 / (wlp1_cubed * (wavelength + 1.0));
    var color = vec3<f32>(0.0);
    var light_color = textureSample(wl_to_color_tex, wl_to_color_samp, wavelength).rgb
        * 1.0 / (wlp1_cubed * (exp(0.1 / (wavelength + 1.0)) - 1.0))
        * vec3<f32>(1.0, 0.8, 1.0);
    var pos = start;
    var dir = d;

    // compute indirect lighting
    for (var depth = 0; ; depth += 1) {
        var ray = raycast_planet(pos, dir, density);
        var ray2 = raycast(pos, dir, select(1.0e12, ray.distance, ray.hit));
        if ray.hit && ray2.hit && ray2.distance < ray.distance || !ray.hit && ray2.hit {
            ray = ray2;
        } else if !ray.hit && !ray2.hit {
            // special-case initial ray hitting the sun
            if depth == 0 && (dot(dir, uniforms.sun) > COS_SUN_RADIUS) {
                color += light_color * SUN_COLOR;
            }
            // we escaped the scene, there is no more lighting contribution
            break;
        }

        pos = pos + dir * ray.distance;

        // L_r = L_direct + L_indirect

        // compute L_direct
        // let direct_dir = cos_hemisphere(-ray.normal);
        // let direct_dir = uniform_hemisphere(-ray.normal);
        let direct_dir = sample_sun();
        // compute visibility
        // only continue if direction is not into the surface
        if dot(direct_dir, -ray.normal) > 0.0 {
            // cast planet and block rays
            var sun_ray = raycast_planet(pos, direct_dir, density);
            if !sun_ray.hit {
                sun_ray = raycast(pos, direct_dir, select(1.0e12, sun_ray.distance, sun_ray.hit));
            }
            if !sun_ray.hit {
                // hit the sun, so add sun contribution
                color += light_color
                    * SUN_COLOR
                    * ray.color.rgb
                    * brdf(dir, direct_dir, -ray.normal)
                    * dot(-ray.normal, direct_dir) * 2
                    * PI
                    * SUN_WEIGHT;
            }
        }

        // compute L_indirect
        // emissive surface contribution (nb emission spectra == diffuse reflection spectra)
        if all(ray.color == vec4<f32>(1.0)) {
            color += light_color * 10.0 * ray.color.rgb * ray.color.a;
        }
        let indirect_dir = cos_hemisphere(-ray.normal);
        // let indirect_dir = uniform_hemisphere(-ray.normal);
        // L_indirect contribution is recursive, so we will recurse here.
        light_color *= ray.color.rgb
            * brdf(dir, indirect_dir, -ray.normal)
            // * dot(-ray.normal, indirect_dir) * 2
            * PI;
        dir = indirect_dir;

        // indirect lighting calculation recurses

        // russian roulette - basically importance sampling for bounce paths
        const T: f32 = 0.5;
        if all(light_color < vec3<f32>(T)) {
            if random().x < T {
                light_color *= 1.0/T;
            } else {
                break;
            }
        }
    }

    return color;
}

@fragment
fn fragment_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    rng = uniforms.rng ^ bitcast<vec3<u32>>(pos.xyz);

    var ld = 4.0 * (pos.xy - uniforms.vp_size / 2.0) / uniforms.vp_size.y;
    let px_size = vec2<f32>(dpdx(ld.x), dpdy(ld.y));
    var result = vec3<f32>(0.0);
    for (var i = 0; i < 1; i = i + 1) {
        let rng = random();
        let rnd = (rng.xy - 0.5) * px_size + ld;
        var d = uniforms.looking * normalize(vec3<f32>(rnd.x, -rnd.y, 1.0));
        result = result + raytrace(uniforms.pos, d, rng.z);
    }
    return vec4<f32>(result / 1.0, 0.0);
}
