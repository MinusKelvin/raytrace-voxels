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
    sun: vec3<f32>,
    vp_size: vec2<f32>,
    rng: vec3<u32>,
    root: u32,
    height: u32,
};

struct Node {
    children: array<u32, 8>
}

struct Space {
    nodes: array<Node>,
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

const EPS: f32 = 1.0e-6;
const PI: f32 = 3.1415926535;

fn hmax(v: vec3<f32>) -> f32 {
    return max(v.x, max(v.y, v.z));
}

fn hmin(v: vec3<f32>) -> f32 {
    return min(v.x, min(v.y, v.z));
}

fn to_bits(v: vec3<bool>) -> u32 {
    return u32(v.x) | u32(v.y) << 1u | u32(v.z) << 2u;
}

var<private> error: bool;

fn raycast(start_: vec3<f32>, d_: vec3<f32>, distance: f32) -> RaycastResult {
    var result: RaycastResult;
    result.hit = false;
    result.color = vec4<f32>(0.0);
    result.distance = 0.0;
    result.normal = vec3<f32>(0.0);

    let flip = d_ < vec3<f32>(0.0);
    let d_sign = sign(d_);
    let mirror_mask = to_bits(flip);
    let d = max(abs(d_), vec3<f32>(EPS));
    let space_bound = vec3<f32>(f32(1u << uniforms.height));
    let start = select(start_, space_bound - start_, flip);

    let enter = -start / d;
    var t = max(hmax(enter), 0.0);
    var enter_dir = vec3<bool>(false);
    if enter.x == t {
        enter_dir.x = true;
    } else if enter.y == t {
        enter_dir.y = true;
    } else if enter.z == t {
        enter_dir.z = true;
    }

    var height = uniforms.height;
    var stack_node: array<u32, 32>;
    var stack_t_midplanes: array<vec3<f32>, 32>;
    var stack_subvoxel: array<vec3<bool>, 32>;
    var stack_t_end: array<f32, 32>;
    var stack_offset: array<vec3<f32>, 32>;

    stack_node[height] = uniforms.root;
    stack_t_end[height] = min(hmin((space_bound - start) / d), distance);
    stack_t_midplanes[height] = (vec3<f32>(f32(1u << height - 1)) - start) / d;
    stack_subvoxel[height] = stack_t_midplanes[height] < vec3<f32>(t);
    if stack_t_end[height] < t {
        return result;
    }
    height -= 1u;
    stack_node[height] = 0xFFFFFFFFu;

    var iters = 0;
    while height <= uniforms.height {
        if iters >= 10000 {
            error = true;
            break;
        }
        iters += 1;
        if stack_node[height] == 0xFFFFFFFFu {
            let subvoxel = stack_subvoxel[height + 1u];
            let p_midplanes = vec3<f32>(f32(1u << height));
            let offset = stack_offset[height + 1u] + select(vec3<f32>(0.0), p_midplanes, subvoxel);

            let node = space.nodes[stack_node[height + 1u]]
                .children[to_bits(subvoxel) ^ mirror_mask];
            if node == 0xFFFFFFFFu {
                height += 1u;
                continue;
            };

            if height == 0 {
                if t == 0.0 {
                    height += 1u;
                    continue;
                }
                result.hit = true;
                result.color = vec4<f32>(
                    bitcast<f32>(space.nodes[node].children[0]),
                    bitcast<f32>(space.nodes[node].children[1]),
                    bitcast<f32>(space.nodes[node].children[2]),
                    1.0,
                );
                result.distance = t;
                result.normal = -select(vec3<f32>(0.0), d_sign, enter_dir);
                break;
            }

            let midplanes = vec3<f32>(f32(1u << height - 1u));
            stack_t_midplanes[height] = (offset + midplanes - start) / d;
            stack_t_end[height] = min(hmin((offset + midplanes * 2.0 - start) / d), distance);
            stack_node[height] = node;
            stack_offset[height] = offset;
            stack_subvoxel[height] = stack_t_midplanes[height] < vec3<f32>(t);
            height -= 1u;
            stack_node[height] = 0xFFFFFFFFu;
            continue;
        };

        let t_next = select(
            stack_t_midplanes[height],
            vec3<f32>(stack_t_end[height]),
            stack_subvoxel[height]
        );
        let min = hmin(t_next);

        if min == stack_t_end[height] {
            height += 1u;
            continue;
        } else if min == t_next.x {
            t = t_next.x;
            stack_subvoxel[height].x = true;
            enter_dir = vec3<bool>(true, false, false);
        } else if min == t_next.y {
            t = t_next.y;
            stack_subvoxel[height].y = true;
            enter_dir = vec3<bool>(false, true, false);
        } else if min == t_next.z {
            t = t_next.z;
            stack_subvoxel[height].z = true;
            enter_dir = vec3<bool>(false, false, true);
        }

        height -= 1u;
        stack_node[height] = 0xFFFFFFFFu;
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

const PLANET_RADIUS: f32 = 6371000.0;
const FOG_HALFLIFE: f32 = 10400.0 / log(2.0);
const FOG_RADIUS: f32 = PLANET_RADIUS + 10 * FOG_HALFLIFE;
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
    var t1 = (-b + sqrt(det)) / 2.0;

    if t1 < 0 {
        return result;
    }

    let c_planet = dot(p, p) - PLANET_RADIUS*PLANET_RADIUS;
    let det_planet = b*b - 4.0 * c_planet;
    if det_planet >= 0.0 {
        let t0_planet = (-b - sqrt(det_planet)) / 2.0;
        if t0_planet > 0.0 {
            t1 = t0_planet;
            result.hit = true;
            result.distance = t1;
            result.normal = normalize(p + dir * t1);
            result.color = vec4<f32>(0.25, 0.25, 0.25, 0.0);
        }
    }

    const N: i32 = 1000;

    let density_scaled = (t1 - t0) / f32(N) * sea_level_density;
    var y = -log(1.0 - random().x);

    for (var i = 0; i < N; i += 1) {
        let a1 = f32(i + 1) / f32(N);
        let a0 = f32(i) / f32(N);
        let t_s0 = a0 * t0 + (1 - a0) * t1;
        let t_s1 = a1 * t0 + (1 - a1) * t1;
        let altitude = length(p + dir * t_s1) - PLANET_RADIUS;
        let d = density_scaled * exp(-altitude * FOG_FACTOR);
        if y < d {
            result.hit = true;
            result.normal = cos_hemisphere(dir);
            result.distance = (y/d) * t_s0 + (1 - y/d) * t_s1;
            result.color = vec4<f32>(1.0, 1.0, 1.0, 0.0);
            break;
        }
        y -= d;
    }

    return result;
}

fn raytrace(start: vec3<f32>, d: vec3<f32>, wavelength: f32) -> vec3<f32> {
    let wl = wavelength*400.0e-9 + 400.0e-9;
    let density = 8.346829234302236e-05 / (7.512000000000001e+25*wl*wl*wl*wl);
    let wlp1_cubed = (wavelength + 1.0) * (wavelength + 1.0) * (wavelength + 1.0);
    var color = vec3<f32>(0.0);
    var light_color = textureSample(wl_to_color_tex, wl_to_color_samp, wavelength).rgb
        * 1.0 / (wlp1_cubed * (exp(0.1 / (wavelength + 1.0)) - 1.0))
        * vec3<f32>(1.0, 0.8, 1.0);
    var pos = start;
    var dir = d;

    // compute indirect lighting
    for (var depth = 0; ; depth += 1) {
        if depth >= 25 || error {
            error = true;
            break;
        }

        var ray = raycast_planet(pos, dir, density);
        var ray2 = raycast(pos, dir, select(1.0e12, ray.distance, ray.hit));
        if ray2.hit {
            ray = ray2;
            ray.distance = ray2.distance;
        }
        if !ray.hit {
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
        // let direct_dir = cos_hemisphere(ray.normal);
        // let direct_dir = uniform_hemisphere(ray.normal);
        let sun_dir = sample_sun();
        // compute visibility
        // only continue if direction is not into the surface
        if dot(sun_dir, ray.normal) > 0.0 {
            // cast planet and block rays
            var sun_ray = raycast_planet(pos, sun_dir, density);
            if !sun_ray.hit {
                sun_ray = raycast(pos, sun_dir, 1.0e12);
            }
            if !sun_ray.hit {
                // hit the sun, so add sun contribution
                color += light_color
                    * SUN_COLOR
                    * ray.color.rgb
                    * brdf(dir, sun_dir, ray.normal)
                    * dot(ray.normal, sun_dir) * 2
                    * PI
                    * SUN_WEIGHT;
            }
        }

        // compute L_indirect
        // emissive surface contribution (nb emission spectra == diffuse reflection spectra)
        if all(ray.color == vec4<f32>(1.0)) {
            color += light_color * 10.0 * ray.color.rgb * ray.color.a;
        }
        let indirect_dir = cos_hemisphere(ray.normal);
        // let indirect_dir = uniform_hemisphere(ray.normal);
        // L_indirect contribution is recursive, so we will recurse here.
        light_color *= ray.color.rgb
            * brdf(dir, indirect_dir, ray.normal)
            // * dot(ray.normal, indirect_dir) * 2
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

    var ld = 2.0 * (pos.xy - uniforms.vp_size / 2.0) / uniforms.vp_size.y;
    let px_size = vec2<f32>(dpdx(ld.x), dpdy(ld.y));
    var result = vec3<f32>(0.0);
    for (var i = 0; i < 1; i = i + 1) {
        let rng = random();
        let rnd = (rng.xy - 0.5) * px_size + ld;
        var d = uniforms.looking * normalize(vec3<f32>(rnd.x, -rnd.y, 1.0));
        result = result + raytrace(uniforms.pos, d, rng.z);
        // result = raycast(uniforms.pos, d, 1.0e24).color.rgb;
        if error {
            return vec4<f32>(1.0e9, -1.0e9, 1.0e9, 0.0);
        }
    }
    return vec4<f32>(result / 1.0, 0.0);
}
