struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
};

var<private> coords: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0,  1.0),
);

@group(0) @binding(0)
var blocks: texture_2d<f32>;

struct Uniforms {
    samples: f32,
};

@group(0) @binding(1)
var<uniform> samples: Uniforms;

@vertex
fn vertex_main(
    @builtin(vertex_index) vertex_index: u32
) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(coords[vertex_index], 0.0, 1.0);
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureLoad(blocks, vec2<i32>(in.pos.xy), 0) / samples.samples;
}
