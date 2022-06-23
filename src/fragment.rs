use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use glam::{EulerRot, IVec4, Mat3, Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::{Space, WgpuState};

pub struct FragmentRaytracer {
    pipeline: wgpu::RenderPipeline,
    uniform_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    space_group: wgpu::BindGroup,
}

impl FragmentRaytracer {
    pub(super) fn new(gpu: &WgpuState, space: &Space) -> Self {
        let shader = gpu
            .device
            .create_shader_module(&wgpu::include_wgsl!("raytrace.wgsl"));

        let uniform_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<Uniforms>() as u64
                            ),
                        },
                        count: None,
                    }],
                });

        let uniform_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &uniform_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(uniform_buffer.as_entire_buffer_binding()),
            }],
        });

        let buffer: Vec<_> = space
            .voxels
            .iter()
            .map(|&v| match v {
                Some(a) => [
                    (a[0] * 255.0) as u8,
                    (a[1] * 255.0) as u8,
                    (a[2] * 255.0) as u8,
                    255,
                ],
                None => [0; 4],
            })
            .collect();

        let space_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&buffer),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let space_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let space_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &space_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(space_buffer.as_entire_buffer_binding()),
            }],
        });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&uniform_group_layout, &space_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vertex_main",
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fragment_main",
                    targets: &[wgpu::ColorTargetState {
                        format: gpu.config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }],
                }),
                multiview: None,
            });

        FragmentRaytracer {
            pipeline,
            uniform_group,
            uniform_buffer,
            space_group,
        }
    }

    pub(super) fn render(
        &mut self,
        gpu: &WgpuState,
        target: &wgpu::TextureView,
        space: &Space,
        camera: Vec3,
        yaw: f32,
        pitch: f32,
    ) -> wgpu::CommandBuffer {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let looking = Mat3::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);

        let uniforms = Uniforms {
            looking: [
                looking.x_axis.extend(0.0),
                looking.y_axis.extend(0.0),
                looking.z_axis.extend(0.0),
            ],
            pos: camera.extend(0.0),
            size: space.size.extend(0),
            sun: Vec3::new(0.1, 1.0, 0.2).normalize(),
            aspect: (gpu.size.width / 2) as f32 / gpu.size.height as f32,
        };
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });

        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.uniform_group, &[]);
        rp.set_bind_group(1, &self.space_group, &[]);
        rp.draw(0..6, 0..1);

        drop(rp);

        encoder.finish()
    }
}

#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
struct Uniforms {
    looking: [Vec4; 3],
    pos: Vec4,
    size: IVec4,
    sun: Vec3,
    aspect: f32,
}
