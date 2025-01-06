use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use glam::{EulerRot, IVec3, Mat3, Vec2, Vec3, Vec4};
use rand::prelude::*;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

use crate::{Cell, Space, WgpuState};

pub struct FragmentRaytracer {
    pipeline: wgpu::RenderPipeline,
    uniform_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    space_buffer: wgpu::Buffer,
    space_group: wgpu::BindGroup,
    accumulator: wgpu::Texture,
    accumulator_desc: wgpu::TextureDescriptor<'static>,
    accumulator_view: wgpu::TextureView,
    copy_bind_group_layout: wgpu::BindGroupLayout,
    copy_bind_group: wgpu::BindGroup,
    copy_pipeline: wgpu::RenderPipeline,
    samples_buffer: wgpu::Buffer,
    pub samples: f32,

    prev_yaw: f32,
    prev_pitch: f32,
    prev_pos: Vec3,
    prev_size: PhysicalSize<u32>,
}

impl FragmentRaytracer {
    pub(super) fn new(gpu: &WgpuState, space: &Space) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("raytrace.wgsl"));

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
                Cell::Solid(a) => [
                    (a[0] * 255.0) as u8,
                    (a[1] * 255.0) as u8,
                    (a[2] * 255.0) as u8,
                    255,
                ],
                Cell::Empty([up, down]) => (up | down << 12).to_le_bytes(),
            })
            .collect();

        let space_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&buffer),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
                resource: space_buffer.as_entire_binding(),
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
                    entry_point: None,
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: None,
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview: None,
                cache: None,
            });

        let accumulator_desc = wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: gpu.size.width,
                height: gpu.size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let accumulator = gpu.device.create_texture(&accumulator_desc);

        let accumulator_view = accumulator.create_view(&wgpu::TextureViewDescriptor::default());

        let samples_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let copy_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let copy_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &copy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&accumulator_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: samples_buffer.as_entire_binding(),
                },
            ],
        });

        let cp_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&copy_bind_group_layout],
                push_constant_ranges: &[],
            });

        let copy_shader = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("copy.wgsl"));

        let copy_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&cp_layout),
                vertex: wgpu::VertexState {
                    module: &copy_shader,
                    entry_point: None,
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &copy_shader,
                    entry_point: None,
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview: None,
                cache: None,
            });

        FragmentRaytracer {
            pipeline,
            uniform_group,
            uniform_buffer,
            space_buffer,
            space_group,
            accumulator,
            accumulator_desc,
            accumulator_view,
            copy_bind_group_layout,
            copy_bind_group,
            copy_pipeline,
            samples_buffer,
            samples: 0.0,
            prev_yaw: 0.0,
            prev_pitch: 0.0,
            prev_pos: Vec3::ZERO,
            prev_size: gpu.size,
        }
    }

    pub(super) fn update_space(&mut self, gpu: &WgpuState, space: &Space) {
        let buffer: Vec<_> = space
            .voxels
            .iter()
            .map(|&v| match v {
                Cell::Solid(a) => [
                    (a[0] * 255.0) as u8,
                    (a[1] * 255.0) as u8,
                    (a[2] * 255.0) as u8,
                    255,
                ],
                Cell::Empty([up, down]) => (up | down << 12).to_le_bytes(),
            })
            .collect();

        gpu.queue
            .write_buffer(&self.space_buffer, 0, bytemuck::cast_slice(&buffer));

        self.prev_pitch = f32::NAN;
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
        if camera != self.prev_pos
            || yaw != self.prev_yaw
            || pitch != self.prev_pitch
            || gpu.size != self.prev_size
        {
            self.samples = 0.0;
            self.prev_pitch = pitch;
            self.prev_yaw = yaw;
            self.prev_pos = camera;
            self.prev_size = gpu.size;

            self.accumulator_desc.size.width = gpu.size.width;
            self.accumulator_desc.size.height = gpu.size.height;
            self.accumulator = gpu.device.create_texture(&self.accumulator_desc);

            self.accumulator_view = self
                .accumulator
                .create_view(&wgpu::TextureViewDescriptor::default());

            self.copy_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.copy_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.accumulator_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.samples_buffer.as_entire_binding(),
                    },
                ],
            });
        }

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
            pos: camera,
            size: space.size,
            sun: Vec3::new(0.1, 1.0, 0.2).normalize(),
            vp_size: Vec2::new(gpu.size.width as f32, gpu.size.height as f32),
            rng: thread_rng().gen(),
            _padding0: 0,
            _padding1: 0,
            _padding2: 0,
            _padding3: [0; 2],
            _padding4: 0,
        };
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        self.samples += 1.0;
        gpu.queue
            .write_buffer(&self.samples_buffer, 0, bytemuck::bytes_of(&self.samples));

        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.accumulator_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.uniform_group, &[]);
        rp.set_bind_group(1, &self.space_group, &[]);
        rp.draw(0..6, 0..1);

        drop(rp);

        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rp.set_pipeline(&self.copy_pipeline);
        rp.set_bind_group(0, &self.copy_bind_group, &[]);
        rp.draw(0..6, 0..1);

        drop(rp);

        encoder.finish()
    }
}

#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
struct Uniforms {
    looking: [Vec4; 3],
    pos: Vec3,
    _padding0: u32,
    size: IVec3,
    _padding1: u32,
    sun: Vec3,
    _padding2: u32,
    vp_size: Vec2,
    _padding3: [u32; 2],
    rng: [u32; 3],
    _padding4: u32,
}
