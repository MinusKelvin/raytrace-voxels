use std::num::NonZeroU64;
use std::path::Path;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{EulerRot, IVec3, Mat3, Vec2, Vec3, Vec4};
use image::{Rgba32FImage, RgbaImage};
use rand::prelude::*;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

use crate::{Cell, ShowPipeline, Space, WgpuState};

pub struct FragmentRaytracer {
    pipeline: wgpu::RenderPipeline,
    uniform_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    space_buffer: wgpu::Buffer,
    space_group: wgpu::BindGroup,
    wl_to_color_group: wgpu::BindGroup,
    accumulator: wgpu::Texture,
    accumulator_desc: wgpu::TextureDescriptor<'static>,
    accumulator_view: wgpu::TextureView,
    show_pipeline: Option<ShowPipeline>,
    pub samples: usize,

    prev_yaw: f32,
    prev_pitch: f32,
    prev_pos: Vec3,
    prev_sun: Vec3,
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

        let wl_to_color_image = image::load_from_memory(include_bytes!("wl-to-color.png"))
            .unwrap()
            .to_rgba8();

        let wl_to_color_texture = gpu.device.create_texture_with_data(
            &gpu.queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: wl_to_color_image.width(),
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D1,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &wl_to_color_image,
        );

        let wl_to_color_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let wl_to_color_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D1,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let wl_to_color_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &wl_to_color_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &wl_to_color_texture.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&wl_to_color_sampler),
                },
            ],
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
                bind_group_layouts: &[
                    &uniform_group_layout,
                    &space_group_layout,
                    &wl_to_color_group_layout,
                ],
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        let accumulator = gpu.device.create_texture(&accumulator_desc);

        let accumulator_view = accumulator.create_view(&wgpu::TextureViewDescriptor::default());

        let show_pipeline = ShowPipeline::new(gpu, &accumulator_view);

        FragmentRaytracer {
            pipeline,
            uniform_group,
            uniform_buffer,
            space_buffer,
            space_group,
            wl_to_color_group,
            accumulator,
            accumulator_desc,
            accumulator_view,
            show_pipeline,
            samples: 0,
            prev_yaw: 0.0,
            prev_pitch: 0.0,
            prev_pos: Vec3::ZERO,
            prev_size: gpu.size,
            prev_sun: Vec3::ZERO,
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

    pub(super) fn show(&mut self, gpu: &WgpuState, target: &wgpu::TextureView) {
        self.show_pipeline
            .as_ref()
            .unwrap()
            .show(gpu, target, self.samples);
    }

    pub(super) fn sample(
        &mut self,
        gpu: &WgpuState,
        space: &Space,
        camera: Vec3,
        yaw: f32,
        pitch: f32,
        sun: Vec3,
    ) {
        if camera != self.prev_pos
            || yaw != self.prev_yaw
            || pitch != self.prev_pitch
            || gpu.size != self.prev_size
            || sun != self.prev_sun
        {
            self.samples = 0;
            self.prev_pitch = pitch;
            self.prev_yaw = yaw;
            self.prev_pos = camera;
            self.prev_size = gpu.size;
            self.prev_sun = sun;

            self.accumulator_desc.size.width = gpu.size.width;
            self.accumulator_desc.size.height = gpu.size.height;
            self.accumulator = gpu.device.create_texture(&self.accumulator_desc);
            self.accumulator_view = self
                .accumulator
                .create_view(&wgpu::TextureViewDescriptor::default());

            if let Some(show) = self.show_pipeline.as_mut() {
                show.update_accumulator(gpu, &self.accumulator_view);
            }
        }

        // if self.samples == 10 {
        //     std::thread::sleep(std::time::Duration::from_millis(1));
        //     return;
        // }

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
            sun,
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

        self.samples += 1;

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
        rp.set_bind_group(2, &self.wl_to_color_group, &[]);
        rp.draw(0..6, 0..1);

        drop(rp);

        let index = gpu.queue.submit([encoder.finish()]);

        gpu.device.poll(wgpu::Maintain::wait_for(index));
    }

    pub(super) fn save_image(&self, gpu: &WgpuState, path: impl AsRef<Path> + Send + 'static) {
        let samples = self.samples;
        let size = self.prev_size;

        let bytes_per_row = size.width * 16;
        let next_bpr = bytes_per_row.next_multiple_of(256);
        let effective_width = next_bpr as usize / 16;

        let buffer = Arc::new(gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: next_bpr as u64 * size.height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.accumulator,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(next_bpr),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
        );
        let index = gpu.queue.submit([encoder.finish()]);

        buffer
            .clone()
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                result.unwrap();
                let range = buffer.slice(..).get_mapped_range();
                let mut image = Rgba32FImage::new(size.width, size.height);
                let pixels: &[[f32; 4]] = bytemuck::cast_slice(&range);
                for (x, y, v) in image.enumerate_pixels_mut() {
                    let color = pixels[y as usize * effective_width + x as usize];
                    v.0 = color.map(|c| c / samples as f32);
                    v.0[3] = 1.0;
                }
                image.save(path).unwrap();
            });

        gpu.device.poll(wgpu::Maintain::wait_for(index));
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
