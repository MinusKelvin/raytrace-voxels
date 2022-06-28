use std::collections::VecDeque;
use std::num::NonZeroU32;

use glam::{EulerRot, IVec3, Vec3, Vec3A};
use image::{EncodableLayout, Rgba};
use rayon::prelude::*;
use winit::dpi::PhysicalSize;

use crate::{Space, WgpuState};

pub struct SoftwareRaytracer {
    tex: wgpu::Texture,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    tex_view: wgpu::TextureView,
    size: PhysicalSize<u32>,

    space: Space<Cell>,
}

impl SoftwareRaytracer {
    pub(super) fn new(gpu: &WgpuState, space: &Space) -> Self {
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: gpu.size.width / 2,
                height: gpu.size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        let tex_view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    }],
                });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&tex_view),
            }],
        });

        let cp_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let copy_shader = gpu
            .device
            .create_shader_module(&wgpu::include_wgsl!("copy.wgsl"));

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&cp_layout),
                vertex: wgpu::VertexState {
                    module: &copy_shader,
                    entry_point: "vertex_main",
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &copy_shader,
                    entry_point: "fragment_main",
                    targets: &[wgpu::ColorTargetState {
                        format: gpu.config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }],
                }),
                multiview: None,
            });

        let mut df_space = Space::new_from(space.size, Cell::Empty(u32::MAX));
        let mut decreases = VecDeque::new();
        for x in 0..space.size.x {
            for y in 0..space.size.y {
                for z in 0..space.size.z {
                    let p = IVec3::new(x, y, z);
                    if let Some(c) = space.get(p).unwrap() {
                        df_space.set(p, Cell::Solid(c));
                        decreases.push_back((p, 0));
                    }
                }
            }
        }

        while let Some((p, v)) = decreases.pop_front() {
            let propogate;
            match df_space.get(p).unwrap() {
                Cell::Solid(_) => {
                    propogate = v == 0;
                }
                Cell::Empty(a) => {
                    propogate = v < a;
                    if propogate {
                        df_space.set(p, Cell::Empty(v));
                    }
                }
            }
            if propogate {
                for x in p.x - 1..=p.x + 1 {
                    for y in p.y - 1..=p.y + 1 {
                        for z in p.z - 1..=p.z + 1 {
                            let p = IVec3::new(x, y, z);
                            if df_space.idx(p).is_some() {
                                decreases.push_back((p, v + 1));
                            }
                        }
                    }
                }
            }
        }

        SoftwareRaytracer {
            tex,
            pipeline,
            bind_group,
            bind_group_layout,
            tex_view,
            size: gpu.size,

            space: df_space,
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
        if self.size != gpu.size {
            self.size = gpu.size;
            self.tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: gpu.size.width / 2,
                    height: gpu.size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            });
            self.tex_view = self
                .tex
                .create_view(&wgpu::TextureViewDescriptor::default());
            self.bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.tex_view),
                }],
            });
        }

        let looking = glam::Mat3A::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);

        let halfwidth = (self.size.width / 2) as f32 / 2.0;
        let halfheight = self.size.height as f32 / 2.0;
        let sun = Vec3::new(0.1, 1.0, 0.2).normalize();

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut raycast_image =
            image::ImageBuffer::<Rgba<u8>, Vec<u8>>::new(self.size.width / 2, self.size.height);
        raycast_image
            .enumerate_rows_mut()
            .par_bridge()
            .for_each(|(_, row)| {
                for (x, y, pixel) in row {
                    let d = Vec3::from(
                        looking
                            * Vec3A::new(
                                (x as f32 - halfwidth) / halfheight,
                                (halfheight - y as f32) / halfheight,
                                1.0,
                            )
                            .normalize(),
                    );
                    let color = raytrace(&self.space, camera, d, sun, 4);
                    *pixel = Rgba([
                        (color[0] * 255.0) as u8,
                        (color[1] * 255.0) as u8,
                        (color[2] * 255.0) as u8,
                        0xff,
                    ])
                }
            });

        gpu.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            raycast_image.as_bytes(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * (self.size.width / 2)),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: self.size.width / 2,
                height: self.size.height,
                depth_or_array_layers: 1,
            },
        );

        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });

        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.draw(0..6, 0..1);

        drop(rp);

        encoder.finish()
    }
}

fn raycast(space: &Space<Cell>, from: Vec3, mut d: Vec3) -> Option<([f32; 3], f32, Vec3)> {
    if d.x.abs() < f32::EPSILON {
        d.x = match d.x >= 0.0 {
            true => f32::EPSILON,
            false => -f32::EPSILON,
        };
    }
    if d.y.abs() < f32::EPSILON {
        d.y = match d.y >= 0.0 {
            true => f32::EPSILON,
            false => -f32::EPSILON,
        };
    }
    if d.z.abs() < f32::EPSILON {
        d.z = match d.z >= 0.0 {
            true => f32::EPSILON,
            false => -f32::EPSILON,
        };
    }

    let step = d.signum();
    let t_delta = step / d;
    let fudge = (1.0 + step) / 2.0;
    let mut t_curr = t_delta * (fudge - from.fract() * step - 1.0);
    let mut p = from.floor().as_ivec3();
    let step = step.as_ivec3();
    let mut empty_size = 1.0;
    loop {
        let t_max = t_curr + t_delta * empty_size;
        let t = t_max.min_element();
        let mut f = Vec3::ZERO;
        if t_max.x < t_max.y {
            if t_max.x < t_max.z {
                let mut step_size = (t_max.x - t_curr) / t_delta;
                step_size.x = empty_size;
                t_curr += t_delta * step_size.floor();
                p += step * step_size.as_ivec3();
                f.x = step.x as f32;
            } else {
                let mut step_size = (t_max.z - t_curr) / t_delta;
                step_size.z = empty_size;
                t_curr += t_delta * step_size.floor();
                p += step * step_size.as_ivec3();
                f.z = step.z as f32;
            }
        } else {
            if t_max.y < t_max.z {
                let mut step_size = (t_max.y - t_curr) / t_delta;
                step_size.y = empty_size;
                t_curr += t_delta * step_size.floor();
                p += step * step_size.as_ivec3();
                f.y = step.y as f32;
            } else {
                let mut step_size = (t_max.z - t_curr) / t_delta;
                step_size.z = empty_size;
                t_curr += t_delta * step_size.floor();
                p += step * step_size.as_ivec3();
                f.z = step.z as f32;
            }
        }

        match space.get(p)? {
            Cell::Solid(color) => return Some((color, t, f)),
            Cell::Empty(space) => empty_size = space as f32,
        }
    }
}

fn raytrace(space: &Space<Cell>, from: Vec3, d: Vec3, sun: Vec3, depth_limit: usize) -> [f32; 3] {
    if depth_limit == 0 {
        return [0.0; 3];
    }
    if let Some((mut c, t, f)) = raycast(space, from, d) {
        let is_reflective = c == [1.0; 3];
        let p = from + d * t;
        let lighting = sun.dot(-f).max(0.0) / 2.0 + 0.5;
        let shadow = raycast(space, p - f * 0.001, sun).is_none() as i32 as f32;
        let lighting = lighting.min(shadow / 2.0 + 0.5);
        c.iter_mut().for_each(|v| *v *= lighting);
        if is_reflective {
            let reflected = raytrace(
                space,
                p - f * 0.001,
                d - 2.0 * d.project_onto(f),
                sun,
                depth_limit - 1,
            );
            for (v, &r) in c.iter_mut().zip(reflected.iter()) {
                *v = *v / 2.0 + r / 2.0;
            }
        }
        c
    } else {
        [0.0; 3]
    }
}

#[derive(Clone, Copy)]
enum Cell {
    Solid([f32; 3]),
    Empty(u32),
}
