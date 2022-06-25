use std::num::NonZeroU32;

use glam::{BVec3, EulerRot, IVec3, Vec3, Vec3A};
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

    svo: Svo,
    scale: i32,
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

        let scale = (space.size.max_element() as u32).next_power_of_two() as i32;
        let svo = Svo::build(space, IVec3::ZERO, scale);

        SoftwareRaytracer {
            tex,
            pipeline,
            bind_group,
            bind_group_layout,
            tex_view,
            size: gpu.size,
            svo,
            scale,
        }
    }

    pub(super) fn render(
        &mut self,
        gpu: &WgpuState,
        target: &wgpu::TextureView,
        _space: &Space,
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
                    let color = raytrace(&self.svo, self.scale, camera, d, sun, 4);
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

fn raycast(svo: &Svo, scale: i32, mut from: Vec3, mut d: Vec3) -> Option<([f32; 3], f32, Vec3)> {
    let mut a = BVec3::new(false, false, false);
    if d.x < 0.0 {
        a.x = true;
        d.x = -d.x;
        from.x = scale as f32 - from.x;
    }
    if d.y < 0.0 {
        a.y = true;
        d.y = -d.y;
        from.y = scale as f32 - from.y;
    }
    if d.z < 0.0 {
        a.z = true;
        d.z = -d.z;
        from.z = scale as f32 - from.z;
    }

    if d.x < f32::EPSILON {
        d.x = f32::EPSILON;
    }
    if d.y < f32::EPSILON {
        d.y = f32::EPSILON;
    }
    if d.z < f32::EPSILON {
        d.z = f32::EPSILON;
    }

    let t0 = -from / d;
    let t1 = (Vec3::splat(scale as f32) - from) / d;

    if t0.max_element() < t1.min_element() {
        iter_raycast_impl(svo, a, t0, t1).map(|(c, t, n)| (c, t, Vec3::select(a, -n, n)))
    } else {
        None
    }
}

fn min_spot(v: Vec3) -> BVec3 {
    if v.x < v.y {
        if v.x < v.z {
            BVec3::new(true, false, false)
        } else {
            BVec3::new(false, false, true)
        }
    } else {
        if v.y < v.z {
            BVec3::new(false, true, false)
        } else {
            BVec3::new(false, false, true)
        }
    }
}

fn xor(a: BVec3, b: BVec3) -> BVec3 {
    BVec3::new(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z)
}

fn first_child(t0: Vec3, tm: Vec3) -> BVec3 {
    if t0.x > t0.y {
        if t0.x > t0.z {
            BVec3::new(false, tm.y < t0.x, tm.z < t0.x)
        } else {
            BVec3::new(tm.x < t0.z, tm.y < t0.z, false)
        }
    } else {
        if t0.y > t0.z {
            BVec3::new(tm.x < t0.y, false, tm.z < t0.y)
        } else {
            BVec3::new(tm.x < t0.z, tm.y < t0.z, false)
        }
    }
}

fn iter_raycast_impl(svo: &Svo, a: BVec3, t0: Vec3, t1: Vec3) -> Option<([f32; 3], f32, Vec3)> {
    let tm = (t0 + t1) * 0.5;

    let mut stack = Vec::with_capacity(32);
    stack.push((svo, t0, t1, first_child(t0, tm)));

    while let Some(&mut (svo, t0, t1, ref mut child)) = stack.last_mut() {
        if t1.x < 0.0 || t1.y < 0.0 || t1.z < 0.0 {
            stack.pop();
            continue;
        }
        let octants = match svo {
            Svo::Transparent => {
                stack.pop();
                continue;
            }
            Svo::Color(c) => {
                return Some((
                    *c,
                    t0.max_element(),
                    if t0.x > t0.y {
                        match t0.x > t0.z {
                            true => Vec3::X,
                            false => Vec3::Z,
                        }
                    } else {
                        match t0.y > t0.z {
                            true => Vec3::Y,
                            false => Vec3::Z,
                        }
                    },
                ));
            }
            Svo::Recurse(octants) => octants,
        };

        let tm = (t0 + t1) * 0.5;
        let diff = min_spot(Vec3::select(*child, t1, tm));

        let nt0 = Vec3::select(*child, tm, t0);
        let nt1 = Vec3::select(*child, t1, tm);
        let ntm = (nt0 + nt1) * 0.5;
        let next_frame = (
            &octants[xor(*child, a).bitmask() as usize],
            nt0,
            nt1,
            first_child(nt0, ntm),
        );

        if (*child & diff).any() {
            stack.pop();
        } else {
            *child |= diff;
        }
        stack.push(next_frame);
    }

    None
}

fn raycast_impl(svo: &Svo, a: BVec3, t0: Vec3, t1: Vec3) -> Option<([f32; 3], f32, Vec3)> {
    if t1.x < 0.0 || t1.y < 0.0 || t1.z < 0.0 {
        return None;
    }

    let octants = match svo {
        Svo::Transparent => return None,
        Svo::Color(c) => {
            return Some((
                *c,
                t0.max_element(),
                if t0.x > t0.y {
                    match t0.x > t0.z {
                        true => Vec3::X,
                        false => Vec3::Z,
                    }
                } else {
                    match t0.y > t0.z {
                        true => Vec3::Y,
                        false => Vec3::Z,
                    }
                },
            ))
        }
        Svo::Recurse(o) => o,
    };

    let tm = (t0 + t1) * 0.5;

    let mut child = first_child(t0, tm);

    loop {
        if let Some(result) = raycast_impl(
            &octants[xor(child, a).bitmask() as usize],
            a,
            Vec3::select(child, tm, t0),
            Vec3::select(child, t1, tm),
        ) {
            return Some(result);
        }
        let diff = min_spot(Vec3::select(child, t1, tm));
        if (child & diff).any() {
            return None;
        }
        child |= diff;
    }
}

fn raytrace(svo: &Svo, scale: i32, from: Vec3, d: Vec3, sun: Vec3, depth_limit: usize) -> [f32; 3] {
    if depth_limit == 0 {
        return [0.0; 3];
    }
    if let Some((mut c, t, f)) = raycast(svo, scale, from, d) {
        let is_reflective = c == [1.0; 3];
        let p = from + d * t;
        let lighting = sun.dot(-f).max(0.0) / 2.0 + 0.5;
        let shadow = raycast(svo, scale, p - f * 0.001, sun).is_none() as i32 as f32;
        let lighting = lighting.min(shadow / 2.0 + 0.5);
        c.iter_mut().for_each(|v| *v *= lighting);
        if is_reflective {
            let reflected = raytrace(
                svo,
                scale,
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

#[derive(Debug)]
enum Svo {
    Recurse(Box<[Svo; 8]>),
    Color([f32; 3]),
    Transparent,
}

impl Svo {
    fn build(space: &Space, min: IVec3, size: i32) -> Svo {
        if size == 1 {
            match space.get(min) {
                Some(Some(color)) => Svo::Color(color),
                Some(None) | None => Svo::Transparent,
            }
        } else {
            let ns = size / 2;
            let octants = [
                Svo::build(space, min + IVec3::new(0, 0, 0), ns),
                Svo::build(space, min + IVec3::new(ns, 0, 0), ns),
                Svo::build(space, min + IVec3::new(0, ns, 0), ns),
                Svo::build(space, min + IVec3::new(ns, ns, 0), ns),
                Svo::build(space, min + IVec3::new(0, 0, ns), ns),
                Svo::build(space, min + IVec3::new(ns, 0, ns), ns),
                Svo::build(space, min + IVec3::new(0, ns, ns), ns),
                Svo::build(space, min + IVec3::new(ns, ns, ns), ns),
            ];
            if let Svo::Color(target) = octants[0] {
                if octants
                    .iter()
                    .all(|v| matches!(v, &Svo::Color(c) if c == target))
                {
                    return Svo::Color(target);
                }
            }
            if octants.iter().all(|v| matches!(v, Svo::Transparent)) {
                Svo::Transparent
            } else {
                Svo::Recurse(Box::new(octants))
            }
        }
    }
}
