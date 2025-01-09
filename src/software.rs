use std::path::Path;

use glam::{EulerRot, IVec3, Vec3, Vec3A};
use image::{EncodableLayout, ImageBuffer, Rgba, RgbaImage};
use rayon::prelude::*;
use winit::dpi::PhysicalSize;

use crate::{Cell, ShowPipeline, Space, WgpuState};

pub struct SoftwareRaytracer {
    tex: wgpu::Texture,
    tex_view: wgpu::TextureView,
    show: Option<ShowPipeline>,
    size: PhysicalSize<u32>,
    img: image::ImageBuffer<Rgba<u8>, Vec<u8>>,
    pub samples: usize,
}

impl SoftwareRaytracer {
    pub(super) fn new(gpu: &WgpuState, _space: &Space) -> Self {
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: gpu.size.width,
                height: gpu.size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let tex_view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        let show = ShowPipeline::new(gpu, &tex_view);

        SoftwareRaytracer {
            tex,
            tex_view,
            show,
            size: gpu.size,
            img: ImageBuffer::new(gpu.size.width, gpu.size.height),
            samples: 1,
        }
    }

    pub(super) fn update_space(&mut self, gpu: &WgpuState, space: &Space) {}

    pub(super) fn sample(
        &mut self,
        gpu: &WgpuState,
        space: &Space,
        camera: Vec3,
        yaw: f32,
        pitch: f32,
        sun: Vec3,
    ) {
        if self.size != gpu.size {
            self.size = gpu.size;
            self.tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: gpu.size.width,
                    height: gpu.size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.tex_view = self
                .tex
                .create_view(&wgpu::TextureViewDescriptor::default());
            if let Some(show) = self.show.as_mut() {
                show.update_accumulator(gpu, &self.tex_view);
            }

            self.img = RgbaImage::new(gpu.size.width, gpu.size.height);
        }

        let looking = glam::Mat3A::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);

        let halfwidth = self.size.width as f32 / 2.0;
        let halfheight = self.size.height as f32 / 2.0;

        self.img
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
                    let color = raytrace(space, camera, d, sun, 4);
                    *pixel = Rgba([
                        (color[0] * 255.0) as u8,
                        (color[1] * 255.0) as u8,
                        (color[2] * 255.0) as u8,
                        0xff,
                    ])
                }
            });
    }

    pub(super) fn show(&mut self, gpu: &WgpuState, view: &wgpu::TextureView) {
        gpu.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            self.img.as_bytes(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * self.size.width),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: self.size.width,
                height: self.size.height,
                depth_or_array_layers: 1,
            },
        );

        self.show.as_ref().unwrap().show(gpu, view, self.samples);
    }

    pub(super) fn save_image(&self, gpu: &WgpuState, path: impl AsRef<Path> + Send + 'static) {
        self.img.save(path).unwrap();
    }
}

pub(super) fn raycast(
    space: &Space,
    from: Vec3,
    mut d: Vec3,
) -> Option<([f32; 3], f32, Vec3, IVec3)> {
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
    let down = (d.y < 0.0) as bool;
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
            Cell::Solid(color) => return Some((color, t, f, p)),
            Cell::Empty(space) => empty_size = space[down as usize] as f32,
        }
    }
}

fn raytrace(space: &Space, from: Vec3, d: Vec3, sun: Vec3, depth_limit: usize) -> [f32; 3] {
    if depth_limit == 0 {
        return [0.0; 3];
    }
    if let Some((mut c, t, f, _)) = raycast(space, from, d) {
        let p = from + d * t;
        let lighting = sun.dot(-f) / 2.0 + 1.0;
        // let shadow = raycast(space, p - f * 0.001, sun).is_none() as i32 as f32;
        // let lighting = lighting * (shadow / 2.0 + 0.5);
        c.iter_mut().for_each(|v| *v *= lighting);
        c
    } else {
        [0.0; 3]
    }
}
