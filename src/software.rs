use std::path::Path;

use glam::{BVec3, EulerRot, IVec3, Vec3, Vec3A};
use image::{EncodableLayout, ImageBuffer, Rgba, RgbaImage};
use rayon::prelude::*;
use winit::dpi::PhysicalSize;

use crate::svo::{Node, SvoCell, SvoSpace};
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
    pub(super) fn new(gpu: &WgpuState, _space: &SvoSpace) -> Self {
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

    pub(super) fn update_space(&mut self, gpu: &WgpuState, space: &SvoSpace) {}

    pub(super) fn sample(
        &mut self,
        gpu: &WgpuState,
        space: &SvoSpace,
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
                    ]);
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

#[derive(Debug, PartialEq)]
pub struct RayHit {
    pub color: [f32; 3],
    pub t: f32,
    pub normal: Vec3,
    pub hit: IVec3,
}

pub(super) fn raycast(space: &SvoSpace, from: Vec3, d: Vec3) -> Option<RayHit> {
    // let rr = raycast_recurse(space, from, d);
    let ri = raycast_iter(space, from, d);
    // assert_eq!(rr, ri);
    ri
}

pub(super) fn raycast_iter(space: &SvoSpace, from: Vec3, d: Vec3) -> Option<RayHit> {
    let flip = d.cmplt(Vec3::ZERO);
    let d_sign = d.signum();
    let mirror_mask = flip.bitmask() as usize;
    let d = d.abs().max(Vec3::splat(f32::EPSILON));
    let space_bound = IVec3::splat(1 << space.height);
    let from = Vec3::select(flip, space_bound.as_vec3() - from, from);

    let mut t = (-from / d).max_element().max(0.0);
    let mut enter_dir = BVec3::new(false, false, false);

    let mut height = space.height as usize;
    let mut stack_node = [None; 32];
    let mut stack_t_midplanes = [Vec3::ZERO; 32];
    let mut stack_subvoxel = [BVec3::new(false, false, false); 32];
    let mut stack_t_end = [0.0; 32];
    let mut stack_offset = [IVec3::ZERO; 32];

    stack_node[height] = Some(space.root_node()?);
    stack_t_end[height] = ((space_bound.as_vec3() - from) / d).min_element();
    stack_t_midplanes[height] = (IVec3::splat(1 << height - 1).as_vec3() - from) / d;
    stack_offset[height] = IVec3::ZERO;
    stack_subvoxel[height] = stack_t_midplanes[height].cmplt(Vec3::splat(t));
    if stack_t_end[height] < t {
        return None;
    }
    height -= 1;

    while height <= space.height as usize {
        if stack_node[height].is_none() {
            let subvoxel = stack_subvoxel[height + 1];
            let midplanes = IVec3::splat(1 << height);
            let offset = stack_offset[height + 1] + IVec3::select(subvoxel, midplanes, IVec3::ZERO);

            let Some(node) = space
                .get_node(stack_node[height + 1].unwrap())
                .unwrap_children()[subvoxel.bitmask() as usize ^ mirror_mask]
            else {
                height += 1;
                continue;
            };

            if height == 0 {
                let SvoCell::Block(color) = space.get_node(node) else {
                    unreachable!()
                };
                return Some(RayHit {
                    color: color.map(|c| c.0),
                    t,
                    normal: -Vec3::select(enter_dir, d_sign, Vec3::ZERO),
                    hit: IVec3::select(flip, space_bound - offset - 1, offset),
                });
            }

            let midplanes = IVec3::splat(1 << height - 1);
            stack_t_midplanes[height] = ((offset + midplanes).as_vec3() - from) / d;
            stack_t_end[height] = (((offset + midplanes * 2).as_vec3() - from) / d).min_element();
            stack_node[height] = Some(node);
            stack_offset[height] = offset;
            stack_subvoxel[height] = stack_t_midplanes[height].cmplt(Vec3::splat(t));
            stack_node[height - 1] = None;
            height -= 1;
            continue;
        };

        let subvoxel = &mut stack_subvoxel[height];
        let t_end = stack_t_end[height];
        let t_mid_plane = stack_t_midplanes[height];
        if subvoxel.all() {
            height += 1;
            continue;
        }

        let min = Vec3::select(*subvoxel, Vec3::splat(f32::INFINITY), t_mid_plane).min_element();
        if min > t_end {
            height += 1;
            continue;
        }

        if min == t_mid_plane.x {
            t = t_mid_plane.x;
            subvoxel.x = true;
            enter_dir = BVec3::new(true, false, false);
        }
        if min == t_mid_plane.y {
            t = t_mid_plane.y;
            subvoxel.y = true;
            enter_dir = BVec3::new(false, true, false);
        }
        if min == t_mid_plane.z {
            t = t_mid_plane.z;
            subvoxel.z = true;
            enter_dir = BVec3::new(false, false, true);
        }

        stack_node[height - 1] = None;
        height -= 1;
    }

    None
}

pub(super) fn raycast_recurse(space: &SvoSpace, from: Vec3, d: Vec3) -> Option<RayHit> {
    let flip = d.cmplt(Vec3::ZERO);
    let d_sign = d.signum();
    let mirror_mask = flip.bitmask() as usize;
    let d = d.abs().max(Vec3::splat(f32::EPSILON));
    let space_bound = Vec3::splat((1 << space.height) as f32);
    let from = Vec3::select(flip, space_bound - from, from);

    let t_enter = -from / d;
    let t_0 = t_enter.max_element().max(0.0);
    let t_1 = ((space_bound - from) / d).min_element();

    if t_0 > t_1 {
        return None;
    }

    let start = from + d * t_0;

    raycast_impl(
        space,
        space.root_node(),
        mirror_mask,
        d_sign,
        space.height,
        Vec3::ZERO,
        start,
        d,
        t_0,
        t_enter.cmpeq(Vec3::splat(t_0)),
    )
}

fn raycast_impl(
    space: &SvoSpace,
    node: Option<Node>,
    mirror_mask: usize,
    d_sign: Vec3,
    height: u32,
    offset: Vec3,
    ray_start: Vec3,
    ray_d: Vec3,
    mut ray_t: f32,
    mut ray_enter: BVec3,
) -> Option<RayHit> {
    let node = node?;

    if height == 0 {
        let SvoCell::Block(color) = space.get_node(node) else {
            unreachable!()
        };
        return Some(RayHit {
            color: color.map(|c| c.0),
            t: ray_t,
            normal: -Vec3::select(ray_enter, d_sign, Vec3::ZERO),
            hit: offset.as_ivec3(),
        });
    }

    let mid_planes = Vec3::splat((1 << height - 1) as f32);
    let t_mid_plane = (offset + mid_planes - ray_start) / ray_d;
    let t_end = ((offset + mid_planes * 2.0 - ray_start) / ray_d).min_element();

    let mut subvoxel = t_mid_plane.cmplt(Vec3::splat(ray_t));
    let children = space.get_node(node).unwrap_children();

    loop {
        if let Some(hit) = raycast_impl(
            space,
            children[subvoxel.bitmask() as usize ^ mirror_mask],
            mirror_mask,
            d_sign,
            height - 1,
            offset + Vec3::select(subvoxel, mid_planes, Vec3::ZERO),
            ray_start,
            ray_d,
            ray_t,
            ray_enter,
        ) {
            return Some(hit);
        }

        if subvoxel.all() {
            return None;
        }

        let min = Vec3::select(subvoxel, Vec3::splat(f32::INFINITY), t_mid_plane).min_element();
        if min > t_end {
            return None;
        }

        if min == t_mid_plane.x {
            ray_t = t_mid_plane.x;
            subvoxel.x = true;
            ray_enter = BVec3::new(true, false, false);
        }
        if min == t_mid_plane.y {
            ray_t = t_mid_plane.y;
            subvoxel.y = true;
            ray_enter = BVec3::new(false, true, false);
        }
        if min == t_mid_plane.z {
            ray_t = t_mid_plane.z;
            subvoxel.z = true;
            ray_enter = BVec3::new(false, false, true);
        }
    }
}

fn raytrace(space: &SvoSpace, from: Vec3, d: Vec3, sun: Vec3, depth_limit: usize) -> [f32; 3] {
    if depth_limit == 0 {
        return [0.0; 3];
    }
    if let Some(RayHit {
        color, t, normal, ..
    }) = raycast(space, from, d)
    {
        let p = from + d * t;
        let lighting = sun.dot(normal) / 2.0 + 1.0;
        // let shadow = raycast(space, p - f * 0.001, sun).is_none() as i32 as f32;
        // let lighting = lighting * (shadow / 2.0 + 0.5);
        color.map(|v| v * lighting)
    } else {
        [0.0; 3]
    }
}
