use std::num::NonZeroU32;

use glam::{EulerRot, IVec3, Mat3, Vec3, Vec3A};
use image::{EncodableLayout, Rgba};
use rand::prelude::*;
use rayon::prelude::*;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

fn main() {
    let mut space = Space::new(IVec3::splat(32));
    for x in 0..32 {
        for y in 0..32 {
            for z in 0..32 {
                if y == 0 {
                    space.set(IVec3::new(x, y, z), Some([1.0; 3]));
                } else if thread_rng().gen_bool(0.01) {
                    space.set(IVec3::new(x, y, z), Some(thread_rng().gen()));
                }
            }
        }
    }
    let mut yaw = 0.0f32;
    let mut pitch = 0.0;
    let mut camera = Vec3::new(16.0, 16.0, 16.0);
    let mut grabbed = false;
    let mut left = false;
    let mut right = false;
    let mut forward = false;
    let mut backward = false;
    let mut up = false;
    let mut down = false;

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = pollster::block_on(WgpuState::new(&window));

    let mut last_time = std::time::Instant::now();

    event_loop.run(move |event, _, cf| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *cf = ControlFlow::Exit,
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => state.resize(*new_inner_size),
            WindowEvent::KeyboardInput { input, .. } => {
                let state = input.state == ElementState::Pressed;
                match input.virtual_keycode {
                    Some(VirtualKeyCode::Escape) if state => {
                        grabbed ^= true;
                        window.set_cursor_grab(grabbed).unwrap();
                        window.set_cursor_visible(!grabbed);
                    }
                    Some(VirtualKeyCode::A) => left = state,
                    Some(VirtualKeyCode::D) => right = state,
                    Some(VirtualKeyCode::W) => forward = state,
                    Some(VirtualKeyCode::S) => backward = state,
                    Some(VirtualKeyCode::Space) => up = state,
                    Some(VirtualKeyCode::LShift) => down = state,
                    _ => {}
                }
            }
            _ => {}
        },
        Event::MainEventsCleared => {
            let now = std::time::Instant::now();
            let delta = (now - last_time).as_secs_f32();
            window.set_title(&format!(
                "cpu voxel raytrace: {:.2?} frametime",
                now - last_time
            ));
            last_time = now;

            let mut d = Vec3::ZERO;
            if left {
                d.x -= 1.0;
            }
            if right {
                d.x += 1.0;
            }
            if forward {
                d.z += 1.0;
            }
            if backward {
                d.z -= 1.0;
            }
            let dir = Mat3::from_euler(EulerRot::YXZ, yaw, 0.0, 0.0);
            camera += dir * d.normalize_or_zero() * delta * 10.0;

            camera.y += (up as i32 - down as i32) as f32 * delta * 10.0;

            match state.surface.get_current_texture() {
                Ok(frame) => render(&mut state, frame, &space, camera, yaw, pitch),
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                Err(wgpu::SurfaceError::OutOfMemory) => *cf = ControlFlow::Exit,
                Err(_) => {}
            }
        }
        Event::DeviceEvent { event, .. } => match event {
            DeviceEvent::MouseMotion { delta } => {
                if grabbed {
                    yaw += delta.0 as f32 * 0.01;
                    pitch += delta.1 as f32 * 0.01;
                }
            }
            _ => {}
        },
        _ => {}
    })
}

fn raycast(space: &Space, from: Vec3, d: Vec3) -> Option<([f32; 3], f32, Vec3)> {
    let step = d.signum();
    let t_delta = step / d;
    let fudge = (1.0 + step) / 2.0;
    let mut t_max = t_delta * (fudge - from.fract() * step);
    if t_max.x.is_nan() {
        t_max.x = f32::INFINITY;
    }
    if t_max.y.is_nan() {
        t_max.y = f32::INFINITY;
    }
    if t_max.z.is_nan() {
        t_max.z = f32::INFINITY;
    }
    let mut p = from.floor().as_ivec3();
    let step = step.as_ivec3();
    let mut i = 0;
    loop {
        let t = t_max.min_element();
        let mut f = Vec3::ZERO;
        if t_max.x < t_max.y {
            if t_max.x < t_max.z {
                t_max.x += t_delta.x;
                p.x += step.x;
                f.x = step.x as f32;
            } else {
                t_max.z += t_delta.z;
                p.z += step.z;
                f.z = step.z as f32;
            }
        } else {
            if t_max.y < t_max.z {
                t_max.y += t_delta.y;
                p.y += step.y;
                f.y = step.y as f32;
            } else {
                t_max.z += t_delta.z;
                p.z += step.z;
                f.z = step.z as f32;
            }
        }

        if let Some(color) = space.get(p)? {
            return Some((color, t, f));
        }

        i += 1;
        if i > 100000 {
            dbg!(step, t_max, from, d, p);
            std::process::exit(1);
        }
    }
}

fn raytrace(space: &Space, from: Vec3, d: Vec3, sun: Vec3, depth_limit: usize) -> [f32; 3] {
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

fn render(
    state: &mut WgpuState,
    frame: wgpu::SurfaceTexture,
    space: &Space,
    camera: Vec3,
    yaw: f32,
    pitch: f32,
) {
    let mut encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let looking = glam::Mat3A::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);

    let halfwidth = state.size.width as f32 / 2.0;
    let halfheight = state.size.height as f32 / 2.0;
    let sun = Vec3::new(0.1, 1.0, 0.2);

    let mut raycast_image =
        image::ImageBuffer::<Rgba<u8>, Vec<u8>>::new(state.size.width, state.size.height);
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
                let color = raytrace(space, camera, d, sun, 4);
                *pixel = Rgba([
                    (color[0] * 255.0) as u8,
                    (color[1] * 255.0) as u8,
                    (color[2] * 255.0) as u8,
                    0xff,
                ])
            }
        });

    state.queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &state.cp_tex,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        raycast_image.as_bytes(),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(NonZeroU32::new(4 * state.size.width).unwrap()),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: state.size.width,
            height: state.size.height,
            depth_or_array_layers: 1,
        },
    );

    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: true,
            },
        }],
        depth_stencil_attachment: None,
    });

    rp.set_pipeline(&state.cp_rp);
    rp.set_bind_group(0, &state.cp_bind, &[]);
    rp.draw(0..6, 0..1);

    drop(rp);

    state.queue.submit(std::iter::once(encoder.finish()));

    frame.present();
}

struct WgpuState {
    surface: wgpu::Surface,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: PhysicalSize<u32>,

    cp_tex: wgpu::Texture,
    cp_rp: wgpu::RenderPipeline,
    cp_bind: wgpu::BindGroup,
    cp_bind_layout: wgpu::BindGroupLayout,
    cp_tex_view: wgpu::TextureView,
}

impl WgpuState {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        let cp_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        let cp_tex_view = cp_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let cp_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            ],
        });

        let cp_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &cp_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&cp_tex_view),
                },
            ],
        });

        let cp_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&cp_bind_layout],
            push_constant_ranges: &[],
        });

        let copy_shader = device.create_shader_module(&wgpu::include_wgsl!("copy.wgsl"));

        let cp_rp = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            multiview: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            cp_tex,
            cp_rp,
            cp_bind,
            cp_bind_layout,
            cp_tex_view,
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            self.cp_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            });
            self.cp_tex_view = self.cp_tex.create_view(&wgpu::TextureViewDescriptor::default());

            self.cp_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.cp_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.cp_tex_view),
                    },
                ],
            });
        }
    }
}

struct Space {
    size: IVec3,
    voxels: Vec<Option<[f32; 3]>>,
}

impl Space {
    fn new(size: IVec3) -> Space {
        Space {
            size,
            voxels: vec![None; (size.x * size.y * size.z) as usize],
        }
    }

    fn idx(&self, p: IVec3) -> Option<usize> {
        if p.cmplt(IVec3::ZERO).any() || p.cmpge(self.size).any() {
            None
        } else {
            Some(((p.x * self.size.y + p.y) * self.size.z + p.z) as usize)
        }
    }

    fn get(&self, p: IVec3) -> Option<Option<[f32; 3]>> {
        Some(self.voxels[self.idx(p)?])
    }

    fn set(&mut self, p: IVec3, v: Option<[f32; 3]>) {
        let i = self.idx(p).unwrap();
        self.voxels[i] = v;
    }
}
