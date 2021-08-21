use std::num::NonZeroU32;

use glam::{EulerRot, IVec3, Mat3, Vec3, Vec3A};
use image::{Bgra, EncodableLayout};
use rand::prelude::*;
use rayon::prelude::*;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
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

            match state.surface.get_current_frame() {
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
    let mut t_max = t_delta * (1.0 - (from * step).fract() + f32::EPSILON);
    if t_max.x.is_nan() {
        t_max.x = f32::INFINITY;
    }
    if t_max.y.is_nan() {
        t_max.y = f32::INFINITY;
    }
    if t_max.z.is_nan() {
        t_max.z = f32::INFINITY;
    }
    let mut p = from.floor().as_i32();
    let step = step.as_i32();
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
    }
}

fn render(
    state: &mut WgpuState,
    frame: wgpu::SurfaceFrame,
    space: &Space,
    camera: Vec3,
    yaw: f32,
    pitch: f32,
) {
    let encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let looking = glam::Mat3A::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);

    let halfwidth = state.size.width as f32 / 2.0;
    let halfheight = state.size.height as f32 / 2.0;
    let sun = Vec3::new(0.1, 1.0, 0.2);

    let mut raycast_image =
        image::ImageBuffer::<Bgra<u8>, Vec<u8>>::new(state.size.width, state.size.height);
    raycast_image
        .enumerate_rows_mut()
        .par_bridge()
        .for_each(|(_, row)| for (x, y, pixel) in row {
            let d = Vec3::from(
                looking
                    * Vec3A::new(
                        (x as f32 - halfwidth) / halfheight,
                        (halfheight - y as f32) / halfheight,
                        1.0,
                    )
                    .normalize(),
            );
            if let Some((c, t, f)) = raycast(space, camera, d) {
                let p = camera + d * t;
                let lighting = sun.dot(-f).max(0.0) / 2.0 + 0.5;
                let shadow = raycast(space, p - f * 0.001, sun).is_none() as i32 as f32;
                let lighting = lighting.min(shadow / 2.0 + 0.5);
                *pixel = image::Bgra([
                    (c[0] * lighting * 255.0) as u8,
                    (c[1] * lighting * 255.0) as u8,
                    (c[2] * lighting * 255.0) as u8,
                    0xFF,
                ]);
            }
        });

    state.queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &frame.output.texture,
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

    state.queue.submit(std::iter::once(encoder.finish()));
}

struct WgpuState {
    surface: wgpu::Surface,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: PhysicalSize<u32>,
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
            usage: wgpu::TextureUsages::COPY_DST,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        Self {
            surface,
            device,
            queue,
            config,
            size,
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
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
