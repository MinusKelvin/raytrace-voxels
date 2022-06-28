use std::collections::VecDeque;
use std::time::Duration;

use glam::{EulerRot, IVec3, Mat3, Vec3};
use rand::prelude::*;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

mod fragment;
mod software;

fn main() {
    let mut space = Space::new(IVec3::splat(256));
    let seed = [
        218, 29, 221, 89, 183, 102, 2, 53, 176, 211, 63, 26, 195, 8, 107, 217, 90, 70, 178, 102,
        69, 8, 249, 220, 44, 31, 182, 202, 20, 106, 91, 98,
    ];
    let mut rng = rand::rngs::StdRng::from_seed(seed);
    for x in 0..space.size.x {
        for z in 0..space.size.z {
            let h = match rng.gen_bool(0.001) {
                true => rng.gen_range(128..192),
                false => {
                    ((x as f32 / 10.0).sin() * 3.0 + (z as f32 / 10.0).sin() * 6.0) as i32 + 96
                }
            };
            for y in 0..h {
                space.set(IVec3::new(x, y, z), Cell::Solid([0.99; 3]));
            }
        }
    }
    space.calculate_distances();
    let mut yaw = 0.95f32;
    let mut pitch = 0.52;
    let mut camera = (space.size / 2).as_vec3();
    let mut grabbed = false;
    let mut left = false;
    let mut right = false;
    let mut forward = false;
    let mut backward = false;
    let mut up = false;
    let mut down = false;

    let mut times = [Duration::ZERO; 1000];
    let mut framecount = 0;

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut gpu = pollster::block_on(WgpuState::new(&window));

    // let mut software = software::SoftwareRaytracer::new(&gpu, &space);
    let mut fragment = fragment::FragmentRaytracer::new(&gpu, &space);

    let mut last_time = std::time::Instant::now();

    event_loop.run(move |event, _, cf| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *cf = ControlFlow::Exit,
            WindowEvent::Resized(size) => gpu.resize(size),
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => gpu.resize(*new_inner_size),
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
            let delta = now - last_time;
            times[framecount % times.len()] = delta;
            let total_time = times.iter().copied().sum::<Duration>();
            let fps = times.len() as f64 / total_time.as_secs_f64();
            window.set_title(&format!("{:.0} FPS", fps));
            let delta = delta.as_secs_f32();
            last_time = now;

            if framecount == 5000 {
                println!("{total_time:.2?}");
            }
            framecount += 1;

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

            match gpu.surface.get_current_texture() {
                Ok(frame) => {
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    // let sw_cmd = software.render(&gpu, &view, &space, camera, yaw, pitch);
                    let fg_cmd = fragment.render(&gpu, &view, &space, camera, yaw, pitch);

                    gpu.queue.submit([/*sw_cmd,*/ fg_cmd]);

                    frame.present();
                }
                Err(wgpu::SurfaceError::Lost) => gpu.resize(gpu.size),
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
            present_mode: wgpu::PresentMode::Mailbox,
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
    voxels: Vec<Cell>,
}

impl Space {
    fn new(size: IVec3) -> Self {
        Space {
            size,
            voxels: vec![Cell::Empty(u32::MAX); (size.x * size.y * size.z) as usize],
        }
    }

    fn idx(&self, p: IVec3) -> Option<usize> {
        if p.cmplt(IVec3::ZERO).any() || p.cmpge(self.size).any() {
            None
        } else {
            Some(((p.x * self.size.y + p.y) * self.size.z + p.z) as usize)
        }
    }

    fn get(&self, p: IVec3) -> Option<Cell> {
        Some(self.voxels[self.idx(p)?])
    }

    fn set(&mut self, p: IVec3, v: Cell) {
        let i = self.idx(p).unwrap();
        self.voxels[i] = v;
    }

    fn calculate_distances(&mut self) {
        let mut decreases = VecDeque::new();

        for x in 0..self.size.x {
            for y in 0..self.size.y {
                for z in 0..self.size.z {
                    let p = IVec3::new(x, y, z);
                    match self.get(p).unwrap() {
                        Cell::Solid(_) => decreases.push_back((p, 0)),
                        Cell::Empty(_) => self.set(p, Cell::Empty(u32::MAX)),
                    }
                }
            }
        }

        while let Some((p, v)) = decreases.pop_front() {
            let propogate;
            match self.get(p).unwrap() {
                Cell::Solid(_) => {
                    propogate = v == 0;
                }
                Cell::Empty(a) => {
                    propogate = v < a;
                    if propogate {
                        self.set(p, Cell::Empty(v));
                    }
                }
            }
            if propogate {
                for x in p.x - 1..=p.x + 1 {
                    for y in p.y - 1..=p.y + 1 {
                        for z in p.z - 1..=p.z + 1 {
                            let p = IVec3::new(x, y, z);
                            if matches!(self.get(p), Some(Cell::Empty(a)) if a > v + 1) {
                                decreases.push_back((p, v + 1));
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Cell {
    Solid([f32; 3]),
    Empty(u32),
}
