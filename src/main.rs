use glam::{EulerRot, IVec3, Mat3, Vec3};
use rand::prelude::*;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

mod fragment;
mod software;

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

    let mut gpu = pollster::block_on(WgpuState::new(&window));

    let mut software = software::SoftwareRaytracer::new(&gpu);
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

            match gpu.surface.get_current_texture() {
                Ok(frame) => {
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    let sw_cmd = software.render(&gpu, &view, &space, camera, yaw, pitch);
                    let fg_cmd = fragment.render(&gpu, &view, &space, camera, yaw, pitch);

                    gpu.queue.submit([sw_cmd, fg_cmd]);

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
