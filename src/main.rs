use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use fragment::FragmentRaytracer;
use glam::{EulerRot, IVec3, Mat3, Vec3};
use rand::prelude::*;
use svo::SvoSpace;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

mod fragment;
mod software;
mod svo;

struct App {
    gpu: Option<WgpuState>,
    window: Option<Arc<Window>>,
    fragment: Option<FragmentRaytracer>,

    yaw: f32,
    pitch: f32,
    camera: Vec3,
    grabbed: bool,

    left: bool,
    right: bool,
    forward: bool,
    backward: bool,
    up: bool,
    down: bool,

    last_time: Instant,
    times: [Duration; 250],
    framecount: usize,

    space: Space,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(WindowAttributes::default().with_inner_size(PhysicalSize::new(853, 480)))
                .unwrap(),
        );

        let gpu = pollster::block_on(WgpuState::new(&window));

        let fragment = FragmentRaytracer::new(&gpu, &self.space);

        self.gpu = Some(gpu);
        self.window = Some(window);
        self.fragment = Some(fragment);

        event_loop.set_control_flow(ControlFlow::Poll);
    }

    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
        if cause != StartCause::Poll {
            return;
        }
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let Some(gpu) = self.gpu.as_mut() else { return };
        let Some(fragment) = self.fragment.as_mut() else {
            return;
        };

        let now = std::time::Instant::now();
        let delta = now - self.last_time;
        self.times[self.framecount % self.times.len()] = delta;
        let total_time = self.times.iter().copied().sum::<Duration>();
        let fps = self.times.len() as f64 / total_time.as_secs_f64();
        window.set_title(&format!(
            "{fps:.0} FPS, {} samples",
            fragment.samples as i32
        ));
        let delta = delta.as_secs_f32();
        self.last_time = now;

        if self.framecount == 5000 {
            println!(
                "{total_time:.2?} {:?} {} {}",
                self.camera, self.yaw, self.pitch
            );
        }
        self.framecount += 1;

        let mut d = Vec3::ZERO;
        if self.left {
            d.x -= 1.0;
        }
        if self.right {
            d.x += 1.0;
        }
        if self.forward {
            d.z += 1.0;
        }
        if self.backward {
            d.z -= 1.0;
        }
        let dir = Mat3::from_euler(EulerRot::YXZ, self.yaw, 0.0, 0.0);
        if self.grabbed {
            self.camera += dir * d.normalize_or_zero() * delta * 10.0;
            self.camera.y += (self.up as i32 - self.down as i32) as f32 * delta * 10.0;
        }

        match gpu.surface.get_current_texture() {
            Ok(frame) => {
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                // let sw_cmd = software.render(&gpu, &view, &space, camera, yaw, pitch);
                let fg_cmd =
                    fragment.render(&gpu, &view, &self.space, self.camera, self.yaw, self.pitch);

                gpu.queue.submit([/*sw_cmd,*/ fg_cmd]);

                window.pre_present_notify();
                frame.present();

                window.request_redraw();
            }
            Err(wgpu::SurfaceError::Lost) => gpu.resize(gpu.size),
            Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
            Err(_) => {}
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let Some(gpu) = self.gpu.as_mut() else { return };
        let Some(fragment) = self.fragment.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => gpu.resize(size),
            WindowEvent::ScaleFactorChanged {
                mut inner_size_writer,
                ..
            } => {
                inner_size_writer.request_inner_size(gpu.size);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let state = event.state == ElementState::Pressed;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) if state => {
                        self.grabbed ^= true;
                        window
                            .set_cursor_grab(match self.grabbed {
                                true => CursorGrabMode::Locked,
                                false => CursorGrabMode::None,
                            })
                            .unwrap();
                        window.set_cursor_visible(!self.grabbed);
                    }
                    PhysicalKey::Code(KeyCode::KeyA) => self.left = state,
                    PhysicalKey::Code(KeyCode::KeyD) => self.right = state,
                    PhysicalKey::Code(KeyCode::KeyW) => self.forward = state,
                    PhysicalKey::Code(KeyCode::KeyS) => self.backward = state,
                    PhysicalKey::Code(KeyCode::Space) => self.up = state,
                    PhysicalKey::Code(KeyCode::ShiftLeft) => self.down = state,
                    _ => {}
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } if self.grabbed => {
                let looking = glam::Mat3A::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0);
                let result = software::raycast(&self.space, self.camera, looking.mul_vec3(Vec3::Z));
                if let Some((_, _, n, p)) = result {
                    match button {
                        MouseButton::Left => self.space.set(p, Cell::Empty([0, 0])),
                        MouseButton::Right => {
                            let np = p - n.as_ivec3();
                            if self.space.idx(np).is_some() {
                                self.space.set(np, Cell::Solid([1.0; 3]));
                            }
                        }
                        _ => {}
                    }

                    self.space.calculate_distances();
                    fragment.update_space(&gpu, &self.space);
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                if self.grabbed {
                    self.yaw += delta.0 as f32 * 0.01;
                    self.pitch += delta.1 as f32 * 0.01;
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let mut space = Space::new(IVec3::splat(256));
    let seed = [
        218, 29, 221, 89, 183, 102, 2, 53, 176, 211, 63, 26, 195, 8, 107, 217, 90, 70, 178, 102,
        69, 8, 249, 220, 44, 31, 182, 202, 20, 106, 91, 98,
    ];
    let mut rng = rand::rngs::StdRng::from_seed(seed);
    for x in 1..space.size.x - 1 {
        for z in 1..space.size.z - 1 {
            let h = ((x as f32 / 10.0).sin() * 6.0 + (z as f32 / 10.0).sin() * 20.0) as i32 + 96;
            // let h2 = match rng.gen_bool(0.001) {
            //     true => h + 15,
            //     false => h,
            // };
            for y in 0..h {
                space.set(IVec3::new(x, y, z), Cell::Solid([0.5; 3]));
            }
            // if h != h2 {
            //     space.set(IVec3::new(x, h2 - 1, z), Cell::Solid([1.0; 3]));
            //     space.set(IVec3::new(x, h2, z), Cell::Solid([0.75; 3]));
            //     space.set(IVec3::new(x - 1, h2 - 2, z), Cell::Solid([0.75; 3]));
            //     space.set(IVec3::new(x + 1, h2 - 2, z), Cell::Solid([0.75; 3]));
            //     space.set(IVec3::new(x, h2 - 2, z - 1), Cell::Solid([0.75; 3]));
            //     space.set(IVec3::new(x, h2 - 2, z + 1), Cell::Solid([0.75; 3]));
            //     space.set(IVec3::new(x - 1, h2 - 1, z), Cell::Solid([0.75; 3]));
            //     space.set(IVec3::new(x + 1, h2 - 1, z), Cell::Solid([0.75; 3]));
            //     space.set(IVec3::new(x, h2 - 1, z - 1), Cell::Solid([0.75; 3]));
            //     space.set(IVec3::new(x, h2 - 1, z + 1), Cell::Solid([0.75; 3]));
            // }
            if h > 111 {
                space.set(IVec3::new(x, 111, z), Cell::Solid([1.0, 0.5, 0.3]));
            }
            // space.set(IVec3::new(x, 255, z), Cell::Solid([1.0; 3]));
        }
    }
    for i in -2..=16 {
        for j in -2..=16 {
            space.set(IVec3::new(i + 100, j + 90, 100), Cell::Solid([0.75; 3]));
            space.set(IVec3::new(100, j + 90, i + 100), Cell::Solid([0.75; 3]));
            space.set(IVec3::new(i + 100, j + 90, 116), Cell::Solid([0.75; 3]));
            space.set(IVec3::new(116, j + 90, i + 100), Cell::Solid([0.75; 3]));
            space.set(IVec3::new(100 + j, 106, i + 100), Cell::Solid([0.75; 3]));
            space.set(IVec3::new(100 + j, 90, i + 100), Cell::Solid([0.75; 3]));
        }
    }
    // space.set(IVec3::new(113, 106, 113), Cell::Empty([0; 2]));
    // space.set(IVec3::new(112, 106, 113), Cell::Empty([0; 2]));
    // space.set(IVec3::new(112, 106, 112), Cell::Empty([0; 2]));
    // space.set(IVec3::new(113, 106, 112), Cell::Empty([0; 2]));
    space.set(IVec3::new(113, 107, 113), Cell::Solid([1.0; 3]));
    space.set(IVec3::new(112, 107, 113), Cell::Solid([1.0; 3]));
    space.set(IVec3::new(112, 107, 112), Cell::Solid([1.0; 3]));
    space.set(IVec3::new(113, 107, 112), Cell::Solid([1.0; 3]));
    space.set(IVec3::new(110, 88, 110), Cell::Solid([1.0, 0.5, 0.3]));
    space.set(IVec3::new(111, 88, 110), Cell::Solid([1.0, 0.5, 0.3]));
    space.set(IVec3::new(111, 88, 109), Cell::Solid([0.3, 0.5, 1.0]));
    space.set(IVec3::new(110, 88, 109), Cell::Solid([1.0, 0.5, 0.3]));
    space.set(IVec3::new(111, 88, 108), Cell::Solid([1.0, 0.5, 0.3]));
    space.set(IVec3::new(110, 88, 108), Cell::Solid([1.0, 0.5, 0.3]));
    space.set(IVec3::new(112, 88, 110), Cell::Solid([1.0, 0.5, 0.3]));
    space.set(IVec3::new(112, 88, 109), Cell::Solid([1.0, 0.5, 0.3]));
    space.set(IVec3::new(112, 88, 108), Cell::Solid([1.0, 0.5, 0.3]));
    space.calculate_distances();

    // println!("making svo");
    // let t = Instant::now();
    // let mut svo_space = SvoSpace::new(space.size);
    // for x in 0..space.size.x {
    //     for y in 0..space.size.y {
    //         for z in 0..space.size.z {
    //             let p = IVec3::new(x, y, z);
    //             let v = match space.get(p) {
    //                 Some(Cell::Solid(v)) => Some(v),
    //                 Some(Cell::Empty(_)) => None,
    //                 None => None,
    //             };
    //             svo_space.set(p, v);
    //         }
    //     }
    // }
    // println!("checking svo {:.3?}", t.elapsed());
    // let t = Instant::now();
    // for x in 0..space.size.x {
    //     for y in 0..space.size.y {
    //         for z in 0..space.size.z {
    //             let p = IVec3::new(x, y, z);
    //             let v = match space.get(p) {
    //                 Some(Cell::Solid(v)) => Some(v),
    //                 Some(Cell::Empty(_)) => None,
    //                 None => None,
    //             };
    //             assert_eq!(v, svo_space.get(p));
    //         }
    //     }
    // }
    // println!("all g {:.2?}", t.elapsed());

    // println!("array size: {}", space.mem_usage());
    // println!("  svo size: {}", svo_space.mem_usage());

    EventLoop::new()
        .unwrap()
        .run_app(&mut App {
            window: None,
            gpu: None,

            yaw: 0.62996435,
            pitch: 0.09000018,
            camera: Vec3::new(34.811836, 114.02207, 74.028244),
            grabbed: false,
            left: false,
            right: false,
            forward: false,
            backward: false,
            up: false,
            down: false,

            last_time: Instant::now(),
            times: [Duration::ZERO; 250],
            framecount: 0,

            space,
            fragment: None,
        })
        .unwrap();

    // let mut software = software::SoftwareRaytracer::new(&gpu, &space);
}

struct WgpuState {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: PhysicalSize<u32>,
}

impl WgpuState {
    async fn new(window: &Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).unwrap();
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
                    required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                    required_limits: wgpu::Limits::default(),
                    label: None,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            // format: wgpu::TextureFormat::Rgba16Float,
            format: surface
                .get_capabilities(&adapter)
                .formats
                .into_iter()
                .find(|s| s.is_srgb())
                .unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Mailbox,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
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
            voxels: vec![Cell::Empty([u32::MAX; 2]); (size.x * size.y * size.z) as usize],
        }
    }

    fn mem_usage(&self) -> usize {
        self.voxels.capacity() * std::mem::size_of::<Cell>()
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
        let mut decreases_up = VecDeque::new();
        let mut decreases_down = VecDeque::new();

        for x in 0..self.size.x {
            for y in 0..self.size.y {
                for z in 0..self.size.z {
                    let p = IVec3::new(x, y, z);
                    match self.get(p).unwrap() {
                        Cell::Solid(_) => {
                            decreases_up.push_back((p, 0));
                            decreases_down.push_back((p, 0));
                        }
                        Cell::Empty(_) => {
                            self.set(p, Cell::Empty([(self.size.y - y) as u32, y as u32 + 1]))
                        }
                    }
                }
            }
        }

        while let Some((p, v)) = decreases_up.pop_front() {
            let propogate;
            match self.get(p).unwrap() {
                Cell::Solid(_) => {
                    propogate = v == 0;
                }
                Cell::Empty([up, down]) => {
                    propogate = v < up;
                    if propogate {
                        self.set(p, Cell::Empty([v, down]));
                    }
                }
            }
            if propogate {
                for x in p.x - 1..=p.x + 1 {
                    for y in p.y - 1..=p.y {
                        for z in p.z - 1..=p.z + 1 {
                            let p = IVec3::new(x, y, z);
                            if matches!(self.get(p), Some(Cell::Empty([up, _])) if up > v + 1) {
                                decreases_up.push_back((p, v + 1));
                            }
                        }
                    }
                }
            }
        }

        while let Some((p, v)) = decreases_down.pop_front() {
            let propogate;
            match self.get(p).unwrap() {
                Cell::Solid(_) => {
                    propogate = v == 0;
                }
                Cell::Empty([up, down]) => {
                    propogate = v < down;
                    if propogate {
                        self.set(p, Cell::Empty([up, v]));
                    }
                }
            }
            if propogate {
                for x in p.x - 1..=p.x + 1 {
                    for y in p.y..=p.y + 1 {
                        for z in p.z - 1..=p.z + 1 {
                            let p = IVec3::new(x, y, z);
                            if matches!(self.get(p), Some(Cell::Empty([_, down])) if down > v + 1) {
                                decreases_down.push_back((p, v + 1));
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
    Empty([u32; 2]),
}
