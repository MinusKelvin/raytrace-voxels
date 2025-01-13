use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use fragment::FragmentRaytracer;
use glam::{EulerRot, IVec3, Mat3, Quat, Vec3};
use image::{DynamicImage, Pixel, Rgba32FImage};
use rand::prelude::*;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use software::{RayHit, SoftwareRaytracer};
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

type Raytracer = SoftwareRaytracer;

struct App {
    gpu: Option<WgpuState>,
    renderer: Option<Raytracer>,
    window: Option<Arc<Window>>,

    yaw: f32,
    pitch: f32,
    camera: Vec3,
    sun: Vec3,
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

    seq: usize,
    iter: usize,
    frame_start: Instant,

    space: SvoSpace,

    headless: bool,
}

impl App {
    fn init(&mut self) {
        let gpu = pollster::block_on(WgpuState::new(self.window.as_ref()));
        let raytracer = Raytracer::new(&gpu, &self.space);

        self.gpu = Some(gpu);
        self.renderer = Some(raytracer);
    }

    fn sample(&mut self) {
        let Some(gpu) = self.gpu.as_mut() else { return };
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        let now = std::time::Instant::now();
        let delta = now - self.last_time;
        self.times[self.framecount % self.times.len()] = delta;
        let delta = delta.as_secs_f32();
        self.last_time = now;

        if self.framecount == 5000 {}
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

        // let sw_cmd = software.render(&gpu, &view, &space, camera, yaw, pitch);
        renderer.sample(
            &gpu,
            &self.space,
            self.camera,
            self.yaw,
            self.pitch,
            self.sun,
        );

        if self.headless && renderer.samples % 1_000 == 0 {
            self.seq = 5;
            renderer.save_image(gpu, format!("frames/{:04}-{:03}.exr", self.iter, self.seq));

            let (axis, angle) = Quat::from_rotation_arc(
                Vec3::new(0.8, 1.0, 3.7).normalize(),
                Vec3::new(0.8, 0.0, 3.7).normalize(),
            )
            .to_axis_angle();
            let quat = Quat::from_axis_angle(axis, 0.01 * angle.signum());
            self.seq += 1;
            self.sun = quat * self.sun;

            let now = Instant::now();
            println!(
                "{:>4.0} paths/px/sec    iter {:>2} frame {:>3}",
                1000.0 / (now - self.frame_start).as_secs_f64(),
                self.iter,
                self.seq,
            );
            self.frame_start = now;

            // if renderer.samples == 10_000 {
            //     std::process::exit(0);
            // }

            if self.sun.y < -0.3 {
                self.iter += 1;
                self.sun = Vec3::new(0.8, 10.2743, 3.7).normalize();
                self.seq = 0;
            }

            if self.seq == 0 {
                println!("Finished iter {}", self.iter - 1);
                if self.iter == 1 {
                    std::process::exit(0);
                }
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.window = Some(Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default().with_inner_size(PhysicalSize::new(853, 480)),
                )
                .unwrap(),
        ));

        event_loop.set_control_flow(ControlFlow::Poll);

        self.init();
    }

    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
        if cause != StartCause::Poll {
            return;
        }

        self.sample();

        let Some(window) = self.window.as_ref() else {
            return;
        };
        let Some(gpu) = self.gpu.as_mut() else { return };
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        let total_time = self.times.iter().copied().sum::<Duration>();
        let fps = self.times.len() as f64 / total_time.as_secs_f64();
        window.set_title(&format!(
            "{} samples, {fps:.0}/sec",
            renderer.samples as i32
        ));

        match gpu.surface.as_ref().map(|s| s.get_current_texture()) {
            Some(Ok(frame)) => {
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                renderer.show(&gpu, &view);

                window.pre_present_notify();
                frame.present();

                window.request_redraw();
            }
            Some(Err(wgpu::SurfaceError::Lost)) => gpu.resize(gpu.size),
            Some(Err(wgpu::SurfaceError::OutOfMemory)) => event_loop.exit(),
            _ => {}
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let Some(gpu) = self.gpu.as_mut() else { return };
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => gpu.resize(size),
            WindowEvent::ScaleFactorChanged {
                mut inner_size_writer,
                ..
            } => {
                inner_size_writer.request_inner_size(gpu.size).unwrap();
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

                        if !self.grabbed {
                            println!("{:?} {} {}", self.camera, self.yaw, self.pitch);
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyA) => self.left = state,
                    PhysicalKey::Code(KeyCode::KeyD) => self.right = state,
                    PhysicalKey::Code(KeyCode::KeyW) => self.forward = state,
                    PhysicalKey::Code(KeyCode::KeyS) => self.backward = state,
                    PhysicalKey::Code(KeyCode::Space) => self.up = state,
                    PhysicalKey::Code(KeyCode::ShiftLeft) => self.down = state,
                    PhysicalKey::Code(KeyCode::KeyG) if state => {
                        self.camera.y += 100000.0;
                    }
                    PhysicalKey::Code(KeyCode::KeyR) if state => {
                        let (axis, angle) = Quat::from_rotation_arc(
                            Vec3::new(0.8, 1.0, 3.7).normalize(),
                            Vec3::new(0.8, 0.0, 3.7).normalize(),
                        )
                        .to_axis_angle();
                        let quat = Quat::from_axis_angle(axis, 0.05 * angle.signum());
                        self.seq += 1;
                        self.sun = quat * self.sun;
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } if self.grabbed => {
                let looking = glam::Mat3A::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0);
                let result = dbg!(software::raycast(
                    &self.space,
                    self.camera,
                    looking.mul_vec3(Vec3::Z)
                ));
                if let Some(RayHit { hit, normal, .. }) = result {
                    match button {
                        MouseButton::Left => self.space.set(hit, None),
                        MouseButton::Right => {
                            let np = hit + normal.as_ivec3();
                            // if self.space.idx(np).is_some() {
                            self.space.set(np, Some([1.0; 3]));
                            // }
                        }
                        _ => {}
                    }

                    // self.space.calculate_distances();
                    renderer.update_space(&gpu, &self.space);
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
                    self.pitch = self
                        .pitch
                        .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
                }
            }
            _ => {}
        }
    }
}

fn main() {
    if std::env::args().any(|s| s == "combine") {
        std::fs::create_dir_all("movie").unwrap();
        let mut files = vec![];
        for f in std::fs::read_dir("frames").unwrap() {
            let path = f.unwrap().path();
            if path.extension() != Some("exr".as_ref()) {
                continue;
            }
            let seq: usize = path
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .split_once('-')
                .unwrap()
                .1
                .parse()
                .unwrap();
            while seq >= files.len() {
                files.push(vec![]);
            }
            files[seq].push(path);
        }

        files
            .into_par_iter()
            .enumerate()
            .for_each(|(i, to_combine)| {
                let divisor = to_combine.len() as f32;
                let mut iter = to_combine.into_iter();
                let mut acc = image::open(iter.next().unwrap()).unwrap().into_rgba32f();
                for f in iter {
                    let img = image::open(f).unwrap().into_rgba32f();
                    assert_eq!(acc.width(), img.width());
                    assert_eq!(acc.height(), img.height());
                    for (acc, p) in acc.pixels_mut().zip(img.pixels()) {
                        acc.apply2(p, |a, b| a + b);
                    }
                }

                for p in acc.pixels_mut() {
                    p.apply(|v| {
                        let v = v / divisor;
                        if v < 0.0031308 {
                            v * 12.92
                        } else {
                            v.powf(1.0 / 2.4) * 1.055 - 0.055
                        }
                    });
                }

                DynamicImage::ImageRgba32F(acc)
                    .to_rgba8()
                    .save(format!("movie/{i}.png"))
                    .unwrap();
            });

        return;
    }

    let mut space = SvoSpace::new(IVec3::splat(256));
    let seed = [
        218, 29, 221, 89, 183, 102, 2, 53, 176, 211, 63, 26, 195, 8, 107, 217, 90, 70, 178, 102,
        69, 8, 249, 220, 44, 31, 182, 202, 20, 106, 91, 98,
    ];
    let mut rng = rand::rngs::StdRng::from_seed(seed);
    for x in 1..space.size.x - 1 {
        for z in 1..space.size.z - 1 {
            let h = ((x as f32 / 10.0).sin() * 3.0 + (z as f32 / 10.0).sin() * 6.0) as i32 + 96;
            let h2 = match rng.gen_bool(0.002) {
                true => h + 10,
                false => h,
            };
            for y in 0..h {
                space.set(IVec3::new(x, y, z), Some([0.5; 3]));
            }
            if h != h2 {
                space.set(IVec3::new(x, h2 - 1, z), Some([1.0; 3]));
                // }
                space.set(IVec3::new(x, h2, z), Some([0.75; 3]));
                space.set(IVec3::new(x - 1, h2 - 2, z), Some([0.75; 3]));
                space.set(IVec3::new(x + 1, h2 - 2, z), Some([0.75; 3]));
                space.set(IVec3::new(x, h2 - 2, z - 1), Some([0.75; 3]));
                space.set(IVec3::new(x, h2 - 2, z + 1), Some([0.75; 3]));
                space.set(IVec3::new(x - 1, h2 - 1, z), Some([0.75; 3]));
                space.set(IVec3::new(x + 1, h2 - 1, z), Some([0.75; 3]));
                space.set(IVec3::new(x, h2 - 1, z - 1), Some([0.75; 3]));
                space.set(IVec3::new(x, h2 - 1, z + 1), Some([0.75; 3]));
            }
            if h > 98 {
                space.set(IVec3::new(x, 98, z), Some([1.0, 0.5, 0.3]));
            }
            // space.set(IVec3::new(x, 255, z), Some([1.0; 3]));
        }
    }
    for i in -2..=24 {
        for j in -2..=24 {
            space.set(IVec3::new(i + 100, 103, j + 100), Some([1.0, 0.1, 0.1]));
            space.set(IVec3::new(i + 100, 103, j + 100), Some([1.0, 0.1, 0.1]));
            space.set(IVec3::new(i + 100, 103, j + 100), Some([1.0, 0.1, 0.1]));
        }
    }
    // space.set(IVec3::new(113, 106, 113), Cell::Empty([0; 2]));
    // space.set(IVec3::new(112, 106, 113), Cell::Empty([0; 2]));
    // space.set(IVec3::new(112, 106, 112), Cell::Empty([0; 2]));
    // space.set(IVec3::new(113, 106, 112), Cell::Empty([0; 2]));
    space.set(IVec3::new(113, 102, 113), Some([1.0; 3]));
    space.set(IVec3::new(112, 102, 113), Some([1.0; 3]));
    space.set(IVec3::new(112, 102, 112), Some([1.0; 3]));
    space.set(IVec3::new(113, 102, 112), Some([1.0; 3]));
    space.set(IVec3::new(110, 90, 110), Some([1.0, 0.5, 0.3]));
    space.set(IVec3::new(111, 90, 110), Some([1.0, 0.5, 0.3]));
    space.set(IVec3::new(111, 90, 109), Some([0.3, 0.5, 1.0]));
    space.set(IVec3::new(110, 90, 109), Some([1.0, 0.5, 0.3]));
    space.set(IVec3::new(111, 90, 108), Some([1.0, 0.5, 0.3]));
    space.set(IVec3::new(110, 90, 108), Some([1.0, 0.5, 0.3]));
    space.set(IVec3::new(112, 90, 110), Some([1.0, 0.5, 0.3]));
    space.set(IVec3::new(112, 90, 109), Some([1.0, 0.5, 0.3]));
    space.set(IVec3::new(112, 90, 108), Some([1.0, 0.5, 0.3]));
    // space.calculate_distances();

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

    let mut app = App {
        window: None,
        gpu: None,

        yaw: -5.770068,
        pitch: -0.13000016,
        camera: Vec3::new(38.89386, 100.9141, 75.09488),
        sun: Vec3::new(0.8, 10.2743, 3.7).normalize(),
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

        seq: 0,
        iter: 0,
        frame_start: Instant::now(),

        space,
        renderer: None,

        headless: std::env::args().any(|s| s == "headless"),
    };
    if app.headless {
        app.init();
        std::fs::create_dir_all("frames").unwrap();
        loop {
            app.sample();
        }
    } else {
        EventLoop::new().unwrap().run_app(&mut app).unwrap();
    }
}

struct WgpuState {
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: PhysicalSize<u32>,
}

impl WgpuState {
    async fn new(window: Option<&Arc<Window>>) -> Self {
        let size = window.map_or(PhysicalSize::new(853, 480), |w| w.inner_size());

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = window.map(|w| instance.create_surface(w.clone()).unwrap());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: surface.as_ref(),
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

        let config = surface.as_ref().map(|surface| wgpu::SurfaceConfiguration {
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
        });
        if let Some((surface, config)) = surface.as_ref().zip(config.as_ref()) {
            surface.configure(&device, config);
        }

        Self {
            surface,
            device,
            queue,
            config,
            size,
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if let Some((surface, config)) = self.surface.as_ref().zip(self.config.as_mut()) {
            if new_size.width > 0 && new_size.height > 0 {
                self.size = new_size;
                config.width = new_size.width;
                config.height = new_size.height;
                surface.configure(&self.device, config);
            }
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

struct ShowPipeline {
    copy_bind_group_layout: wgpu::BindGroupLayout,
    copy_bind_group: wgpu::BindGroup,
    copy_pipeline: wgpu::RenderPipeline,
    samples_buffer: wgpu::Buffer,
}

impl ShowPipeline {
    fn new(gpu: &WgpuState, accumulator_view: &wgpu::TextureView) -> Option<Self> {
        let config = gpu.config.as_ref()?;

        let samples_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let copy_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let copy_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &copy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&accumulator_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: samples_buffer.as_entire_binding(),
                },
            ],
        });

        let cp_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&copy_bind_group_layout],
                push_constant_ranges: &[],
            });

        let copy_shader = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("copy.wgsl"));

        let copy_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&cp_layout),
                vertex: wgpu::VertexState {
                    module: &copy_shader,
                    entry_point: None,
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &copy_shader,
                    entry_point: None,
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview: None,
                cache: None,
            });

        Some(ShowPipeline {
            copy_bind_group_layout,
            copy_bind_group,
            copy_pipeline,
            samples_buffer,
        })
    }

    fn update_accumulator(&mut self, gpu: &WgpuState, accumulator_view: &wgpu::TextureView) {
        self.copy_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.copy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(accumulator_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.samples_buffer.as_entire_binding(),
                },
            ],
        });
    }

    fn show(&self, gpu: &WgpuState, target: &wgpu::TextureView, samples: usize) {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        gpu.queue.write_buffer(
            &self.samples_buffer,
            0,
            bytemuck::bytes_of(&(samples as f32)),
        );

        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rp.set_pipeline(&self.copy_pipeline);
        rp.set_bind_group(0, &self.copy_bind_group, &[]);
        rp.draw(0..6, 0..1);

        drop(rp);

        gpu.queue.submit([encoder.finish()]);
    }
}
