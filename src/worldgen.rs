use std::mem::MaybeUninit;
use std::time::Instant;

use ndarray::{prelude::*, DataOwned, RawData, Zip};
use noise::{Fbm, MultiFractal, NoiseFn, OpenSimplex};
use ordered_float::OrderedFloat;

use crate::svo::{Node, SvoCell, SvoSpace};

pub fn generate() -> SvoSpace {
    let t0 = Instant::now();
    let mut world = SvoSpace::new();

    let scale_noise = OpenSimplex::new(0xafe29979);
    let noise = Fbm::<OpenSimplex>::new(0x1c766e2e).set_octaves(7);

    let log2_size = 13;
    let heightmap = par_array((1 << log2_size, 1 << log2_size), |(i, j)| {
        let base = scale_noise.get([i as f64 / 2048.0, j as f64 / 2048.0]) + 1.0;
        let scale = base.powi(3);
        let n = noise.get([i as f64 / 512.0, j as f64 / 512.0]) + 1.0 + base;
        let v = (n * 256.0 * scale + 64.0) as usize;
        (v, v)
    });

    let mut min_max_maps = vec![heightmap];
    while min_max_maps.last().unwrap().dim().0 > 1 {
        let last_map = min_max_maps.last().unwrap();
        let new_map = par_array((last_map.dim().0 / 2, last_map.dim().1 / 2), |(i, j)| {
            let mut min = usize::MAX;
            let mut max = 0;
            for ii in 2 * i..2 * i + 2 {
                for jj in 2 * j..2 * j + 2 {
                    min = min.min(last_map[(ii, jj)].0);
                    max = max.max(last_map[(ii, jj)].1);
                }
            }
            (min, max)
        });
        min_max_maps.push(new_map);
    }

    let t1 = Instant::now();
    println!("heightmaps in {:.3?}", t1 - t0);

    let node = generate_node(
        &mut world,
        min_max_maps.iter().map(|m| m.view()).collect(),
        0,
    );
    world.set_root(node);

    let t2 = Instant::now();
    println!("octree in {:.3?}", t2 - t1);

    println!("worldgen in {:.3?}", t2 - t0);
    println!("world size: {} mib", world.mem_usage() / 1024 / 1024);
    world
}

pub fn generate_node(
    world: &mut SvoSpace,
    hmaps: Vec<ArrayView2<(usize, usize)>>,
    base_y: usize,
) -> Option<Node> {
    let height = hmaps[0].dim().0;
    if base_y + height <= hmaps.last().unwrap()[(0, 0)].0 {
        let mut node = world.create_node(SvoCell::Block([0.5; 3].map(OrderedFloat)));
        for _ in 1..hmaps.len() {
            node = world.create_node(SvoCell::Children([node; 8]));
        }
        return node;
    } else if base_y >= hmaps.last().unwrap()[(0, 0)].1 {
        return None;
    }
    let mut children = [None; 8];
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let i = z * 4 + y * 2 + x;
                children[i] = generate_node(
                    world,
                    hmaps[..hmaps.len() - 1]
                        .iter()
                        .map(|m| {
                            let half = m.dim().0 / 2;
                            m.slice(s![x * half..(x + 1) * half, z * half..(z + 1) * half])
                        })
                        .collect(),
                    base_y + y * hmaps[0].dim().0 / 2,
                );
            }
        }
    }
    world.create_node(SvoCell::Children(children))
}

fn par_array<Sh, D, T>(shape: Sh, f: impl Fn(D::Pattern) -> T + Send + Sync) -> Array<T, D>
where
    D: Dimension + Copy,
    D::Pattern: Send,
    Sh: ShapeBuilder<Dim = D>,
    T: Send,
{
    let mut array = Array::uninit(shape);
    Zip::indexed(&mut array).par_for_each(|i, v| *v = MaybeUninit::new(f(i)));
    unsafe { array.assume_init() }
}
