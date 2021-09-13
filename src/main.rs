use std::cmp::{max, min};
use std::env;

use fastrand;
use rayon::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufWriter;
use std::time::SystemTime;
use std::{f64, i32};
use unchecked_index::unchecked_index;

const MIN_LEN: usize = 16;

fn radius(n: usize) -> f64 {
    (1.0 / n as f64).sqrt() * 0.5
}

#[inline(always)]
fn choose<T: Copy>(rng: &mut fastrand::Rng, v: &[T]) -> T {
    unsafe { *v.get_unchecked(rng.usize(0..v.len())) }
}

fn dithering_matrix(size: u32, seed: u64) -> Vec<u32> {
    print!("\nPreprocess... ");

    let size = size as usize;
    let num = size * size;
    let last = num - 1;
    let size_i32 = size as i32;
    let num_u32 = num as u32;

    let mut force_field = unsafe { unchecked_index(vec![0_u64; num]) };

    let mut _result = vec![num_u32; num];
    let mut result = unsafe { unchecked_index(&mut _result) };

    let mut candidates: Vec<usize> = Vec::with_capacity(num);
    let mut free_locations = unsafe { unchecked_index(Vec::with_capacity(num)) };

    for i in 0..num as u32 {
        free_locations.push(i);
    }

    // let _force_field_fn = precalculate_force_field(size);
    let force_field_fn = unsafe { unchecked_index(precalculate_force_field(size)) };

    let idx = |x: i32, y: i32| -> usize { (y * size_i32 + x) as usize };

    let pos = |pid: i32| -> (i32, i32) {
        let quotient = pid / size_i32;
        (pid - (quotient * size_i32), quotient)
    };

    let mut rng: fastrand::Rng = fastrand::Rng::with_seed(seed as u64);

    println!("Ok.");

    // first point
    {
        let x = 0.25;
        let y = rng.f32() * 0.5 + 0.25;

        let minloc_x = (x * size as f32) as i32;
        let minloc_y = (y * size as f32) as i32;
        let minloc_pid = idx(minloc_x, minloc_y);

        let index = free_locations
            .iter()
            .position(|x| *x == minloc_pid as u32)
            .unwrap();
        free_locations.remove(index);

        force_field
            .par_iter_mut()
            .with_min_len(MIN_LEN)
            .enumerate()
            .for_each(|(i, v)| {
                let (x, y) = pos(i as i32);

                let dx = (minloc_x - x).abs();
                let dy = (minloc_y - y).abs();
                let f = force_field_fn[idx(dx, dy)];
                *v += f;
            });

        result[minloc_pid as usize] = 0;
    }
    // other points
    for dither_val in 1..num {
        rng.shuffle(&mut free_locations);
        let half_pos = max(free_locations.len() / 2, 1);
        let half_free_locations = &free_locations[..half_pos];

        let min_field_val = half_free_locations
            .par_iter()
            .with_min_len(MIN_LEN)
            .map(|&pid| force_field[pid as usize])
            .min()
            .unwrap();

        candidates.par_extend(
            half_free_locations
                .par_iter()
                .with_min_len(MIN_LEN * 2)
                .enumerate()
                .filter_map(|(i, &pid)| {
                    if force_field[pid as usize] == min_field_val {
                        Some(i)
                    } else {
                        None
                    }
                }),
        );

        let minloc_i = choose(&mut rng, &candidates);
        candidates.truncate(0);

        let minloc_pid = free_locations[minloc_i];
        let (minloc_x, minloc_y) = pos(minloc_pid as i32);

        force_field
            .par_iter_mut()
            .with_min_len(MIN_LEN)
            .enumerate()
            .for_each(|(i, v)| {
                let (x, y) = pos(i as i32);

                let dx = (minloc_x - x).abs();
                let dy = (minloc_y - y).abs();
                let f = force_field_fn[idx(dx, dy)];
                *v += f;
            });

        result[minloc_pid as usize] = dither_val as u32;

        free_locations.remove(minloc_i);

        print!(
            "\rGenerate matrix: {:>2}% = {} pts",
            100 * (dither_val + 1) / last,
            dither_val
        );
    }

    println!(" ");
    _result
}

fn precalculate_force_field(size: usize) -> Vec<u64> {
    // The integral over the whole force field is smaller than 20. The force field function is therefore
    // scaled such that the sum can be represented in one int and its influence ranges over half the matrix.
    let scale = f64::min(
        std::u64::MAX as f64 / 20.0,
        1.0 / force_fn((size / 2) as f64),
    );

    let mut result = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            let dx = min(x, size - x);
            let dy = min(y, size - y);

            let r2 = dx * dx + dy * dy;

            let r = (r2 as f64).sqrt();
            let f = force_fn(r) * scale;
            result.push(f as u64);
        }
    }
    result
}

#[inline(always)]
fn force_fn(radius: f64) -> f64 {
    f64::exp(-f64::sqrt(2.0 * radius))
}

fn main() {
    let usage = "
Usage:
    ./dithering-matrix <size> <seed>

Make dithring matrix.
";

    let args = env::args();

    if args.len() != 3 {
        println!("\nWrong number of args...\n");
        println!("{}", usage);
        return;
    }

    let size_str = env::args().nth(1).unwrap();
    let size;
    if let Ok(n) = size_str.parse::<u32>() {
        size = n;
    } else {
        println!("\n'{}': wrong type for matrix size.\n", size_str);
        println!("{}", usage);
        return;
    }

    let seed_str = env::args().nth(2).unwrap();
    let seed;
    if let Ok(n) = seed_str.parse::<u64>() {
        seed = n;
    } else {
        println!("\n'{}': wrong type for random seed.\n", seed_str);
        println!("{}", usage);
        return;
    }

    println!("\nGenerate dithering matrix.");
    println!("Size: {0:?}x{0:?} = {1} samples", size, size * size);
    println!("Seed: {}", seed);

    let start = SystemTime::now();

    let matrix = dithering_matrix(size, seed);

    let time = start.elapsed().expect("Failed to get render time?");
    println!(
        "\nGenerting matrix took {:4}s\n",
        time.as_secs() as f64 + time.subsec_nanos() as f64 * 1e-9
    );

    let filename = &format!("matrix_{0}x{0}_{1}.bin", size, seed);
    let mut buffer = BufWriter::new(File::create(filename).unwrap());
    buffer
        .write_all(unsafe {
            ::std::slice::from_raw_parts(matrix.as_ptr() as *const u8, matrix.len() * 4)
        })
        .unwrap();
    buffer.flush().unwrap();
    println!("Write to file: '{}'", filename);
}
