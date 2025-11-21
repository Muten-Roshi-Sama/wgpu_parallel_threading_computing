// src/parallel_sums_64.rs
use std::error::Error;
use std::time::Instant;

use futures_intrusive::channel::shared::oneshot_channel;
use bytemuck::cast_slice;
use wgpu::util::DeviceExt;



// 1. Host gives array = [1,2,3,...,16384]
// 2. chunk this array into 256 groups of 64
// 3. each workgroup reduces this sum into a partial sum.
// 4. repeat until you get one answer

// OUTPUT:
    // num_groups = 256
    // First partials (up to 16) = [2080, 6176, 10272, 14368, 18464, 22560, 26656, 30752, 34848, 38944, 43040, 47136, 51232, 55328, 59424, 63520]
    // GPU dispatch+execute time: 3.009ms
    // Full roundtrip (copy+map+read) time: 98.700µs
    // Total from GPU partials = 134225920
    // expected answer = 134225920


// ====================================================================================

/// Initialize WebGPU and return (adapter, device, queue).
pub async fn init_wgpu() -> Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue), Box<dyn Error>> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or("No suitable GPU adapter found")?;
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await?;
    Ok((adapter, device, queue))
}

/// Create input, partials and staging buffers.
/// - `input` is a slice of u32 values (host).
/// - `num_groups` is how many partial sums (one per workgroup).
pub fn create_buffers(
    device: &wgpu::Device,
    input: &[u32],
    num_groups: u32,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let input_bytes = cast_slice(input);
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input"),
        contents: input_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let partials_size = (num_groups as u64) * std::mem::size_of::<u32>() as u64;
    let partials_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("partials"),
        size: partials_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: partials_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    (input_buffer, partials_buffer, staging)
}

/// Create the compute pipeline and bind group for `parallel_sums_64.wgsl`.
/// Returns (pipeline, bind_group).
pub fn create_pipeline_and_bindgroup(
    device: &wgpu::Device,
    input_buffer: &wgpu::Buffer,
    partials_buffer: &wgpu::Buffer,
) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
    // compile-time include keeps path issues away
    let module = device.create_shader_module(wgpu::include_wgsl!("../shaders/parallel_sums_64.wgsl"));

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl"),
        entries: &[
            // binding 0: input (read-only storage)
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 1: partials (read-write storage)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: "main",
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: partials_buffer.as_entire_binding(),
            },
        ],
    });

    (pipeline, bind_group)
}

/// Dispatch the compute shader, copy partials to staging, map and return a Vec<u32> of partials.
/// This function performs the submit + wait and reads back the CPU-visible data.
pub async fn dispatch_and_read_partials(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    partials_buffer: &wgpu::Buffer,
    staging: &wgpu::Buffer,
    num_groups: u32,
) -> Result<(Vec<u32>, std::time::Duration, std::time::Duration), Box<dyn Error>> {
    // Encode compute pass and copy to staging
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("compute-encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute-pass"),
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(num_groups, 1, 1);
    }

    let partials_bytes = (num_groups as u64) * std::mem::size_of::<u32>() as u64;
    encoder.copy_buffer_to_buffer(partials_buffer, 0, staging, 0, partials_bytes);

    // Submit and wait for compute
    let gpu_start = Instant::now();
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);
    let gpu_elapsed = gpu_start.elapsed();

    // Map staging and read
    let full_start = Instant::now();
    let slice = staging.slice(..);
    let (sender, receiver) = oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |v| {
        sender.send(v).ok();
    });
    device.poll(wgpu::Maintain::Wait);
    receiver.receive().await.ok_or("map callback failed")??;

    let data = slice.get_mapped_range();
    let partials: Vec<u32> = cast_slice(&data).to_vec();
    let full_elapsed = full_start.elapsed();

    drop(data);
    staging.unmap();

    Ok((partials, gpu_elapsed, full_elapsed))
}

/// Thin orchestrator: uses the helpers above to run the 64-thread reduction, sum partials and print results.
pub async fn run() -> Result<(), Box<dyn Error>> {
    // config
    let elements_per_group: u32 = 64;
    let n: u32 = 16384;
    assert!(n % elements_per_group == 0);
    let num_groups = n / elements_per_group;

    // prepare input on host
    let input_data: Vec<u32> = (1..=n).collect();

    // init GPU
    let (adapter, device, queue) = init_wgpu().await?;
    println!("Adapter: {:?}", adapter.get_info());

    // create buffers
    let (input_buffer, partials_buffer, staging) = create_buffers(&device, &input_data, num_groups);

    // pipeline + bind group
    let (pipeline, bind_group) = create_pipeline_and_bindgroup(&device, &input_buffer, &partials_buffer);

    // dispatch and read partials
    let (partials, gpu_elapsed, full_elapsed) = dispatch_and_read_partials(
        &device,
        &queue,
        &pipeline,
        &bind_group,
        &partials_buffer,
        &staging,
        num_groups,
    )
    .await?;

    // sum and verify on host
    let gpu_total: u64 = partials.iter().map(|&v| v as u64).sum();
    let expected: u64 = (n as u64) * (n as u64 + 1) / 2;

    println!("num_groups = {}", num_groups);
    println!("First partials (up to 16) = {:?}", &partials[..partials.len().min(16)]);
    println!("GPU dispatch+execute time: {:.3?}", gpu_elapsed);
    println!("Full roundtrip (copy+map+read) time: {:.3?}", full_elapsed);
    println!("Total from GPU partials = {}", gpu_total);
    println!("Expected answer = {}", expected);
    println!("Match: {}", gpu_total == expected);

    Ok(())
}