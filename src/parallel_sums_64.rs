// 

use std::error::Error;
use std::time::Instant;


use futures_intrusive::channel::shared::oneshot_channel;

// Read buffer
use bytemuck::cast_slice;


use wgpu::util::{BufferInitDescriptor, DeviceExt};




pub async fn run() -> Result<(), Box<dyn Error>> {

    // Configuration
    let elements_per_group: u32 = 64;
    let n: u32 = 16384;
    assert!(n % elements_per_group == 0, "n must be divisible by {}", elements_per_group);
    let num_groups = n / elements_per_group;

    // Prepare host data (1..=n)
    let input_data: Vec<u32> = (1..=n).collect();
    let byte_size_input = (input_data.len() * std::mem::size_of::<u32>()) as u64;
    let partials_byte_size = (num_groups as u64) * std::mem::size_of::<u32>() as u64;


    // === Initialize wgpu ===
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or("No suitable GPU adapter found")?;
    println!("Adapter: {:?}", adapter.get_info());

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


    // === Buffers ===
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input"),
        contents: cast_slice(&input_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let partials_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("partials"),
        size: partials_byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Staging buffer for readback
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: partials_byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });


    // === Shader & pipeline ===
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/parallel_sums_64.wgsl"));

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
        label: Some("pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
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

    // === Encode compute dispatch ===
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("compute-encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute-pass"),
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        // dispatch num_groups workgroups in X dimension
        cpass.dispatch_workgroups(num_groups, 1, 1);
    }

    // Copy partials -> staging for readback
    encoder.copy_buffer_to_buffer(&partials_buffer, 0, &staging, 0, partials_byte_size);

    // Submit and time GPU execution (submit then wait)
    let gpu_start = Instant::now();
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);
    let gpu_elapsed = gpu_start.elapsed();

    // Now map staging and wait for the callback
    let full_start = Instant::now();
    let slice = staging.slice(..);
    let (sender, receiver) = oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |v| {
        sender.send(v).ok();
    });
    // Ensure callback is processed
    device.poll(wgpu::Maintain::Wait);

    // Wait for map result
    receiver.receive().await.ok_or("map callback failed")??;

    // Read mapped range
    let data = slice.get_mapped_range();
    let partials: &[u32] = cast_slice(&data);

    // Sum partials on CPU
    let gpu_total: u64 = partials.iter().map(|&v| v as u64).sum();

    let full_elapsed = full_start.elapsed();

    // Expected total: sum 1..=n = n*(n+1)/2
    let expected: u64 = (n as u64) * (n as u64 + 1) / 2;

    println!("num_groups = {}", num_groups);
    println!("First partials (up to 16) = {:?}", &partials[..(partials.len().min(16))]);
    println!("GPU dispatch+execute time: {:.3?}", gpu_elapsed);
    println!("Full roundtrip (copy+map+read) time: {:.3?}", full_elapsed);
    println!("Total from GPU partials = {}", gpu_total);
    println!("Expected total = {}", expected);
    println!("Match: {}", gpu_total == expected);

    // Unmap
    drop(data);
    staging.unmap();




    Ok(())
}

