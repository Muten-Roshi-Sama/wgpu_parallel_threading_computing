// introduction.rs

use std::error::Error;
// use flume::bounded;
use futures_intrusive::channel::shared::oneshot_channel;
use bytemuck::cast_slice;
use wgpu::util::{BufferInitDescriptor, DeviceExt};




pub async fn run() -> Result<(), Box<dyn Error>> {
    // Create instance / adapter / device + queue
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .ok_or("No suitable adapter found")?;
    let (device, queue) = adapter.request_device(&Default::default(), None).await?;

    // Load shader and create compute pipeline
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/introduction.wgsl"));

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Introduction Compute Pipeline"),
        layout: None,
        module: &shader,
        entry_point: "main",
        // compilation_options: Default::default(),
        // cache: Default::default(),
    });

    // Prepare input data (u32 values)
    let input_data = (0..10_000u32).collect::<Vec<_>>();
    let element_count = input_data.len();
    let byte_size = (element_count * std::mem::size_of::<u32>()) as u64;

    // GPU buffers
    let input_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("input"),
        contents: bytemuck::cast_slice(&input_data),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // Staging buffer for readback (COPY_DST + MAP_READ)
    let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Bind group (the pipeline's WGSL must expect these bindings)
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    // -----------------------------------------------------------------------
    // Encode compute pass
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("compute-encoder") });
    
    // Compute how many workgroups we need (shader assumes 64 threads per workgroup)
    let num_dispatches = ((element_count as u32) + 63) / 64;

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("compute-pass") });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_dispatches, 1, 1);
    }

    // Copy result to staging buffer for CPU read
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &temp_buffer, 0, byte_size);

    // Submit and wait
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Map staging buffer and wait for the callback via oneshot channel
    let slice = temp_buffer.slice(..);
    let (sender, receiver) = oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        // send the result (Ok or Err) through the channel
        sender.send(res).ok();
    });

    // Ensure the device processes the mapping callback
    device.poll(wgpu::Maintain::Wait);

    // Wait for the mapping result
    let map_result = receiver.receive().await.ok_or("Failed to get mapping result")?;
    if let Err(e) = map_result {
        return Err(Box::from(format!("buffer map error: {:?}", e)));
    }

    // Read mapped data
    let data = slice.get_mapped_range();
    let output_u32: &[u32] = cast_slice(&data);
    println!("First 32 output values: {:?}", &output_u32[..32.min(output_u32.len())]);

    // Verify data (here we assert equality between input and output)
    // Modify this check according to what your shader actually computes
    let mismatches: usize = output_u32.iter().zip(input_data.iter()).filter(|(o,i)| o != i).count();    println!("Mismatches = {}", mismatches);
    // assert_eq!(&input_data[..], output_u32);

    // Unmap the buffer so GPU can reuse it
    drop(data);
    temp_buffer.unmap();

    log::info!("Success!");



    Ok(())
}




