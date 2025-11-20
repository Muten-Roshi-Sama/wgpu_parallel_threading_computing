//parallel_sums_64.wgsl
//

// GOAL :
    // Input : an array of u32 values to add
    // Output : partial sum of each WG


// Pseudo-Code :
    // Host: 
        // - split input into chunk of 64      (ex. 128 values)
        // - dispatch one workgroup per chunk. ()
        // - 

    // Shader :
        // Each workgroup:
            // 1) Each of the 64 threads loads one input value into fast shared-memory
            // 2) Selected threads do a tree-reduction inside shared-memory.
            // 3) Thread 0 writes the group's sum into `partial_sums[group_id.x]`.

// _______________________________________________________________________________
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> partial_sums: array<u32>;

// u32 Fast-memory slot shared by the 64 threads in workgroup.
var<workgroup> shared_data: array<u32, 64>;

// This shader runs with 64 * 1 * 1 threads per workgroup
@compute @workgroup_size(64, 1, 1) 


fn main(@builtin(local_invocation_id) local_id: vec3<u32>,     // thread id inside WG (0..63)
        @builtin(global_invocation_id) global_id: vec3<u32>,  // global thread id -> index into input
        @builtin(workgroup_id) group_id: vec3<u32>) {        // workgroup id -> index into partial_sums

    // ---------- Step 1 : load (one-element per thread) ---------
    let idx = global_id.x;         // index of data to load from input
    let local_index = local_id.x; // index inside WG (0..63)

    // Load data into shared memory
    shared_data[local_index] = input[idx];  // thread reads value from input and Write to shared_data
    workgroupBarrier();                    // sync


    // ---------- Step 2: in-place tree-reduction in shared memory ----------
    // Keep combining pairs of values apart from multiples of stride
    // Reduction loop
    var stride = 1u;
    while (stride < 64u) {

        let index = 2u * stride * local_index;   // which threads to use (halves each loop)

        // Only perform the add if the partner (index+stride) exists in the array.
        if (index + stride < 64u) {
            // Add the partner element into the target slot
            shared_data[index] += shared_data[index + stride];
        }
        workgroupBarrier();
        // Next pass combines pairs that are twice as far apart
        stride = stride * 2u;
    }

    // ---------- Step 3: write the group's final sum ----------
    // After the loop, shared_data[0] contains the sum of the 64 values.
    if (local_index == 0u) {
        partial_sums[group_id.x] = shared_data[0];
    }
}


/* ----------- Execution Example -----------

Let input = [a0, a1, a2, a3, a4, a5, a6, a7]
Let shared_data = input

Pass 1: stride = 1
  index = 2 * 1 * local_index-----------------(0, 2, 4, 6)
  threads with local_index = 0..3 compute:
    index(0) = 0 :  shared[0] = a0 + shared[1] = a0 + a1
    index(1) = 2 :  shared[2] = a2 + shared[3] = a2 + a3
    index(2) = 4 :  shared[4] = a4 + shared[5] = a4 + a5
    index(3) = 6 :  shared[6] = a6 + shared[7] = a6 +a7
  After barrier, shared ~ [a0_1, a1, a2_3, a3, a4_5, a5, a6_7, a7]

Pass 2: stride = 2
  index = 2 * 2 * local_index---------------(0, 4, 8, 16)
  threads with local_index = 0..1 compute:
    index(0) = 0 : shared[0] += shared[2]    -> shared[0] = (a0+a1) + (a2+a3)
    index(1) = 4 : shared[4] += shared[6]    -> shared[4] = (a4+a5) + (a6+a7)
  After barrier, shared ~ [sum0..3, ., ., ., sum4..7, ., ., .]

Pass 3: stride = 4
  index = 2 * 4 * local_index
  thread local_index = 0 computes:
    index(0) = 0 : shared[0] += shared[4]    -> shared[0] = total sum of a0..a7

End: shared[0] contains the full sum for this chunk; thread 0 writes it out.

*/
