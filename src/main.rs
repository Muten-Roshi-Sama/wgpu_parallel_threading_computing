// use pollster::FutureExt;
use pollster::block_on;

mod introduction;
mod parallel_sums_64;

fn main() {
    env_logger::init();
    
    // Call the async function from the library crate and block on it
    // block_on(introduction::run()).unwrap();
    /* Result:
        First 32 output values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        Mismatches = 0
     */

    block_on(parallel_sums_64::run()).unwrap();

    // #[cfg(feature="sort")]
    // compute::sort::run().block_on().unwrap();


}








