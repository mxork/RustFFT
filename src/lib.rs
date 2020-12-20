#![cfg_attr(all(feature = "bench", test), feature(test))]

//! RustFFT is a high-performance FFT library written in pure Rust.
//!
//! This is an experimental release of RustFFT that enables AVX acceleration. It currently requires a nightly compiler,
//! mainly for the `min_specialization` feature. The eventual plan is to release this experimental version as version 5.0 of RustFFT,
//! but that will not happen until it compiles on stable Rust.
//!
//! ### Usage
//!
//! The recommended way to use RustFFT is to create a [`FftPlanner`](crate::FftPlanner) instance and then call its
//! [`plan_fft`](crate::FftPlanner::plan_fft) method. This method will automatically choose which FFT algorithms are best
//! for a given size and initialize the required buffers and precomputed data.
//!
//! ```
//! // Perform a forward FFT of size 1234
//! use rustfft::{FftPlanner, num_complex::Complex};
//!
//! let mut planner = FftPlanner::new(false);
//! let fft = planner.plan_fft(1234);
//!
//! let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 1234];
//! fft.process_inplace(&mut buffer);
//! ```
//! The planner returns trait objects of the [`Fft`](crate::Fft) trait, allowing for FFT sizes that aren't known
//! until runtime.
//! 
//! RustFFT also exposes individual FFT algorithms. For example, if you know beforehand that you need a power-of-two FFT, you can
//! avoid the overhead of the planner and trait object by directly creating instances of the [`Radix4`](crate::algorithm::Radix4) algorithm:
//!
//! ```
//! // Computes a forward FFT of size 4096
//! use rustfft::{Fft, num_complex::Complex, algorithm::Radix4};
//!
//! let fft = Radix4::new(4096, false);
//!
//! let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 4096];
//! fft.process_inplace(&mut buffer);
//! ```
//!
//! For the vast majority of situations, simply using the [`FftPlanner`](crate::FftPlanner) will be enough, but
//! advanced users may have better insight than the planner into which algorithms are best for a specific size. See the
//! [`algorithm`](crate::algorithm) module for a complete list of scalar (non-AVX) algorithms implemented by RustFFT. As noted below,
//! bypassing the planner will prevent any use of AVX instructions.
//!
//! ### AVX Acceleration
//!
//! RustFFT includes algorithms designed to take advantage of the AVX instruction set. To use AVX,
//! simply plan a FFT through the [`FftPlanner`](crate::FftPlanner) on a machine which supports the `avx` and `fma` features.
//! Benchmarking shows that while using AVX, RustFFT computes FFTs at equal or faster speeds than [FFTW](http://www.fftw.org/)!
//!
//! If your machine doesn't support AVX, the [`FftPlanner`](crate::FftPlanner) will fall back to scalar algorithms. If you'd rather just not compute
//! a FFT at all if AVX isn't available, you can instead create an instance of the [`FftPlannerAvx`](crate::FftPlannerAvx) struct and plan through that.
//!
//! For the time being, individual AVX algorithms can't be constructed outside of the planner. This may change eventually.
//!
//! ### AVX Performance Tips
//!
//! The performance of any given FFT size is heavily dependent on that size's [prime factorization](https://en.wikipedia.org/wiki/Prime_number#Unique_factorization).
//! It's common in FFT libraries (including RustFFT's scalar implementation) for powers of two to be the fastest, but that's not the case for RustFFT's AVX implementation.
//! RustFFT's AVX implementation is fastest when computing any size of the form `2^n * 3^m` -- which includes powers of two, but isn't restricted to them.
//!
//! Any FFT where all prime factors are 11 or smaller (For example, 10164 = `2*2*3*7*11*11`) can be computed very quickly.
//! 
//! All other FFT sizes, such as prime numbers, and composite numbers where the largest prime factor is greater than 11, will be noticeably slower.
//! For example, 1201 (prime number) takes 3x longer to compute than 1200 = `2*2*2*2*3*5*5`.
//! However, they will still be computed in O(nlogn) time, they still benefit from AVX acceleration, 
//! and according to benchmarks we've run, are still faster than the same size computed by FFTW.
//!
//! ### Normalization
//!
//! RustFFT does not normalize outputs. Callers must manually normalize the results by scaling each element by
//! `1/len().sqrt()`. Multiple normalization steps can be merged into one via pairwise multiplication, so when
//! doing a forward FFT followed by an inverse callers can normalize once by scaling each element by `1/len()`
//!
//! ### Output Order
//!
//! Elements in the output are ordered by ascending frequency, with the first element corresponding to frequency 0.

#![feature(min_specialization)]
#![feature(maybe_uninit_extra)]
#![feature(maybe_uninit_slice)]


pub use num_complex;
pub use num_traits;


#[macro_use]
mod common;

/// Individual FFT algorithms
pub mod algorithm;
mod math_utils;
mod array_utils;
mod plan;
mod twiddles;

use num_complex::Complex;
use num_traits::Zero;

pub use crate::plan::FftPlanner;
pub use crate::common::FFTnum;



/// A trait that allows FFT algorithms to report their expected input/output size
pub trait Length {
    /// The FFT size that this algorithm can process
    fn len(&self) -> usize;
}

/// A trait that allows FFT algorithms to report whether they compute forward FFTs or inverse FFTs
pub trait IsInverse {
    /// Returns false if this instance computes forward FFTs, true for inverse FFTs
    fn is_inverse(&self) -> bool;
}

/// Trait for algorithms that compute FFTs.
/// 
/// This trait has two main methods:
/// - [`process_inplace(buffer)`](crate::Fft::process_inplace) will compute a FFT using `buffer` as input and store the result back into `buffer`.
/// - [`process(input, output)`](crate::Fft::process) will compute a FFT using `input` as input and store the result into `output`. 
///
/// Both methods may need to allocate additional scratch space. If you'd like re-use that allocation across multiple FFT computations, call 
/// `process_inplace_with_scratch` or `process_with_scratch`, respectively.
pub trait Fft<T: FFTnum>: Length + IsInverse + Sync + Send {
    /// Convenience method that allocates the required scratch space and and calls `self.process_with_scratch`.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling.
    ///
    /// The output is not normalized. Callers must manually normalize the results by scaling each element by
    /// `1/len().sqrt()`.
    ///
    /// # Panics
    /// 
    /// This method panics if
    /// - `input.len() != self.len()`
    /// - `output.len() != self.len()`
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::zero(); self.get_out_of_place_scratch_len()];
        self.process_with_scratch(input, output, &mut scratch);
    }
    /// Convenience method that allocates the required scratch space and calls `self.process_inplace_with_scratch`.
    ///
    /// The output is not normalized. Callers must manually normalize the results by scaling each element by
    /// `1/len().sqrt()`.
    ///
    /// # Panics
    /// 
    /// This method panics if
    /// - `buffer.len() != self.len()`
    fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::zero(); self.get_inplace_scratch_len()];
        self.process_inplace_with_scratch(buffer, &mut scratch);
    }

    /// Computes an FFT on the data in `input` and stores the result into `output`.
    ///
    /// This method uses both the `input` buffer and `scratch` buffer as scratch space, so the contents of both should
    /// be considered garbage after calling.
    ///
    /// The output is not normalized. Callers must manually normalize the results by scaling each element by
    /// `1/len().sqrt()`.
    ///
    /// # Panics
    /// 
    /// This method panics if
    /// - `input.len() != self.len()`
    /// - `output.len() != self.len()`
    /// - `scratch.len() < self.get_out_of_place_scratch_len()`
    fn process_with_scratch(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], scratch: &mut [Complex<T>]);

    /// Computes an FFT on `buffer` and stores the result back into `buffer`.
    ///
    /// This method uses the `scratch` buffer as scratch space, so the contents of `scratch` should be considered garbage
    /// after calling.
    ///
    /// The output is not normalized. Callers must manually normalize the results by scaling each element by
    /// `1/len().sqrt()`.
    ///
    /// # Panics
    /// 
    /// This method panics if
    /// - `buffer.len() != self.len()`
    /// - `scratch.len() < self.get_inplace_scratch_len()`
    fn process_inplace_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]);

    /// Divides `input` and `output` into chunks of size `self.len()`, computes an FFT on each input chunk,
    /// and stores the result in the corresponding output chunk.
    ///
    /// This method uses both the `input` buffer and `scratch` buffer as scratch space, so the contents of both should
    /// be considered garbage after calling.
    ///
    /// The output is not normalized. Callers must manually normalize the results by scaling each element by
    /// `1/len().sqrt()`.
    ///
    /// # Panics
    /// 
    /// This method panics if
    /// - `input.len() % self.len() != 0`
    /// - `output.len() != input.len()`
    /// - `scratch.len() < self.get_out_of_place_scratch_len()`
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], scratch: &mut [Complex<T>]);

    /// Divides `buffer` into chunks of size `self.len()`, computes an FFT on each chunk, and stores the result back
    /// into `buffer`.
    ///
    /// This method uses the `scratch` buffer as scratch space, so its contents should be considered garbage after
    /// calling.
    ///
    /// The output is not normalized. Callers must manually normalize the results by scaling each element by
    /// `1/len().sqrt()`.
    ///
    /// # Panics
    /// 
    /// This method panics if
    /// - `buffer.len() % self.len() != 0`
    /// - `scratch.len() < self.get_inplace_scratch_len()`
    fn process_inplace_multi(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]);

    /// Returns the size of the scratch buffer required by `process_inplace_with_scratch` and `process_inplace_multi`
    fn get_inplace_scratch_len(&self) -> usize;

    /// Returns the size of the scratch buffer required by `process_with_scratch` and `process_multi`
    fn get_out_of_place_scratch_len(&self) -> usize;
}

// Algorithms implemented to use AVX instructions. Only compiled on x86_64.
#[cfg(target_arch="x86_64")]
mod avx;

// When we're not on avx, keep a stub implementation around that just does nothing
#[cfg(not(target_arch="x86_64"))]
mod avx{
   pub mod avx_planner {
        use crate::{Fft, FFTnum};
        use std::sync::Arc;
        pub struct FftPlannerAvx<T: FFTnum> {
            _phantom: std::marker::PhantomData<T>,
        }
        impl<T: FFTnum> FftPlannerAvx<T> {
            pub fn new(_inverse: bool) -> Result<Self, ()> {
                Err(())
            }
            pub fn plan_fft(&mut self, _len: usize) -> Arc<dyn Fft<T>> {
                unreachable!();
            }
        }
    }
}

pub use self::avx::avx_planner::FftPlannerAvx;

#[cfg(test)]
mod test_utils;
