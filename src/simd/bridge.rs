use core::simd::{Which, Simd,simd_swizzle as swizzle, SimdElement, LaneCount, SupportedLaneCount};
// use Which::{First,Second as B};

pub(crate) use core::simd::{
    f32x2 as float32x2_t,
    f32x4 as float32x4_t,
    f64x2 as float64x2_t,
};

use core::simd::{
    f32x2,
    f32x4,
    f64x2,
};

use super::intrinsics::*;

pub type f32x1 = Simd<f32,1>;
pub type f64x1 = Simd<f64,1>;
type float32x1_t = f32x1;
type float64x1_t = f64x1;
type uint64x2_t = Simd<u64,2>;
type uint32x2_t = Simd<u32,2>;

pub fn vcombine_f32(a: f32x2, b: f32x2) -> f32x4 {
    swizzle!(a, b, [Which::First(0), Which::First(1), Which::Second(0), Which::Second(1)])
}

// not in use
pub fn vcombine_f64(a: f64x1, b: f64x1) -> f64x2 {
    swizzle!(a, b, [Which::First(0), Which::Second(0)])
}

// not in use
pub fn vget_high_f64(a: float64x2_t) -> float64x1_t {
  swizzle!(a, [1])
}
pub fn vget_high_f32(a: float32x4_t) -> float32x2_t{
  swizzle!(a, [2,3])
}

//not in use
pub fn vget_low_f64(a: float64x2_t) -> float64x1_t{
  swizzle!(a, [0])
}
pub fn vget_low_f32(a: float32x4_t) -> float32x2_t{
  swizzle!(a, [0,1])
}

// pub unsafe fn vld1_f32(ptr: *const f32) -> float32x2_t{
//   unsafe { *std::mem::transmute(ptr) }
// }
// pub unsafe fn vld1q_f32(ptr: *const f32) -> float32x4_t{
//   unsafe { *std::mem::transmute(ptr) }
// }
// pub unsafe fn vld1q_f64(ptr: *const f64) -> float64x2_t{
//   unsafe { *std::mem::transmute(ptr) }
// }

// not in use
pub unsafe fn vld1q_dup_u64(ptr: *const u64) -> uint64x2_t{
    // unimplemented!(); // used for 5xx
  // dbg!(*ptr);
  let ret = uint64x2_t::splat(*ptr);
  // dbg!(ret);
  ret
}

pub fn vld1q_lane_u64<const LANE: i32>( ptr: *const u64, src: uint64x2_t) -> uint64x2_t{
    // unsafe {
    //   simd_insert(src, LANE as u32, *ptr)
    // }
  unsafe {
      let mut a = src.to_array();
      a[LANE as usize] = *ptr;
      let ret = a.into();
      ret
  }
}

// #[trace] // used, not this one
pub unsafe fn vrev64q_f32(a: float32x4_t) -> float32x4_t {
    // simd_shuffle4!(a, a, [1, 0, 3, 2])
    swizzle!(a, [1,0,3,2])
}

pub unsafe fn vrev64_u32(a: uint32x2_t) -> uint32x2_t {
    // simd_shuffle2!(a, a, [1, 0])
    swizzle!(a, [1,0])
}

use core::ptr::write_unaligned;
pub unsafe fn vst1_f32(ptr: *mut f32, a: float32x2_t) {
    write_unaligned(ptr.cast(), a);
}

pub unsafe fn vst1q_f32(ptr: *mut f32, a: float32x4_t) {
    write_unaligned(ptr.cast(), a);
}
pub unsafe fn vst1q_f64(ptr: *mut f64, a: float64x2_t) {
    write_unaligned(ptr.cast(), a);
}

// not in use
pub unsafe fn vtrn1q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    // unimplemented!(); // used for 9xx
    swizzle!(a, b, [Which::First(0), Which::Second(0), Which::First(2), Which::Second(2)])
}

// not in use
pub unsafe fn vtrn1q_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    // unimplemented!();
    swizzle!(a, b, [Which::First(0), Which::Second(0)])
}

// not in use
pub unsafe fn vtrn2q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    // unimplemented!(); // used for 9xx
    swizzle!(a, b, [Which::First(1), Which::Second(1), Which::First(3), Which::Second(3)])
}

// not in use
pub unsafe fn vtrn2q_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    // unimplemented!();
    swizzle!(a, b, [Which::First(1), Which::Second(1)])
}

pub unsafe fn vld1_f32(a: *const f32) -> f32x2 {
    let Ap: *const [f32; 2] = a.cast();
    let A = *Ap;
    Simd::from(A)
}

pub unsafe fn vld1q_f64(a: *const f64) -> f64x2 {
    let Ap: *const [f64; 2] = a.cast();
    let A = *Ap;
    Simd::from(A)
}

use std::ptr::read_unaligned;
mod rename {
    use super::*;
    pub unsafe fn simd_dup<T, const N: usize>(value: T) -> Simd<T,N>
    where T: SimdElement, LaneCount<N>: SupportedLaneCount {
        Simd::splat(value)
    }

    // not in use
    pub unsafe fn simd_mul_lane<T, const N: usize, const LANE: usize>(a: Simd<T,N>, b: Simd<T,N>) -> Simd<T,N>
        where T: SimdElement, LaneCount<N>: SupportedLaneCount {
            simd_mul(a, Simd::splat(b[LANE]))
    }
}

use rename::simd_dup;
pub unsafe fn vmovq_n_f32(value: f32) -> Simd<f32, 4> {
    simd_dup(value)
}

pub unsafe fn veorq_u32(a: Simd<u32, 4>, b: Simd<u32, 4>) -> Simd<u32, 4> {
    simd_xor(a, b)
}

pub unsafe fn veorq_u64(a: Simd<u64, 2>, b: Simd<u64, 2>) -> Simd<u64, 2> {
    simd_xor(a, b)
}

use rename::simd_mul_lane;
// #[trace] // not this one
pub unsafe fn vmulq_laneq_f64<const LANE: usize>(a: Simd<f64, 2>, b: Simd<f64, 2>) -> Simd<f64, 2> {
    unsafe {
        simd_mul_lane::<f64, 2, LANE>(a, b)
    }
}

pub unsafe fn vfmaq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    // simd_add(simd_mul(a,b), c)
    simd_add(a, simd_mul(b,c))
}

pub unsafe fn vfmaq_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    // simd_add(simd_mul(a,b), c)
    simd_add(a, simd_mul(b,c))
}

pub unsafe fn vfmaq_laneq_f64<const LANE: usize>(
    a: float64x2_t,
    b: float64x2_t,
    c: float64x2_t,
) -> float64x2_t {
    simd_add(a, simd_mul(b, Simd::splat(c[LANE])))
}

pub unsafe fn vld1q_f32(ptr: *const f32) -> float32x4_t {
    read_unaligned(ptr.cast())
}

pub use rename::{
    // simd_dup as vmovq_n_f32,
    simd_dup as vmovq_n_f64,

    // simd_mul_lane as vmulq_laneq_f64,
};


pub(crate) use std::mem::{
    transmute as vreinterpret_f32_u32,
    transmute as vreinterpretq_f32_f64,
    transmute as vreinterpretq_f32_u32,
    transmute as vreinterpretq_f32_u64,
    transmute as vreinterpretq_f64_f32,
    transmute as vreinterpretq_f64_u64,
    transmute as vreinterpretq_u32_f32,
    transmute as vreinterpretq_u64_f32,
    transmute as vreinterpretq_u64_f64,
    transmute as vreinterpret_u32_f32,
};


// use super::intrinsics::*;
pub(crate) use super::intrinsics::{
    simd_add as vadd_f32,
    simd_add as vaddq_f32,
    simd_add as vaddq_f64,

    // not actually used, thank god
    // vcmlaq_f32,
    // vcmlaq_f64,

    // vcmulq_f32,
    // vcmulq_f64,

    // vcombine_f32,
    // vcombine_f64,

    // simd_xor as veorq_u32,
    // simd_xor as veorq_u64,
    simd_xor as veor_u32,


    // vget_high_f32,
    // vget_high_f64,
    // vget_low_f32,
    // vget_low_f64,

    // vld1_f32,
    // vld1q_dup_u64,
    // vld1q_f32,
    // vld1q_f64,
    // vld1q_lane_u64,

    // vmovq_n_f32,
    // vmovq_n_f64,

    simd_mul as vmulq_f32,
    simd_mul as vmulq_f64,

    // vmulq_laneq_f64,

    simd_neg as vneg_f32,
    simd_neg as vneg_f64,
    simd_neg as vnegq_f32,

    // vreinterpret_f32_u32,
    // vreinterpretq_f32_f64,
    // vreinterpretq_f32_u32,
    // vreinterpretq_f32_u64,
    // vreinterpretq_f64_f32,
    // vreinterpretq_f64_u64,
    // vreinterpretq_u32_f32,
    // vreinterpretq_u64_f32,
    // vreinterpretq_u64_f64,
    // vreinterpret_u32_f32,

    // vrev64q_f32,
    // vrev64_u32,

    // vst1_f32,
    // vst1q_f32,
    // vst1q_f64,

    simd_sub as vsub_f32,
    simd_sub as vsubq_f32,
    simd_sub as vsubq_f64,

    // vtrn1q_f32,
    // vtrn1q_f64,
    // vtrn2q_f32,
    // vtrn2q_f64,
};

#[cfg(test)]
#[test]
 fn test_bridge() {
    // const a = float32x4_t::from([1.,2.,3.,4.]);
    // const b = float32x4_t::from([5.,6.,7.,8.]);
    // assert_eq!(simd_)
    unsafe {
        let k: [f32; 2] = [0.,1.];
        let v: float32x2_t = vld1_f32(k.as_ptr());
        assert_eq!(v, float32x2_t::from([0.,1.]));
    }

    unsafe {
        let k: [f64; 2] = [0.,1.];
        let v: float64x2_t = vld1q_f64(k.as_ptr());
        assert_eq!(v, float64x2_t::from([0.,1.]));
    }

    // pub unsafe fn vst1_f32(ptr: *mut f32, a: float32x2_t) {
    //     write_unaligned(ptr.cast(), a);
    // }

    // pub unsafe fn vst1q_f32(ptr: *mut f32, a: float32x4_t) {
    //     write_unaligned(ptr.cast(), a);
    // }
    // pub unsafe fn vst1q_f64(ptr: *mut f64, a: float64x2_t) {
    //     write_unaligned(ptr.cast(), a);
    // }

    unsafe {
        let k: [f32; 2] = [2.,3.];
        let mut buf: [f32; 2] = [5.,7.];
        vst1_f32(buf.as_mut_ptr(), Simd::from(k));
        assert_eq!(buf, k);
    }

    unsafe {
        let k: [f32; 4] = [2.,3.,5.,6.];
        let mut buf: [f32; 4] = [-1.,-1.,-1.,-1.];
        vst1q_f32(buf.as_mut_ptr(), Simd::from(k));
        assert_eq!(buf, k);
    }

    unsafe {
        let k: [f64; 2] = [2.,3.];
        let mut buf: [f64; 2] = [5.,7.];
        vst1q_f64(buf.as_mut_ptr(), Simd::from(k));
        assert_eq!(buf, k);
    }

    unsafe {
    let a = float64x2_t::from([1.,2.]);
    let b = float64x2_t::from([5.,6.]);
    let c = float64x2_t::from([0.,1.]);
    assert_eq!(vfmaq_laneq_f64::<1>(a,b,c), float64x2_t::from([6.,8.]))

    }

    unsafe {
    let a = float32x4_t::from([1.,2.,3.,4.]);
    let b = float32x4_t::from([5.,6.,7.,8.]);
    assert_eq!(vaddq_f32(a,b), float32x4_t::from([6.,8.,10.,12.]));
    assert_eq!(vsubq_f32(a,b), float32x4_t::from([-4.,-4.,-4.,-4.]));
    }

    unsafe {
    let a = float32x4_t::from([1.,2.,3.,4.]);
    let b = float32x4_t::from([5.,6.,7.,8.]);
    assert_eq!(vtrn2q_f32(a,b), float32x4_t::from([2.,6.,4.,8.]))
    }

    unsafe {
    let a = float32x4_t::from([1.,2.,3.,4.]);
    let b = float32x4_t::from([5.,6.,7.,8.]);
    assert_eq!(vneg_f32(vget_low_f32(a)), float32x2_t::from([-1.,-2.]));
    assert_eq!(
        vcombine_f32(vget_high_f32(a), vneg_f32(vget_low_f32(a))),
        float32x4_t::from([3.,4.,-1.,-2.]),

    );
    }
}

