pub mod mul;

use std::fmt::Debug;
use std::marker::PhantomData;

use tract_data::TractResult;
use tract_data::internal::TensorView;

use crate::frame::element_wise_helper::TempBuffer;
use crate::{LADatum, LinalgFn};

macro_rules! unicast_impl_wrap {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $run: item) => {
        paste! {
            #[derive(Copy, Clone, Debug)]
            #[allow(non_camel_case_types)]
            pub struct $func;

            impl crate::frame::unicast::UnicastKer<$ti> for $func {
                #[inline(always)]
                fn name() -> &'static str {
                    stringify!($func)
                }
                #[inline(always)]
                fn nr() -> usize {
                    $nr
                }
                #[inline(always)]
                fn alignment_items() -> usize {
                    $alignment_items
                }
                $run
            }
        }
    };
}

pub trait Unicast<T>: Send + Sync + Debug + dyn_clone::DynClone
where
    T: Copy + Debug + PartialEq + Send + Sync,
{
    fn name(&self) -> &'static str;
    fn run(&self, a: &mut [T], b: &[T]) -> TractResult<()>;
}

dyn_clone::clone_trait_object!(<T> Unicast<T> where T: Copy);

#[derive(Debug, Clone, new)]
pub struct UnicastImpl<K, T>
where
    T: LADatum,
    K: UnicastKer<T> + Clone,
{
    phantom: PhantomData<(K, T)>,
}


impl<K, T> UnicastImpl<K, T>
where
    T: LADatum,
    K: UnicastKer<T> + Clone,
{
}
impl<K, T> Unicast<T> for UnicastImpl<K, T>
where
    T: LADatum,
    K: UnicastKer<T> + Clone,
{
    fn name(&self) -> &'static str {
        K::name()
    }
    fn run(&self, a: &mut [T], b: &[T]) -> TractResult<()> {
        unicast_with_alignment(a, b, |a, b| K::run(a, b), K::nr(), K::alignment_bytes())
    }
}

pub trait UnicastKer<T>: Send + Sync + Debug + dyn_clone::DynClone + Clone + 'static
where
    T: LADatum,
{
    fn name() -> &'static str;
    fn alignment_bytes() -> usize {
        Self::alignment_items() * T::datum_type().size_of()
    }
    fn alignment_items() -> usize;
    fn nr() -> usize;
    fn run(a: &mut [T], b: &[T]);
    fn bin() -> Box<dyn Unicast<T>> {
        Box::new(UnicastImpl::<Self, T>::new())
    }
    fn bin_1() -> LinalgFn {
        Box::new(|a: &mut TensorView, b: &TensorView| {
            let a_slice = a.as_slice_mut()?;
            let b_slice = b.as_slice()?;
            Self::bin().run(a_slice, b_slice)
        })
    }
}

std::thread_local! {
    static TMP: std::cell::RefCell<(TempBuffer, TempBuffer)> = std::cell::RefCell::new((TempBuffer::default(), TempBuffer::default()));
}

fn create_incomplete_tile<'a, T: LADatum>(a: &'a mut [T], b: &'a [T], a_prefix_len: usize, b_prefix_len: usize) -> (&'a mut [T], &'a [T], usize) {
    let effective_prefix = if (a_prefix_len == 0) || (b_prefix_len == 0) {
        // One of the two slice is aligned, the target size is the number of unaligned elements of
        // the other slice, the max value between the two.
        a_prefix_len.max(b_prefix_len)
    } else {
        // Both are unaligned, the minimal common subset is the one including elements from a and b
        // so it's the min value between the two.
        a_prefix_len.min(b_prefix_len)
    };
    (&mut a[..effective_prefix], &b[..effective_prefix], effective_prefix)
}


pub(crate) fn unicast_with_alignment<T>(
    a: &mut [T],
    b: &[T],
    f: impl Fn(&mut [T], &[T]),
    nr: usize,
    alignment_bytes: usize,
) -> TractResult<()>
where
    T: LADatum,
{
    if a.is_empty() {
        return Ok(());
    }
    unsafe {
        TMP.with(|buffers| {
            let mut buffers = buffers.borrow_mut();
            buffers.0.ensure(nr * T::datum_type().size_of(), alignment_bytes);
            buffers.1.ensure(nr * T::datum_type().size_of(), alignment_bytes);
            let tmp_a = std::slice::from_raw_parts_mut(buffers.0.buffer as *mut T, nr);
            let tmp_b = std::slice::from_raw_parts_mut(buffers.1.buffer as *mut T, nr);
            let mut compute_via_temp_buffer = |a: &mut [T], b: &[T]| {
                tmp_a[..a.len()].copy_from_slice(a);
                tmp_b[..b.len()].copy_from_slice(b);
                f(tmp_a, tmp_b);
                a.copy_from_slice(&tmp_a[..a.len()])
            };

            let mut num_element_processed = 0;
            let a_prefix_len = a.as_ptr().align_offset(alignment_bytes).min(a.len());
            let b_prefix_len = b.as_ptr().align_offset(alignment_bytes).min(b.len());
            let mut applied_prefix_len = 0;
            if (a_prefix_len > 0) || (b_prefix_len > 0) {
                // Incomplete tile needs to be created to process unaligned data.
                let (mut sub_a, sub_b, applied_prefix) = create_incomplete_tile(a, b, a_prefix_len, b_prefix_len);
                applied_prefix_len = applied_prefix;
                compute_via_temp_buffer(&mut sub_a, &sub_b);
                num_element_processed += applied_prefix_len;
            }

            let num_complete_tiles = (a.len() - applied_prefix_len) / nr;
            if num_complete_tiles > 0 {
                // Process all tiles that are complete.
                let mut sub_a = &mut a[applied_prefix_len..][..(num_complete_tiles * nr)];
                let sub_b = &b[applied_prefix_len..][..(num_complete_tiles * nr)];
                f(&mut sub_a, &sub_b);
                num_element_processed += num_complete_tiles * nr;
            }

            if num_element_processed < a.len() {
                // Incomplete tile needs to be created to process remaining elements.
                compute_via_temp_buffer(
                    &mut a[num_element_processed..],
                    &b[num_element_processed..],
                );
            }
        })
    }
    Ok(())
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::LADatum;
    use proptest::test_runner::{TestCaseError, TestCaseResult};
    use tract_data::internal::*;

    pub fn test_unicast<K: UnicastKer<T>, T: LADatum>(
        a: &[T],
        b: &[T],
        reference: impl Fn(T, T) -> T,
    ) -> TestCaseResult {
        crate::setup_test_logger();
        let op = UnicastImpl::<K, T>::new();
        let expected = a.iter().zip(b.iter()).map(|(a, b)| (reference)(*a, *b)).collect::<Vec<_>>();
        let mut found = a.to_vec();
        op.run(&mut found, b).unwrap();
        tensor1(&found)
            .close_enough(&tensor1(&expected), true)
            .map_err(|e| TestCaseError::fail(e.root_cause().to_string()))?;
        Ok(())
    }
}
