//! Utilities related to random numbers.

use num;
use rand;
use rand::distributions::{Distribution, Uniform};
use rand::distributions::uniform::SampleUniform;

/// Samples a given number of indices from the given probability distribution.
/// The distribution does not have to be normalized.
pub fn draw_from_distribution<F>(pdf: &[F], n_samples: usize) -> Vec<usize>
where F: num::Float + SampleUniform
{
    let cdf: Vec<F> = pdf.iter().scan(F::zero(),
        |state, &value| {
            *state = *state + value;
            Some(*state)
        }
    ).collect();

    let uniform_cdf_values = Uniform::new(cdf[0], cdf[cdf.len() - 1]);
    let rng = rand::thread_rng();
    uniform_cdf_values.sample_iter(rng).take(n_samples).map(
        |sampled_cdf_value| {
            match cdf.binary_search_by(|cdf_value| cdf_value.partial_cmp(&sampled_cdf_value).unwrap()) {
                Result::Ok(exact_idx) => exact_idx,
                Result::Err(adjacent_idx) => adjacent_idx
            }
        }
    ).collect()
}