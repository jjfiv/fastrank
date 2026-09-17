use fastrand::Rng;

/// Sample with replacement.
pub fn sample_with_replacement<T: Clone>(data: &[T], rand: &mut Rng, count: usize) -> Vec<T> {
    let mut output = Vec::new();
    let n = data.len();
    for _ in 0..count {
        let idx = rand.usize(0..n);
        output.push(data[idx].clone());
    }
    output
}

pub fn sample_without_replacement<T: Clone>(data: &[T], rand: &mut Rng, count: usize) -> Vec<T> {
    let mut in_vec: Vec<T> = data.to_vec();
    shuffle(&mut in_vec, rand);
    in_vec.into_iter().take(count).collect()
}

/// Shuffle a vector.
pub fn shuffle<T>(vec: &mut Vec<T>, rand: &mut Rng) {
    let n = vec.len();
    for i in 0..n {
        let j = rand.usize(i..n);
        vec.swap(i, j);
    }
}
