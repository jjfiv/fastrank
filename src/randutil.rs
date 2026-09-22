use fastrand::Rng;

/// Sample with replacement.
pub fn sample_with_replacement<T: Clone>(data: &[T], rand: &mut Rng, count: usize) -> Vec<T> {
    let mut output = Vec::with_capacity(count);
    let n = data.len();
    for _ in 0..count {
        let idx = rand.usize(0..n);
        output.push(data[idx].clone());
    }
    output
}

/// Sample without replacement.
pub fn sample_without_replacement<T: Clone>(data: &[T], rand: &mut Rng, count: usize) -> Vec<T> {
    let mut values = data.to_vec();
    let count = count.min(values.len());
    for i in 0..count {
        let j = rand.usize(i..values.len());
        values.swap(i, j);
    }
    values.truncate(count);
    values
}

/// Shuffle a vector.
pub fn shuffle<T>(vec: &mut Vec<T>, rand: &mut Rng) {
    rand.shuffle(vec);
}
