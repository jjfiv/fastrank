use oorandom::Rand64;

/// Sample with replacement.
pub fn sample_with_replacement<T: Clone>(data: &[T], rand: &mut Rand64, count: usize) -> Vec<T> {
    let mut output = Vec::new();
    let n = data.len() as u64;
    for _ in 0..count {
        let idx = rand.rand_range(0..n) as usize;
        output.push(data[idx].clone());
    }
    output
}

pub fn sample_without_replacement<T: Clone>(data: &[T], rand: &mut Rand64, count: usize) -> Vec<T> {
    let mut in_vec: Vec<T> = data.iter().cloned().collect();
    shuffle(&mut in_vec, rand);
    in_vec.into_iter().take(count).collect()
}

/// Shuffle a vector.
pub fn shuffle<T>(vec: &mut Vec<T>, rand: &mut Rand64) {
    let n = vec.len() as u64;
    for i in 0..n {
        let j = rand.rand_range(i..n) as usize;
        vec.swap(i as usize, j);
    }
}
