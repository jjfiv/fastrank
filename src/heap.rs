use std::collections::BinaryHeap;

pub struct ScoringHeap<T> {
    heap: BinaryHeap<T>,
    max_depth: usize,
}

impl<T> ScoringHeap<T>
where
    T: Ord,
{
    pub fn new(max_depth: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(max_depth + 1),
            max_depth: max_depth,
        }
    }
    pub fn len(&self) -> usize {
        self.heap.len()
    }
    pub fn clear(&mut self) {
        self.heap.clear();
    }
    pub fn offer(&mut self, item: T) {
        if self.heap.len() < self.max_depth {
            self.heap.push(item);
            return;
        }
        if self.heap.peek().unwrap() > &item {
            self.heap.push(item);
            if self.heap.len() >= self.max_depth {
                self.heap.pop();
            }
            return;
        }
    }
    #[cfg(test)]
    pub fn into_vec(&mut self) -> Vec<T> {
        let mut output = Vec::with_capacity(self.heap.len());
        self.drain_into_mapping(|x| x, &mut output);
        output
    }

    pub fn drain_unordered(&mut self) -> Vec<T> {
        self.heap.drain().collect()
    }

    pub fn drain_into_mapping<F, M>(&mut self, mut mapper: F, destination: &mut Vec<M>)
    where
        F: FnMut(T) -> M,
    {
        destination.clear();
        while self.heap.len() > 0 {
            destination.push(mapper(self.heap.pop().unwrap()));
        }
        destination.reverse();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heap() {
        let items = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut heap = ScoringHeap::new(4);
        for it in items.iter() {
            heap.offer(it);
        }
        assert_eq!(Some(&&4), heap.heap.peek());
        let shift = 5;
        let mut keep: Vec<i32> = Vec::with_capacity(4);
        heap.drain_into_mapping(|x| x + shift, &mut keep);

        assert_eq!(vec![1 + shift, 2 + shift, 3 + shift, 4 + shift], keep);
    }

    #[test]
    fn test_heap_rev() {
        let items = [9, 8, 7, 6, 5, 4, 3, 2, 1];
        let mut heap = ScoringHeap::new(4);
        for it in items.iter() {
            heap.offer(it);
        }
        assert_eq!(Some(&&4), heap.heap.peek());
        let shift = 5;
        let mut keep: Vec<i32> = Vec::with_capacity(4);
        heap.drain_into_mapping(|x| x + shift, &mut keep);

        assert_eq!(vec![1 + shift, 2 + shift, 3 + shift, 4 + shift], keep);
    }
}
