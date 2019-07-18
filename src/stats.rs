//! Derived from https://github.com/jjfiv/chai/blob/6e0e57f0924f9b4c99b5f8b01034681dcd69c76d/src/main/java/ciir/jfoley/chai/math/StreamingStats.java

#[derive(Debug, Clone)]
pub struct StreamingStats {
    num_elements: u64,
    mean: f64,
    s_value: f64,
    max: f64,
    min: f64,
    total: f64,
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self {
            num_elements: 0,
            mean: 0.0,
            s_value: 0.0,
            max: std::f64::MIN,
            min: std::f64::MAX,
            total: 0.0,
        }
    }
}

impl StreamingStats {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push(&mut self, x: f64) {
        debug_assert!(!x.is_nan());

        self.num_elements += 1;
        let old_mean = self.mean;
        let old_s = self.s_value;

        if self.max < x {
            self.max = x;
        }
        if self.min > x {
            self.min = x;
        }
        self.total += x;

        // Knuth TAOCP vol 2, 3ed, p. 232
        if self.num_elements == 1 {
            self.mean = x;
            return;
        }

        self.mean = old_mean + (x - old_mean) / (self.num_elements as f64);
        self.s_value = old_s + (x - old_mean) * (x - self.mean);
    }
    pub fn clear(&mut self) {
        *self = Self::default()
    }
    pub fn get_mean(&self) -> f64 {
        self.mean
    }
    pub fn get_variance(&self) -> Option<f64> {
        if self.num_elements <= 1 {
            return None;
        }
        Some(self.s_value / ((self.num_elements - 1) as f64))
    }
    pub fn get_stddev(&self) -> Option<f64> {
        self.get_variance().map(|var| var.sqrt())
    }
    pub fn get_max(&self) -> Option<f64> {
        if self.num_elements == 0 {
            return None;
        }
        Some(self.max)
    }
    pub fn get_min(&self) -> Option<f64> {
        if self.num_elements == 0 {
            return None;
        }
        Some(self.min)
    }
    pub fn get_total(&self) -> f64 {
        self.total
    }
    pub fn get_count(&self) -> f64 {
        self.num_elements as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const DELTA: f64 = 1e-5;
    fn assert_float_eq(attr: &str, x: f64, y: f64) {
        if (x - y).abs() > DELTA {
            panic!("{} failure: {} != {} at tolerance={}", attr, x, y, DELTA);
        }
    }

    #[test]
    fn test_stats_simple() {
        let data = vec![0, 1, 1, 0];
        let mut ss = StreamingStats::new();
        for x in data {
            ss.push(x as f64);
        }
        assert_float_eq("mean", ss.get_mean(), 0.5);
        assert_float_eq("variance", ss.get_variance().unwrap(), 0.33333);
        assert_float_eq("stddev", ss.get_stddev().unwrap(), 0.57735);
        assert_float_eq("max", ss.get_max().unwrap(), 1.0);
        assert_float_eq("min", ss.get_min().unwrap(), 0.0);
        assert_float_eq("total", ss.get_total(), 2.0);
    }
}
