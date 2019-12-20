//! Derived from https://github.com/jjfiv/chai/blob/6e0e57f0924f9b4c99b5f8b01034681dcd69c76d/src/main/java/ciir/jfoley/chai/math/StreamingStats.java
use ordered_float::NotNan;
use std::cmp;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputedStats {
    pub num_elements: u64,
    pub mean: f64,
    pub variance: f64,
    pub max: f64,
    pub min: f64,
    pub total: f64,
}

impl ComputedStats {
    pub fn mean(&self) -> NotNan<f64> {
        NotNan::new(self.mean).expect("stddev::mean")
    }
    pub fn stddev(&self) -> NotNan<f64> {
        NotNan::new(self.variance.sqrt()).expect("stddev::NaN")
    }
    pub fn max(&self) -> NotNan<f64> {
        NotNan::new(self.max).expect("stddev::max")
    }
    pub fn min(&self) -> NotNan<f64> {
        NotNan::new(self.min).expect("stddev::min")
    }
}

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
    pub fn finish(&self) -> Option<ComputedStats> {
        if let Some(var) = self.get_variance() {
            Some(ComputedStats {
                num_elements: self.num_elements,
                mean: self.mean,
                max: self.max,
                min: self.min,
                variance: var,
                total: self.total,
            })
        } else {
            None
        }
    }
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

pub struct PercentileStats {
    dataset: Vec<NotNan<f64>>,
}

impl PercentileStats {
    pub fn new(dataset: &[f64]) -> Self {
        let mut dataset: Vec<NotNan<f64>> = dataset
            .iter()
            .map(|f| NotNan::new(*f).expect("PercentileStats::NaN"))
            .collect();
        dataset.sort_unstable();
        PercentileStats { dataset }
    }
    pub fn median(&self) -> f64 {
        self.percentile(0.5)
    }
    pub fn percentile(&self, percentile: f64) -> f64 {
        if percentile < 0.0 || percentile > 1.0 {
            panic!("Bad percentile: {}, should be 0<x<1", percentile);
        }
        let n = percentile * ((self.dataset.len() - 1) as f64);
        let lhs = n.trunc() as usize;
        let rhs = cmp::min(self.dataset.len(), n.ceil() as usize);
        let interp = n.fract();
        if lhs == rhs {
            return self.dataset[lhs].into_inner();
        }
        // LERP:
        (interp * self.dataset[lhs].into_inner()) + (1.0 - interp) * self.dataset[rhs].into_inner()
    }
    pub fn summary(&self) -> (f64, f64, f64, f64, f64) {
        (
            self.percentile(0.05),
            self.percentile(0.25),
            self.percentile(0.5),
            self.percentile(0.75),
            self.percentile(0.95),
        )
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

    #[test]
    fn test_percentile_stats() {
        let data = PercentileStats::new(&(0..10).map(|i| i as f64).collect::<Vec<_>>());
        assert_float_eq("median", data.median(), 4.5);
        let data = PercentileStats::new(&(0..9).map(|i| i as f64).collect::<Vec<_>>());
        assert_float_eq("median", data.median(), 4.0);
    }
}
