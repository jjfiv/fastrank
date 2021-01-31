//! This module defines data structures and parsing code for LibSVM input formats.
//!
//! In the following example libsvm file, the first instance has a comment "instance zero" and
//! features 1,2, and 4 are defined. It also has a positive label (+1). The second instance has a negative
//! label (-1) and features 2 and 3 are defined.
//!
//! ```text
//! 1 1:7 2:3 4:-1 # instance zero
//! -1 2:1 3:14
//! ```
//!
//!
//!
use ordered_float::{FloatIsNan, NotNan};
use fast_float;
use std::fmt;
use std::io;
use std::num;

/// Custom error class to produce readable errors when input files are not correctly formatted.
#[derive(Debug)]
pub enum ParseError {
    /// Any number of IO errors that could occur from a lower-level system.
    IO(io::Error),
    /// Something is wrong with the label; it couldn't be parsed as a float.
    Label(num::ParseFloatError),
    /// Why would you have NaN labels?
    LabelIsNan(FloatIsNan),
    /// A token was found without a colon; therefore it can't be a valid fnum:fvalue pair.
    FeatureNoColon(),
    /// A feature was defined multiple times (same index) and we don't know what to do.
    MultipleDefinitions(Feature, Feature),
    /// A feature number could not be parsed.
    FeatureNum(num::ParseIntError),
    /// A feature value could not be parsed.
    FeatureVal(num::ParseFloatError),
    /// A feature value could not be parsed (v2).
    FeatureValNotFloat(fast_float::Error)
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// This enum represents errors that can occur when processing a whole file.
#[derive(Debug)]
pub enum FileParseError {
    /// A file error with no line context; probably discovered on opening.
    ReadErr(io::Error),
    /// A file error with a particular line context.
    LineIO(u64, io::Error),
    /// A semantic error with a particular line.
    LineParseError(u64, ParseError),
}

impl fmt::Display for FileParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for FileParseError {
    fn description(&self) -> &str {
        "LibSVM file issue."
    }
}

/// Represents a sparse feature: numerNoic id and floating point value.
#[derive(Debug, Clone)]
pub struct Feature {
    /// The number identifying this feature.
    pub idx: u32,
    /// The value of this numerical feature.
    pub value: f32,
}

impl Feature {
    /// Parse an individual LibSVM feature entry.
    pub fn parse(tok: &str) -> Result<Feature, ParseError> {
        match tok.find(':') {
            Some(idx) => {
                let (fid_str, fval_str) = tok.split_at(idx);
                let fid = fid_str.parse::<u32>().map_err(ParseError::FeatureNum)?;
                // Ditch first character of fval_str, which will be :
                let fval: f32 = fast_float::parse(&fval_str[1..])
                    .map_err(ParseError::FeatureValNotFloat)?;
                Ok(Feature {
                    idx: fid,
                    value: fval,
                })
            }
            None => {
                println!("FeatureNoColon: {}", tok);
                // TODO flag for boolean features?
                Err(ParseError::FeatureNoColon())
            }
        }
    }
}

/// Represents a line of input from a LibSVM file: a label (for classification or regression) a
/// number of features and then an optional comment, which is used for instance ids, typically,
/// though not all parsers support comments.
#[derive(Debug)]
pub struct Instance {
    /// This is a float value to support regression inputs.
    pub label: NotNan<f32>,
    /// This is the query identifier provided with the instance.
    pub query: Option<String>,
    /// This is the sparse feature representation. These *are* expected to be sorted and unique.
    /// We check to see if they are sorted as we read them from an input file.
    pub features: Vec<Feature>,
    /// This is the comment, if available.
    pub comment: Option<String>,
}

impl Instance {
    /// Constructor. Creates an empty instance with no comment and a negative label.
    pub fn new() -> Instance {
        Instance {
            label: NotNan::new(0.0).unwrap(),
            query: None,
            features: Vec::new(),
            comment: None,
        }
    }

    /// Parse a line of input in LibSVM format.
    pub fn parse(line: &str) -> Result<Instance, ParseError> {
        let mut inst = Instance::new();
        let data = match line.find('#') {
            Some(idx) => {
                let (features, comment) = line.split_at(idx);
                inst.comment = Some(comment[1..].trim().to_owned());
                features
            }
            None => line,
        };

        let mut tokens = data.split_whitespace().peekable();

        // Parse label if we can:
        inst.label = NotNan::new(
            tokens
                .next()
                .unwrap()
                .parse::<f64>()
                .map_err(ParseError::Label)? as f32,
        )
        .map_err(ParseError::LabelIsNan)?;

        // Parse qid if we can:
        if let Some(t) = tokens.peek().cloned() {
            if t.starts_with("qid:") {
                let qid = tokens.next().unwrap().trim_start_matches("qid:");
                inst.query = Some(qid.to_owned());
            }
        }

        for tok in tokens {
            inst.features.push(Feature::parse(tok)?);
        }

        // Only invoke sort on data we've observed to be unsorted.
        // Check order and repeats correctness by assuming best-case.
        let mut needs_sorting = false;
        for i in 0..(inst.features.len() - 1) {
            if inst.features[i].idx >= inst.features[i + 1].idx {
                needs_sorting = true;
            }
        }

        // Sort features by index so we have some guarantees about them.
        if needs_sorting {
            inst.features.sort_unstable_by(|f1, f2| f1.idx.cmp(&f2.idx));
            for i in 0..(inst.features.len() - 1) {
                if inst.features[i].idx == inst.features[i + 1].idx {
                    return Err(ParseError::MultipleDefinitions(
                        inst.features[i].clone(),
                        inst.features[i + 1].clone(),
                    ));
                }
            }
        }

        Ok(inst)
    }

    /// Return the largest feature number in this instance (if any). This allows us to convert to
    /// dense representations later, depending on need.
    pub fn max_feature_index(&self) -> Option<u32> {
        self.features.iter().map(|ftr| ftr.idx).max()
    }
}

/// This class implements an iterator over a file of libsvm/ranklib style instances.
pub struct InstanceIter {
    reader: Box<dyn io::BufRead>,
    line: String,
    line_num: u64,
}

impl InstanceIter {
    /// Construct a new iterator from a file.
    fn new(reader: Box<dyn io::BufRead>) -> Self {
        Self {
            reader,
            line: String::new(),
            line_num: 0,
        }
    }
}

impl Iterator for InstanceIter {
    /// Each line may be an error or a valid instance.
    type Item = Result<Instance, FileParseError>;

    /// When iterators return None, they are done.
    fn next(&mut self) -> Option<Self::Item> {
        self.line_num += 1;
        // Clear internal state:
        self.line.clear();

        // Can we successfully read the next line?
        let amt_read = self
            .reader
            .read_line(&mut self.line)
            .map_err(|io| FileParseError::LineIO(self.line_num, io));

        match amt_read {
            Ok(amt) => {
                // All done?
                if amt <= 0 {
                    return None;
                }
            }
            // Some kind of I/O error.
            Err(e) => return Some(Err(e)),
        }

        // Parse instance if we can!
        Some(
            Instance::parse(&self.line)
                .map_err(|e| FileParseError::LineParseError(self.line_num, e)),
        )
    }
}

/// Public interface to construct an iterator of instances.
pub fn instances(reader: Box<dyn io::BufRead>) -> InstanceIter {
    InstanceIter::new(reader)
}

/// This method allows us to parse all lines of a LibSVM input file and perform an action (given as
/// a closure) on the parsed instances. On any parse failure, it will short-circuit and return an
/// error. In comparison to an iterator, this callback approach allows us to re-use a single buffer
/// for our line-based file-IO.
pub fn foreach<F>(reader: Box<dyn io::BufRead>, handler: &mut F) -> Result<(), FileParseError>
where
    F: FnMut(Instance),
{
    for inst in instances(reader) {
        handler(inst?);
    }
    Ok(())
}

pub fn collect_reader(reader: Box<dyn io::BufRead>) -> Result<Vec<Instance>, FileParseError> {
    instances(reader).collect()
}

#[cfg(test)]
mod tests {
    use super::ParseError::*;
    use super::*;
    use std::result::Result;

    /// This calculates the dot product between two instances efficiently, given that their
    /// features are sorted already.
    fn dot_product(lhs: &Instance, rhs: &Instance) -> f32 {
        let ref a = lhs.features;
        let ref b = rhs.features;
        let mut i = 0;
        let mut j = 0;

        let mut sum = 0.0;
        while i < a.len() && j < b.len() {
            if a[i].idx < b[j].idx {
                i += 1;
                continue;
            } else if b[j].idx < a[i].idx {
                j += 1;
                continue;
            } else {
                // both have feature
                assert_eq!(a[i].idx, b[j].idx);
                sum += a[i].value * b[j].value;
                i += 1;
                j += 1;
            }
        }
        sum
    }

    #[test]
    fn test_qid() {
        let lhs = Instance::parse("1 qid:A 1:1 2:1 3:1").unwrap();
        let rhs = Instance::parse("2 qid:B 4:1 5:1 6:1").unwrap();

        assert_eq!(Some("A".to_owned()), lhs.query);
        assert_eq!(Some("B".to_owned()), rhs.query);
    }

    /// Compare floats with a given epsilon.
    fn flt_eq(lhs: f32, rhs: f32) -> bool {
        return (lhs - rhs).abs() < 1e-7;
    }

    #[test]
    fn test_dotp_zero() {
        let lhs = Instance::parse("1 1:1 2:1 3:1").unwrap();
        let rhs = Instance::parse("2 4:1 5:1 6:1").unwrap();
        assert_eq!(0, dot_product(&lhs, &rhs) as i32)
    }

    #[test]
    fn test_dotp_one() {
        let lhs = Instance::parse("1 1:1 2:1 3:1").unwrap();
        let rhs = Instance::parse("2 2:1 5:1 6:1").unwrap();
        assert_eq!(1, dot_product(&lhs, &rhs) as i32)
    }

    #[test]
    fn test_dotp_one_unsorted() {
        let lhs = Instance::parse("1 1:1 2:1 3:1").unwrap();
        let rhs = Instance::parse("2 6:1 3:1").unwrap();
        assert_eq!(1, dot_product(&lhs, &rhs) as i32)
    }

    /// Implement equality (as much as we can), for tests below.
    impl PartialEq for ParseError {
        fn eq(&self, other: &ParseError) -> bool {
            match *self {
                IO(_) => panic!("Can't compare IO(io::Error) instances."),
                Label(ref lhs) => {
                    if let Label(ref rhs) = *other {
                        return lhs == rhs;
                    }
                }
                LabelIsNan(_) => {
                    if let LabelIsNan(_) = *other {
                        return true;
                    }
                }
                FeatureNum(ref lhs) => {
                    if let FeatureNum(ref rhs) = *other {
                        return lhs == rhs;
                    }
                }
                FeatureVal(ref lhs) => {
                    if let FeatureVal(ref rhs) = *other {
                        return lhs == rhs;
                    }
                }
                FeatureValNotFloat(ref lhs) => {
                    if let FeatureValNotFloat(ref rhs) = *other {
                        return lhs == rhs;
                    }
                }
                FeatureNoColon() => {
                    if let FeatureNoColon() = *other {
                        return true;
                    }
                }
                MultipleDefinitions(ref a, ref b) => {
                    if let MultipleDefinitions(ref c, ref d) = *other {
                        return a.idx == c.idx && b.idx == d.idx;
                    }
                }
            };
            false
        }
        fn ne(&self, other: &ParseError) -> bool {
            !self.eq(other)
        }
    }

    #[test]
    fn feature_happy_path() {
        let f = Feature::parse("13:1.7").unwrap();
        assert_eq!(f.idx, 13);
        assert!(flt_eq(f.value, 1.7));
    }

    #[test]
    fn not_even_close() {
        let f = Feature::parse("what");
        assert!(f.is_err());
        assert_eq!(f.unwrap_err(), FeatureNoColon())
    }

    #[test]
    fn must_have_int_features() {
        let f: Result<Feature, ParseError> = Feature::parse("what:1.7");
        assert!(f.is_err());
        if let Err(FeatureNum(pfe)) = f {
            assert_eq!(pfe, "what".parse::<usize>().unwrap_err());
        } else {
            panic!("Error should be complaining about bad feature number.");
        }
    }

    #[test]
    fn must_have_float_values() {
        let f: Result<Feature, ParseError> = Feature::parse("1:what");
        assert!(f.is_err());
        if let Err(FeatureValNotFloat(pfe)) = f {
            assert_eq!(pfe, fast_float::parse::<f32, _>("what").unwrap_err());
        } else {
            panic!("Error should be complaining about bad feature value.");
        }
    }

}
