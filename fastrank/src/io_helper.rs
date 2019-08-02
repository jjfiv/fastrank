//! This module defines methods to read from compressed files seamlessly. Currently this means that
//! we box the reader, but using it should be seamless.
//!
//! We support ``.bz2``, ``.gz``, and ``.zst`` for now.

extern crate bzip2;
extern crate flate2;
extern crate zstd;

use self::bzip2::read::BzDecoder;
use self::flate2::read::MultiGzDecoder;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Result;

/// Open a file based on its extension; seamlessly supporting different compression styles.
pub fn open_reader(file_name: &str) -> Result<Box<BufRead>> {
    let fp = File::open(file_name)?;
    return if file_name.ends_with(".zst") {
        Ok(Box::new(BufReader::new(zstd::Decoder::new(fp)?)))
    } else if file_name.ends_with(".gz") {
        Ok(Box::new(BufReader::new(MultiGzDecoder::new(fp))))
    } else if file_name.ends_with(".bz2") {
        Ok(Box::new(BufReader::new(BzDecoder::new(fp))))
    } else {
        Ok(Box::new(BufReader::new(fp)))
    };
}
