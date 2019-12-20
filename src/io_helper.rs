//! This module defines methods to read from compressed files seamlessly. Currently this means that
//! we box the reader, but using it should be seamless.
//!
//! We support ``.bz2``, ``.gz``, and ``.zst`` for now.

use bzip2::read::BzDecoder;
use bzip2::write::BzEncoder;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Result;
use zstd;

/// Open a file based on its extension; seamlessly supporting different compression styles.
pub fn open_reader(file_name: &str) -> Result<Box<dyn BufRead>> {
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

pub fn open_writer(file_name: &str) -> Result<Box<dyn Write>> {
    let fp = File::create(file_name)?;
    return if file_name.ends_with(".zst") {
        Ok(Box::new(BufWriter::new(zstd::Encoder::new(fp, 0)?)))
    } else if file_name.ends_with(".gz") {
        Ok(Box::new(BufWriter::new(GzEncoder::new(
            fp,
            flate2::Compression::default(),
        ))))
    } else if file_name.ends_with(".bz2") {
        Ok(Box::new(BufWriter::new(BzEncoder::new(
            fp,
            bzip2::Compression::Default,
        ))))
    } else {
        Ok(Box::new(BufWriter::new(fp)))
    };
}
