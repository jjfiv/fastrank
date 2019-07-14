use clap::{App, Arg};
use fastrank::coordinate_ascent::*;
use fastrank::io_helper;
use fastrank::libsvm;
use std::error::Error;

fn main() -> Result<(), Box<Error>> {
    let matches = App::new("coordinate_ascent_learn")
        .version("0.1")
        .about("Learn a linear ranking model.")
        .arg(Arg::with_name("TRAIN_FILE").required(true))
        .arg(Arg::with_name("quiet"))
        .get_matches();
    let mut params = CoordinateAscentParams::default();
    params.quiet = matches.is_present("quiet");
    let input = matches
        .value_of("TRAIN_FILE")
        .ok_or("You need a training file to learn a model!")?;

    let mut reader = io_helper::open_reader(input)?;

    let instances: Vec<libsvm::Instance> = libsvm::collect_reader(&mut reader)?;
    let instances: Result<Vec<_>, _> = instances
        .into_iter()
        .map(|i| TrainingInstance::try_new(i))
        .collect();
    let instances = instances?;

    let mut m = CoordinateAscentModel::new_with_params(params);
    m.learn(instances);
    Ok(())
}
