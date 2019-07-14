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
        .arg(Arg::with_name("seed").long("seed").takes_value(true))
        .arg(
            Arg::with_name("restarts")
                .long("num_restarts")
                .short("r")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("iterations")
                .long("num_max_iterations")
                .short("i")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("tolerance")
                .long("tolerance")
                .short("t")
                .takes_value(true),
        )
        .arg(Arg::with_name("quiet").long("quiet").short("q"))
        .get_matches();

    let mut params = CoordinateAscentParams::default();
    params.quiet = matches.is_present("quiet");
    if let Some(seed) = matches.value_of("seed") {
        params.seed = seed.parse::<u64>()?;
    } else {
        println!("Selected random seed: {}", params.seed);
    }
    if let Some(restarts) = matches.value_of("restarts") {
        params.num_restarts = restarts.parse::<u32>()?;
    }
    if let Some(iterations) = matches.value_of("iterations") {
        params.num_max_iterations = iterations.parse::<u32>()?;
    }
    if let Some(tolerance) = matches.value_of("tolerance") {
        params.tolerance = tolerance.parse::<f64>()?;
    }
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

    let final_score = params.learn(instances);
    println!("TRAINING mAP: {:.3}", final_score);
    Ok(())
}
