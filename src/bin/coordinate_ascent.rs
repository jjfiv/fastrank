use clap::{App, Arg};
use fastrank::coordinate_ascent::*;
use fastrank::dataset;
use fastrank::dataset::RankingDataset;
use fastrank::qrel;
use std::error::Error;

fn main() -> Result<(), Box<Error>> {
    let matches = App::new("coordinate_ascent_learn")
        .version("0.1")
        .about("Learn a linear ranking model.")
        .arg(Arg::with_name("TRAIN_FILE").required(true))
        .arg(Arg::with_name("TEST_FILE").long("test").takes_value(true))
        .arg(
            Arg::with_name("FEATURE_NAMES_FILE")
                .long("feature_names")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("ignore_features")
                .long("ignore")
                .short("i")
                .takes_value(true)
                .multiple(true),
        )
        .arg(Arg::with_name("seed").long("seed").takes_value(true))
        // Optional loading of query relevance files.
        .arg(Arg::with_name("qrel").long("qrel").takes_value(true))
        .arg(
            Arg::with_name("restarts")
                .long("num_restarts")
                .short("r")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("iterations")
                .long("num_max_iterations")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("tolerance")
                .long("tolerance")
                .takes_value(true),
        )
        .arg(Arg::with_name("normalize_weights").long("normalize_weights"))
        .arg(Arg::with_name("init_random").long("init_random"))
        .arg(Arg::with_name("output_ensemble").long("output_ensemble"))
        .arg(
            Arg::with_name("training_measure")
                .long("metric2t")
                .takes_value(true),
        )
        .arg(Arg::with_name("quiet").long("quiet").short("q"))
        .get_matches();

    let mut params = CoordinateAscentParams::default();
    params.init_random = matches.is_present("init_random");
    params.quiet = matches.is_present("quiet");
    if let Some(seed) = matches.value_of("seed") {
        params.seed = seed.parse::<u64>()?;
    } else {
        println!("Selected random seed: {}", params.seed);
    }
    if let Some(restarts) = matches.value_of("restarts") {
        params.num_restarts = restarts.parse::<u32>()?;
    }
    params.output_ensemble = matches.is_present("output_ensemble");
    if let Some(iterations) = matches.value_of("iterations") {
        params.num_max_iterations = iterations.parse::<u32>()?;
    }
    if let Some(tolerance) = matches.value_of("tolerance") {
        params.tolerance = tolerance.parse::<f64>()?;
    }
    params.normalize = matches.is_present("normalize_weights");
    let judgments = match matches.value_of("qrel") {
        None => None,
        Some(path) => Some(qrel::read_file(path)?),
    };
    let input = matches
        .value_of("TRAIN_FILE")
        .ok_or("You need a training file to learn a model!")?;

    let feature_names = matches
        .value_of("FEATURE_NAMES_FILE")
        .map(|path| dataset::load_feature_names_json(path))
        .transpose()?;
    let mut train_dataset = RankingDataset::load_libsvm(input, feature_names.as_ref())?;
    let test_dataset = matches
        .value_of("TEST_FILE")
        .map(|test_file| RankingDataset::load_libsvm(test_file, feature_names.as_ref()))
        .transpose()?;

    if let Some(features_to_ignore) = matches.values_of("ignore_features") {
        // Only need to remove it from training.
        for ftr in features_to_ignore {
            train_dataset.try_remove_feature(ftr)?;
        }
    }

    let evaluator = train_dataset.make_evaluator(
        matches.value_of("training_measure").unwrap_or("map"),
        judgments.clone(),
    )?;
    let model = params.learn(&train_dataset, evaluator.as_ref());
    println!("MODEL {:?}", model);
    println!("Training Performance:");
    for measure in &["map", "rr", "ndcg@5", "ndcg"] {
        let evaluator = train_dataset.make_evaluator(measure, judgments.clone())?;
        println!(
            "\t{}: {:.3}",
            evaluator.name(),
            train_dataset.evaluate_mean(model.as_ref(), evaluator.as_ref())
        );
    }

    if let Some(test_dataset) = test_dataset {
        println!("Test Performance:");
        for measure in &["map", "rr", "ndcg@5", "ndcg"] {
            let evaluator = test_dataset.make_evaluator(measure, judgments.clone())?;
            println!(
                "\t{}: {:.3}",
                evaluator.name(),
                test_dataset.evaluate_mean(model.as_ref(), evaluator.as_ref())
            );
        }
    }

    Ok(())
}
