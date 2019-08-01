use clap::{App, Arg};
use fastrank::dataset;
use fastrank::dataset::RankingDataset;
use fastrank::qrel;
use fastrank::random_forest::*;
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
            Arg::with_name("num_trees")
                .long("num_trees")
                .short("n")
                .takes_value(true),
        )
        .arg(Arg::with_name("weight_by_perf").long("weight_by_perf"))
        .arg(
            Arg::with_name("split_method")
                .long("split_method")
                .short("m")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("min_leaf_support")
                .long("min_leaf_support")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("split_candidates")
                .long("split_candidates")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("max_depth")
                .long("max_depth")
                .takes_value(true),
        )
        .arg(Arg::with_name("frate").long("frate").takes_value(true))
        .arg(Arg::with_name("srate").long("srate").takes_value(true))
        .arg(
            Arg::with_name("training_measure")
                .long("metric2t")
                .takes_value(true),
        )
        .arg(Arg::with_name("quiet").long("quiet").short("q"))
        .get_matches();

    let mut params = RandomForestParams::default();
    params.quiet = matches.is_present("quiet");
    if let Some(seed) = matches.value_of("seed") {
        params.seed = seed.parse::<u64>()?;
    } else {
        println!("Selected random seed: {}", params.seed);
    }
    if let Some(num_trees) = matches.value_of("num_trees") {
        params.num_trees = num_trees.parse::<u32>()?;
    }
    params.weight_trees = matches.is_present("weight_by_perf");
    params.split_method = match matches.value_of("split_method") {
        Some("l2") | None => SplitSelectionStrategy::SquaredError(),
        Some("gini") => SplitSelectionStrategy::BinaryGiniImpurity(),
        Some("entropy") | Some("information_gain") => SplitSelectionStrategy::InformationGain(),
        Some("variance") => SplitSelectionStrategy::TrueVarianceReduction(),
        Some(unkn) => panic!("No such split_method={}", unkn),
    };
    if let Some(min_leaf_support) = matches.value_of("min_leaf_support") {
        params.min_leaf_support = min_leaf_support.parse::<u32>()?;
    }
    if let Some(split_candidates) = matches.value_of("split_candidates") {
        params.split_candidates = split_candidates.parse::<u32>()?;
    }
    if let Some(max_depth) = matches.value_of("max_depth") {
        params.max_depth = max_depth.parse::<u32>()?;
    }
    if let Some(srate) = matches.value_of("srate") {
        params.instance_sampling_rate = srate.parse::<f64>()?;
    }
    if let Some(frate) = matches.value_of("frate") {
        params.feature_sampling_rate = frate.parse::<f64>()?;
    }
    if let Some(max_depth) = matches.value_of("max_depth") {
        params.max_depth = max_depth.parse::<u32>()?;
    }
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

    // TODO: we open the test dataset up early to quickly get errors, but maybe we want to save the RAM? idk.
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
    let model = Box::new(learn_ensemble(&params, &train_dataset, evaluator.as_ref()));
    //println!("MODEL {:?}", model);
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
