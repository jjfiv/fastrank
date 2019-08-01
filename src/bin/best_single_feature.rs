use clap::{App, Arg};
use fastrank::dataset;
use fastrank::dataset::Normalizer;
use fastrank::dataset::RankingDataset;
use fastrank::model::Model;
use fastrank::qrel;
use fastrank::Scored;
use ordered_float::NotNan;
use std::error::Error;

#[derive(Debug, Clone, Copy)]
pub struct SingleFeatureModel {
    fid: u32,
    dir: f64,
}

impl Model for SingleFeatureModel {
    fn score(&self, features: &dataset::Features) -> NotNan<f64> {
        let val = features.get(self.fid).unwrap_or(0.0);
        NotNan::new(self.dir * val).unwrap()
    }
}

fn main() -> Result<(), Box<Error>> {
    let matches = App::new("best_single_feature")
        .version("0.1")
        .about("Select the best single feature from the training dataset.")
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
        .arg(
            Arg::with_name("normalize_features")
                .long("norm")
                .takes_value(true),
        )
        // Optional loading of query relevance files.
        .arg(Arg::with_name("qrel").long("qrel").takes_value(true))
        .arg(
            Arg::with_name("training_measure")
                .long("metric2t")
                .takes_value(true),
        )
        .arg(Arg::with_name("quiet").long("quiet").short("q"))
        .get_matches();

    let quiet = matches.is_present("quiet");
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

    let normalizer = matches
        .value_of("normalize_features")
        .map(|norm_name| Normalizer::new(norm_name, &train_dataset))
        .transpose()?;
    if let Some(norm) = &normalizer {
        train_dataset.apply_normalization(norm);
    }

    let evaluator = train_dataset.make_evaluator(
        matches.value_of("training_measure").unwrap_or("map"),
        judgments.clone(),
    )?;

    let mut models = Vec::new();

    if !quiet {
        println!("------------------------------------------");
        println!(
            "{:3} | {:16} | {:4} | {:9} |",
            "#",
            "Feature Name",
            "Dir",
            evaluator.name()
        );
        println!("------------------------------------------");
    }
    // explore all features:
    let multiplier = &[-1.0, 1.0];
    let mut features: Vec<u32> = train_dataset.features.iter().cloned().collect();
    features.sort_unstable();
    for fid in features {
        let feature_name = feature_names
            .as_ref()
            .and_then(|names| names.get(&fid).cloned())
            .unwrap_or(format!("{}", fid));
        let best_by_dir = multiplier
            .iter()
            .cloned()
            .map(|dir| {
                let model = SingleFeatureModel { fid, dir };
                let perf = train_dataset.evaluate_mean(&model, evaluator.as_ref());
                Scored::new(perf, model)
            })
            .max()
            .unwrap();

        if !quiet {
            println!(
                "{:3} | {:16} | {:4.0} | {:9.3}",
                fid, feature_name, best_by_dir.item.dir, best_by_dir.score
            );
        }

        models.push(best_by_dir);
    }
    if !quiet {
        println!("------------------------------------------");
    }

    models.sort_unstable();
    let model = models.iter().last().unwrap().item;

    println!("MODEL {:?}", model);
    println!("Training Performance:");
    for measure in &["map", "rr", "ndcg@5", "ndcg"] {
        let evaluator = train_dataset.make_evaluator(measure, judgments.clone())?;
        println!(
            "\t{}: {:.3}",
            evaluator.name(),
            train_dataset.evaluate_mean(&model, evaluator.as_ref())
        );
    }

    if let Some(mut test_dataset) = test_dataset {
        if let Some(norm) = &normalizer {
            test_dataset.apply_normalization(&norm);
        }
        println!("Test Performance:");
        for measure in &["map", "rr", "ndcg@5", "ndcg"] {
            let evaluator = test_dataset.make_evaluator(measure, judgments.clone())?;
            println!(
                "\t{}: {:.3}",
                evaluator.name(),
                test_dataset.evaluate_mean(&model, evaluator.as_ref())
            );
        }
    }

    Ok(())
}
