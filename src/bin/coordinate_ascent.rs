use clap::{App, Arg};
use fastrank::coordinate_ascent::*;
use fastrank::io_helper;
use fastrank::libsvm;
use std::error::Error;
use fastrank::dataset::RankingDataset;
use fastrank::evaluators::*;
use fastrank::qrel;

fn main() -> Result<(), Box<Error>> {
    let matches = App::new("coordinate_ascent_learn")
        .version("0.1")
        .about("Learn a linear ranking model.")
        .arg(Arg::with_name("TRAIN_FILE").required(true))
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
                .short("i")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("tolerance")
                .long("tolerance")
                .short("t")
                .takes_value(true),
        )
        .arg(Arg::with_name("training_measure").long("train_with").long("metric2t").takes_value(true))
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
    let judgments = match matches.value_of("qrel") {
        None => None,
        Some(path) => Some(qrel::read_file(path)?),
    };
    let input = matches
        .value_of("TRAIN_FILE")
        .ok_or("You need a training file to learn a model!")?;

    let mut reader = io_helper::open_reader(input)?;


    let instances: Vec<libsvm::Instance> = libsvm::collect_reader(&mut reader)?;
    let dataset = RankingDataset::import(instances)?;

    let m_ap = AveragePrecision::new(&dataset, judgments.clone());
    let m_rr = ReciprocalRank;
    let ndcg1000 = NDCG::new(1000, &dataset, judgments.clone());
    let ndcg10 = NDCG::new(10, &dataset, judgments.clone());

    let measure = matches.value_of("training_measure").unwrap_or("map").to_lowercase();
    // TODO: support at rank syntax: rr@10
    let evaluator: &Evaluator = match measure.as_str() {
        // TODO: support loading qrel file norms
        "ap" | "map" => &m_ap,
        "rr" | "mrr" => &m_rr,
        "ndcg" | "ndcg@1000" => &ndcg1000,
        "ndcg@10" => &ndcg10,
        _ => panic!("Invalid training measure: \"{}\"", measure)
    };
    let model = params.learn(&dataset, evaluator);
    println!("MODEL {:?}", model);
    println!("Training Performance:");
    println!("    mAP: {:.3}", dataset.evaluate_mean(model.as_ref(), &m_ap));
    println!("    mRR: {:.3}", dataset.evaluate_mean(model.as_ref(), &m_rr));
    println!("    ndcg: {:.3}", dataset.evaluate_mean(model.as_ref(), &ndcg1000));
    println!("    ndcg@10: {:.3}", dataset.evaluate_mean(model.as_ref(), &ndcg10));
    Ok(())
}
