use crate::io_helper;
use ordered_float::NotNan;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Serialize, Deserialize)]
pub struct QueryJudgments {
    #[serde(flatten)]
    docid_to_rel: Arc<HashMap<String, NotNan<f32>>>,
}

impl QueryJudgments {
    fn new(data: HashMap<String, NotNan<f32>>) -> Self {
        Self {
            docid_to_rel: Arc::new(data),
        }
    }
    pub fn num_judged(&self) -> u32 {
        self.docid_to_rel.len() as u32
    }
    pub fn num_relevant(&self) -> u32 {
        self.docid_to_rel
            .iter()
            .map(|(_, gain)| gain)
            .filter(|gain| gain.into_inner() > 0.0)
            .count() as u32
    }
    pub fn get_gain(&self, docid: &str) -> NotNan<f32> {
        *self
            .docid_to_rel
            .get(docid)
            .unwrap_or(&NotNan::new(0.0).unwrap())
    }
    pub fn gain_vector(&self) -> Vec<NotNan<f32>> {
        self.docid_to_rel
            .values()
            .filter(|g| g.into_inner() > 0.0)
            .cloned()
            .collect()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct QuerySetJudgments {
    #[serde(flatten)]
    pub query_to_judgments: Arc<HashMap<String, QueryJudgments>>,
}

impl QuerySetJudgments {
    fn new(data: HashMap<String, QueryJudgments>) -> Self {
        Self {
            query_to_judgments: Arc::new(data),
        }
    }
    pub fn get_queries(&self) -> Vec<String> {
        self.query_to_judgments
            .keys()
            .map(|s| s.to_string())
            .collect()
    }
    pub fn get(&self, qid: &str) -> Option<QueryJudgments> {
        self.query_to_judgments.get(qid).cloned()
    }
}

pub fn read_file(path: &str) -> Result<QuerySetJudgments, Box<dyn std::error::Error>> {
    let mut reader = io_helper::open_reader(path)?;

    let mut line = String::new();
    let mut num = 0;
    let mut output: HashMap<String, HashMap<String, NotNan<f32>>> = HashMap::new();

    loop {
        num += 1;
        let amt = reader.read_line(&mut line)?;
        if amt <= 0 {
            break;
        }
        let row: Vec<&str> = line.split_whitespace().collect();
        let qid = row[0].to_string();
        let _unused = row[1];
        let docid = row[2].to_string();
        let gain = row[3]
            .parse::<f32>()
            .map_err(|_| format!("{}:{}: Invalid relevance judgment {}", path, num, row[3]))?;
        let gain =
            NotNan::new(gain).map_err(|_| format!("{}:{}: NaN relevance judgment.", path, num))?;

        output
            .entry(qid)
            .or_insert_with(|| HashMap::new())
            .insert(docid, gain);
        line.clear();
    }

    let mut query_to_judgments: HashMap<String, QueryJudgments> = HashMap::new();

    for (qid, docid_to_rel) in output.into_iter() {
        query_to_judgments.insert(qid, QueryJudgments::new(docid_to_rel));
    }

    Ok(QuerySetJudgments::new(query_to_judgments))
}
