use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

use serde::Deserialize;

#[derive(Deserialize, Clone)]
struct CondData {
    styles: HashMap<String, Vec<f32>>
}


#[derive(Deserialize, Clone)]
struct ConfigMeta {
    compatability: String,
    model_dir: String
}

#[derive(Deserialize, Clone)]
struct ModelMeta {
    size: String,
    parameters: String,
    created:i64,
    updated:i64,
    changelog: Vec<String>
}

#[derive(Deserialize, Clone)]
struct WorkerConfig {
    scale: usize,
    tile_size: usize,
    tile_pad: usize
}


#[derive(Deserialize, Clone)]
struct Module {
    src: String,
}

#[derive(Deserialize, Clone)]
struct Source {
    url: Option<String> 
}

#[derive(Deserialize, Clone)]
struct Sources {
    bin: Option<Source>,
    onnx: Option<Source>,
    param: Option<Source>,
    pth: Option<Source>, 
    safetensors: Option<Source>, 
}

#[derive(Deserialize, Clone)]
struct Model {
    name: String,
    description: String,
    sunset: bool,
    meta: ModelMeta,
    config: WorkerConfig,
    modules: HashMap<String, Option<Module>>,
    src: Sources
}


#[derive(Deserialize, Clone)]
pub struct Config {
    meta: ConfigMeta,
    model_configs: Vec<Model>,
}

impl Config {
    pub fn new() -> Result<Self, serde_json::Error> {
        let file = File::open("config.json").unwrap();
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)
    }

    pub fn names(&self) -> Vec<String> {
        let mut result:Vec<String> = vec![];
        for model in self.model_configs.iter() {
            result.push(model.name.clone())
        }

        return result;
    }
}