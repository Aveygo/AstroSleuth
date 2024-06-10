use crate::models;

use clap::{Args, Parser, Subcommand};

#[derive(Args, Debug)]
pub struct HeadlessArgs {
    /// Name of the model to utilize 
    #[arg(short, long)]
    pub model: models::Model,
    
    /// Source image to process
    #[arg(short, long)]
    pub src: Box<std::path::Path>,

    /// Destination to save image
    #[arg(short, long)]
    pub dst: Box<std::path::Path>,
}

#[derive(Args, Debug)]
pub struct ListArgs {
   /// Where to load json data from
   #[arg(short, long, default_value="data.json")]
   pub src: Option<String>,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
   #[command(subcommand)]
   pub command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    Headless(HeadlessArgs),
    List(ListArgs)
}
