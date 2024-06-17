use clap::Parser;

mod cli;
mod tui;
mod models;


fn main() {

    let args = cli::Cli::parse();
    match &args.command {
        Some(cli::Commands::Headless(cmd_args)) => {
            let worker = models::Worker::new(256, 16, 4, &cmd_args.model);
            worker.upscale(&cmd_args.src, &cmd_args.dst);
        } 
        Some(cli::Commands::List(cmd_args)) => {println!("{:?}", cmd_args)}
        None => {tui::tui();}
    }
}