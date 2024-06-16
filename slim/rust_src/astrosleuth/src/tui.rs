
use std::io::{self, Write, stdin,stdout};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use image::io::Reader as ImageReader;

use crate::models;

const LOGO: &str = r#"
    ___         __            _____ __           __  __  
   /   |  _____/ /__________ / ___// /__  __  __/ /_/ /_ 
  / /| | / ___/ __/ ___/ __ \\__ \/ / _ \/ / / / __/ __ \
 / ___ |(__  ) /_/ /  / /_/ /__/ / /  __/ /_/ / /_/ / / /
/_/  |_/____/\__/_/   \____/____/_/\___/\____/\__/_/ /_/ 
"#;


fn get_image_path() -> Option<String> {
    // Request user input
    let mut input = String::new();
    stdin().read_line(&mut input).expect("Failed to read line");

    if let Some('\n')=input.chars().next_back() {
        input.pop();
    }
    if let Some('\r')=input.chars().next_back() {
        input.pop();
    }
    
    // Remove any trailing whitespace or newline characters
    let input = input.trim();

    // Check if the input is a valid file path and if it's an image
    if Path::new(input).is_file() {
        match ImageReader::open(input) {
            Ok(_) => Some(input.to_string()),
            Err(_) => {
                println!("The file is not a valid image.");
                None
            }
        }
    } else {
        println!("The path is not a valid file.");
        None
    }
}

fn get_destination_path() -> Option<String> {
    // Request user input
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read line");
    
    // Remove any trailing whitespace or newline characters
    let input = input.trim();

    if input.len() == 0 {
        return Some("".to_string());
    }
    
    // Check if the file has a valid image extension
    let valid_extensions = ["avif", "jpeg", "jpg", "png", "tiff"];
    let path = Path::new(input);
    if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
        if !valid_extensions.contains(&extension.to_lowercase().as_str()) {
            println!("The file does not have a valid image extension.");
            return None;
        }
    } else {
        println!("The file does not have an extension.");
        return None;
    }

    // Check if the directory exists
    if let Some(parent) = path.parent() {
        if parent.exists() {
            Some(input.to_string())
        } else {
            println!("The directory does not exist.");
            None
        }
    } else {
        println!("Invalid path.");
        None
    }
}

fn append_out_to_filename(image_path: &String) -> String {
    // Convert the image_path string to a Path
    let path = Path::new(image_path);

    // Get the file name and extension separately
    if let Some(file_stem) = path.file_stem() {
        if let Some(extension) = path.extension() {
            // Create a new file name with "_out" appended
            let new_file_name = format!("{}_out.{}", file_stem.to_string_lossy(), extension.to_string_lossy());
            
            // Get the parent directory of the path
            if let Some(parent) = path.parent() {
                // Create a new path with the parent directory and the new file name
                let mut new_path = PathBuf::from(parent);
                new_path.push(new_file_name);
                
                return new_path.to_string_lossy().to_string();
            }
        }
    }

    // If something went wrong, return the original path
    image_path.to_string()
}

fn get_model() -> Option<models::Model> {
    // Request user input
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read line");
    
    // Remove any trailing whitespace or newline characters
    let input = input.trim();
    match models::Model::from_str(input) {
        Ok(model) => {
            return Some(model);
        },
        Err(msg) => {
            println!("{:?}", msg);
            return None;
        }
    }
}

pub fn tui() {
    println!("{}", LOGO);
    
    
    let src:String;
    loop {
        print!("Source image path > ");
        let _=stdout().flush();
        match get_image_path() {
            Some(image_path) => {
                src = image_path;
                break;
            },
            None => {},
        }
    }

    let mut dst:String = append_out_to_filename(&src);
    
    loop {
        print!("Destination path (default={:?}) > ", dst);
        let _=stdout().flush();
        match get_destination_path() {
            Some(image_path) => {
                if image_path.len() != 0 {
                    dst = image_path;
                }
                break;
            },
            None => {},
        }
    }

    let model:models::Model;

    loop {
        print!("Model name > ");
        let _=stdout().flush();
        match get_model() {
            Some(possible_model) => {
                model = possible_model;
                break;
            },
            _ => {}
        }
    }

    let worker = models::Worker::new(64, 8, 4, &model);
    worker.upscale(&src, &dst);

}