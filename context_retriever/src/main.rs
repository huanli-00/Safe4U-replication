mod analyzer;

pub(crate) mod __private;

use analyzer::FnInfo;
pub use analyzer::Processor;
use config::Config;
use log::{Level, Metadata, Record};
use serde_json::Value as JsonValue;
use std::path::Path;

struct SimpleLogger;

impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Debug
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            println!("{} - {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: SimpleLogger = SimpleLogger;

pub fn init_logger() -> Result<(), log::SetLoggerError> {
    log::set_logger(&LOGGER).map(|()| log::set_max_level(log::LevelFilter::Info))
}

pub(crate) fn get_config() -> Config {
    Config::builder()
        .add_source(config::File::with_name("config.toml"))
        .build()
        .unwrap()
}

fn main() {
    let config = get_config();
    let args: Vec<String> = std::env::args().collect();
    let target = match args.get(1) {
        Some(target) => target,
        None => "risky_function",
    };
    init_logger().unwrap();
    let repo_root_path = config
        .get_string(format!("{}.repo_root_path", target).as_str())
        .unwrap();
    let repo_root_path = Path::new(&repo_root_path);
    let item_root_path = config
        .get_string(format!("{}.item_root_path", target).as_str())
        .unwrap();
    let item_root_path = Path::new(&item_root_path);
    let result_root = config
        .get_string(format!("{}.result_root_path", target).as_str())
        .unwrap();
    let result_root = Path::new(&result_root);
    if !result_root.exists() {
        std::fs::create_dir_all(result_root).unwrap();
    }
    let locate_method = config
        .get_string(format!("{}.function_locate", target).as_str())
        .unwrap_or("start_line".to_string());
    // list file in the directory
    for entry in item_root_path
        .read_dir()
        .expect("read item directory failed")
    {
        let entry = entry.unwrap();
        let item_path = entry.path();
        let item_file = item_path.file_name().unwrap();
        let repo_name = item_file.to_str().unwrap().rsplit_once(".json").unwrap().0;
        log::info!("Processing file: {:?}", &repo_name);
        let mut fns = Vec::<FnInfo>::new();
        let samples_content = std::fs::read_to_string(&item_path).unwrap();
        if let JsonValue::Array(samples) = serde_json::from_str(&samples_content).unwrap() {
            if samples.is_empty() {
                continue;
            }
            let mut root_processor = Processor::new(repo_root_path, repo_name);
            for sample in samples {
                let relative_file = sample["relative_file"].as_str().unwrap();

                // relative_file: path relative to the crates root
                let mut process_sample = |processor: &mut Processor, relative_file: &str| {
                    let fn_info = match locate_method.as_str() {
                        "function_range" => {
                            let range = (
                                sample["start"]["byte"].as_u64().unwrap() as usize,
                                sample["end"]["byte"].as_u64().unwrap() as usize,
                            );
                            processor.process_fn_in_range(relative_file, range)
                        }
                        _ => {
                            let line = sample["start_line"].as_u64().unwrap() as usize;
                            processor.process_fn_in_line(relative_file, line)
                        }
                    };
                    match fn_info {
                        Ok(fn_info) => {
                            fns.push(fn_info);
                        }
                        _ => {}
                    }
                };
                match &mut root_processor {
                    Ok(p) => {
                        process_sample(p, relative_file);
                    }
                    Err(_) => {
                        let (mut crate_processor, relative_file) =
                            match Processor::processor_for_crate(
                                repo_root_path,
                                repo_name,
                                relative_file,
                            ) {
                                Ok(p) => p,
                                Err(e) => {
                                    log::error!("Failed to create processor: {:?}", e);
                                    break;
                                }
                            };
                        process_sample(&mut crate_processor, &relative_file);
                    }
                }
            }
        }
        let result_file = result_root.join(item_file);
        std::fs::write(result_file, serde_json::to_string_pretty(&fns).unwrap()).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_fn_in_range() {
        let config = get_config();
        let repo_name = "kivikakk.comrak";
        let file = "src/parser/autolink.rs";
        let range = (2796, 3525);
        let mut fn_processor = Processor::new(
            Path::new(config.get_string("test.repo_root_path").unwrap().as_str()),
            repo_name,
        )
        .unwrap();
        let fn_info = fn_processor.process_fn_in_range(file, range).unwrap();
        print_fn_info(fn_info);
    }

    #[test]
    fn test_process_fn_with_line() {
        let config = get_config();
        let repo_name = "hyperium.http";
        let file = "src/method.rs";
        let line = 332;
        let mut fn_processor = Processor::new(
            Path::new(config.get_string("test.repo_root_path").unwrap().as_str()),
            repo_name,
        )
        .unwrap();
        let fn_info = fn_processor.process_fn_in_line(file, line).unwrap();
        print_fn_info(fn_info);
    }

    #[test]
    fn test_process_fn_with_line2() {
        let config = get_config();
        let repo_name = "Stebalien.xattr";
        let file = "src/lib.rs";
        let line = 117;
        let mut fn_processor = Processor::new(
            Path::new(config.get_string("test.repo_root_path").unwrap().as_str()),
            repo_name,
        )
        .unwrap();
        let fn_info = fn_processor.process_fn_in_line(file, line).unwrap();
        print_fn_info(fn_info);
    }

    #[test]
    fn test_process_fn_with_line3() {
        let config = get_config();
        let repo_name = "Alexhuszagh.rust-lexical";
        let file = "lexical-parse-float/src/bigint.rs";
        let line = 917;
        let mut fn_processor = Processor::new(
            Path::new(config.get_string("test.repo_root_path").unwrap().as_str()),
            repo_name,
        )
        .unwrap();
        let fn_info = fn_processor.process_fn_in_line(file, line).unwrap();
        print_fn_info(fn_info);
    }

    #[test]
    fn test_process_fn_with_line4() {
        let config = get_config();
        let repo_name = "matklad.once_cell";
        let file = "src/imp_std.rs";
        let line = 60;
        let mut fn_processor = Processor::new(
            Path::new(config.get_string("test.repo_root_path").unwrap().as_str()),
            repo_name,
        )
        .unwrap();
        let fn_info = fn_processor.process_fn_in_line(file, line).unwrap();
        print_fn_info(fn_info);
    }

    #[test]
    fn test_repo_require_build_script() {
        // repository that require build.rs
        let config = get_config();
        let repo_name = "bytecodealliance.rustix";
        let file = "src/io/ioctl.rs";
        let line = 42;
        let mut fn_processor = Processor::new(
            Path::new(config.get_string("test.repo_root_path").unwrap().as_str()),
            repo_name,
        )
        .unwrap();
        let fn_info = fn_processor.process_fn_in_line(file, line).unwrap();
        print_fn_info(fn_info);
    }

    fn print_fn_info(fn_info: analyzer::FnInfo) {
        println!("## name: {}", fn_info.name);
        println!("## parameters: {:?}", fn_info.params);
        println!("## return type: {}", fn_info.return_type);
        println!("## code: {:?}", fn_info.raw_code);
        println!("## code with hints: {:?}", fn_info.code_with_hints);
        println!("## code with type: {:?}", fn_info.code_with_type);
        println!("## code with param: {:?}", fn_info.code_with_param);
        println!("## comment: {}", fn_info.comment);
        println!("## attributs: {}", fn_info.attributes);
        println!("## There are {} detail infos:", fn_info.detail_info.len());
        for info in fn_info.detail_info {
            println!("{:?} - {:?}:", info.item_type, info.name);
            println!("declaration: {:?}", info.declaration);
            println!("documentation: {:?}", info.documentation);
        }
        println!("## self: {:?}", fn_info.self_info);
    }
}
