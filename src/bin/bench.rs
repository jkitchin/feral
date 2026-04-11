use std::path::Path;

fn main() {
    println!("FERAL benchmark harness");

    let config_path = Path::new("data/benchmark-config.toml");
    print!("Loading matrices from {} ... ", config_path.display());

    if config_path.exists() {
        println!("found");
        // Future: parse config and run benchmarks
    } else {
        println!("not found");
    }

    println!("0 matrices benchmarked");
}
