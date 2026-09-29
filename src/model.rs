use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const DEFAULT_REPO: &str = "https://huggingface.co/onnx-community/Nemotron-3-Diarization-ONNX/resolve/353b6f8ad2cac3580e982d7fbdf0a010786b0406/onnx";
pub const MODEL_FILES: [(&str, u64); 2] = [
    ("model_quantized.onnx", 200_000),
    ("model_quantized.onnx_data", 100_000_000),
];

pub fn default_model_dir() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        let dot_cache = home.join(".cache").join("tiginal").join("models").join("nemotron-3-diarization");
        return dot_cache;
    }
    if let Some(cache) = dirs::cache_dir() {
        cache.join("tiginal").join("models").join("nemotron-3-diarization")
    } else {
        PathBuf::from(".cache/tiginal/models/nemotron-3-diarization")
    }
}

pub fn resolve_model(
    custom_path: Option<&Path>,
    allow_download: bool,
    verbose: bool,
) -> Result<PathBuf, String> {
    if let Some(path) = custom_path {
        if !path.exists() {
            return Err(format!("Specified model file does not exist: {}", path.display()));
        }
        return Ok(path.to_path_buf());
    }

    let dir = default_model_dir();
    let main_model = dir.join(MODEL_FILES[0].0);
    let weights_file = dir.join(MODEL_FILES[1].0);

    let main_ok = main_model.is_file() && fs::metadata(&main_model).map(|m| m.len()).unwrap_or(0) >= MODEL_FILES[0].1;
    let weights_ok = weights_file.is_file() && fs::metadata(&weights_file).map(|m| m.len()).unwrap_or(0) >= MODEL_FILES[1].1;

    if main_ok && weights_ok {
        if verbose {
            eprintln!("[info] Found cached model at: {}", main_model.display());
        }
        return Ok(main_model);
    }

    if !allow_download {
        return Err(format!(
            "Nemotron-3 Diarization model is missing in {}\n\
             Please download it in Tiginal Settings > Model Engines, or pass `--download` to download automatically.",
            dir.display()
        ));
    }

    // Download files
    fs::create_dir_all(&dir).map_err(|e| format!("Failed to create model directory {}: {e}", dir.display()))?;

    for (filename, min_size) in MODEL_FILES {
        let dest = dir.join(filename);
        let exists_valid = dest.is_file() && fs::metadata(&dest).map(|m| m.len()).unwrap_or(0) >= min_size;
        if exists_valid {
            continue;
        }

        let url = format!("{DEFAULT_REPO}/{filename}");
        eprintln!("[info] Downloading {filename} from Hugging Face...");
        download_file(&url, &dest, verbose)?;
    }

    Ok(main_model)
}

fn download_file(url: &str, dest: &Path, verbose: bool) -> Result<(), String> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("HTTP request failed for {url}: {e}"))?;

    let total_bytes = response
        .header("Content-Length")
        .and_then(|val| val.parse::<u64>().ok());

    let mut reader = response.into_reader();
    let temp_dest = dest.with_extension("tmp");
    let mut file = File::create(&temp_dest)
        .map_err(|e| format!("Failed to create temporary file {}: {e}", temp_dest.display()))?;

    let mut buffer = [0u8; 64 * 1024];
    let mut downloaded = 0u64;
    let mut last_reported_pct = 0;

    loop {
        let bytes_read = reader
            .read(&mut buffer)
            .map_err(|e| format!("Read error while downloading {url}: {e}"))?;
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])
            .map_err(|e| format!("Write error to {}: {e}", temp_dest.display()))?;
        downloaded += bytes_read as u64;

        if verbose && total_bytes.is_some() {
            let total = total_bytes.unwrap();
            let pct = (downloaded * 100 / total) as usize;
            if pct >= last_reported_pct + 10 {
                last_reported_pct = pct;
                eprintln!("[download] {}% ({:.1} MB / {:.1} MB)", 
                    pct, 
                    downloaded as f64 / 1_048_576.0, 
                    total as f64 / 1_048_576.0);
            }
        }
    }

    file.flush()
        .map_err(|e| format!("Failed to flush {}: {e}", temp_dest.display()))?;
    drop(file);

    fs::rename(&temp_dest, dest)
        .map_err(|e| format!("Failed to rename temporary file to {}: {e}", dest.display()))?;

    eprintln!("[info] Successfully downloaded to {}", dest.display());
    Ok(())
}
