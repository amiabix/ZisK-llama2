use std::error::Error;
use std::path::PathBuf;
use std::process::Command;

use llama2_common::model::{Model, dot_soft, generate, read_config};
use llama2_common::params::GenParams;
use llama2_common::tokenizer::Tokenizer;
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const GUEST_ELF: &str = env!("ZISK_ELF_llama2-guest");

const USAGE: &str = "\
llama2-zisk [OPTIONS]
  --model PATH        llama2.c checkpoint (default: models/stories260K.bin)
  --tokenizer PATH    tokenizer (default: tok512.bin next to the model)
  --prompt TEXT       (default: \"Once upon a\")
  --max-new N         tokens to generate (default: 2)
  --floatdot          use the FloatDot precompile
  --ziskemu PATH      (default: ziskemu)
  --save-input PATH   write the guest input and exit";

fn record(input: &mut Vec<u8>, data: &[u8]) {
    input.extend_from_slice(&(data.len() as u64).to_le_bytes());
    input.extend_from_slice(data);
    input.resize(input.len().next_multiple_of(8), 0);
}

fn main() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut model_path = root.join("models/stories260K.bin");
    let mut tokenizer_path = None;
    let mut prompt = "Once upon a".to_string();
    let mut max_new = 2;
    let mut floatdot = false;
    let mut ziskemu = "ziskemu".to_string();
    let mut save_input = None;

    let mut args = std::env::args().skip(1);
    while let Some(name) = args.next() {
        if name == "--floatdot" {
            floatdot = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| format!("{name} needs a value\n{USAGE}"))?;
        match name.as_str() {
            "--model" => model_path = value.into(),
            "--tokenizer" => tokenizer_path = Some(PathBuf::from(value)),
            "--prompt" => prompt = value,
            "--max-new" => max_new = value.parse()?,
            "--ziskemu" => ziskemu = value,
            "--save-input" => save_input = Some(value),
            _ => return Err(format!("unknown option {name}\n{USAGE}").into()),
        }
    }

    let checkpoint = std::fs::read(&model_path)?;
    let (config, _) = read_config(&checkpoint)?;
    let tokenizer_path = tokenizer_path.unwrap_or_else(|| model_path.with_file_name("tok512.bin"));
    let tokenizer = Tokenizer::parse(&std::fs::read(tokenizer_path)?, config.vocab_size)?;
    let params = GenParams {
        prompt: tokenizer.encode(&prompt),
        max_new,
    };

    let mut input = Vec::new();
    record(&mut input, &checkpoint);
    record(&mut input, &params.to_bytes());
    record(&mut input, &[floatdot as u8]);
    if let Some(path) = save_input {
        std::fs::write(&path, input)?;
        println!("wrote {path}");
        return Ok(());
    }

    let dir = std::env::temp_dir().join(format!("llama2-zisk-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let (input_path, output_path) = (dir.join("input.bin"), dir.join("output.bin"));
    std::fs::write(&input_path, input)?;
    let run = Command::new(&ziskemu)
        .args(["-e", GUEST_ELF, "-m", "-i"])
        .arg(&input_path)
        .arg("-o")
        .arg(&output_path)
        .output()?;
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    if !run.status.success() {
        return Err(format!(
            "ziskemu failed: {stdout}{}",
            String::from_utf8_lossy(&run.stderr)
        )
        .into());
    }
    let committed = std::fs::read(&output_path)?;
    std::fs::remove_dir_all(&dir).ok();

    let tokens: Vec<u32> = stdout
        .lines()
        .find_map(|l| l.strip_prefix("tokens:"))
        .ok_or("the guest printed no tokens")?
        .split_whitespace()
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let steps = stdout
        .lines()
        .find_map(|l| l.split("steps=").nth(1)?.split_whitespace().next());

    let (native, trace) = generate(&Model::parse(&checkpoint), &params, dot_soft);
    let mut request = params.to_bytes();
    request.extend(tokens.iter().flat_map(|t| t.to_le_bytes()));
    let trace: Vec<u8> = trace.iter().flat_map(|v| v.to_le_bytes()).collect();
    let expected = [
        Sha256::digest(&request),
        Sha256::digest(&trace),
        Sha256::digest(&checkpoint),
    ];

    let mut text = prompt.clone();
    let mut prev = *params.prompt.last().unwrap();
    for &t in &tokens {
        text.push_str(&String::from_utf8_lossy(&tokenizer.decode(prev, t)));
        prev = t;
    }
    let ok = |b: bool| if b { "yes" } else { "NO" };
    println!("{text}");
    println!("steps: {}", steps.unwrap_or("?"));
    println!("same tokens as native: {}", ok(native == tokens));
    for (i, name) in ["request", "trace", "model"].iter().enumerate() {
        let same = committed.get(32 * i..32 * (i + 1)) == Some(&expected[i][..]);
        println!("committed {name} digest matches: {}", ok(same));
    }
    Ok(())
}
