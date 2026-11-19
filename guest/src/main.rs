//! Input: a llama2.c checkpoint, the generation parameters, and one byte
//! (1 = FloatDot). Public output: sha256(params || tokens), sha256(trace),
//! sha256(checkpoint).

#![no_main]

ziskos::entrypoint!(main);

use llama2_common::model::{Model, dot_floatdot, dot_soft, generate};
use llama2_common::params::GenParams;
use ziskos::zisklib::sha256;

fn main() {
    let checkpoint = ziskos::io::read_slice();
    let params = ziskos::io::read_slice();
    let floatdot = ziskos::io::read_slice()[0] == 1;

    let model = Model::parse(&checkpoint);
    let dot = if floatdot { dot_floatdot } else { dot_soft };
    let (tokens, trace) = generate(&model, &GenParams::from_bytes(&params), dot);

    let line: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
    println!("tokens: {}", line.join(" "));

    let mut request = params.to_vec();
    request.extend(tokens.iter().flat_map(|t| t.to_le_bytes()));
    let trace: Vec<u8> = trace.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut output = Vec::with_capacity(96);
    output.extend_from_slice(&sha256(&request));
    output.extend_from_slice(&sha256(&trace));
    output.extend_from_slice(&sha256(&checkpoint));
    ziskos::io::commit_slice(&output);
}
