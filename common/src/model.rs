//! llama2.c's `run.c` in `f32`. Every dot product goes through one [`DotFn`]
//! over operands packed two `f32` per `u64`, low half first, the layout the
//! FloatDot precompile reads.

use alloc::vec;
use alloc::vec::Vec;

use crate::params::GenParams;
use crate::{BOS, read_f32, read_u32};

/// `((0 + x[0]·y[0]) + x[1]·y[1]) + …` over `n` values, each operation
/// rounded separately.
pub type DotFn = fn(&[u64], &[u64], usize) -> f32;

pub fn dot_soft(x: &[u64], y: &[u64], n: usize) -> f32 {
    let lo = |w: u64| f32::from_bits(w as u32);
    let hi = |w: u64| f32::from_bits((w >> 32) as u32);
    let mut acc = 0.0f32;
    for (&a, &b) in x[..n / 2].iter().zip(&y[..n / 2]) {
        acc += lo(a) * lo(b);
        acc += hi(a) * hi(b);
    }
    if n % 2 == 1 {
        acc += lo(x[n / 2]) * lo(y[n / 2]);
    }
    acc
}

/// Calls of at most 4096 values, each starting from the previous result.
#[cfg(feature = "floatdot")]
pub fn dot_floatdot(x: &[u64], y: &[u64], n: usize) -> f32 {
    const MAX: usize = 4096;
    let words = n.div_ceil(2);
    let mut acc = 0u32;
    for (i, (xc, yc)) in x[..words]
        .chunks(MAX / 2)
        .zip(y[..words].chunks(MAX / 2))
        .enumerate()
    {
        let len = (n - i * MAX).min(MAX);
        let r =
            ziskos::syscalls::syscall_fp32_dot(xc, yc, len, acc, 0).expect("valid FloatDot call");
        acc = r as u32;
    }
    f32::from_bits(acc)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
}

impl Config {
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.dim * self.n_kv_heads / self.n_heads
    }
}

/// The checkpoint header; `true` when the classifier is the embedding.
pub fn read_config(bytes: &[u8]) -> Result<(Config, bool), &'static str> {
    if bytes.len() < 28 {
        return Err("checkpoint is too short");
    }
    let h: Vec<i32> = (0..7).map(|i| read_u32(bytes, 4 * i) as i32).collect();
    if h.iter()
        .enumerate()
        .any(|(i, &v)| v == 0 || (i != 5 && v < 0))
    {
        return Err("checkpoint header is not a llama2.c config");
    }
    let config = Config {
        dim: h[0] as usize,
        hidden_dim: h[1] as usize,
        n_layers: h[2] as usize,
        n_heads: h[3] as usize,
        n_kv_heads: h[4] as usize,
        vocab_size: h[5].unsigned_abs() as usize,
        seq_len: h[6] as usize,
    };
    Ok((config, h[5] > 0))
}

fn pack(v: &[f32], words: &mut [u64]) {
    for (w, pair) in words.iter_mut().zip(v.chunks(2)) {
        let hi = pair.get(1).map_or(0, |h| h.to_bits());
        *w = pair[0].to_bits() as u64 | (hi as u64) << 32;
    }
}

fn push(trace: &mut Vec<u32>, values: &[f32]) {
    trace.extend(values.iter().map(|v| v.to_bits()));
}

struct Matrix {
    words: Vec<u64>,
    cols: usize,
}

pub struct Model {
    pub config: Config,
    rms_att: Vec<f32>,
    rms_ffn: Vec<f32>,
    rms_final: Vec<f32>,
    embedding: Matrix,
    wq: Matrix,
    wk: Matrix,
    wv: Matrix,
    wo: Matrix,
    w1: Matrix,
    w2: Matrix,
    w3: Matrix,
    wcls: Option<Matrix>,
}

impl Model {
    /// Parses a llama2.c checkpoint with even dimensions.
    pub fn parse(bytes: &[u8]) -> Self {
        let (c, shared) = read_config(bytes).expect("not a llama2.c checkpoint");
        let (dim, hidden, layers, kv_dim) = (c.dim, c.hidden_dim, c.n_layers, c.kv_dim());
        assert!(c.head_size() % 2 == 0 && kv_dim % 2 == 0, "odd head size");
        let mut at = 28;
        let mut next = |n: usize| {
            at += 4 * n;
            assert!(at <= bytes.len(), "checkpoint is truncated");
            at - 4 * n
        };
        let floats =
            |from: usize, n: usize| (0..n).map(|i| read_f32(bytes, from + 4 * i)).collect();
        let matrix = |from: usize, rows: usize, cols: usize| {
            assert!(cols.is_multiple_of(2), "odd matrix width");
            let b = &bytes[from..from + 4 * rows * cols];
            let words = b
                .chunks_exact(8)
                .map(|w| u64::from_le_bytes(w.try_into().unwrap()));
            Matrix {
                words: words.collect(),
                cols,
            }
        };
        let embedding = matrix(next(c.vocab_size * dim), c.vocab_size, dim);
        let rms_att = floats(next(layers * dim), layers * dim);
        let wq = matrix(next(layers * dim * dim), layers * dim, dim);
        let wk = matrix(next(layers * kv_dim * dim), layers * kv_dim, dim);
        let wv = matrix(next(layers * kv_dim * dim), layers * kv_dim, dim);
        let wo = matrix(next(layers * dim * dim), layers * dim, dim);
        let rms_ffn = floats(next(layers * dim), layers * dim);
        let w1 = matrix(next(layers * hidden * dim), layers * hidden, dim);
        let w2 = matrix(next(layers * dim * hidden), layers * dim, hidden);
        let w3 = matrix(next(layers * hidden * dim), layers * hidden, dim);
        let rms_final = floats(next(dim), dim);
        next(c.seq_len * c.head_size()); // unused RoPE tables
        let wcls = (!shared).then(|| matrix(next(c.vocab_size * dim), c.vocab_size, dim));
        Model {
            config: c,
            rms_att,
            rms_ffn,
            rms_final,
            embedding,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            wcls,
        }
    }
}

fn matmul(dot: DotFn, out: &mut [f32], xw: &[u64], w: &Matrix, first_row: usize) {
    let half = w.cols / 2;
    for (i, o) in out.iter_mut().enumerate() {
        *o = dot(&w.words[(first_row + i) * half..][..half], xw, w.cols);
    }
}

fn rmsnorm(
    dot: DotFn,
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    xw: &mut [u64],
    trace: &mut Vec<u32>,
) {
    pack(x, xw);
    let mut ss = dot(xw, xw, x.len());
    push(trace, &[ss]);
    ss /= x.len() as f32;
    ss += 1e-5;
    ss = 1.0 / libm::sqrtf(ss);
    for ((o, &v), &w) in out.iter_mut().zip(x).zip(weight) {
        *o = w * (ss * v);
    }
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().fold(x[0], |m, &v| m.max(v));
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = libm::expf(*v - max);
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

struct Run<'a> {
    model: &'a Model,
    dot: DotFn,
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    logits: Vec<f32>,
    xw: Vec<u64>,
    qw: Vec<u64>,
    aw: Vec<u64>,
    cw: Vec<u64>,
    key_cache: Vec<u64>,
    value_cache: Vec<f32>,
    rope_freq: Vec<f32>,
    trace: Vec<u32>,
}

impl<'a> Run<'a> {
    fn new(model: &'a Model, dot: DotFn) -> Self {
        let c = &model.config;
        let (dim, kv_dim, hidden, hs) = (c.dim, c.kv_dim(), c.hidden_dim, c.head_size());
        let rope_freq = (0..hs)
            .step_by(2)
            .map(|hd| 1.0 / libm::powf(10000.0, hd as f32 / hs as f32))
            .collect();
        Run {
            model,
            dot,
            x: vec![0.0; dim],
            xb: vec![0.0; dim],
            xb2: vec![0.0; dim],
            hb: vec![0.0; hidden],
            hb2: vec![0.0; hidden],
            q: vec![0.0; dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            att: vec![0.0; c.seq_len],
            logits: vec![0.0; c.vocab_size],
            xw: vec![0; dim.max(hidden).div_ceil(2)],
            qw: vec![0; dim / 2],
            aw: vec![0; c.seq_len.div_ceil(2)],
            cw: vec![0; c.seq_len.div_ceil(2)],
            key_cache: vec![0; c.n_layers * c.seq_len * kv_dim / 2],
            value_cache: vec![0.0; c.n_layers * c.seq_len * kv_dim],
            rope_freq,
            trace: Vec::new(),
        }
    }

    fn forward(&mut self, token: u32, pos: usize) -> &[f32] {
        let (m, dot) = (self.model, self.dot);
        let s = self;
        let c = &m.config;
        let (dim, kv_dim, hidden, hs) = (c.dim, c.kv_dim(), c.hidden_dim, c.head_size());
        let kv_mul = c.n_heads / c.n_kv_heads;

        let row = &m.embedding.words[token as usize * dim / 2..][..dim / 2];
        for (pair, &w) in s.x.chunks_mut(2).zip(row) {
            pair[0] = f32::from_bits(w as u32);
            pair[1] = f32::from_bits((w >> 32) as u32);
        }

        let sqrt_hs = libm::sqrtf(hs as f32);
        for l in 0..c.n_layers {
            let rms = &m.rms_att[l * dim..(l + 1) * dim];
            rmsnorm(dot, &mut s.xb, &s.x, rms, &mut s.xw, &mut s.trace);
            pack(&s.xb, &mut s.xw);
            matmul(dot, &mut s.q, &s.xw, &m.wq, l * dim);
            matmul(dot, &mut s.k, &s.xw, &m.wk, l * kv_dim);
            matmul(dot, &mut s.v, &s.xw, &m.wv, l * kv_dim);
            push(&mut s.trace, &s.q);
            push(&mut s.trace, &s.k);
            push(&mut s.trace, &s.v);

            for i in (0..dim).step_by(2) {
                let val = pos as f32 * s.rope_freq[(i % hs) / 2];
                let (fcr, fci) = (libm::cosf(val), libm::sinf(val));
                let rotn = if i < kv_dim { 2 } else { 1 };
                for r in 0..rotn {
                    let v = if r == 0 { &mut s.q } else { &mut s.k };
                    let (v0, v1) = (v[i], v[i + 1]);
                    v[i] = v0 * fcr - v1 * fci;
                    v[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            let loff = l * c.seq_len * kv_dim;
            pack(
                &s.k,
                &mut s.key_cache[(loff + pos * kv_dim) / 2..][..kv_dim / 2],
            );
            s.value_cache[loff + pos * kv_dim..][..kv_dim].copy_from_slice(&s.v);
            pack(&s.q, &mut s.qw);

            for h in 0..c.n_heads {
                let qh = &s.qw[h * hs / 2..][..hs / 2];
                let kv_off = (h / kv_mul) * hs;
                let att = &mut s.att[..pos + 1];
                for (t, a) in att.iter_mut().enumerate() {
                    let kh = &s.key_cache[(loff + t * kv_dim + kv_off) / 2..][..hs / 2];
                    let score = dot(qh, kh, hs);
                    push(&mut s.trace, &[score]);
                    *a = score / sqrt_hs;
                }
                softmax(att);
                pack(att, &mut s.aw);
                for i in 0..hs {
                    let value = |t: usize| match t <= pos {
                        true => s.value_cache[loff + t * kv_dim + kv_off + i].to_bits() as u64,
                        false => 0,
                    };
                    for (j, w) in s.cw[..(pos + 1).div_ceil(2)].iter_mut().enumerate() {
                        *w = value(2 * j) | value(2 * j + 1) << 32;
                    }
                    s.xb[h * hs + i] = dot(&s.aw, &s.cw, pos + 1);
                }
                push(&mut s.trace, &s.xb[h * hs..(h + 1) * hs]);
            }

            pack(&s.xb, &mut s.xw);
            matmul(dot, &mut s.xb2, &s.xw, &m.wo, l * dim);
            push(&mut s.trace, &s.xb2);
            for (x, &d) in s.x.iter_mut().zip(&s.xb2) {
                *x += d;
            }

            let rms = &m.rms_ffn[l * dim..(l + 1) * dim];
            rmsnorm(dot, &mut s.xb, &s.x, rms, &mut s.xw, &mut s.trace);
            pack(&s.xb, &mut s.xw);
            matmul(dot, &mut s.hb, &s.xw, &m.w1, l * hidden);
            matmul(dot, &mut s.hb2, &s.xw, &m.w3, l * hidden);
            push(&mut s.trace, &s.hb);
            push(&mut s.trace, &s.hb2);
            for (h, &h2) in s.hb.iter_mut().zip(&s.hb2) {
                let mut val = *h;
                val *= 1.0 / (1.0 + libm::expf(-val));
                *h = val * h2;
            }
            pack(&s.hb, &mut s.xw);
            matmul(dot, &mut s.xb, &s.xw, &m.w2, l * dim);
            push(&mut s.trace, &s.xb);
            for (x, &d) in s.x.iter_mut().zip(&s.xb) {
                *x += d;
            }
        }

        let x = s.x.clone();
        rmsnorm(dot, &mut s.x, &x, &m.rms_final, &mut s.xw, &mut s.trace);
        pack(&s.x, &mut s.xw);
        matmul(
            dot,
            &mut s.logits,
            &s.xw,
            m.wcls.as_ref().unwrap_or(&m.embedding),
            0,
        );
        push(&mut s.trace, &s.logits);
        &s.logits
    }
}

/// Greedy generation. Returns the generated tokens and the trace: the bits of
/// every dot product result and logit, in order.
pub fn generate(model: &Model, p: &GenParams, dot: DotFn) -> (Vec<u32>, Vec<u32>) {
    assert!(!p.prompt.is_empty(), "the prompt must start with BOS");
    let mut run = Run::new(model, dot);
    let mut out = Vec::new();
    let mut token = p.prompt[0];
    let mut pos = 0;
    while pos < model.config.seq_len && out.len() < p.max_new as usize {
        let logits = run.forward(token, pos);
        token = if pos + 1 < p.prompt.len() {
            p.prompt[pos + 1]
        } else {
            let next = argmax(logits);
            if next == BOS {
                break;
            }
            out.push(next);
            next
        };
        pos += 1;
    }
    (out, run.trace)
}
