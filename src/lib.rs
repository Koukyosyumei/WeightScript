use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const VOCAB: &str = "ABCD";
pub const WIDTH: usize = 4;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Program {
    pub triggers: BTreeSet<char>,
}
impl Program {
    pub fn new(triggers: &str) -> Result<Self, String> {
        if !triggers.chars().all(|c| VOCAB.contains(c)) {
            return Err("triggers must contain only A, B, C, D".into());
        }
        Ok(Self {
            triggers: triggers.chars().collect(),
        })
    }
    pub fn predict(&self, input: &str) -> Result<String, String> {
        let ids = tokens(input)?;
        Ok(ids
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                let source = if i > 0 && self.triggers.contains(&symbol(t)) {
                    ids[i - 1]
                } else {
                    t
                };
                symbol(source)
            })
            .collect())
    }
    pub fn source(&self) -> String {
        format!("fn predict(tokens: &[char]) -> Vec<char> {{\n    tokens.iter().enumerate().map(|(i, &t)| {{\n        if i > 0 && {:?}.contains(t) {{ tokens[i - 1] }} else {{ t }}\n    }}).collect()\n}}\n", self.triggers.iter().collect::<String>())
    }
}

// Exact integer, single-head hard-attention Transformer. The residual stream
// contains current-token one-hot (4) and attention output (4). A 16-unit
// ReLU MLP adds corrections to current-token logits. No source metadata.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Model {
    pub format: String,
    pub q: [[i32; 2]; 2],
    pub k: [[i32; 2]; 2],
    pub v: [[i32; 4]; 4],
    pub w_in: Vec<[i32; 8]>,
    pub bias: Vec<i32>,
    pub w_out: Vec<[i32; 4]>,
}
fn identity() -> [[i32; 4]; 4] {
    std::array::from_fn(|i| std::array::from_fn(|j| i32::from(i == j)))
}
pub fn compile(p: &Program) -> Model {
    let mut m = Model {
        format: "weightscript-hard-attention-v1".into(),
        q: [[1, 0], [0, 1]],
        k: [[2, 0], [0, -1]],
        v: identity(),
        w_in: vec![],
        bias: vec![-1; 16],
        w_out: vec![],
    };
    for t in 0..4 {
        for prev in 0..4 {
            let mut row = [0; 8];
            row[t] = 1;
            row[4 + prev] = 1;
            m.w_in.push(row);
            let mut out = [0; 4];
            if p.triggers.contains(&symbol(t)) {
                out[prev] += 1;
                out[t] -= 1;
            }
            m.w_out.push(out);
        }
    }
    m
}
impl Model {
    pub fn validate(&self) -> Result<(), String> {
        if self.w_in.len() != 16 || self.bias.len() != 16 || self.w_out.len() != 16 {
            return Err("expected a 16-unit MLP".into());
        }
        // Restrict imported files to the supported circuit family. Also bounds
        // all integer arithmetic used in the verifier.
        if self.format != "weightscript-hard-attention-v1"
            || self.q != [[1, 0], [0, 1]]
            || self.k != [[2, 0], [0, -1]]
            || self.v != identity()
        {
            return Err("unsupported attention circuit".into());
        }
        let base = compile(&Program::new("")?);
        if self.w_in != base.w_in || self.bias != base.bias {
            return Err("unsupported MLP input circuit".into());
        }
        for t in 0..4 {
            let active = compile(&Program::new(&symbol(t).to_string())?);
            let rows = &self.w_out[t * 4..t * 4 + 4];
            if rows != &base.w_out[t * 4..t * 4 + 4] && rows != &active.w_out[t * 4..t * 4 + 4] {
                return Err(format!("unsupported output circuit for {}", symbol(t)));
            }
        }
        Ok(())
    }
    pub fn predict(&self, input: &str) -> Result<String, String> {
        self.validate()?;
        let ids = tokens(input)?;
        let mut result = String::new();
        for (i, &t) in ids.iter().enumerate() {
            // Positional features: query [i-1, 1], key [j, j*j].
            // q.k = 2(i-1)j-j², uniquely maximized at previous position.
            let pos = [i as i32 - 1, 1];
            let query: [i32; 2] =
                std::array::from_fn(|a| (0..2).map(|b| self.q[a][b] * pos[b]).sum());
            let mut best = i64::MIN;
            let mut selected = 0;
            for j in 0..=i {
                let key_pos = [j as i32, (j * j) as i32];
                let key: [i32; 2] =
                    std::array::from_fn(|a| (0..2).map(|b| self.k[a][b] * key_pos[b]).sum());
                let score: i64 = (0..2)
                    .map(|a| i64::from(query[a]) * i64::from(key[a]))
                    .sum();
                if score > best {
                    best = score;
                    selected = j;
                }
            }
            let mut residual = [0; 8];
            residual[t] = 1;
            for a in 0..4 {
                residual[4 + a] = self.v[a][ids[selected]];
            }
            let mut logits = [0; 4];
            logits[t] = 1;
            for h in 0..16 {
                let activation = (self.bias[h]
                    + (0..8).map(|a| self.w_in[h][a] * residual[a]).sum::<i32>())
                .max(0);
                for (a, logit) in logits.iter_mut().enumerate() {
                    *logit += activation * self.w_out[h][a];
                }
            }
            let winner = (0..4).max_by_key(|&a| (logits[a], -(a as i32))).unwrap();
            result.push(symbol(winner));
        }
        Ok(result)
    }
}
pub fn decompile(m: &Model) -> Result<Program, String> {
    m.validate()?;
    let mut triggers = BTreeSet::new();
    for t in 0..4 {
        if m.w_out[t * 4..t * 4 + 4].iter().flatten().any(|&w| w != 0) {
            triggers.insert(symbol(t));
        }
    }
    let p = Program { triggers };
    if compile(&p) != *m {
        return Err("round-trip mismatch".into());
    }
    Ok(p)
}
pub fn patch(m: &Model, p: &Program) -> Result<Model, String> {
    decompile(m)?;
    let mut updated = m.clone();
    updated.w_out = compile(p).w_out;
    Ok(updated)
}
fn symbol(i: usize) -> char {
    VOCAB.as_bytes()[i] as char
}
fn tokens(input: &str) -> Result<Vec<usize>, String> {
    if input.len() > 64 {
        return Err("maximum supported input length is 64".into());
    }
    input
        .chars()
        .map(|c| VOCAB.find(c).ok_or_else(|| format!("invalid token {c:?}")))
        .collect()
}
#[derive(Debug, Serialize)]
pub struct Verification {
    pub domain: String,
    pub checked: usize,
    pub changed: usize,
    pub preserved: usize,
    pub counterexample: Option<String>,
    pub method: String,
}
pub fn verify(
    before: &Model,
    after: &Model,
    target: &Program,
    max_len: usize,
) -> Result<Verification, String> {
    if !(1..=8).contains(&max_len) {
        return Err("exhaustive verification length must be 1..8".into());
    }
    let original = decompile(before)?;
    after.validate()?;
    let mut report = Verification {
        domain: format!("ABCD strings of length 0..={max_len}"),
        checked: 0,
        changed: 0,
        preserved: 0,
        counterexample: None,
        method: "exhaustive enumeration; exact integer inference; no floating point or SMT".into(),
    };
    for n in 0..=max_len {
        for mut code in 0..4usize.pow(n as u32) {
            let mut input = String::new();
            for _ in 0..n {
                input.push(symbol(code % 4));
                code /= 4;
            }
            let a = before.predict(&input)?;
            let b = after.predict(&input)?;
            let expected = target.predict(&input)?;
            report.checked += 1;
            if a == b {
                report.preserved += 1;
            } else {
                report.changed += 1;
            }
            if a != original.predict(&input)? || b != expected {
                report.counterexample = Some(input);
                return Ok(report);
            }
        }
    }
    Ok(report)
}
