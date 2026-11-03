use std::{env, fs, path::Path};
use weightscript::{compile, decompile, patch, verify, Model, Program};
fn read(path: &str) -> Result<Model, String> {
    serde_json::from_str(&fs::read_to_string(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}
fn write(path: impl AsRef<Path>, value: &impl serde::Serialize) -> Result<(), String> {
    fs::write(
        path,
        serde_json::to_string_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
fn run(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str).unwrap_or("demo") {
        "demo" => {
            let dir=args.get(1).map(String::as_str).unwrap_or("artifacts");
            fs::create_dir_all(dir).map_err(|e|e.to_string())?;
            let path=Path::new(dir);
            let before=compile(&Program::new("A")?);
            write(path.join("before.weights.json"), &before)?;
            // Deliberately read only the serialized weights for extraction.
            let before=read(path.join("before.weights.json").to_str().ok_or("invalid path")?)?;
            let original=decompile(&before)?;
            fs::write(path.join("before.rs"),original.source()).map_err(|e|e.to_string())?;
            let edited=Program::new("AB")?;
            write(path.join("edited.program.json"),&edited)?;
            fs::write(path.join("after.rs"),edited.source()).map_err(|e|e.to_string())?;
            let after=patch(&before,&edited)?;
            write(path.join("after.weights.json"),&after)?;
            let report=verify(&before,&after,&edited,6)?;
            write(path.join("verification.json"),&report)?;
            if report.counterexample.is_some() { return Err(format!("verification failed: {report:?}")); }
            let changed=before.w_out.iter().flatten().zip(after.w_out.iter().flatten()).filter(|(a,b)|a!=b).count();
            println!("WeightScript · decompile → edit → patch → verify\n\nModel: compiled integer hard-attention Transformer (not gradient-trained)\nEdit mechanism: deterministic structured edit (not a live LLM agent)\n\nExtracted from weights:\n{}\nEdited program:\n{}",original.source(),edited.source());
            println!("Same architecture. {changed} / {} scalar weights changed. Gradient updates: 0.",before.q.iter().flatten().count()+before.k.iter().flatten().count()+16+16*8+16+16*4);
            println!("\nInput     Before    After");
            for s in ["ABCD","CBDA","DABA","BBBB","AAAA"] { println!("{s:10}{:10}{}", before.predict(s)?,after.predict(s)?); }
            println!("\nPASS: {} inputs checked over {}.\n{} changed, {} preserved.\nExact bounded check; no claim of general Transformer decompilation.\nArtifacts: {dir}/",report.checked,report.domain,report.changed,report.preserved);
        }
        "compile" if args.len()==3 => {
            let p:Program=serde_json::from_str(&fs::read_to_string(&args[1]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            // Validate also when deserializing externally supplied programs.
            Program::new(&p.triggers.iter().collect::<String>())?;
            write(&args[2],&compile(&p))?;
        }
        "decompile" if args.len()==2 => { println!("{}",decompile(&read(&args[1])?)?.source()); }
        "patch" if args.len()==4 => {
            let m=read(&args[1])?;
            let p=Program::new(&args[2])?;
            write(&args[3],&patch(&m,&p)?)?;
        }
        "predict" if args.len()==3 => { println!("{}",read(&args[1])?.predict(&args[2])?); }
        "verify" if args.len()==5 || args.len()==6 => {
            let n=args.get(5).map(|s|s.parse::<usize>()).transpose().map_err(|e|e.to_string())?.unwrap_or(6);
            let report=verify(&read(&args[1])?,&read(&args[2])?,&Program::new(&args[3])?,n)?;
            write(&args[4],&report)?;
            println!("{}",serde_json::to_string_pretty(&report).map_err(|e|e.to_string())?);
            if report.counterexample.is_some() { return Err("verification failed".into()); }
        }
        _ => return Err("usage:\n  weightscript demo [artifact-directory]\n  weightscript decompile MODEL\n  weightscript compile PROGRAM.json OUTPUT\n  weightscript patch MODEL TRIGGERS OUTPUT\n  weightscript predict MODEL INPUT\n  weightscript verify BEFORE AFTER TRIGGERS REPORT.json [MAX_LEN]".into())
    }
    Ok(())
}
fn main() {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments.first().map(String::as_str) == Some("hf") {
        let python = env::var("WEIGHTSCRIPT_PYTHON").unwrap_or_else(|_| "python3".into());
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("python");
        let mut paths = vec![source];
        if let Some(existing) = env::var_os("PYTHONPATH") {
            paths.extend(env::split_paths(&existing));
        }
        let status = std::process::Command::new(&python)
            .env(
                "PYTHONPATH",
                env::join_paths(paths).expect("invalid Python path"),
            )
            .args(["-m", "weightscript.cli"])
            .args(&arguments[1..])
            .status();
        match status {
            Ok(s) => std::process::exit(s.code().unwrap_or(1)),
            Err(e) => {
                eprintln!("cannot launch {python}: {e}; set WEIGHTSCRIPT_PYTHON to your environment's Python");
                std::process::exit(1);
            }
        }
    }
    if let Err(e) = run(&env::args().skip(1).collect::<Vec<_>>()) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
