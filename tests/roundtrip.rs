use weightscript::*;
#[test]
fn all_programs_round_trip_and_execute() {
    for mask in 0..16 {
        let triggers: String = VOCAB
            .chars()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, c)| c)
            .collect();
        let p = Program::new(&triggers).unwrap();
        let model = compile(&p);
        assert_eq!(decompile(&model).unwrap(), p);
        let report = verify(&model, &model, &p, 4).unwrap();
        assert_eq!(report.checked, 341);
        assert!(report.counterexample.is_none());
    }
}
#[test]
fn patch_changes_only_requested_positions_and_output_weights() {
    let before = compile(&Program::new("A").unwrap());
    let target = Program::new("AB").unwrap();
    let after = patch(&before, &target).unwrap();
    let mut restored = after.clone();
    restored.w_out = before.w_out.clone();
    assert_eq!(restored, before);
    for mut code in 0..4096 {
        let mut input = String::new();
        for _ in 0..6 {
            input.push(VOCAB.as_bytes()[code % 4] as char);
            code /= 4;
        }
        let a = before.predict(&input).unwrap();
        let b = after.predict(&input).unwrap();
        for (i, ((x, y), t)) in a.chars().zip(b.chars()).zip(input.chars()).enumerate() {
            if i == 0 || t != 'B' {
                assert_eq!(x, y);
            }
        }
    }
    assert_eq!(before.predict("ABCD").unwrap(), "ABCD");
    assert_eq!(after.predict("ABCD").unwrap(), "AACD");
}
#[test]
fn verifier_finds_wrong_patch_and_imports_fail_closed() {
    let before = compile(&Program::new("A").unwrap());
    assert!(verify(&before, &before, &Program::new("AB").unwrap(), 3)
        .unwrap()
        .counterexample
        .is_some());
    let mut corrupt = before.clone();
    corrupt.w_out[0][0] = 99;
    assert!(decompile(&corrupt).is_err());
    assert!(corrupt.predict("AB").is_err());
    assert!(Program::new("Z").is_err());
    assert!(before.predict("A💥").is_err());
    assert!(before.predict(&"A".repeat(65)).is_err());
    assert!(verify(&before, &before, &Program::new("A").unwrap(), 9).is_err());
}
