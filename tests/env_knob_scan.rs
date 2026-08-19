//! Issue #176 guard: no env read outside `src/env.rs` may parse its own
//! value.
//!
//! The bug in #176 was not one call site but one *shape*, copied to
//! sixteen places: `env::var(NAME).ok().and_then(|v| v.parse().ok())`
//! throws the parse error away, so a value the parser cannot use is
//! replaced by the built-in default without a word. `feral::env` is now
//! the single place that decides what a knob accepts and what happens
//! to a value it cannot use.
//!
//! This is a source scan, not a behavioural test, so it lives apart from
//! `env_knob_parsing.rs` — that binary mutates process-global env vars
//! and must not share a process with anything else.

/// The scan matches `env::var(` with **any** argument, not just a
/// literal `"FERAL_…"`. The four sneakiest sites #176 touched read
/// `env::var(key)` through a local `fn env_usize(key: &str, …)` helper,
/// so a literal-name scan would have missed exactly the cases that are
/// easiest to reintroduce.
#[test]
fn no_env_read_parses_its_own_value() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    collect_rs(&root.join("src"), &mut files);
    collect_rs(&root.join("crates"), &mut files);
    assert!(
        files.len() > 50,
        "source scan found only {} files — the walk is broken, not the tree clean",
        files.len()
    );

    let mut offenders = Vec::new();
    for path in files {
        // The one file allowed to parse: the shared policy itself.
        if path.ends_with("src/env.rs") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for (start, _) in text.match_indices("env::var(") {
            // Window = the rest of that statement, capped so a `;`-less
            // construct cannot swallow the whole file. The cap is walked
            // back to a char boundary: this tree's comments are full of
            // multibyte characters (`×`, `‖`, `≥`), and slicing a byte
            // offset inside one panics.
            let rest = &text[start..];
            let hard = rest.find(';').unwrap_or(rest.len()).min(400);
            let end = rest
                .char_indices()
                .map(|(i, _)| i)
                .take_while(|&i| i <= hard)
                .last()
                .unwrap_or(0);
            let window = &rest[..end];
            if window.contains(".parse") {
                let line = text[..start].matches('\n').count() + 1;
                offenders.push(format!("{}:{}", path.display(), line));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "these env reads parse locally instead of going through \
         feral::env, which is how issue #176 happened: {offenders:?}"
    );
}

fn collect_rs(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}
