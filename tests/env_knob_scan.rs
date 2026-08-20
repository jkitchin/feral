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
//!
//! **Scope.** The guard is *statement*-scoped: it catches a parse in the
//! same statement as the read, which is the shape #176 found sixteen
//! copies of. It does not catch a read whose value is bound to a local
//! and parsed further down — `SIZES` in `bench_intrafront.rs` and
//! `STATIC_PIVOTS` in `probe_static_pivot_inertia.rs` were both live
//! instances of that second shape, found by review rather than by this
//! test and converted to `feral::env` alongside it. Widening the window
//! to catch them generically would flag legitimate `argv` parsing, so
//! the scan stays narrow and this note records what it cannot see.

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
            //
            // The window ends at whichever comes first: the statement's
            // `;`, a blank line, or 400 bytes. The blank line matters —
            // a read used as a function's *tail expression* has no `;`
            // at all, so `find(';')` runs on into whatever follows and
            // only the byte cap stops it. `Solver::pool_num_threads`
            // (`src/numeric/solver.rs`) is exactly that: its next `;` is
            // 1708 bytes away and the `.parse` inside the *separate*
            // `pool_num_threads_from` helper below it sits at +485, so
            // the scan passed only because 400 < 485 — an 85-byte
            // margin that trimming a doc comment would have erased.
            // Statements do not span blank lines under rustfmt, so
            // cutting there is the real statement boundary.
            let rest = &text[start..];
            let hard = rest
                .find(';')
                .unwrap_or(rest.len())
                .min(rest.find("\n\n").unwrap_or(rest.len()))
                .min(400);
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
