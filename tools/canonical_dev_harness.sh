#!/bin/bash
# Development harness audit for research code.
# Usage: .agents/scripts/dev-harness-audit.sh [project_dir]

set -euo pipefail

PROJECT_DIR="${1:-.}"
FAIL=0
WARN=0
PASS=0

note_pass() {
    echo "  [PASS] $*"
    PASS=$((PASS + 1))
}

note_warn() {
    echo "  [WARN] $*"
    WARN=$((WARN + 1))
}

note_fail() {
    echo "  [FAIL] $*"
    FAIL=$((FAIL + 1))
}

has_julia() {
    command -v julia >/dev/null 2>&1
}

is_excluded_project_toml() {
    case "$1" in
        */.agents/*|*/.claude/*|*/paper_dst_superlearner/*)
            return 0
            ;;
    esac
    # Exclude APPLICATION / environment Project.tomls (no name+uuid) — they are not Julia packages,
    # so the package-structure + Pkg.test checks do not apply to them (e.g. the bundled dashboard app/,
    # docs/). Only a real package (declares both name and uuid) is audited as a package.
    if ! grep -qiE '^[[:space:]]*name[[:space:]]*=' "$1" 2>/dev/null \
       || ! grep -qiE '^[[:space:]]*uuid[[:space:]]*=' "$1" 2>/dev/null; then
        return 0
    fi
    return 1
}

echo "# Development Harness Audit"
echo "Project: $PROJECT_DIR"
echo ""

PROJECT_TOMLS=""
while IFS= read -r toml; do
    if ! is_excluded_project_toml "$toml"; then
        PROJECT_TOMLS+="$toml"$'\n'
    fi
done < <(find "$PROJECT_DIR" -maxdepth 3 -name "Project.toml" 2>/dev/null | sort)
PROJECT_TOMLS="$(printf "%s" "$PROJECT_TOMLS" | sed '/^$/d')"

RESEARCH_PROJECT_TOMLS=""
while IFS= read -r toml; do
    [ -n "$toml" ] || continue
    pkg_dir="$(dirname "$toml")"
    if [ -s "$pkg_dir/examples/experiments.jl" ]; then
        RESEARCH_PROJECT_TOMLS+="$toml"$'\n'
    fi
done <<< "$PROJECT_TOMLS"
RESEARCH_PROJECT_TOMLS="$(printf "%s" "$RESEARCH_PROJECT_TOMLS" | sed '/^$/d')"

if [ -n "$RESEARCH_PROJECT_TOMLS" ]; then
    PROJECT_TOMLS="$RESEARCH_PROJECT_TOMLS"
fi

if [ -z "$PROJECT_TOMLS" ]; then
    note_warn "no Julia Project.toml found; development harness is ready but no package exists yet"
else
    if has_julia; then
        note_pass "Julia executable found"
    else
        note_fail "Julia executable not found; cannot verify research code"
    fi
fi

while IFS= read -r toml; do
    [ -n "$toml" ] || continue
    PKG_DIR="$(dirname "$toml")"
    IS_RESEARCH_PACKAGE=0
    if [ -s "$PKG_DIR/examples/experiments.jl" ]; then
        IS_RESEARCH_PACKAGE=1
    fi
    echo ""
    echo "## Package: $PKG_DIR"

    [ -s "$toml" ] && note_pass "Project.toml is present and nonempty" || note_fail "Project.toml is empty"

    if [ -d "$PKG_DIR/src" ]; then
        SRC_FILES="$(find "$PKG_DIR/src" -name "*.jl" 2>/dev/null | sort)"
        if [ -n "$SRC_FILES" ]; then
            note_pass "src/ contains Julia source"
        else
            note_fail "src/ exists but contains no .jl files"
        fi
    else
        note_fail "missing src/ directory"
        SRC_FILES=""
    fi

    TEST_FILE="$PKG_DIR/test/runtests.jl"
    if [ -s "$TEST_FILE" ]; then
        note_pass "test/runtests.jl exists"
    else
        note_fail "missing test/runtests.jl"
    fi

    if [ "$IS_RESEARCH_PACKAGE" -eq 1 ]; then
        CODE_NOTES="$PROJECT_DIR/CODE_NOTES.md"
        TEST_REPORT="$PROJECT_DIR/TEST_REPORT.md"

        if [ -s "$CODE_NOTES" ]; then
            CODE_NOTES_MISSING=""
            grep -Eiq 'Development Ledger' "$CODE_NOTES" || CODE_NOTES_MISSING+="Development Ledger; "
            grep -Eiq 'Implementation Correctness Gate|Independent oracle|Independent oracles' "$CODE_NOTES" || CODE_NOTES_MISSING+="implementation correctness/oracle record; "
            if [ -z "$CODE_NOTES_MISSING" ]; then
                note_pass "CODE_NOTES.md records implementation correctness context"
            else
                note_fail "CODE_NOTES.md missing: $CODE_NOTES_MISSING"
            fi
        else
            note_fail "missing CODE_NOTES.md for research package"
        fi

        if [ -s "$TEST_REPORT" ]; then
            TEST_REPORT_MISSING=""
            grep -Eiq 'Coverage|coverage' "$TEST_REPORT" || TEST_REPORT_MISSING+="coverage; "
            grep -Eiq 'Tolerance|tolerance' "$TEST_REPORT" || TEST_REPORT_MISSING+="tolerance; "
            grep -Eiq 'Anti-False-Test|anti-false|false-test|False-Test' "$TEST_REPORT" || TEST_REPORT_MISSING+="anti-false-test; "
            grep -Eiq 'oracle|independent' "$TEST_REPORT" || TEST_REPORT_MISSING+="oracle/independent expectation; "
            if [ -z "$TEST_REPORT_MISSING" ]; then
                note_pass "TEST_REPORT.md records test coverage, tolerances, oracles, or anti-false-test checks"
            else
                note_fail "TEST_REPORT.md missing: $TEST_REPORT_MISSING"
            fi
        else
            note_fail "missing TEST_REPORT.md for research package"
        fi
    fi

    ALL_JL="$(find "$PKG_DIR" -path "$PKG_DIR/.git" -prune -o -type f -name "*.jl" -print 2>/dev/null | sort)"
    if [ -n "$ALL_JL" ] && has_julia; then
        while IFS= read -r file; do
            [ -n "$file" ] || continue
            if julia --startup-file=no -e 'Meta.parseall(read(ARGS[1], String))' "$file" >/dev/null 2>&1; then
                note_pass "parse ok: ${file#$PROJECT_DIR/}"
            else
                note_fail "parse failed: ${file#$PROJECT_DIR/}"
            fi
        done <<< "$ALL_JL"
    fi

    # Bundled application suites (for example app/test) ship inside the package and are run by the
    # package's own runtests.jl, so their assertions are part of the package's evidence and must be
    # scanned with it. Scanning only $PKG_DIR/test leaves them unaudited.
    TEST_DIRS=("$PKG_DIR/test")
    for EXTRA_TEST_DIR in "$PKG_DIR"/*/test; do
        [ -d "$EXTRA_TEST_DIR" ] || continue
        [ "$EXTRA_TEST_DIR" = "$PKG_DIR/test" ] && continue
        TEST_DIRS+=("$EXTRA_TEST_DIR")
    done

    if [ -s "$TEST_FILE" ]; then
        ASSERT_MACROS="$(grep -RhoE '@test(_throws|_broken|_skip)?([[:space:]]|\(|$)' "${TEST_DIRS[@]}" 2>/dev/null | wc -l | tr -d ' ')"
        TESTSET_MACROS="$(grep -RhoE '@testset([[:space:]]|\(|$)' "${TEST_DIRS[@]}" 2>/dev/null | wc -l | tr -d ' ')"
        if [ "$ASSERT_MACROS" -gt 0 ] && [ "$TESTSET_MACROS" -gt 0 ]; then
            note_pass "tests contain @testset and test assertions"
        else
            note_fail "tests lack real @testset/@test coverage"
        fi

        BAD_TESTS=""
        while IFS= read -r test_jl; do
            [ -n "$test_jl" ] || continue
            # NOTE: the @ in @test MUST be escaped (\@). Unescaped, perl interpolates @test as an (empty)
            # array, collapsing /@test\s+(true|false)/ to /\s+(true|false)/ — which then false-flags EVERY
            # line containing the word true/false (comments, == true/false comparisons, boolean args).
            HITS="$(perl -nle 'print "$ARGV:$.:$_" if /\@test\s+(true|false)\b/ || /\@test\s+([A-Za-z_]\w*)\s*==\s*\1\b/ || /isapprox\(\s*([A-Za-z_]\w*)\s*,\s*\1[\s,\)]/ || /;\s*true\s*\)?\s*$/' "$test_jl" 2>/dev/null || true)"
            if [ -n "$HITS" ]; then
                BAD_TESTS+="$HITS"$'\n'
            fi
        done < <(find "${TEST_DIRS[@]}" -name "*.jl" 2>/dev/null | sort)
        if [ -n "$BAD_TESTS" ]; then
            echo "$BAD_TESTS" | sed 's/^/    /'
            note_fail "suspicious tautological tests found"
        else
            note_pass "no obvious tautological tests found"
        fi

        LOOSE_TOL="$(grep -RInE '(r|a)tol[[:space:]]*=[[:space:]]*(0\.[0-9]+|1e-[0-3]([^0-9]|$))' "${TEST_DIRS[@]}" 2>/dev/null || true)"
        if [ -n "$LOOSE_TOL" ]; then
            echo "$LOOSE_TOL" | sed 's/^/    /'
            note_warn "loose tolerances found; confirm each is justified"
        else
            note_pass "no obviously loose tolerances found"
        fi
    fi

    RANDOM_LINES="$(grep -RInE 'rand\(|randn\(|Random\.' "$PKG_DIR/src" "${TEST_DIRS[@]}" 2>/dev/null || true)"
    if [ -n "$RANDOM_LINES" ]; then
        SEEDED_LINES="$(echo "$RANDOM_LINES" | grep -Ei 'seed!|MersenneTwister|StableRNG|rng' || true)"
        if [ -n "$SEEDED_LINES" ]; then
            note_pass "randomness appears to use explicit seed/RNG"
        else
            echo "$RANDOM_LINES" | sed 's/^/    /'
            note_warn "randomness found without obvious seed/RNG"
        fi
    else
        note_pass "no randomness detected"
    fi

    MARKERS="$(grep -RInE 'TODO|FIXME|HACK|XXX' "$PKG_DIR/src" "${TEST_DIRS[@]}" 2>/dev/null || true)"
    if [ -n "$MARKERS" ]; then
        echo "$MARKERS" | sed 's/^/    /'
        note_warn "open TODO/FIXME/HACK markers in code"
    else
        note_pass "no TODO/FIXME/HACK markers in source/tests"
    fi

    if has_julia && [ -s "$TEST_FILE" ]; then
        PKG_TEST_LOG="$(mktemp "${TMPDIR:-/tmp}/dev-harness-pkg-test.XXXXXX")"
        if (cd "$PKG_DIR" && julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()') >"$PKG_TEST_LOG" 2>&1; then
            note_pass "Pkg.test() passed"
        else
            sed -n '1,80p' "$PKG_TEST_LOG" | sed 's/^/    /'
            note_fail "Pkg.test() failed"
        fi
    fi

    if [ "$IS_RESEARCH_PACKAGE" -eq 1 ]; then
        if has_julia; then
            EXPERIMENT_LOG="$(mktemp "${TMPDIR:-/tmp}/dev-harness-experiments.XXXXXX")"
            if (cd "$PKG_DIR" && julia --startup-file=no --project=. examples/experiments.jl) >"$EXPERIMENT_LOG" 2>&1; then
                note_pass "examples/experiments.jl executed"
            else
                sed -n '1,80p' "$EXPERIMENT_LOG" | sed 's/^/    /'
                note_fail "examples/experiments.jl failed"
            fi
        else
            note_fail "cannot run examples/experiments.jl without Julia"
        fi
    fi
done <<< "$PROJECT_TOMLS"

echo ""
echo "## Paper Data/Figure Coupling"
if [ -d "$PROJECT_DIR/paper_dst_superlearner/data" ]; then
    CSV_COUNT="$(find "$PROJECT_DIR/paper_dst_superlearner/data" -name "*.csv" 2>/dev/null | wc -l | tr -d ' ')"
    if [ "$CSV_COUNT" -gt 0 ]; then
        note_pass "$CSV_COUNT CSV data files found"
        if "$PROJECT_DIR/.agents/scripts/csv-audit.sh" "$PROJECT_DIR/paper_dst_superlearner/data" >/tmp/dev-harness-csv.log 2>&1; then
            note_pass "CSV audit completed"
        else
            sed -n '1,80p' /tmp/dev-harness-csv.log | sed 's/^/    /'
            note_fail "CSV audit failed"
        fi
    else
        note_warn "paper_dst_superlearner/data exists but contains no CSV files"
    fi
else
    note_warn "paper_dst_superlearner/data directory not found"
fi

if [ -d "$PROJECT_DIR/paper_dst_superlearner/figs" ]; then
    FIG_COUNT="$(find "$PROJECT_DIR/paper_dst_superlearner/figs" \( -name "*.pdf" -o -name "*.png" -o -name "*.svg" \) 2>/dev/null | wc -l | tr -d ' ')"
    if [ "$FIG_COUNT" -gt 0 ]; then
        note_pass "$FIG_COUNT figure files found"
    else
        note_warn "paper_dst_superlearner/figs exists but contains no figure files"
    fi
else
    note_warn "paper_dst_superlearner/figs directory not found"
fi

echo ""
echo "## Summary"
echo "PASS: $PASS"
echo "WARN: $WARN"
echo "FAIL: $FAIL"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
