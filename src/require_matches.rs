/// Checks that the provided expression matches a pattern, or returns with an error.
macro_rules! require_matches {
    ($expression:expr, $pattern:pat $(if $guard:expr)?, $then: expr) => {
        match $expression {
            $pattern $(if $guard)? => $then,
            _ => bail!("Incorrect type.")
        }
    };
}

pub(crate) use require_matches;
