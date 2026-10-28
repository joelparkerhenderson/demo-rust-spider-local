//! Predicate for visible (non-hidden) directory entries.

use ::walkdir::DirEntry;

/// Returns `true` if the entry's file name does not start with a dot.
///
/// This is the opposite of
/// [`dir_entry_is_hidden`](super::dir_entry_is_hidden::dir_entry_is_hidden)
/// for valid UTF-8 names, and is convenient as a
/// `WalkDir::into_iter().filter_entry(dir_entry_is_visible)` predicate to
/// skip hidden files and whole hidden directories.
///
/// Note: names that are not valid UTF-8 return `false` here (treated as not
/// visible), whereas `dir_entry_is_hidden` also returns `false` for them.
///
/// # Arguments
///
/// * `entry` - the directory entry to test.
#[allow(dead_code)]
pub fn dir_entry_is_visible(entry: &DirEntry) -> bool {
    entry.file_name()
         // Convert the OS string to `&str`; `None` if not valid UTF-8.
         .to_str()
         .map(|s| !s.starts_with("."))
         // Non-UTF-8 names are excluded.
         .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::walkdir::WalkDir;

    /// Creates a unique temp directory holding `.dotfile` and `plainfile`,
    /// returns the result of `dir_entry_is_visible` for each as `(dotfile, plainfile)`.
    fn results(tag: &str) -> (bool, bool) {
        let dir = std::env::temp_dir().join(format!("demo-nhs-llm-{}-{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(".dotfile"), "").unwrap();
        std::fs::write(dir.join("plainfile"), "").unwrap();
        let mut dot = None;
        let mut plain = None;
        for entry in WalkDir::new(&dir).min_depth(1) {
            let entry = entry.unwrap();
            match entry.file_name().to_str().unwrap() {
                ".dotfile" => dot = Some(dir_entry_is_visible(&entry)),
                "plainfile" => plain = Some(dir_entry_is_visible(&entry)),
                _ => {}
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
        (dot.unwrap(), plain.unwrap())
    }

    /// A name starting with a dot gives `false`.
    #[test]
    fn test_dotfile() {
        assert_eq!(results("visible-dot").0, false);
    }

    /// A name not starting with a dot gives `true`.
    #[test]
    fn test_plain_file() {
        assert_eq!(results("visible-plain").1, true);
    }
}
