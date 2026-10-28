//! Helpers for filtering `walkdir` directory entries.

/// Predicate: is a directory entry hidden (name starts with `.`)?
pub mod dir_entry_is_hidden;

/// Predicate: is a directory entry visible (name does not start with `.`)?
pub mod dir_entry_is_visible;
