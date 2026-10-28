//! Maps a URL onto a relative file system path, as a website mirror would.

/// From URL host segments into PathBuf.
///
/// Builds a relative path whose first component is the host name, followed
/// by each path segment. Scheme, port, query, and fragment are ignored.
/// Empty segments (such as the one produced by a trailing slash) are dropped,
/// so `https://example.com/alfa/bravo/` and `https://example.com/alfa/bravo`
/// give the same result, `example.com/alfa/bravo`.
///
/// If the URL has no host (e.g. `mailto:`), the path starts at the first
/// path segment, or is empty.
///
/// # Arguments
///
/// * `url` - the parsed URL to convert.
pub async fn from_url_host_segments_into_path_buf(url: &url::Url) -> std::path::PathBuf {
    // `host_str()` is an `Option<&str>`; `into_iter()` makes it a 0-or-1 item iterator.
    url.host_str()
        .into_iter()
        // Append the path segments. `path_segments()` is `None` for
        // cannot-be-a-base URLs, which `flatten()` turns into no items.
        .chain(url.path_segments().into_iter().flatten())
        // Drop empty segments, e.g. from a trailing or doubled slash.
        .filter(|segment| !segment.is_empty())
        // `PathBuf` implements `FromIterator`, joining items as components.
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use url::Url;

    /// A bare host (no path) maps to just the host name.
    #[tokio::test]
    async fn test_with_host_name() {
        let url = Url::parse("https://example.com").unwrap();
        let actual = from_url_host_segments_into_path_buf(&url).await;
        let expect = PathBuf::from("example.com");
        assert_eq!(actual, expect);
    }

    /// A trailing slash does not add an empty component.
    #[tokio::test]
    async fn test_with_trailing_slash() {
        let url = Url::parse("https://example.com/alfa/bravo/").unwrap();
        let actual = from_url_host_segments_into_path_buf(&url).await;
        let expect = PathBuf::from("example.com/alfa/bravo");
        assert_eq!(actual, expect);
    }

    /// Without a trailing slash, the result is the same as with one.
    #[tokio::test]
    async fn test_without_trailing_slash() {
        let url = Url::parse("https://example.com/alfa/bravo").unwrap();
        let actual = from_url_host_segments_into_path_buf(&url).await;
        let expect = PathBuf::from("example.com/alfa/bravo");
        assert_eq!(actual, expect);
    }
}