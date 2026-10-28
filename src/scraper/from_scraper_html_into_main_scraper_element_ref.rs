//! Locates the `<main>` element of a parsed HTML document.

/// From scraper HTML into main element ref.
///
/// Returns the first `<main>` element in document order. NHS pages put their
/// primary content in `<main>`, which excludes navigation, headers, and
/// footers.
///
/// The returned reference borrows from `html`, so it cannot outlive it.
///
/// # Arguments
///
/// * `html` - the parsed HTML document to search.
///
/// # Returns
///
/// `Some(element)` for the first `<main>` element, or `None` if the document
/// has none.
pub async fn from_scraper_html_into_main_scraper_element_ref(
    html: &scraper::Html,
) -> Option<scraper::ElementRef<'_>> {
    // The selector is parsed once, on first use, and shared afterwards;
    // parsing is comparatively expensive and this runs once per page.
    // The `unwrap` is safe because `"main"` is a valid CSS selector.
    static SCRAPER_SELECTOR_MAIN: std::sync::LazyLock<scraper::Selector> =
        std::sync::LazyLock::new(|| scraper::Selector::parse("main").unwrap());

    // `select` yields all matches in document order; we want only the first.
    html.select(&SCRAPER_SELECTOR_MAIN).next()
}

#[cfg(test)]
mod tests_from_scraper_html_into_main_scraper_element_ref {
    use super::*;

    /// A document with a `<main>` element yields that element.
    #[tokio::test]
    async fn test() {
        let html_str = "<html><body><main>hello world</main></body></html>";
        let html = scraper::Html::parse_document(html_str);
        let actual = from_scraper_html_into_main_scraper_element_ref(&html);
        // The element's tag name should be "main".
        assert_eq!(actual.await.unwrap().value().name(), "main");
    }
}
