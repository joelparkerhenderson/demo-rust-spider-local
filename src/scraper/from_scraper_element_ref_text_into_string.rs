//! Flattens the text content of a scraper element into one string.

/// From scraper element ref into text string.
///
/// Collects every descendant text node of the element, in document order,
/// and joins them with a single space.
///
/// Because text nodes are joined with a space, markup boundaries introduce
/// spaces (e.g. `a<b>c</b>` becomes `"a c"`), and whitespace-only text nodes
/// between tags are preserved as-is.
///
/// This function is `async` for consistency with the other extraction
/// helpers, but it does no awaiting.
///
/// # Arguments
///
/// * `element` - the element whose text should be collected.
///
/// # Returns
///
/// The joined text; empty if the element has no text nodes.
pub async fn from_scraper_element_ref_text_into_string(element: scraper::ElementRef<'_>) -> String {
    // `text()` yields each descendant text node as a `&str`.
    element.text().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests_from_scraper_element_ref_text_into_string {
    use super::*;

    /// A single text node comes back unchanged.
    #[tokio::test]
    async fn test() {
        // Parse a minimal document containing one `<main>` element.
        let html_str = "<html><body><main>hello world</main></body></html>";
        let html = scraper::Html::parse_document(html_str);
        let selector = scraper::Selector::parse("main").unwrap();
        // Select the `<main>` element; unwrap is fine because the fixture has one.
        let element = html.select(&selector).next().unwrap();
        let actual = from_scraper_element_ref_text_into_string(element);
        let expect = "hello world";
        assert_eq!(actual.await, expect);
    }
}
