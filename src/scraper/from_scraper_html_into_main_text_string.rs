//! Extracts the text of the `<main>` element of a parsed HTML document.

use crate::scraper::from_scraper_html_into_main_scraper_element_ref::from_scraper_html_into_main_scraper_element_ref;
use crate::scraper::from_scraper_element_ref_text_into_string::from_scraper_element_ref_text_into_string;

/// From scraper HTML into main text string.
///
/// Finds the first `<main>` element, then joins its text nodes into one
/// string. This composes
/// [`from_scraper_html_into_main_scraper_element_ref`] and
/// [`from_scraper_element_ref_text_into_string`].
///
/// # Arguments
///
/// * `html` - the parsed HTML document.
///
/// # Returns
///
/// `Some(text)` with the main text, or `None` if the document has no
/// `<main>` element.
pub async fn from_scraper_html_into_main_text_string(
    html: &::scraper::Html,
) -> Option<String> {
    // Find `<main>`; the `?` returns `None` early when there isn't one.
    let main_element_ref = from_scraper_html_into_main_scraper_element_ref(html).await?;

    // Flatten the element's text nodes into a single string.
    let main_text_string = from_scraper_element_ref_text_into_string(main_element_ref).await;
    Some(main_text_string)
}

#[cfg(test)]
mod tests_from_scraper_html_into_main_text_string {
    use super::*;

    /// The text inside `<main>` is returned.
    #[tokio::test]
    async fn test() {
        let html_str = "<html><body><main>hello world</main></body></html>";
        let html = scraper::Html::parse_document(html_str);
        let actual = from_scraper_html_into_main_text_string(&html);
        let expect = "hello world";
        assert_eq!(actual.await, Some(expect.to_string()));
    }

    /// A document without `<main>` yields `None`.
    #[tokio::test]
    async fn test_without_main() {
        let html = scraper::Html::parse_document("<html><body>hello</body></html>");
        assert_eq!(from_scraper_html_into_main_text_string(&html).await, None);
    }
}
