//! Helpers for extracting text from HTML using the `scraper` crate.
//!
//! The functions compose: find the `<main>` element, then flatten its text
//! nodes into one string.

/// Joins the text nodes of a scraper element into a single string.
pub mod from_scraper_element_ref_text_into_string;

/// Finds the first `<main>` element in a parsed HTML document.
pub mod from_scraper_html_into_main_scraper_element_ref;

/// Extracts the text of the `<main>` element of a parsed HTML document.
pub mod from_scraper_html_into_main_text_string;
