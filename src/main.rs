use ::std::path::Path;
use ::std::sync::LazyLock;
use ::regex::Regex;

#[allow(unused_imports)]
use log::{Level, debug, error, info, log_enabled, warn};

mod optimizer;
mod optimizer_functions;
mod tensor;
mod tensor_functions;
mod tokenizer;
mod tokenizer_functions;
mod tokenizer_stats;
mod scraper;
mod walkdir;

#[tokio::main]
async fn main() {
    env_logger::init();
    info!("main");
    do_directory("/Users/jph/mirrors/www.nhs.uk").await;
}

static REGEX_DELETE_REVIEWED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(Media|Page) last reviewed: .*\n\s+(Media|Page|Next) review due: .*?\n").unwrap()
});

static REGEX_DELETE_BLANK_LINES: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\n\s*\n").unwrap()
});

pub async fn do_directory(dir_str: &str) {
    use ::walkdir::WalkDir;
    use ::futures::StreamExt;

    futures::stream::iter(WalkDir::new(dir_str))
        .filter_map(|e| async { e.ok() })
        .filter(|e| std::future::ready(e.file_type().is_file()))
        .map(|e| e.path().to_path_buf())
        .for_each(|path| async move {
            // You have to handle or unwrap the error here because for_each absorbs it
            if let Err(e) = do_file(&path).await {
                eprintln!("Error processing file: {}", e);
            }
        })
        .await;
}

pub async fn do_file(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use crate::scraper::from_scraper_html_into_main_text_string::from_scraper_html_into_main_text_string;
    let html = ::scraper::Html::parse_document(&std::fs::read_to_string(path)?);
    let text = from_scraper_html_into_main_text_string(&html).await.ok_or("Failure extracting main text")?;
    let text = REGEX_DELETE_BLANK_LINES.replace_all(&text, "\n").into_owned();
    let text = REGEX_DELETE_REVIEWED.replace_all(&text, "").into_owned();
    log::info!("path: {}, text: {}", path.display(), text);
    Ok(())
}

#[cfg(test)]
mod tests {
    // use super::*;
}
