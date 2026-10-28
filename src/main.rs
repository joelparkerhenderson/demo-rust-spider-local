use spider::{tokio, website::Website};

pub fn website(url_str: impl AsRef<str>) -> Website {
    Website::new(url_str.as_ref())
        .with_limit(50) // concurrent requests
        .with_depth(10) // how deep to follow links
        .with_delay(500) // pause between requests (ms)
        .with_respect_robots_txt(true)
        .with_subdomains(true)
        .with_user_agent(Some("MyBot/1.0"))
        .with_stealth(true)
        .build()
        .expect("website")
}

#[tokio::main]
async fn main() {
    let mut website = website("https://example.com");
    let mut rx = website.subscribe(16);

    tokio::spawn(async move {
        while let Ok(page) = rx.recv().await {
            println!("{}  {}", page.status_code, page.get_url());
        }
    });

    website.crawl().await;
    website.unsubscribe();
}
