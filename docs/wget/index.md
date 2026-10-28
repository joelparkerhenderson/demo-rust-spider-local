# wget

To build the LLM, we first download web pages to our local system so we can process the pages. We can download in a variety of ways, such as by using the command `wget` or our tool `web-mirror`.

This page shows how to do the download by using the command `wget`:

```sh
wget "https://www.nhs.uk/conditions/" \
--recursive \
--continue \
--no-clobber \
--directory-prefix="." \
--domains="www.nhs.uk" \
--reject-regex "%5C|\\" \
--force-directories \
--level=2 \
--no-cookies \
--no-parent \
--wait=1 \
--random-wait
```

We use wget with these settings:

- `-r, --recursive` : turn on recursive retrieving.

- `-c, --continue` : resume getting a partially-downloaded file, site, etc.

- `-nc, --no-clobber` : TODO

- `-P, --directory-prefix` : save all the files and directories to the specified directory.

- `--domains=domains-list` : set domains to be followed.

- `--force-directories` : always create a hierarchy of directories.

- `--reject-regex regex` : reject a regular expression match.

- `-l, --level=depth` : set the maximum number of subdirectories.

- `--no-cookies` : TODO

- `-np, --no-parent` : do not ascend to the parent directory.

- `--wait=SECONDS` : wait wait SECONDS between retrievals

- `--random-wait` : wait from 0.5*WAIT...1.5*WAIT secs between retrievals
