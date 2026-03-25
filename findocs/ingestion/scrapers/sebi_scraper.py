"""Scraper for AMFI mutual-fund scheme-portfolio factsheets.

Downloads factsheet PDFs from the AMFI India research portal and
persists them locally, tracking progress in a SQLite manifest to
prevent duplicate downloads across runs.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Final
from urllib.parse import urljoin

import aiosqlite
import httpx
import structlog
from bs4 import BeautifulSoup, Tag

from findocs.ingestion.models import DownloadedDocument

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_DEFAULT_DATA_DIR: Final[Path] = Path("./data/raw/sebi")
_MANIFEST_DB: Final[str] = "sebi_manifest.db"
_REQUEST_TIMEOUT: Final[float] = 60.0
_DOWNLOAD_TIMEOUT: Final[float] = 120.0
_MAX_RETRIES: Final[int] = 3
_BACKOFF_BASE: Final[float] = 2.0
_RATE_LIMIT_SLEEP: Final[float] = 2.0
_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class SEBIScraper:
    """Asynchronous scraper for AMFI mutual-fund factsheets.

    Downloads scheme-portfolio factsheet PDFs from the AMFI India website,
    stores them locally, and tracks completed downloads in a SQLite
    manifest to support incremental runs.

    Attributes:
        BASE_URL: Root URL of the AMFI website.
        INDEX_URL: Scheme-portfolio factsheets page URL.
    """

    BASE_URL: Final[str] = "https://www.amfiindia.com"
    INDEX_URL: Final[str] = (
        "https://www.amfiindia.com/research-information/other-data/scheme-portfolio"
    )

    def __init__(
        self,
        data_dir: Path = _DEFAULT_DATA_DIR,
        manifest_path: Path | None = None,
    ) -> None:
        """Initialise the scraper.

        Args:
            data_dir: Root directory for downloaded PDFs.
            manifest_path: Path to the SQLite manifest DB.  Defaults to
                ``<data_dir>/sebi_manifest.db``.
        """
        self._data_dir = data_dir
        self._manifest_path = manifest_path or (data_dir / _MANIFEST_DB)
        self._ensure_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scrape_factsheets(
        self,
        year: int | None = None,
        month: int | None = None,
        max_docs: int = 100,
    ) -> list[DownloadedDocument]:
        """Scrape AMFI factsheets for the given period.

        The AMFI portal organises factsheets by fund house and month.
        This method fetches the index page, discovers all available
        factsheet PDF links, and downloads any that have not yet been
        retrieved.

        Args:
            year: Calendar year to target.  Defaults to the current year.
            month: Calendar month to target.  Defaults to the current month.
            max_docs: Maximum number of documents to download in a single
                run.

        Returns:
            A list of ``DownloadedDocument`` instances for every newly
            downloaded document.
        """
        target_year = year or datetime.now().year
        target_month = month or datetime.now().month
        log = logger.bind(year=target_year, month=target_month, max_docs=max_docs)
        log.info("sebi_scraper.start")

        already_downloaded = self.load_manifest()
        downloaded: list[DownloadedDocument] = []

        headers = {
            "User-Agent": _USER_AGENT,
            "Referer": self.INDEX_URL,
        }

        async with httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(_REQUEST_TIMEOUT),
            follow_redirects=True,
        ) as client:
            index_html = await self._fetch_index_page(client)
            if index_html is None:
                log.warning("sebi_scraper.index_fetch_failed")
                return downloaded

            factsheets = self._parse_factsheet_links(
                index_html, target_year, target_month,
            )
            log.info("sebi_scraper.parsed_links", total=len(factsheets))

            for title, pdf_url, pub_date in factsheets:
                if len(downloaded) >= max_docs:
                    log.info("sebi_scraper.max_docs_reached")
                    break

                if pdf_url in already_downloaded:
                    log.debug("sebi_scraper.skip_already_downloaded", url=pdf_url)
                    continue

                period_dir = self._data_dir / f"{target_year}" / f"{target_month:02d}"
                period_dir.mkdir(parents=True, exist_ok=True)
                filename = self._safe_filename(title, pdf_url)
                dest = period_dir / filename

                success = await self.download_pdf(pdf_url, dest, client=client)
                if success:
                    doc = DownloadedDocument(
                        local_path=dest,
                        url=pdf_url,
                        title=title,
                        date=pub_date,
                        doc_type="sebi_factsheet",
                    )
                    downloaded.append(doc)
                    self.update_manifest(pdf_url)
                    log.info(
                        "sebi_scraper.downloaded",
                        title=title,
                        path=str(dest),
                    )

                await asyncio.sleep(_RATE_LIMIT_SLEEP)

        log.info("sebi_scraper.done", new_downloads=len(downloaded))
        return downloaded

    async def download_pdf(
        self,
        url: str,
        destination: Path,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> bool:
        """Download a single PDF with retries and exponential back-off.

        Args:
            url: Fully-qualified URL of the PDF.
            destination: Local filesystem path to write the file.
            client: Optional pre-configured ``httpx.AsyncClient``.

        Returns:
            ``True`` when the file was saved successfully, ``False``
            otherwise.
        """
        log = logger.bind(url=url, destination=str(destination))
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(
                headers={"User-Agent": _USER_AGENT},
                timeout=httpx.Timeout(_DOWNLOAD_TIMEOUT),
                follow_redirects=True,
            )

        try:
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    response = await client.get(url, timeout=_DOWNLOAD_TIMEOUT)
                    response.raise_for_status()

                    content_type = response.headers.get("content-type", "")
                    if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                        log.warning(
                            "sebi_scraper.unexpected_content_type",
                            content_type=content_type,
                            attempt=attempt,
                        )

                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(response.content)
                    log.debug("sebi_scraper.pdf_saved", size=len(response.content))
                    return True

                except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                    wait = _BACKOFF_BASE**attempt
                    log.warning(
                        "sebi_scraper.download_retry",
                        attempt=attempt,
                        error=str(exc),
                        wait=wait,
                    )
                    if attempt < _MAX_RETRIES:
                        await asyncio.sleep(wait)

            log.error("sebi_scraper.download_failed_all_retries")
            return False
        finally:
            if owns_client:
                await client.aclose()

    def load_manifest(self) -> set[str]:
        """Load the set of already-downloaded URLs from the SQLite manifest.

        Returns:
            A set of URL strings that have been previously downloaded.
        """
        conn = sqlite3.connect(str(self._manifest_path))
        try:
            rows = conn.execute("SELECT url FROM downloaded").fetchall()
            return {row[0] for row in rows}
        finally:
            conn.close()

    def update_manifest(self, url: str) -> None:
        """Mark a URL as downloaded in the SQLite manifest.

        Args:
            url: The URL to record as downloaded.
        """
        conn = sqlite3.connect(str(self._manifest_path))
        try:
            conn.execute(
                "INSERT OR IGNORE INTO downloaded (url, downloaded_at) VALUES (?, ?)",
                (url, datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_manifest(self) -> None:
        """Create the manifest DB and table if they do not exist."""
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._manifest_path))
        try:
            conn.execute(
                """\
                CREATE TABLE IF NOT EXISTS downloaded (
                    url           TEXT PRIMARY KEY,
                    downloaded_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    async def _fetch_index_page(
        self,
        client: httpx.AsyncClient,
    ) -> str | None:
        """Fetch the AMFI factsheet index page.

        Args:
            client: An active ``httpx.AsyncClient``.

        Returns:
            The HTML body as a string, or ``None`` on failure.
        """
        try:
            response = await client.get(self.INDEX_URL)
            response.raise_for_status()
            logger.debug("sebi_scraper.index_fetched", status=response.status_code)
            return response.text
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            logger.error("sebi_scraper.index_error", error=str(exc))
            return None

    def _parse_factsheet_links(
        self,
        html: str,
        year: int,
        month: int,
    ) -> list[tuple[str, str, datetime | None]]:
        """Extract factsheet metadata from the index HTML.

        The AMFI scheme-portfolio page lists fund houses and their
        factsheet download links.  This method walks the DOM to find
        those links and returns structured metadata.

        Args:
            html: Raw HTML of the index page.
            year: Target year for filtering.
            month: Target month for filtering.

        Returns:
            A list of tuples ``(title, absolute_pdf_url, date)``.
        """
        soup = BeautifulSoup(html, "html.parser")
        results: list[tuple[str, str, datetime | None]] = []

        # AMFI uses various structures; we look for download anchors
        # across multiple possible containers.
        link_tags = soup.find_all("a", href=True)
        for tag in link_tags:
            href: str = tag["href"]  # type: ignore[index]

            # Keep only links that point to PDFs or known download
            # endpoints.
            is_pdf = href.lower().endswith(".pdf")
            is_download = "download" in href.lower() or "portfolio" in href.lower()
            if not is_pdf and not is_download:
                continue

            title = tag.get_text(strip=True) or "Untitled Factsheet"

            if href.startswith("/"):
                href = urljoin(self.BASE_URL, href)
            elif not href.startswith("http"):
                href = urljoin(self.INDEX_URL, href)

            # Attempt to extract a date from surrounding text or from
            # the link text itself.
            pub_date = self._extract_date_near_tag(tag, year, month)
            results.append((title, href, pub_date))

        return results

    @staticmethod
    def _extract_date_near_tag(
        tag: Tag,
        fallback_year: int,
        fallback_month: int,
    ) -> datetime | None:
        """Attempt to extract a date from the tag or its parent context.

        Args:
            tag: The anchor ``Tag`` to inspect.
            fallback_year: Year to assume when not found explicitly.
            fallback_month: Month to assume when not found explicitly.

        Returns:
            A ``datetime`` if extraction succeeded, otherwise ``None``.
        """
        # Look for dates in the surrounding row or parent element.
        context = tag.parent.get_text(" ", strip=True) if tag.parent else ""
        date_patterns = [
            (r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", "dmy"),
            (r"(\w+)\s+(\d{4})", "my"),
        ]
        for pattern, fmt in date_patterns:
            match = re.search(pattern, context)
            if match is None:
                continue
            groups = match.groups()
            try:
                if fmt == "dmy":
                    return datetime(
                        year=int(groups[2]),
                        month=int(groups[1]),
                        day=int(groups[0]),
                    )
                if fmt == "my":
                    month_dt = datetime.strptime(groups[0][:3], "%b")
                    return datetime(
                        year=int(groups[1]),
                        month=month_dt.month,
                        day=1,
                    )
            except (ValueError, IndexError):
                continue

        # If nothing found, use the fallback year/month.
        try:
            return datetime(year=fallback_year, month=fallback_month, day=1)
        except ValueError:
            return None

    @staticmethod
    def _safe_filename(title: str, url: str) -> str:
        """Derive a filesystem-safe filename from a title and URL.

        Args:
            title: Human-readable document title.
            url: Source URL (used as fallback for naming).

        Returns:
            A sanitised filename string ending in ``.pdf``.
        """
        if url.lower().endswith(".pdf"):
            candidate = url.rsplit("/", 1)[-1]
        else:
            candidate = re.sub(r"[^\w\s-]", "", title)[:80].strip()
            candidate = re.sub(r"[\s]+", "_", candidate)
            candidate = f"{candidate}.pdf" if candidate else "factsheet.pdf"

        candidate = re.sub(r"[^\w.\-]", "_", candidate)
        return candidate
