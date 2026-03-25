"""Scraper for RBI (Reserve Bank of India) circulars.

Scrapes RBI circulars from the official circular index page, downloads
PDFs into a local directory tree organised by year, and tracks progress
in a lightweight SQLite manifest so that re-runs never re-download.
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

_DEFAULT_DATA_DIR: Final[Path] = Path("./data/raw/rbi")
_MANIFEST_DB: Final[str] = "rbi_manifest.db"
_REQUEST_TIMEOUT: Final[float] = 60.0
_DOWNLOAD_TIMEOUT: Final[float] = 120.0
_MAX_RETRIES: Final[int] = 3
_BACKOFF_BASE: Final[float] = 2.0
_RATE_LIMIT_SLEEP: Final[float] = 2.0
_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class RBIScraper:
    """Asynchronous scraper for Reserve Bank of India circulars.

    Downloads PDF circulars from the RBI website, persists them locally,
    and uses a SQLite manifest to avoid duplicate work across runs.

    Attributes:
        BASE_URL: Root URL of the RBI website.
        INDEX_URL: Circulars index page URL.
    """

    BASE_URL: Final[str] = "https://www.rbi.org.in"
    INDEX_URL: Final[str] = (
        "https://www.rbi.org.in/Scripts/BS_CircularIndexDisplay.aspx"
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
                ``<data_dir>/rbi_manifest.db``.
        """
        self._data_dir = data_dir
        self._manifest_path = manifest_path or (data_dir / _MANIFEST_DB)
        self._ensure_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scrape_circulars(
        self,
        year: int | None = None,
        max_docs: int = 100,
    ) -> list[DownloadedDocument]:
        """Scrape circulars for the given year.

        Args:
            year: Calendar year to scrape.  Defaults to the current year.
            max_docs: Maximum number of documents to download in one run.

        Returns:
            A list of ``DownloadedDocument`` instances for every newly
            downloaded document.
        """
        target_year = year or datetime.now().year
        log = logger.bind(year=target_year, max_docs=max_docs)
        log.info("rbi_scraper.start")

        already_downloaded = self.load_manifest()
        downloaded: list[DownloadedDocument] = []

        headers = {"User-Agent": _USER_AGENT}

        async with httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(_REQUEST_TIMEOUT),
            follow_redirects=True,
        ) as client:
            index_html = await self._fetch_index_page(client, target_year)
            if index_html is None:
                log.warning("rbi_scraper.index_fetch_failed")
                return downloaded

            circulars = self._parse_circular_links(index_html, target_year)
            log.info("rbi_scraper.parsed_links", total=len(circulars))

            for title, pdf_url, pub_date in circulars:
                if len(downloaded) >= max_docs:
                    log.info("rbi_scraper.max_docs_reached")
                    break

                if pdf_url in already_downloaded:
                    log.debug("rbi_scraper.skip_already_downloaded", url=pdf_url)
                    continue

                year_dir = self._data_dir / str(target_year)
                year_dir.mkdir(parents=True, exist_ok=True)
                filename = self._safe_filename(title, pdf_url)
                dest = year_dir / filename

                success = await self.download_pdf(pdf_url, dest, client=client)
                if success:
                    doc = DownloadedDocument(
                        local_path=dest,
                        url=pdf_url,
                        title=title,
                        date=pub_date,
                        doc_type="rbi_circular",
                    )
                    downloaded.append(doc)
                    self.update_manifest(pdf_url)
                    log.info(
                        "rbi_scraper.downloaded",
                        title=title,
                        path=str(dest),
                    )

                # Respectful rate limiting
                await asyncio.sleep(_RATE_LIMIT_SLEEP)

        log.info("rbi_scraper.done", new_downloads=len(downloaded))
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
                            "rbi_scraper.unexpected_content_type",
                            content_type=content_type,
                            attempt=attempt,
                        )

                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(response.content)
                    log.debug("rbi_scraper.pdf_saved", size=len(response.content))
                    return True

                except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                    wait = _BACKOFF_BASE**attempt
                    log.warning(
                        "rbi_scraper.download_retry",
                        attempt=attempt,
                        error=str(exc),
                        wait=wait,
                    )
                    if attempt < _MAX_RETRIES:
                        await asyncio.sleep(wait)

            log.error("rbi_scraper.download_failed_all_retries")
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
        year: int,
    ) -> str | None:
        """Fetch the RBI circulars index page for a given year.

        Args:
            client: An active ``httpx.AsyncClient``.
            year: The target calendar year.

        Returns:
            The HTML body as a string, or ``None`` on failure.
        """
        log = logger.bind(year=year)
        try:
            response = await client.get(
                self.INDEX_URL,
                params={"year": str(year)},
            )
            response.raise_for_status()
            log.debug("rbi_scraper.index_fetched", status=response.status_code)
            return response.text
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            log.error("rbi_scraper.index_error", error=str(exc))
            return None

    def _parse_circular_links(
        self,
        html: str,
        year: int,
    ) -> list[tuple[str, str, datetime | None]]:
        """Extract circular metadata from the index HTML.

        Parses the HTML table on the RBI circulars index page and
        returns a list of ``(title, pdf_url, date)`` tuples.

        Args:
            html: Raw HTML of the index page.
            year: The target year (used for date fallback).

        Returns:
            A list of tuples ``(title, absolute_pdf_url, publication_date)``.
        """
        soup = BeautifulSoup(html, "html.parser")
        results: list[tuple[str, str, datetime | None]] = []

        # The RBI page typically renders circulars inside a <table> with
        # rows containing a date column, a title/link column, and
        # sometimes a department column.
        table = soup.find("table", class_="tablebg") or soup.find("table")
        if table is None:
            logger.warning("rbi_scraper.no_table_found")
            return results

        rows: list[Tag] = table.find_all("tr")  # type: ignore[union-attr]
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue

            # Attempt to find a link to a PDF or circular detail page.
            link_tag = row.find("a", href=True)
            if link_tag is None:
                continue

            href: str = link_tag["href"]  # type: ignore[index]
            title: str = link_tag.get_text(strip=True) or "Untitled Circular"

            # Resolve relative URLs.
            if href.startswith("/"):
                href = urljoin(self.BASE_URL, href)
            elif not href.startswith("http"):
                href = urljoin(self.INDEX_URL, href)

            # Only keep PDF links or circular-detail pages that we can
            # later follow to find the actual PDF.  We normalise detail
            # page links to the PDF notification URL pattern used by RBI.
            if not href.lower().endswith(".pdf"):
                # RBI notification detail pages use NotificationUser.aspx
                # with an Id param — keep those as we can attempt to
                # derive the PDF link from them.
                if "NotificationUser" not in href and "Notification" not in href:
                    continue

            pub_date = self._parse_date_from_cells(cells, year)
            results.append((title, href, pub_date))

        return results

    @staticmethod
    def _parse_date_from_cells(
        cells: list[Tag],
        fallback_year: int,
    ) -> datetime | None:
        """Try to extract a publication date from table cells.

        Args:
            cells: List of ``<td>`` elements from a table row.
            fallback_year: Year to assume if only month/day are found.

        Returns:
            A ``datetime`` if parsing succeeded, otherwise ``None``.
        """
        date_patterns = [
            r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})",  # DD-MM-YYYY
            r"(\w+)\s+(\d{1,2}),?\s+(\d{4})",  # Month DD, YYYY
        ]
        for cell in cells:
            text = cell.get_text(strip=True)
            for pattern in date_patterns:
                match = re.search(pattern, text)
                if match is None:
                    continue
                groups = match.groups()
                try:
                    if groups[0].isdigit():
                        return datetime(
                            year=int(groups[2]),
                            month=int(groups[1]),
                            day=int(groups[0]),
                        )
                    else:
                        return datetime.strptime(
                            f"{groups[0]} {groups[1]} {groups[2]}",
                            "%B %d %Y",
                        )
                except (ValueError, IndexError):
                    continue
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
        # Prefer the last path segment of the URL when it is a PDF.
        if url.lower().endswith(".pdf"):
            candidate = url.rsplit("/", 1)[-1]
        else:
            candidate = re.sub(r"[^\w\s-]", "", title)[:80].strip()
            candidate = re.sub(r"[\s]+", "_", candidate)
            candidate = f"{candidate}.pdf" if candidate else "circular.pdf"

        # Final safety pass — remove anything not alphanumeric, hyphen,
        # underscore, or dot.
        candidate = re.sub(r"[^\w.\-]", "_", candidate)
        return candidate
