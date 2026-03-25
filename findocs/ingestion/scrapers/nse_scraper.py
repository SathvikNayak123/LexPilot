"""Scraper for NSE (National Stock Exchange) annual reports.

Downloads corporate-filing annual-report PDFs from the NSE India
website, stores them locally by year, and tracks progress in a
SQLite manifest for incremental operation.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import urljoin

import aiosqlite
import httpx
import structlog
from bs4 import BeautifulSoup, Tag

from findocs.ingestion.models import DownloadedDocument

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_DEFAULT_DATA_DIR: Final[Path] = Path("./data/raw/nse")
_MANIFEST_DB: Final[str] = "nse_manifest.db"
_REQUEST_TIMEOUT: Final[float] = 60.0
_DOWNLOAD_TIMEOUT: Final[float] = 180.0
_MAX_RETRIES: Final[int] = 3
_BACKOFF_BASE: Final[float] = 2.0
_RATE_LIMIT_SLEEP: Final[float] = 2.0
_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# NSE serves corporate filings data via a JSON API behind the rendered
# page.  The front-end at the INDEX_URL loads data from the API_URL.
_NSE_API_URL: Final[str] = (
    "https://www.nseindia.com/api/corporate-annual-reports"
)


class NSEScraper:
    """Asynchronous scraper for NSE India annual reports.

    Downloads annual-report PDFs from the NSE corporate-filings section,
    saves them locally, and uses a SQLite manifest to skip previously
    downloaded files.

    Attributes:
        BASE_URL: Root URL of the NSE website.
        INDEX_URL: Annual-reports listing page URL.
    """

    BASE_URL: Final[str] = "https://www.nseindia.com"
    INDEX_URL: Final[str] = (
        "https://www.nseindia.com/companies-listing/"
        "corporate-filings-annual-reports"
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
                ``<data_dir>/nse_manifest.db``.
        """
        self._data_dir = data_dir
        self._manifest_path = manifest_path or (data_dir / _MANIFEST_DB)
        self._ensure_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scrape_annual_reports(
        self,
        year: int | None = None,
        max_docs: int = 100,
    ) -> list[DownloadedDocument]:
        """Scrape NSE annual reports for the given year.

        NSE renders the annual-reports page client-side and fetches the
        actual data from an internal JSON API.  This scraper first
        visits the landing page to acquire cookies / session tokens,
        then queries the API to retrieve listing data and PDF links.

        If the API call fails (e.g. the endpoint has changed), the
        scraper falls back to parsing the HTML page directly.

        Args:
            year: Financial year to target.  Defaults to the current
                year.
            max_docs: Maximum number of documents to download in a
                single run.

        Returns:
            A list of ``DownloadedDocument`` instances for every newly
            downloaded document.
        """
        target_year = year or datetime.now().year
        log = logger.bind(year=target_year, max_docs=max_docs)
        log.info("nse_scraper.start")

        already_downloaded = self.load_manifest()
        downloaded: list[DownloadedDocument] = []

        headers = {
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
        }

        async with httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(_REQUEST_TIMEOUT),
            follow_redirects=True,
        ) as client:
            # Step 1: Hit the landing page to obtain session cookies.
            await self._warm_session(client)

            # Step 2: Try the JSON API first; fall back to HTML parsing.
            reports = await self._fetch_reports_via_api(client, target_year)
            if not reports:
                log.info("nse_scraper.api_fallback_to_html")
                index_html = await self._fetch_index_page(client)
                if index_html is not None:
                    reports = self._parse_report_links_html(index_html, target_year)

            log.info("nse_scraper.parsed_links", total=len(reports))

            for title, pdf_url, pub_date in reports:
                if len(downloaded) >= max_docs:
                    log.info("nse_scraper.max_docs_reached")
                    break

                if pdf_url in already_downloaded:
                    log.debug("nse_scraper.skip_already_downloaded", url=pdf_url)
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
                        doc_type="nse_annual_report",
                    )
                    downloaded.append(doc)
                    self.update_manifest(pdf_url)
                    log.info(
                        "nse_scraper.downloaded",
                        title=title,
                        path=str(dest),
                    )

                await asyncio.sleep(_RATE_LIMIT_SLEEP)

        log.info("nse_scraper.done", new_downloads=len(downloaded))
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
                            "nse_scraper.unexpected_content_type",
                            content_type=content_type,
                            attempt=attempt,
                        )

                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(response.content)
                    log.debug("nse_scraper.pdf_saved", size=len(response.content))
                    return True

                except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                    wait = _BACKOFF_BASE**attempt
                    log.warning(
                        "nse_scraper.download_retry",
                        attempt=attempt,
                        error=str(exc),
                        wait=wait,
                    )
                    if attempt < _MAX_RETRIES:
                        await asyncio.sleep(wait)

            log.error("nse_scraper.download_failed_all_retries")
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

    async def _warm_session(self, client: httpx.AsyncClient) -> None:
        """Visit the NSE landing page to acquire session cookies.

        NSE requires valid session cookies (set via the initial page
        load) before it will honour API requests.  This method performs
        that initial request and lets ``httpx`` store the cookies in
        the client's cookie jar.

        Args:
            client: An active ``httpx.AsyncClient``.
        """
        try:
            response = await client.get(self.INDEX_URL)
            response.raise_for_status()
            logger.debug(
                "nse_scraper.session_warmed",
                status=response.status_code,
                cookies=len(response.cookies),
            )
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            logger.warning("nse_scraper.session_warm_failed", error=str(exc))

    async def _fetch_reports_via_api(
        self,
        client: httpx.AsyncClient,
        year: int,
    ) -> list[tuple[str, str, datetime | None]]:
        """Query the NSE JSON API for annual-report filings.

        Args:
            client: An active ``httpx.AsyncClient`` with session cookies.
            year: The target financial year.

        Returns:
            A list of ``(title, pdf_url, date)`` tuples.  Returns an
            empty list on failure so the caller can fall back.
        """
        results: list[tuple[str, str, datetime | None]] = []

        # The NSE API typically expects query parameters for the date
        # range and an index/category filter.
        params: dict[str, str] = {
            "index": "equities",
            "from_date": f"01-01-{year}",
            "to_date": f"31-12-{year}",
        }

        api_headers = {
            "Referer": self.INDEX_URL,
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json, text/javascript, */*; q=0.01",
        }

        try:
            response = await client.get(
                _NSE_API_URL,
                params=params,
                headers=api_headers,
            )
            response.raise_for_status()
            data: Any = response.json()
        except (httpx.HTTPStatusError, httpx.TransportError, ValueError) as exc:
            logger.warning("nse_scraper.api_fetch_failed", error=str(exc))
            return results

        # The API response is typically a list of filing objects.
        records: list[dict[str, Any]] = []
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            # Some NSE APIs wrap the list under a key like "data".
            for key in ("data", "results", "records"):
                if key in data and isinstance(data[key], list):
                    records = data[key]
                    break

        for record in records:
            title = (
                record.get("companyName")
                or record.get("company")
                or record.get("symbol", "Unknown Company")
            )
            pdf_url = record.get("pdfLink") or record.get("fileLink") or ""

            if not pdf_url:
                # Try to build the URL from known NSE patterns.
                file_name = record.get("fileName") or record.get("file")
                if file_name:
                    pdf_url = urljoin(
                        self.BASE_URL,
                        f"/archives/annual/data/{file_name}",
                    )
                else:
                    continue

            if pdf_url.startswith("/"):
                pdf_url = urljoin(self.BASE_URL, pdf_url)

            pub_date = self._parse_date_string(
                record.get("date") or record.get("period") or ""
            )
            results.append((str(title), pdf_url, pub_date))

        logger.debug("nse_scraper.api_parsed", count=len(results))
        return results

    async def _fetch_index_page(
        self,
        client: httpx.AsyncClient,
    ) -> str | None:
        """Fetch the NSE annual-reports HTML page.

        Args:
            client: An active ``httpx.AsyncClient``.

        Returns:
            The HTML body as a string, or ``None`` on failure.
        """
        try:
            response = await client.get(self.INDEX_URL)
            response.raise_for_status()
            logger.debug("nse_scraper.index_fetched", status=response.status_code)
            return response.text
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            logger.error("nse_scraper.index_error", error=str(exc))
            return None

    def _parse_report_links_html(
        self,
        html: str,
        year: int,
    ) -> list[tuple[str, str, datetime | None]]:
        """Extract annual-report links from the HTML page (fallback).

        Used when the JSON API is unavailable.  Walks the DOM looking
        for PDF download links in the corporate-filings table.

        Args:
            html: Raw HTML of the index page.
            year: The target financial year.

        Returns:
            A list of ``(title, absolute_pdf_url, date)`` tuples.
        """
        soup = BeautifulSoup(html, "html.parser")
        results: list[tuple[str, str, datetime | None]] = []

        # NSE renders a DataTable; look for rows in any table body.
        table = soup.find("table") or soup.find("div", class_="table-responsive")
        if table is None:
            # Broaden search to any anchor pointing at a PDF.
            link_tags = soup.find_all("a", href=True)
        else:
            link_tags = table.find_all("a", href=True)

        for tag in link_tags:
            href: str = tag["href"]  # type: ignore[index]

            if not href.lower().endswith(".pdf"):
                continue

            title = tag.get_text(strip=True) or "Untitled Annual Report"

            if href.startswith("/"):
                href = urljoin(self.BASE_URL, href)
            elif not href.startswith("http"):
                href = urljoin(self.INDEX_URL, href)

            # Try to pull a date from the table row.
            pub_date: datetime | None = None
            row = tag.find_parent("tr")
            if row is not None:
                cells = row.find_all("td")
                pub_date = self._parse_date_from_cells(cells, year)

            results.append((title, href, pub_date))

        return results

    @staticmethod
    def _parse_date_from_cells(
        cells: list[Tag],
        fallback_year: int,
    ) -> datetime | None:
        """Try to extract a date from table cells.

        Args:
            cells: List of ``<td>`` elements from a table row.
            fallback_year: Year assumed when only partial info is found.

        Returns:
            A ``datetime`` if parsing succeeded, otherwise ``None``.
        """
        date_patterns = [
            r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})",
            r"(\d{1,2})-(\w{3})-(\d{4})",
            r"(\w+)\s+(\d{1,2}),?\s+(\d{4})",
        ]
        for cell in cells:
            text = cell.get_text(strip=True)
            for pattern in date_patterns:
                match = re.search(pattern, text)
                if match is None:
                    continue
                groups = match.groups()
                try:
                    if groups[0].isdigit() and groups[1].isdigit():
                        return datetime(
                            year=int(groups[2]),
                            month=int(groups[1]),
                            day=int(groups[0]),
                        )
                    if groups[0].isdigit() and groups[1].isalpha():
                        return datetime.strptime(
                            f"{groups[0]}-{groups[1]}-{groups[2]}",
                            "%d-%b-%Y",
                        )
                    if groups[0].isalpha():
                        return datetime.strptime(
                            f"{groups[0]} {groups[1]} {groups[2]}",
                            "%B %d %Y",
                        )
                except (ValueError, IndexError):
                    continue
        return None

    @staticmethod
    def _parse_date_string(raw: str) -> datetime | None:
        """Parse a date string from the API response.

        Tries several common formats used by NSE.

        Args:
            raw: Raw date string from the JSON record.

        Returns:
            A ``datetime`` on success, ``None`` otherwise.
        """
        if not raw:
            return None
        formats = [
            "%d-%m-%Y",
            "%d-%b-%Y",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%b %d, %Y",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(raw.strip(), fmt)
            except ValueError:
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
        if url.lower().endswith(".pdf"):
            candidate = url.rsplit("/", 1)[-1]
        else:
            candidate = re.sub(r"[^\w\s-]", "", title)[:80].strip()
            candidate = re.sub(r"[\s]+", "_", candidate)
            candidate = f"{candidate}.pdf" if candidate else "annual_report.pdf"

        candidate = re.sub(r"[^\w.\-]", "_", candidate)
        return candidate
