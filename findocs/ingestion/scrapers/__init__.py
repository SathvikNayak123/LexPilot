"""Web scrapers for Indian financial document sources."""

from findocs.ingestion.scrapers.nse_scraper import NSEScraper
from findocs.ingestion.scrapers.rbi_scraper import RBIScraper
from findocs.ingestion.scrapers.sebi_scraper import SEBIScraper

__all__ = ["NSEScraper", "RBIScraper", "SEBIScraper"]
