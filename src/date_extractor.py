# news_date_extractor.py
import re, asyncio, random, aiohttp, concurrent.futures
from pathlib import Path
from typing import Iterable, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

import pandas as pd
from htmldate import find_date   # pulled in by `pip install trafilatura[fast]`
# (Trafilatura itself isn't imported directly, but brings the Rust speed-ups)

class NewsDateExtractor:
    # ------------------------------------------------------------------ #
    #                0.  ONE-TIME COMPILED REGEXES                       #
    # ------------------------------------------------------------------ #
    _URL_DATE_RE = re.compile(r'/(?P<Y>\d{4})[/-]?(?P<M>\d{1,2})[/-]?(?P<D>\d{1,2})(?!\d)', re.I)
    
    # Comprehensive meta tag patterns for publication dates
    # Ordered by reliability/specificity
    _META_PATTERNS = [
        # Most specific publication date patterns first
        re.compile(r'<meta[^>]*(?:property|name)=["\']article:published_time["\'][^>]*content=["\']([^"\']+)["\']', re.I),
        re.compile(r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*(?:property|name)=["\']article:published_time["\']', re.I),
        re.compile(r'<meta[^>]*(?:property|name)=["\']datePublished["\'][^>]*content=["\']([^"\']+)["\']', re.I),
        re.compile(r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*(?:property|name)=["\']datePublished["\']', re.I),
        
        # JSON-LD structured data (high priority)
        re.compile(r'"datePublished":\s*"([^"]+)"', re.I),
        re.compile(r'"publishedDate":\s*"([^"]+)"', re.I),
        
        # Time tags with publication semantics
        re.compile(r'<time[^>]*(?:class=["\'][^"\']*publish[^"\']*["\']|itemprop=["\']datePublished["\'])[^>]*datetime=["\']([^"\']+)["\']', re.I),
        re.compile(r'<time[^>]*datetime=["\']([^"\']+)["\'][^>]*(?:class=["\'][^"\']*publish[^"\']*["\']|itemprop=["\']datePublished["\'])', re.I),
        
        # Other publication date patterns
        re.compile(r'<meta[^>]*(?:property|name)=["\'](?:publishedDate|publication_date|pubdate)["\'][^>]*content=["\']([^"\']+)["\']', re.I),
        re.compile(r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*(?:property|name)=["\'](?:publishedDate|publication_date|pubdate)["\']', re.I),
        
        # Generic time tags (lower priority)
        re.compile(r'<time[^>]*datetime=["\']([^"\']+)["\']', re.I),
        
        # Last resort: generic date patterns (lowest priority)
        re.compile(r'<meta[^>]*(?:property|name)=["\']date["\'][^>]*content=["\']([^"\']+)["\']', re.I),
        re.compile(r'(?:published|publication):\s*["\']?(\d{4}-\d{2}-\d{2})', re.I),
    ]

    # A short but varied list; swap in your own or use `fake_useragent`.
    _UA_POOL = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    # ------------------------------------------------------------------ #
    #                   1.  CONSTRUCTOR & HELPERS                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        concurrency: int = 50,
        max_workers: int = 8,
        byte_snippet: int = 25_000,  # Increased for better meta detection
        timeout: int = 10,
        user_agents: Optional[List[str]] = None,
        max_retries: int = 2,
        skip_ssl_verify: bool = True,
        logger: Optional[logging.Logger] = None,
        date_range_years: int = 30,  # Only accept dates within this range from today
    ):
        """
        Parameters
        ----------
        concurrency : simultaneous TCP connections (aiohttp connector limit)
        max_workers : threads for CPU-bound HTML parsing (htmldate)
        byte_snippet: bytes to request when only a <meta> scan is needed
        timeout     : seconds for requests (reduced default for faster failure)
        user_agents : list of UA strings. If None, default pool above.
        max_retries : number of retries for failed requests
        skip_ssl_verify : whether to skip SSL verification (useful for some sites)
        logger      : logger instance for debugging
        date_range_years : only accept dates within this many years of today
        """
        self.concurrency = concurrency
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.byte_snippet = byte_snippet
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.ua_pool = user_agents or self._UA_POOL
        self.max_retries = max_retries
        self.skip_ssl_verify = skip_ssl_verify
        self.logger = logger or logging.getLogger(__name__)
        
        # Date validation range
        today = datetime.now()
        self.min_date = today - timedelta(days=365 * date_range_years)
        self.max_date = today + timedelta(days=30)  # Allow slight future dates for timezone issues

        # Build the range header once (except UA which varies)
        self._base_range_header = {"Range": f"bytes=0-{self.byte_snippet}"}

    # ---------- Date validation and parsing helpers --------------------
    def _validate_date_string(self, date_str: str) -> Optional[str]:
        """
        Validate and normalize a date string.
        Returns ISO format date string if valid, None otherwise.
        """
        if not date_str:
            return None
            
        try:
            # Handle common date formats
            for fmt in [
                "%Y-%m-%d",           # 2024-01-05
                "%Y/%m/%d",           # 2024/01/05
                "%Y-%m-%dT%H:%M:%S",  # 2024-01-05T10:30:00
                "%Y-%m-%dT%H:%M:%SZ", # 2024-01-05T10:30:00Z
                "%Y-%m-%d %H:%M:%S",  # 2024-01-05 10:30:00
                "%d/%m/%Y",           # 05/01/2024
                "%m/%d/%Y",           # 01/05/2024
            ]:
                try:
                    parsed = datetime.strptime(date_str[:19], fmt)  # Take first 19 chars
                    if self.min_date <= parsed <= self.max_date:
                        return parsed.strftime("%Y-%m-%d")
                except ValueError:
                    continue
            
            # Try to extract just YYYY-MM-DD from longer strings
            match = re.search(r'(\d{4}-\d{2}-\d{2})', date_str)
            if match:
                try:
                    parsed = datetime.strptime(match.group(1), "%Y-%m-%d")
                    if self.min_date <= parsed <= self.max_date:
                        return parsed.strftime("%Y-%m-%d")
                except ValueError:
                    pass
                    
        except Exception as e:
            self.logger.debug(f"Date parsing error for '{date_str}': {e}")
            
        return None

    # ---------- cheap string / regex stages --------------------------------
    @staticmethod
    def _date_from_url(url: str) -> Optional[str]:
        try:
            m = NewsDateExtractor._URL_DATE_RE.search(url)
            if m:
                year = int(m['Y'])
                month = int(m['M'])
                day = int(m['D'])
                
                # Basic validation
                if 1995 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                    return f"{year}-{month:02d}-{day:02d}"
        except Exception:
            pass
        return None

    def _date_from_meta(self, snippet: str) -> Optional[str]:
        """
        Extract publication date from HTML snippet using multiple patterns.
        """
        try:
            if "date" not in snippet.lower():
                return None
                
            # Try each pattern in order of preference
            for pattern in self._META_PATTERNS:
                matches = pattern.findall(snippet)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match else ""
                    
                    validated_date = self._validate_date_string(match)
                    if validated_date:
                        self.logger.debug(f"Found date '{validated_date}' using pattern: {pattern.pattern[:50]}...")
                        return validated_date
                        
        except Exception as e:
            self.logger.debug(f"Meta date extraction error: {e}")
            
        return None

    # ---------- network helpers -------------------------------------------
    async def _fetch_meta_with_retry(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """
        Fetch HTML snippet with retry logic and error handling.
        Returns HTML snippet or None if all attempts fail.
        """
        for attempt in range(self.max_retries + 1):
            try:
                return await self._fetch_meta(session, url)
            except Exception as e:
                if attempt == self.max_retries:
                    self.logger.debug(f"Failed to fetch {url} after {self.max_retries + 1} attempts: {e}")
                    return None
                await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff
        return None

    async def _fetch_meta(self, session: aiohttp.ClientSession, url: str) -> str:
        """
        Fetch partial HTML content (first N bytes) for meta tag parsing.
        """
        hdr = {
            **self._base_range_header,
            "User-Agent": random.choice(self.ua_pool),
        }
        async with session.get(url, headers=hdr, timeout=self.timeout) as get_resp:
            return await get_resp.text(errors="ignore")

    def _date_from_full_html(self, html: str, url: str) -> Optional[str]:
        """
        Use htmldate library as fallback, but validate the result.
        Be more conservative and reject future dates.
        """
        try:
            # First try with original_date=True (looks for publication dates specifically)
            result = find_date(html, url=url, extensive_search=False, original_date=True)
            if result:
                validated = self._validate_date_string(result)
                if validated:
                    # Extra check: reject dates more than 7 days in the future
                    from datetime import datetime, timedelta
                    result_date = datetime.strptime(validated, "%Y-%m-%d")
                    if result_date <= datetime.now() + timedelta(days=7):
                        self.logger.debug(f"htmldate found valid publication date: {validated}")
                        return validated
                    else:
                        self.logger.debug(f"htmldate found future date, rejecting: {validated}")
                else:
                    self.logger.debug(f"htmldate found invalid date: {result}")
        except Exception as e:
            self.logger.debug(f"htmldate error: {e}")
            
        return None

    # ---------- per-URL orchestrator --------------------------------------
    async def _process_url(self, url: str, session: aiohttp.ClientSession) -> Tuple[str, Optional[str]]:
        """
        Process a single URL with comprehensive error handling.
        Always returns a tuple, with None for failed extractions.
        """
        try:
            # Layer A : URL slug (highest priority for news sites)
            if (d := self._date_from_url(url)):
                self.logger.debug(f"Found date in URL: {d}")
                return url, d

            # Layer B : Partial GET for meta tags
            html_snippet = await self._fetch_meta_with_retry(session, url)
            if html_snippet is None:
                return url, None
            
            # Try meta tag extraction first
            if (d := self._date_from_meta(html_snippet)):
                self.logger.debug(f"Found date in meta tags: {d}")
                return url, d

            # Layer C : Full GET + htmldate as fallback (only if needed)
            try:
                async with session.get(url, timeout=self.timeout,
                                     headers={"User-Agent": random.choice(self.ua_pool)}) as resp:
                    html_remaining = await resp.text(errors="ignore")
                    html_full = html_snippet + html_remaining

                loop = asyncio.get_running_loop()
                d = await loop.run_in_executor(self._thread_pool, self._date_from_full_html, html_full, url)
                if d:
                    self.logger.debug(f"Found date via htmldate: {d}")
                return url, d
            except Exception as e:
                self.logger.debug(f"Full HTML fetch failed for {url}: {e}")
                return url, None

        except Exception as e:
            self.logger.debug(f"Error processing {url}: {e}")
            return url, None

    # ------------------------------------------------------------------ #
    #                          2.  PUBLIC API                            #
    # ------------------------------------------------------------------ #
    async def _run_async(self, urls: Iterable[str]) -> List[Tuple[str, Optional[str]]]:
        # Create SSL context
        ssl_context = False if self.skip_ssl_verify else None
        
        conn = aiohttp.TCPConnector(
            limit=self.concurrency, 
            ssl=ssl_context,
            ttl_dns_cache=300,  # Cache DNS lookups
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        connector_timeout = aiohttp.ClientTimeout(total=self.timeout.total + 5)
        
        async with aiohttp.ClientSession(
            connector=conn, 
            timeout=connector_timeout,
            headers={"Connection": "keep-alive"}
        ) as session:
            tasks = [self._process_url(u, session) for u in urls]
            # Use asyncio.gather with return_exceptions=True to handle failures gracefully
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and convert to proper tuples
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Use the original URL from the input
                    url = list(urls)[i] if hasattr(urls, '__getitem__') else str(i)
                    self.logger.debug(f"Task failed for URL {url}: {result}")
                    final_results.append((url, None))
                else:
                    final_results.append(result)
                    
        return final_results

    def extract(self, urls: Iterable[str], as_dataframe: bool = True) -> pd.DataFrame | List[Tuple[str, Optional[str]]]:
        """
        Resolve publication dates for a list/Series of URLs.
        
        This method handles both regular Python environments and Jupyter notebooks.

        Returns
        -------
        * pandas.DataFrame with columns ['url', 'pub_date']  (default)
        * OR raw list of (url, pub_date) tuples if `as_dataframe=False`
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in a notebook/async context, create a new thread
            import concurrent.futures
            import threading
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self._run_async(urls))
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                results = future.result()
                
        except RuntimeError:
            # No event loop running, we can use asyncio.run()
            results = asyncio.run(self._run_async(urls))
        
        if as_dataframe:
            return pd.DataFrame(results, columns=["url", "pub_date"])
        return results

    async def aextract(self, urls, as_dataframe: bool = True):
        """Async version – use this in notebooks (`await extractor.aextract(...)`)"""
        results = await self._run_async(urls)
        if as_dataframe:
            return pd.DataFrame(results, columns=["url", "pub_date"])
        return results

# ---------------------------------------------------------------------- #
#                       3.  quick CLI / usage demo                      #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    # Provide a CSV with a 'url' column or list manually
    sample_csv = Path("my_urls.csv")
    if sample_csv.exists():
        url_iterable = pd.read_csv(sample_csv)["url"]
    else:
        url_iterable = [
            "https://www.bbc.com/news/world-europe-66718123",
            "https://edition.cnn.com/2024/01/05/tech/apple-earnings-01-05-2024/index.html",
        ]

    extractor = NewsDateExtractor(concurrency=80, max_workers=10)
    df_dates = extractor.extract(url_iterable)
    print(df_dates.head())
