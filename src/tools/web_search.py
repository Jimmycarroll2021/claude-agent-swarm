"""
Web search tool for Claude Agent Swarm.

Provides web search capabilities through various providers.
"""

import asyncio
import json
import hashlib
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import aiohttp

from . import BaseTool, ToolSchema, ToolResult


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    rank: int
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "rank": self.rank,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class SearchCacheEntry:
    """Cached search result."""
    query: str
    results: List[SearchResult]
    timestamp: datetime
    ttl_seconds: int = 3600  # 1 hour default
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.utcnow() - self.timestamp > timedelta(seconds=self.ttl_seconds)


class WebSearchTool(BaseTool):
    """Tool for web search operations."""
    
    def __init__(
        self,
        provider: str = "brave",
        api_key: Optional[str] = None,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 3600,
        max_results: int = 10,
        timeout: int = 30
    ):
        super().__init__(
            name="web_search",
            description="Search the web for information"
        )
        self.provider = provider
        self.api_key = api_key
        self.cache_enabled = cache_enabled
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_results = max_results
        self.timeout = timeout
        
        self._cache: Dict[str, SearchCacheEntry] = {}
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    def get_schema(self) -> ToolSchema:
        """Get the tool schema."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50
                },
                "offset": {
                    "type": "integer",
                    "description": "Result offset for pagination",
                    "default": 0
                },
                "freshness": {
                    "type": "string",
                    "enum": ["any", "day", "week", "month"],
                    "description": "Filter by result freshness",
                    "default": "any"
                }
            },
            required=["query"],
            returns={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "snippet": {"type": "string"},
                                "rank": {"type": "integer"}
                            }
                        }
                    },
                    "total": {"type": "integer"},
                    "query": {"type": "string"}
                }
            }
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute web search."""
        query = kwargs.get("query", "").strip()
        
        if not query:
            return ToolResult(success=False, error="Empty search query")
        
        num_results = min(kwargs.get("num_results", self.max_results), 50)
        offset = kwargs.get("offset", 0)
        freshness = kwargs.get("freshness", "any")
        
        try:
            # Check cache
            if self.cache_enabled:
                cached = self._get_from_cache(query)
                if cached:
                    return ToolResult(
                        success=True,
                        data={
                            "results": [r.to_dict() for r in cached[:num_results]],
                            "total": len(cached),
                            "query": query,
                            "cached": True
                        }
                    )
            
            # Perform search based on provider
            if self.provider == "brave":
                results = await self._search_brave(query, num_results, offset, freshness)
            elif self.provider == "serper":
                results = await self._search_serper(query, num_results, offset)
            elif self.provider == "duckduckgo":
                results = await self._search_duckduckgo(query, num_results, offset)
            else:
                return ToolResult(success=False, error=f"Unknown provider: {self.provider}")
            
            # Cache results
            if self.cache_enabled:
                self._add_to_cache(query, results)
            
            return ToolResult(
                success=True,
                data={
                    "results": [r.to_dict() for r in results],
                    "total": len(results),
                    "query": query,
                    "cached": False
                }
            )
        
        except Exception as e:
            return ToolResult(success=False, error=f"Search failed: {str(e)}")
    
    async def _search_brave(
        self, 
        query: str, 
        num_results: int,
        offset: int,
        freshness: str
    ) -> List[SearchResult]:
        """Search using Brave Search API."""
        if not self.api_key:
            raise ValueError("Brave Search API key required")
        
        session = await self._get_session()
        
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json"
        }
        params = {
            "q": query,
            "count": num_results,
            "offset": offset
        }
        
        if freshness != "any":
            params["freshness"] = freshness
        
        async with session.get(url, headers=headers, params=params, timeout=self.timeout) as resp:
            if resp.status != 200:
                raise Exception(f"Brave API error: {resp.status}")
            
            data = await resp.json()
            
            results = []
            for i, item in enumerate(data.get("web", {}).get("results", [])):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                    rank=offset + i + 1
                ))
            
            return results
    
    async def _search_serper(
        self, 
        query: str, 
        num_results: int,
        offset: int
    ) -> List[SearchResult]:
        """Search using Serper.dev API."""
        if not self.api_key:
            raise ValueError("Serper API key required")
        
        session = await self._get_session()
        
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
            "num": num_results,
            "start": offset
        }
        
        async with session.post(url, headers=headers, json=payload, timeout=self.timeout) as resp:
            if resp.status != 200:
                raise Exception(f"Serper API error: {resp.status}")
            
            data = await resp.json()
            
            results = []
            for i, item in enumerate(data.get("organic", [])):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    rank=offset + i + 1
                ))
            
            return results
    
    async def _search_duckduckgo(
        self, 
        query: str, 
        num_results: int,
        offset: int
    ) -> List[SearchResult]:
        """Search using DuckDuckGo (no API key required)."""
        # This is a simplified implementation
        # In production, use a proper DuckDuckGo library
        session = await self._get_session()
        
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        
        async with session.post(url, data=params, timeout=self.timeout) as resp:
            if resp.status != 200:
                raise Exception(f"DuckDuckGo error: {resp.status}")
            
            html = await resp.text()
            
            # Simple parsing (in production, use BeautifulSoup)
            results = []
            # This is a placeholder - real implementation would parse HTML
            
            return results
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for a query."""
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def _get_from_cache(self, query: str) -> Optional[List[SearchResult]]:
        """Get cached search results."""
        key = self._get_cache_key(query)
        entry = self._cache.get(key)
        
        if entry and not entry.is_expired():
            return entry.results
        
        if entry and entry.is_expired():
            del self._cache[key]
        
        return None
    
    def _add_to_cache(self, query: str, results: List[SearchResult]) -> None:
        """Add search results to cache."""
        key = self._get_cache_key(query)
        self._cache[key] = SearchCacheEntry(
            query=query,
            results=results,
            timestamp=datetime.utcnow(),
            ttl_seconds=self.cache_ttl_seconds
        )
    
    def clear_cache(self) -> None:
        """Clear search cache."""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = len(self._cache)
        expired = sum(1 for e in self._cache.values() if e.is_expired())
        
        return {
            "total_entries": total,
            "expired_entries": expired,
            "valid_entries": total - expired
        }
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


__all__ = [
    "SearchResult",
    "SearchCacheEntry",
    "WebSearchTool"
]
