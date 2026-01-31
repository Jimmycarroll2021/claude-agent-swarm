"""
File operations tool for Claude Agent Swarm.

Provides safe file read, write, edit, and search operations.
"""

import os
import re
import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from dataclasses import dataclass
from datetime import datetime
import aiofiles
import asyncio

from . import BaseTool, ToolSchema, ToolResult


class FileOpsError(Exception):
    """Base exception for file operations."""
    pass


class PathValidationError(FileOpsError):
    """Raised when a path is invalid or unsafe."""
    pass


@dataclass
class FileInfo:
    """Information about a file or directory."""
    path: str
    name: str
    is_file: bool
    is_dir: bool
    size: int
    modified_time: datetime
    created_time: datetime
    permissions: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "name": self.name,
            "is_file": self.is_file,
            "is_dir": self.is_dir,
            "size": self.size,
            "modified_time": self.modified_time.isoformat(),
            "created_time": self.created_time.isoformat(),
            "permissions": self.permissions
        }


class FileOperationsTool(BaseTool):
    """Tool for file operations with safety checks."""
    
    def __init__(
        self, 
        allowed_paths: Optional[List[str]] = None,
        block_symlinks: bool = True,
        max_file_size: int = 10 * 1024 * 1024  # 10MB
    ):
        super().__init__(
            name="file_operations",
            description="Read, write, edit, and search files safely"
        )
        self.allowed_paths = allowed_paths or ["/mnt/okcomputer"]
        self.block_symlinks = block_symlinks
        self.max_file_size = max_file_size
    
    def get_schema(self) -> ToolSchema:
        """Get the tool schema."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "append", "edit", "delete", "search", "list", "info"],
                    "description": "File operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path"
                },
                "content": {
                    "type": "string",
                    "description": "Content for write/append operations"
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern or glob pattern"
                },
                "old_string": {
                    "type": "string",
                    "description": "String to replace in edit operation"
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement string in edit operation"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively",
                    "default": False
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 100
                }
            },
            required=["operation", "path"],
            returns={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {"type": "any"},
                    "error": {"type": "string"}
                }
            }
        )
    
    def _validate_path(self, path: str) -> Path:
        """Validate and resolve a path."""
        resolved = Path(path).resolve()
        
        # Check for symlinks
        if self.block_symlinks and Path(path).is_symlink():
            raise PathValidationError(f"Symlinks are blocked: {path}")
        
        # Check allowed paths
        allowed = False
        for allowed_path in self.allowed_paths:
            allowed_resolved = Path(allowed_path).resolve()
            try:
                resolved.relative_to(allowed_resolved)
                allowed = True
                break
            except ValueError:
                continue
        
        if not allowed:
            raise PathValidationError(
                f"Path '{path}' is outside allowed directories: {self.allowed_paths}"
            )
        
        return resolved
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute file operation."""
        operation = kwargs.get("operation")
        path = kwargs.get("path", "")
        
        try:
            if operation == "read":
                return await self._read_file(path)
            elif operation == "write":
                return await self._write_file(path, kwargs.get("content", ""))
            elif operation == "append":
                return await self._append_file(path, kwargs.get("content", ""))
            elif operation == "edit":
                return await self._edit_file(
                    path, 
                    kwargs.get("old_string", ""),
                    kwargs.get("new_string", "")
                )
            elif operation == "delete":
                return await self._delete_file(path)
            elif operation == "search":
                return await self._search(
                    path,
                    kwargs.get("pattern", ""),
                    kwargs.get("recursive", False),
                    kwargs.get("limit", 100)
                )
            elif operation == "list":
                return await self._list_dir(path, kwargs.get("recursive", False))
            elif operation == "info":
                return await self._get_info(path)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown operation: {operation}"
                )
        
        except PathValidationError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"File operation failed: {str(e)}")
    
    async def _read_file(self, path: str) -> ToolResult:
        """Read a file."""
        resolved = self._validate_path(path)
        
        if not resolved.exists():
            return ToolResult(success=False, error=f"File not found: {path}")
        
        if not resolved.is_file():
            return ToolResult(success=False, error=f"Not a file: {path}")
        
        if resolved.stat().st_size > self.max_file_size:
            return ToolResult(
                success=False, 
                error=f"File too large (max {self.max_file_size} bytes)"
            )
        
        async with aiofiles.open(resolved, "r", encoding="utf-8") as f:
            content = await f.read()
        
        return ToolResult(success=True, data={"content": content, "path": str(resolved)})
    
    async def _write_file(self, path: str, content: str) -> ToolResult:
        """Write content to a file."""
        resolved = self._validate_path(path)
        
        # Create parent directories if needed
        resolved.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(resolved, "w", encoding="utf-8") as f:
            await f.write(content)
        
        return ToolResult(success=True, data={"path": str(resolved), "bytes_written": len(content)})
    
    async def _append_file(self, path: str, content: str) -> ToolResult:
        """Append content to a file."""
        resolved = self._validate_path(path)
        
        if not resolved.exists():
            resolved.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(resolved, "a", encoding="utf-8") as f:
            await f.write(content)
        
        return ToolResult(success=True, data={"path": str(resolved), "bytes_appended": len(content)})
    
    async def _edit_file(self, path: str, old_string: str, new_string: str) -> ToolResult:
        """Edit a file by replacing a string."""
        resolved = self._validate_path(path)
        
        if not resolved.exists():
            return ToolResult(success=False, error=f"File not found: {path}")
        
        async with aiofiles.open(resolved, "r", encoding="utf-8") as f:
            content = await f.read()
        
        if old_string not in content:
            return ToolResult(
                success=False, 
                error=f"String not found in file: {old_string[:50]}..."
            )
        
        new_content = content.replace(old_string, new_string, 1)
        
        async with aiofiles.open(resolved, "w", encoding="utf-8") as f:
            await f.write(new_content)
        
        return ToolResult(
            success=True, 
            data={"path": str(resolved), "replacements": 1}
        )
    
    async def _delete_file(self, path: str) -> ToolResult:
        """Delete a file or directory."""
        resolved = self._validate_path(path)
        
        if not resolved.exists():
            return ToolResult(success=False, error=f"Path not found: {path}")
        
        if resolved.is_file():
            resolved.unlink()
        else:
            import shutil
            shutil.rmtree(resolved)
        
        return ToolResult(success=True, data={"deleted": str(resolved)})
    
    async def _search(
        self, 
        path: str, 
        pattern: str, 
        recursive: bool,
        limit: int
    ) -> ToolResult:
        """Search for pattern in files."""
        resolved = self._validate_path(path)
        
        results = []
        count = 0
        
        if resolved.is_file():
            files_to_search = [resolved]
        else:
            if recursive:
                files_to_search = list(resolved.rglob("*"))
            else:
                files_to_search = list(resolved.iterdir())
            files_to_search = [f for f in files_to_search if f.is_file()]
        
        for file_path in files_to_search:
            if count >= limit:
                break
            
            try:
                if file_path.stat().st_size > self.max_file_size:
                    continue
                
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                
                matches = []
                for i, line in enumerate(content.split("\n"), 1):
                    if pattern in line:
                        matches.append({"line": i, "content": line.strip()})
                
                if matches:
                    results.append({
                        "file": str(file_path),
                        "matches": matches
                    })
                    count += len(matches)
            
            except Exception:
                # Skip files that can't be read
                continue
        
        return ToolResult(
            success=True,
            data={"results": results, "total_matches": count}
        )
    
    async def _list_dir(self, path: str, recursive: bool) -> ToolResult:
        """List directory contents."""
        resolved = self._validate_path(path)
        
        if not resolved.is_dir():
            return ToolResult(success=False, error=f"Not a directory: {path}")
        
        entries = []
        
        if recursive:
            for item in resolved.rglob("*"):
                try:
                    info = self._get_file_info(item)
                    entries.append(info.to_dict())
                except Exception:
                    continue
        else:
            for item in resolved.iterdir():
                try:
                    info = self._get_file_info(item)
                    entries.append(info.to_dict())
                except Exception:
                    continue
        
        return ToolResult(success=True, data={"entries": entries, "path": str(resolved)})
    
    async def _get_info(self, path: str) -> ToolResult:
        """Get file or directory information."""
        resolved = self._validate_path(path)
        
        if not resolved.exists():
            return ToolResult(success=False, error=f"Path not found: {path}")
        
        info = self._get_file_info(resolved)
        return ToolResult(success=True, data=info.to_dict())
    
    def _get_file_info(self, path: Path) -> FileInfo:
        """Get information about a file or directory."""
        stat = path.stat()
        
        return FileInfo(
            path=str(path),
            name=path.name,
            is_file=path.is_file(),
            is_dir=path.is_dir(),
            size=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            created_time=datetime.fromtimestamp(stat.st_ctime),
            permissions=oct(stat.st_mode)[-3:]
        )


__all__ = [
    "FileOpsError",
    "PathValidationError",
    "FileInfo",
    "FileOperationsTool"
]
