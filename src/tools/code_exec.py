"""
Code execution tool for Claude Agent Swarm.

Provides sandboxed code execution capabilities.
"""

import asyncio
import re
import tempfile
import os
import signal
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import subprocess

from . import BaseTool, ToolSchema, ToolResult


class CodeExecutionError(Exception):
    """Base exception for code execution errors."""
    pass


class SecurityError(CodeExecutionError):
    """Raised when code violates security policy."""
    pass


class TimeoutError(CodeExecutionError):
    """Raised when code execution times out."""
    pass


@dataclass
class ExecutionResult:
    """Result of code execution."""
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "execution_time_ms": self.execution_time_ms
        }


class CodeExecutionTool(BaseTool):
    """Tool for executing code in sandboxed environment."""
    
    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        r"import\s+os\s*;\s*os\.system",
        r"import\s+subprocess",
        r"__import__\s*\(\s*['\"]os['\"]",
        r"__import__\s*\(\s*['\"]subprocess['\"]",
        r"eval\s*\(",
        r"exec\s*\(",
        r"compile\s*\(",
        r"open\s*\(\s*['\"]/",
        r"file\s*\(\s*['\"]/",
        r"import\s+pty",
        r"socket\.",
        r"urllib",
        r"requests\.",
        r"http\.client",
        r"ftplib",
        r"telnetlib",
    ]
    
    def __init__(
        self,
        allowed_imports: Optional[List[str]] = None,
        blocked_imports: Optional[List[str]] = None,
        max_execution_time: int = 30,
        max_output_size: int = 10000,
        allow_network: bool = False,
        allow_file_write: bool = False,
        temp_dir: Optional[str] = None
    ):
        super().__init__(
            name="code_execution",
            description="Execute Python code or bash commands in sandboxed environment"
        )
        self.allowed_imports = allowed_imports or []
        self.blocked_imports = blocked_imports or [
            "os.system", "subprocess", "pty", "socket", "urllib", 
            "requests", "http.client", "ftplib", "telnetlib"
        ]
        self.max_execution_time = max_execution_time
        self.max_output_size = max_output_size
        self.allow_network = allow_network
        self.allow_file_write = allow_file_write
        self.temp_dir = temp_dir or tempfile.gettempdir()
    
    def get_schema(self) -> ToolSchema:
        """Get the tool schema."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "language": {
                    "type": "string",
                    "enum": ["python", "bash"],
                    "description": "Programming language to execute"
                },
                "code": {
                    "type": "string",
                    "description": "Code to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds",
                    "default": 30
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Python packages to install (for python only)"
                }
            },
            required=["language", "code"],
            returns={
                "type": "object",
                "properties": {
                    "stdout": {"type": "string"},
                    "stderr": {"type": "string"},
                    "exit_code": {"type": "integer"},
                    "execution_time_ms": {"type": "number"}
                }
            }
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute code."""
        language = kwargs.get("language", "python")
        code = kwargs.get("code", "")
        timeout = kwargs.get("timeout", self.max_execution_time)
        dependencies = kwargs.get("dependencies", [])
        
        if not code.strip():
            return ToolResult(success=False, error="Empty code")
        
        try:
            if language == "python":
                result = await self.execute_python(code, timeout, dependencies)
            elif language == "bash":
                result = await self.execute_bash(code, timeout)
            else:
                return ToolResult(success=False, error=f"Unsupported language: {language}")
            
            return ToolResult(
                success=result.exit_code == 0,
                data=result.to_dict()
            )
        
        except SecurityError as e:
            return ToolResult(success=False, error=f"Security violation: {str(e)}")
        except TimeoutError:
            return ToolResult(success=False, error=f"Execution timed out after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, error=f"Execution failed: {str(e)}")
    
    def validate_code(self, code: str) -> List[str]:
        """Validate Python code for security issues."""
        violations = []
        
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(f"Dangerous pattern detected: {pattern}")
        
        # Check for blocked imports
        import_pattern = r"(?:from|import)\s+(\S+)"
        for match in re.finditer(import_pattern, code):
            module = match.group(1).split(".")[0]
            if module in self.blocked_imports:
                violations.append(f"Blocked import: {module}")
        
        return violations
    
    async def execute_python(
        self, 
        code: str, 
        timeout: int,
        dependencies: List[str]
    ) -> ExecutionResult:
        """Execute Python code."""
        # Validate code
        violations = self.validate_code(code)
        if violations:
            raise SecurityError("; ".join(violations))
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", 
            suffix=".py", 
            dir=self.temp_dir,
            delete=False
        ) as f:
            # Add safety wrapper
            wrapper_code = self._generate_safety_wrapper(code)
            f.write(wrapper_code)
            temp_file = f.name
        
        try:
            # Install dependencies if needed
            if dependencies:
                await self._install_dependencies(dependencies)
            
            # Execute code
            start_time = asyncio.get_event_loop().time()
            
            proc = await asyncio.create_subprocess_exec(
                "python", temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_sandbox_env()
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), 
                    timeout=timeout
                )
                execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
                
                stdout_str = stdout.decode("utf-8", errors="replace")[:self.max_output_size]
                stderr_str = stderr.decode("utf-8", errors="replace")[:self.max_output_size]
                
                return ExecutionResult(
                    stdout=stdout_str,
                    stderr=stderr_str,
                    exit_code=proc.returncode or 0,
                    execution_time_ms=execution_time
                )
            
            except asyncio.TimeoutError:
                proc.kill()
                raise TimeoutError(f"Execution timed out after {timeout}s")
        
        finally:
            # Cleanup
            try:
                os.unlink(temp_file)
            except:
                pass
    
    async def execute_bash(self, code: str, timeout: int) -> ExecutionResult:
        """Execute bash command."""
        # Validate bash command
        dangerous_commands = [
            "rm -rf /", "rm -rf /*", "> /dev/sda", "mkfs", "dd if=/dev/zero",
            ":(){ :|:& };:", "chmod -R 777 /", "chown -R",
            "wget", "curl", "nc ", "netcat", "ncat",
            "bash -i", "sh -i", "/bin/bash -i",
            "python -c", "python3 -c", "perl -e"
        ]
        
        for cmd in dangerous_commands:
            if cmd in code.lower():
                raise SecurityError(f"Dangerous command detected: {cmd}")
        
        start_time = asyncio.get_event_loop().time()
        
        proc = await asyncio.create_subprocess_shell(
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._get_sandbox_env(),
            cwd=self.temp_dir
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout
            )
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            stdout_str = stdout.decode("utf-8", errors="replace")[:self.max_output_size]
            stderr_str = stderr.decode("utf-8", errors="replace")[:self.max_output_size]
            
            return ExecutionResult(
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=proc.returncode or 0,
                execution_time_ms=execution_time
            )
        
        except asyncio.TimeoutError:
            proc.kill()
            raise TimeoutError(f"Execution timed out after {timeout}s")
    
    def _generate_safety_wrapper(self, code: str) -> str:
        """Generate safety wrapper for Python code."""
        wrapper = '''
import sys
import builtins

# Block dangerous functions
blocked_functions = ['eval', 'exec', 'compile', '__import__']
for func in blocked_functions:
    if hasattr(builtins, func):
        delattr(builtins, func)

# Limit imports
import importlib
_original_import = __builtins__.__import__

def safe_import(name, *args, **kwargs):
    blocked = ''' + str(self.blocked_imports) + '''
    base_module = name.split('.')[0]
    if base_module in blocked:
        raise ImportError(f"Import of '{name}' is not allowed")
    return _original_import(name, *args, **kwargs)

__builtins__.__import__ = safe_import

# Execute user code
'''
        return wrapper + code
    
    async def _install_dependencies(self, dependencies: List[str]) -> None:
        """Install Python dependencies."""
        if not dependencies:
            return
        
        proc = await asyncio.create_subprocess_exec(
            "pip", "install", "--quiet", *dependencies,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await proc.wait()
        
        if proc.returncode != 0:
            raise Exception(f"Failed to install dependencies: {dependencies}")
    
    def _get_sandbox_env(self) -> Dict[str, str]:
        """Get sandboxed environment variables."""
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": self.temp_dir,
            "TMPDIR": self.temp_dir,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1"
        }
        
        if not self.allow_network:
            # This would need additional network isolation in production
            pass
        
        return env


__all__ = [
    "CodeExecutionError",
    "SecurityError",
    "TimeoutError",
    "ExecutionResult",
    "CodeExecutionTool"
]
