"""
Terminal Dashboard for Claude Agent Swarm Framework

Provides real-time visualization of swarm execution using Rich library.
Displays active agents, progress bars, resource utilization, and more.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, TaskID
)
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


@dataclass
class AgentDisplayInfo:
    """Information for displaying an agent in the dashboard."""
    agent_id: str
    status: str = "idle"  # idle, running, completed, failed
    task_name: str = ""
    progress: float = 0.0  # 0-100
    start_time: Optional[float] = None
    tokens_used: int = 0
    tool_calls: int = 0
    error_message: Optional[str] = None
    
    def get_duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_duration_str(self) -> str:
        """Get formatted duration string."""
        duration = self.get_duration()
        return str(timedelta(seconds=int(duration)))


@dataclass
class SwarmStats:
    """Statistics for the swarm."""
    total_agents: int = 0
    active_agents: int = 0
    completed_agents: int = 0
    failed_agents: int = 0
    total_tokens: int = 0
    total_tool_calls: int = 0
    start_time: Optional[float] = None
    estimated_completion: Optional[float] = None
    
    def get_elapsed(self) -> str:
        """Get elapsed time string."""
        if self.start_time is None:
            return "00:00:00"
        elapsed = time.time() - self.start_time
        return str(timedelta(seconds=int(elapsed)))
    
    def get_eta(self) -> str:
        """Get estimated time of arrival string."""
        if self.estimated_completion is None:
            return "--:--:--"
        remaining = self.estimated_completion - time.time()
        if remaining < 0:
            return "00:00:00"
        return str(timedelta(seconds=int(remaining)))


class SwarmDashboard:
    """
    Real-time terminal dashboard for Claude Agent Swarm.
    
    Displays:
    - Active agents with status and progress
    - Real-time progress bars
    - Token usage and tool call counters
    - Resource utilization
    - Estimated time remaining
    
    Example:
        >>> dashboard = SwarmDashboard()
        >>> dashboard.start()
        >>> dashboard.update_agent("agent_1", status="running", progress=50)
        >>> dashboard.update_progress("agent_1", 75)
        >>> dashboard.stop()
    
    Attributes:
        refresh_rate: Dashboard refresh rate in seconds
        console: Rich console instance
        live: Live display instance
    """
    
    def __init__(
        self,
        title: str = "Claude Agent Swarm",
        refresh_rate: float = 0.5,
        console: Optional[Console] = None,
        show_token_usage: bool = True,
        show_tool_calls: bool = True,
        show_timestamps: bool = True
    ):
        """
        Initialize the swarm dashboard.
        
        Args:
            title: Dashboard title
            refresh_rate: Update frequency in seconds
            console: Rich console instance (optional)
            show_token_usage: Show token usage column
            show_tool_calls: Show tool calls column
            show_timestamps: Show timestamps
        """
        self.title = title
        self.refresh_rate = refresh_rate
        self.console = console or Console()
        self.show_token_usage = show_token_usage
        self.show_tool_calls = show_tool_calls
        self.show_timestamps = show_timestamps
        
        # Agent tracking
        self._agents: Dict[str, AgentDisplayInfo] = {}
        self._swarm_stats = SwarmStats()
        
        # Progress tracking
        self._progress_bars: Dict[str, TaskID] = {}
        self._progress: Optional[Progress] = None
        
        # Live display
        self._live: Optional[Live] = None
        self._running = False
        self._lock = threading.RLock()
        
        # Callbacks
        self._update_callbacks: List[Callable[[], None]] = []
        
        # Layout components
        self._layout: Optional[Layout] = None
    
    def start(self) -> None:
        """Start the dashboard display."""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._swarm_stats.start_time = time.time()
            
            # Create progress tracker
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.fields[agent_id]}"),
                BarColumn(bar_width=40),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=True
            )
            
            # Create layout
            self._layout = self._create_layout()
            
            # Start live display
            self._live = Live(
                self._layout,
                console=self.console,
                refresh_per_second=1 / self.refresh_rate,
                screen=True
            )
            self._live.start()
    
    def stop(self) -> None:
        """Stop the dashboard display."""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            
            if self._live:
                self._live.stop()
                self._live = None
    
    def _create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout(name="root")
        
        # Header
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Main area split
        layout["main"].split_row(
            Layout(name="agents", ratio=2),
            Layout(name="stats", ratio=1)
        )
        
        return layout
    
    def _update_display(self) -> None:
        """Update the dashboard display."""
        if not self._layout:
            return
        
        # Update header
        self._layout["header"].update(self._create_header())
        
        # Update agents panel
        self._layout["agents"].update(self._create_agents_panel())
        
        # Update stats panel
        self._layout["stats"].update(self._create_stats_panel())
        
        # Update footer
        self._layout["footer"].update(self._create_footer())
    
    def _create_header(self) -> Panel:
        """Create the header panel."""
        elapsed = self._swarm_stats.get_elapsed()
        eta = self._swarm_stats.get_eta()
        
        header_text = Text()
        header_text.append(f" {self.title} ", style="bold white on blue")
        header_text.append(f" | Elapsed: {elapsed} | ETA: {eta}")
        
        return Panel(header_text, border_style="blue")
    
    def _create_agents_panel(self) -> Panel:
        """Create the agents panel with progress bars."""
        if not self._agents:
            return Panel(
                Text("No active agents", style="dim"),
                title="[bold]Agents[/bold]",
                border_style="cyan"
            )
        
        # Create agent table
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Agent", style="cyan", width=20)
        table.add_column("Status", width=12)
        table.add_column("Task", style="green")
        table.add_column("Progress", width=15)
        if self.show_token_usage:
            table.add_column("Tokens", justify="right", width=10)
        if self.show_tool_calls:
            table.add_column("Tools", justify="right", width=8)
        if self.show_timestamps:
            table.add_column("Duration", width=10)
        
        for agent_id, info in sorted(self._agents.items()):
            # Status styling
            status_style = {
                "idle": "dim",
                "running": "bold yellow",
                "completed": "bold green",
                "failed": "bold red"
            }.get(info.status, "white")
            
            # Progress bar
            progress_bar = "█" * int(info.progress / 5) + "░" * (20 - int(info.progress / 5))
            progress_text = f"{progress_bar} {info.progress:.0f}%"
            
            row = [
                agent_id,
                Text(info.status, style=status_style),
                info.task_name or "-",
                progress_text
            ]
            
            if self.show_token_usage:
                row.append(f"{info.tokens_used:,}")
            if self.show_tool_calls:
                row.append(f"{info.tool_calls}")
            if self.show_timestamps:
                row.append(info.get_duration_str())
            
            table.add_row(*row)
            
            # Add error message if failed
            if info.status == "failed" and info.error_message:
                table.add_row(
                    "",
                    "",
                    Text(f"  ⚠ {info.error_message}", style="red"),
                    "",
                    *([""] * (len(table.columns) - 4))
                )
        
        return Panel(table, title="[bold]Agents[/bold]", border_style="cyan")
    
    def _create_stats_panel(self) -> Panel:
        """Create the statistics panel."""
        stats = self._swarm_stats
        
        # Create stats table
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="white")
        
        table.add_row("Total Agents", str(stats.total_agents))
        table.add_row("Active", f"[yellow]{stats.active_agents}[/yellow]")
        table.add_row("Completed", f"[green]{stats.completed_agents}[/green]")
        table.add_row("Failed", f"[red]{stats.failed_agents}[/red]")
        table.add_row("", "")
        table.add_row("Total Tokens", f"[blue]{stats.total_tokens:,}[/blue]")
        table.add_row("Tool Calls", f"[magenta]{stats.total_tool_calls}[/magenta]")
        table.add_row("", "")
        table.add_row("Elapsed Time", stats.get_elapsed())
        table.add_row("Est. Remaining", stats.get_eta())
        
        # Token breakdown
        if self._agents:
            table.add_row("", "")
            table.add_row("[bold]Tokens by Agent[/bold]", "")
            for agent_id, info in sorted(self._agents.items()):
                table.add_row(f"  {agent_id}", f"{info.tokens_used:,}")
        
        return Panel(table, title="[bold]Statistics[/bold]", border_style="green")
    
    def _create_footer(self) -> Panel:
        """Create the footer panel."""
        footer_text = Text()
        footer_text.append(" Press ", style="dim")
        footer_text.append("Ctrl+C", style="bold")
        footer_text.append(" to stop ", style="dim")
        
        return Panel(footer_text, border_style="dim")
    
    def update_agent(
        self,
        agent_id: str,
        status: Optional[str] = None,
        task_name: Optional[str] = None,
        progress: Optional[float] = None,
        tokens_used: Optional[int] = None,
        tool_calls: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update agent information in the dashboard.
        
        Args:
            agent_id: Agent identifier
            status: New status (idle, running, completed, failed)
            task_name: Current task name
            progress: Progress percentage (0-100)
            tokens_used: Total tokens used by agent
            tool_calls: Total tool calls by agent
            error_message: Error message if failed
        """
        with self._lock:
            if agent_id not in self._agents:
                self._agents[agent_id] = AgentDisplayInfo(agent_id=agent_id)
                self._swarm_stats.total_agents += 1
                
                # Add progress bar
                if self._progress:
                    task_id = self._progress.add_task(
                        f"Agent {agent_id}",
                        agent_id=agent_id,
                        total=100
                    )
                    self._progress_bars[agent_id] = task_id
            
            agent = self._agents[agent_id]
            
            # Track status changes
            old_status = agent.status
            if status:
                agent.status = status
                
                if status == "running" and old_status != "running":
                    agent.start_time = time.time()
                    self._swarm_stats.active_agents += 1
                elif status == "completed" and old_status == "running":
                    self._swarm_stats.active_agents -= 1
                    self._swarm_stats.completed_agents += 1
                elif status == "failed" and old_status == "running":
                    self._swarm_stats.active_agents -= 1
                    self._swarm_stats.failed_agents += 1
            
            # Update other fields
            if task_name is not None:
                agent.task_name = task_name
            if progress is not None:
                agent.progress = max(0, min(100, progress))
                if agent_id in self._progress_bars and self._progress:
                    self._progress.update(
                        self._progress_bars[agent_id],
                        completed=agent.progress
                    )
            if tokens_used is not None:
                token_diff = tokens_used - agent.tokens_used
                agent.tokens_used = tokens_used
                self._swarm_stats.total_tokens += token_diff
            if tool_calls is not None:
                call_diff = tool_calls - agent.tool_calls
                agent.tool_calls = tool_calls
                self._swarm_stats.total_tool_calls += call_diff
            if error_message is not None:
                agent.error_message = error_message
        
        # Update display
        self._update_display()
    
    def update_progress(self, agent_id: str, progress: float) -> None:
        """
        Update agent progress.
        
        Args:
            agent_id: Agent identifier
            progress: Progress percentage (0-100)
        """
        self.update_agent(agent_id, progress=progress)
    
    def update_tokens(self, agent_id: str, tokens: int) -> None:
        """
        Update token count for an agent.
        
        Args:
            agent_id: Agent identifier
            tokens: Total tokens used
        """
        self.update_agent(agent_id, tokens_used=tokens)
    
    def update_tool_calls(self, agent_id: str, calls: int) -> None:
        """
        Update tool call count for an agent.
        
        Args:
            agent_id: Agent identifier
            calls: Total tool calls
        """
        self.update_agent(agent_id, tool_calls=calls)
    
    def remove_agent(self, agent_id: str) -> None:
        """
        Remove an agent from the dashboard.
        
        Args:
            agent_id: Agent identifier
        """
        with self._lock:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                
                # Update stats
                if agent.status == "running":
                    self._swarm_stats.active_agents -= 1
                elif agent.status == "completed":
                    self._swarm_stats.completed_agents -= 1
                elif agent.status == "failed":
                    self._swarm_stats.failed_agents -= 1
                
                self._swarm_stats.total_agents -= 1
                self._swarm_stats.total_tokens -= agent.tokens_used
                self._swarm_stats.total_tool_calls -= agent.tool_calls
                
                del self._agents[agent_id]
                
                if agent_id in self._progress_bars and self._progress:
                    self._progress.remove_task(self._progress_bars[agent_id])
                    del self._progress_bars[agent_id]
        
        self._update_display()
    
    def set_eta(self, eta_seconds: float) -> None:
        """
        Set estimated time of completion.
        
        Args:
            eta_seconds: Estimated seconds until completion
        """
        with self._lock:
            self._swarm_stats.estimated_completion = time.time() + eta_seconds
    
    def add_update_callback(self, callback: Callable[[], None]) -> None:
        """
        Add a callback to be called on each update.
        
        Args:
            callback: Function to call
        """
        self._update_callbacks.append(callback)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the dashboard state.
        
        Returns:
            Dictionary with summary statistics
        """
        with self._lock:
            return {
                "total_agents": self._swarm_stats.total_agents,
                "active_agents": self._swarm_stats.active_agents,
                "completed_agents": self._swarm_stats.completed_agents,
                "failed_agents": self._swarm_stats.failed_agents,
                "total_tokens": self._swarm_stats.total_tokens,
                "total_tool_calls": self._swarm_stats.total_tool_calls,
                "elapsed_time": self._swarm_stats.get_elapsed(),
                "agents": {
                    aid: {
                        "status": info.status,
                        "progress": info.progress,
                        "tokens_used": info.tokens_used,
                        "tool_calls": info.tool_calls,
                        "duration": info.get_duration()
                    }
                    for aid, info in self._agents.items()
                }
            }
    
    def print_summary(self) -> None:
        """Print a summary table after dashboard stops."""
        summary = self.get_summary()
        
        self.console.print("\n[bold]Swarm Execution Summary[/bold]\n")
        
        table = Table(title="Execution Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Total Agents", str(summary["total_agents"]))
        table.add_row("Completed", f"[green]{summary['completed_agents']}[/green]")
        table.add_row("Failed", f"[red]{summary['failed_agents']}[/red]")
        table.add_row("Total Tokens", f"[blue]{summary['total_tokens']:,}[/blue]")
        table.add_row("Total Tool Calls", f"[magenta]{summary['total_tool_calls']}[/magenta]")
        table.add_row("Elapsed Time", summary["elapsed_time"])
        
        self.console.print(table)
        
        # Agent details
        if summary["agents"]:
            self.console.print("\n[bold]Agent Details:[/bold]")
            agent_table = Table()
            agent_table.add_column("Agent ID", style="cyan")
            agent_table.add_column("Status")
            agent_table.add_column("Progress")
            agent_table.add_column("Tokens")
            agent_table.add_column("Tool Calls")
            
            for aid, info in summary["agents"].items():
                status_color = {
                    "completed": "green",
                    "failed": "red",
                    "running": "yellow"
                }.get(info["status"], "white")
                
                agent_table.add_row(
                    aid,
                    f"[{status_color}]{info['status']}[/{status_color}]",
                    f"{info['progress']:.0f}%",
                    f"{info['tokens_used']:,}",
                    str(info["tool_calls"])
                )
            
            self.console.print(agent_table)


class AsyncSwarmDashboard(SwarmDashboard):
    """
    Async-compatible version of SwarmDashboard.
    
    Provides non-blocking updates suitable for async applications.
    
    Example:
        >>> dashboard = AsyncSwarmDashboard()
        >>> await dashboard.start_async()
        >>> await dashboard.update_agent_async("agent_1", progress=50)
        >>> await dashboard.stop_async()
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_queue: asyncio.Queue = None
        self._update_task: Optional[asyncio.Task] = None
    
    async def start_async(self) -> None:
        """Start the dashboard asynchronously."""
        self._update_queue = asyncio.Queue()
        self.start()
        self._update_task = asyncio.create_task(self._process_updates())
    
    async def stop_async(self) -> None:
        """Stop the dashboard asynchronously."""
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        self.stop()
    
    async def _process_updates(self) -> None:
        """Process queued updates."""
        while self._running:
            try:
                update = await asyncio.wait_for(
                    self._update_queue.get(),
                    timeout=0.1
                )
                update()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
    
    async def update_agent_async(
        self,
        agent_id: str,
        status: Optional[str] = None,
        task_name: Optional[str] = None,
        progress: Optional[float] = None,
        tokens_used: Optional[int] = None,
        tool_calls: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update agent information asynchronously.
        
        Args:
            agent_id: Agent identifier
            status: New status
            task_name: Current task name
            progress: Progress percentage
            tokens_used: Total tokens used
            tool_calls: Total tool calls
            error_message: Error message if failed
        """
        if self._update_queue:
            await self._update_queue.put(
                lambda: self.update_agent(
                    agent_id, status, task_name, progress,
                    tokens_used, tool_calls, error_message
                )
            )
        else:
            self.update_agent(
                agent_id, status, task_name, progress,
                tokens_used, tool_calls, error_message
            )
    
    async def update_progress_async(self, agent_id: str, progress: float) -> None:
        """
        Update agent progress asynchronously.
        
        Args:
            agent_id: Agent identifier
            progress: Progress percentage
        """
        await self.update_agent_async(agent_id, progress=progress)


def create_dashboard(
    title: str = "Claude Agent Swarm",
    async_mode: bool = False,
    **kwargs
) -> SwarmDashboard:
    """
    Factory function to create a dashboard.
    
    Args:
        title: Dashboard title
        async_mode: Create async-compatible dashboard
        **kwargs: Additional dashboard options
        
    Returns:
        Configured dashboard instance
    """
    if async_mode:
        return AsyncSwarmDashboard(title=title, **kwargs)
    return SwarmDashboard(title=title, **kwargs)
