"""CLI for Claude Agent Swarm framework."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

console = Console()


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="claude-swarm",
        description="Claude Agent Swarm Framework CLI",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize a new swarm project")
    init_parser.add_argument("name", help="Project name")
    init_parser.add_argument(
        "--template",
        choices=["research", "code", "analysis", "custom"],
        default="custom",
        help="Project template",
    )

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a swarm from configuration")
    run_parser.add_argument(
        "--config", "-c", required=True, help="Path to swarm configuration file"
    )
    run_parser.add_argument("--task", "-t", help="Task to execute")
    run_parser.add_argument(
        "--pattern",
        choices=["auto", "leader", "swarm", "pipeline", "council"],
        default="auto",
        help="Execution pattern",
    )

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor running swarms")
    monitor_parser.add_argument("--swarm-id", help="Specific swarm ID to monitor")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export swarm results")
    export_parser.add_argument(
        "--format", "-f", choices=["json", "markdown", "html"], default="json"
    )
    export_parser.add_argument("--output", "-o", required=True, help="Output file path")

    # Version command
    subparsers.add_parser("version", help="Show version information")

    args = parser.parse_args()

    if args.command == "version":
        from . import __version__

        console.print(f"[bold blue]Claude Agent Swarm[/bold blue] v{__version__}")
        return 0

    if args.command == "init":
        return init_project(args.name, args.template)

    if args.command == "run":
        return asyncio.run(run_swarm(args.config, args.task, args.pattern))

    if args.command == "monitor":
        return monitor_swarm(args.swarm_id)

    if args.command == "export":
        return export_results(args.format, args.output)

    # No command provided, show help
    parser.print_help()
    return 0


def init_project(name: str, template: str) -> int:
    """Initialize a new swarm project."""
    project_path = Path.cwd() / name

    if project_path.exists():
        console.print(f"[red]Error:[/red] Directory '{name}' already exists")
        return 1

    try:
        project_path.mkdir(parents=True)

        # Create basic structure
        (project_path / "configs").mkdir()
        (project_path / "outputs").mkdir()

        # Create default config
        config_content = f"""# {name} Swarm Configuration
version: 1
swarm:
  name: "{name}"
  orchestration_mode: "auto"
  max_agents: 10
  parallel_limit: 5

  orchestrator:
    model: "claude-3-7-sonnet-20250219"
    description: "Main orchestrator"

  agent_templates:
    worker:
      model: "claude-3-7-sonnet-20250219"
      system_prompt: "You are a helpful assistant."
      tools: []

  workflows:
    default:
      pattern: "{template if template != 'custom' else 'swarm'}"
      steps:
        - spawn_agents:
            template: "worker"
            count: 3
        - collect_results:
            timeout: 300
"""

        (project_path / "configs" / "swarm.yml").write_text(config_content)

        # Create .env.example
        env_content = """# Claude Agent Swarm Environment Variables
ANTHROPIC_API_KEY=your_api_key_here
"""
        (project_path / ".env.example").write_text(env_content)

        console.print(
            Panel(
                f"[green]Project '{name}' initialized successfully![/green]\n\n"
                f"Next steps:\n"
                f"  1. cd {name}\n"
                f"  2. cp .env.example .env\n"
                f"  3. Add your ANTHROPIC_API_KEY to .env\n"
                f"  4. claude-swarm run --config configs/swarm.yml --task 'Your task'",
                title="Success",
            )
        )
        return 0

    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to initialize project: {e}")
        return 1


async def run_swarm(config_path: str, task: Optional[str], pattern: str) -> int:
    """Run a swarm from configuration."""
    from . import SwarmOrchestrator, ConfigLoader

    config_file = Path(config_path)
    if not config_file.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config_path}")
        return 1

    try:
        console.print(f"[blue]Loading configuration from {config_path}...[/blue]")

        # Load config and create orchestrator
        orchestrator = await SwarmOrchestrator.create(
            config_path=config_file,
            load_config=True,
        )

        # Create swarm
        swarm_id = await orchestrator.create_swarm(pattern=pattern)
        console.print(f"[green]Swarm created:[/green] {swarm_id}")

        if task:
            console.print(f"[blue]Executing task:[/blue] {task[:100]}...")

            # Execute task
            result = await orchestrator.execute_task(swarm_id, task)

            if result.get("success", False):
                console.print("[green]Task completed successfully![/green]")
                console.print(result.get("content", "")[:500])
            else:
                console.print(f"[red]Task failed:[/red] {result.get('error', 'Unknown error')}")
                return 1
        else:
            console.print("[yellow]No task specified. Swarm is ready.[/yellow]")

        # Cleanup
        await orchestrator.close()
        return 0

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1


def monitor_swarm(swarm_id: Optional[str]) -> int:
    """Monitor running swarms."""
    console.print("[yellow]Monitoring feature coming soon...[/yellow]")
    console.print("Use the SwarmDashboard for real-time monitoring.")
    return 0


def export_results(format: str, output: str) -> int:
    """Export swarm results."""
    console.print("[yellow]Export feature coming soon...[/yellow]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
