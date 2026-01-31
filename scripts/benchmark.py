#!/usr/bin/env python3
"""
Performance benchmarking script for Claude Agent Swarm.

Compares single-agent vs multi-agent performance on various tasks.
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
import argparse
import json
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    task_count: int
    agent_count: int
    total_time_ms: float
    avg_time_per_task_ms: float
    success_rate: float
    speedup_vs_single: float


class SwarmBenchmark:
    """Benchmark suite for agent swarm."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.results: List[BenchmarkResult] = []
    
    async def run_single_agent(self, tasks: List[str]) -> Dict[str, Any]:
        """Run tasks sequentially with a single agent."""
        from claude_agent_swarm import SwarmOrchestrator
        
        orchestrator = SwarmOrchestrator(api_key=self.api_key)
        
        start = time.time()
        results = []
        
        for task in tasks:
            # Execute sequentially
            result = await orchestrator.execute_task(task)
            results.append(result)
        
        elapsed = (time.time() - start) * 1000
        
        await orchestrator.terminate_all()
        
        return {
            "time_ms": elapsed,
            "results": results
        }
    
    async def run_swarm(self, tasks: List[str], agent_count: int) -> Dict[str, Any]:
        """Run tasks in parallel with a swarm."""
        from claude_agent_swarm import SwarmOrchestrator
        from claude_agent_swarm.patterns import SwarmPattern
        
        orchestrator = SwarmOrchestrator(api_key=self.api_key)
        swarm = SwarmPattern(orchestrator, max_agents=agent_count)
        
        start = time.time()
        results = await swarm.execute_parallel(tasks)
        elapsed = (time.time() - start) * 1000
        
        await orchestrator.terminate_all()
        
        return {
            "time_ms": elapsed,
            "results": results
        }
    
    async def benchmark_simple_tasks(self, task_count: int = 10) -> BenchmarkResult:
        """Benchmark simple parallel tasks."""
        tasks = [
            f"What is {i} + {i}?" 
            for i in range(task_count)
        ]
        
        print(f"\nBenchmarking {task_count} simple tasks...")
        
        # Single agent
        print("  Running with single agent...")
        single_result = await self.run_single_agent(tasks)
        
        # Swarm
        print("  Running with swarm (10 agents)...")
        swarm_result = await self.run_swarm(tasks, 10)
        
        # Calculate speedup
        speedup = single_result["time_ms"] / swarm_result["time_ms"]
        
        success_count = sum(1 for r in swarm_result["results"] if r.success)
        
        result = BenchmarkResult(
            name="Simple Tasks",
            task_count=task_count,
            agent_count=10,
            total_time_ms=swarm_result["time_ms"],
            avg_time_per_task_ms=swarm_result["time_ms"] / task_count,
            success_rate=success_count / task_count,
            speedup_vs_single=speedup
        )
        
        self.results.append(result)
        return result
    
    async def benchmark_research_tasks(self, task_count: int = 5) -> BenchmarkResult:
        """Benchmark research tasks."""
        topics = [
            "Python asyncio best practices",
            "Machine learning deployment patterns",
            "API design principles",
            "Database optimization techniques",
            "Cloud architecture patterns"
        ]
        
        tasks = [
            f"Research: {topic}" 
            for topic in topics[:task_count]
        ]
        
        print(f"\nBenchmarking {task_count} research tasks...")
        
        # Single agent
        print("  Running with single agent...")
        single_result = await self.run_single_agent(tasks)
        
        # Swarm
        print("  Running with swarm (5 agents)...")
        swarm_result = await self.run_swarm(tasks, 5)
        
        speedup = single_result["time_ms"] / swarm_result["time_ms"]
        success_count = sum(1 for r in swarm_result["results"] if r.success)
        
        result = BenchmarkResult(
            name="Research Tasks",
            task_count=task_count,
            agent_count=5,
            total_time_ms=swarm_result["time_ms"],
            avg_time_per_task_ms=swarm_result["time_ms"] / task_count,
            success_rate=success_count / task_count,
            speedup_vs_single=speedup
        )
        
        self.results.append(result)
        return result
    
    async def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        print("=" * 60)
        print("Claude Agent Swarm Performance Benchmarks")
        print("=" * 60)
        
        await self.benchmark_simple_tasks(task_count=10)
        await self.benchmark_research_tasks(task_count=5)
        
        return self.results
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("Benchmark Summary")
        print("=" * 60)
        
        for result in self.results:
            print(f"\n{result.name}:")
            print(f"  Tasks: {result.task_count}")
            print(f"  Agents: {result.agent_count}")
            print(f"  Total Time: {result.total_time_ms/1000:.2f}s")
            print(f"  Avg/Task: {result.avg_time_per_task_ms/1000:.2f}s")
            print(f"  Success Rate: {result.success_rate*100:.1f}%")
            print(f"  Speedup: {result.speedup_vs_single:.2f}x")
    
    def export_results(self, filename: str):
        """Export results to JSON."""
        data = [asdict(r) for r in self.results]
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nResults exported to {filename}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark Claude Agent Swarm")
    parser.add_argument("--api-key", help="Anthropic API key")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: API key required. Set ANTHROPIC_API_KEY or use --api-key")
        return 1
    
    benchmark = SwarmBenchmark(api_key)
    await benchmark.run_all()
    benchmark.print_summary()
    benchmark.export_results(args.output)
    
    return 0


if __name__ == "__main__":
    import os
    import sys
    sys.exit(asyncio.run(main()))
