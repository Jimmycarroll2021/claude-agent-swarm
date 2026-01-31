#!/usr/bin/env python3
"""
Example 1: Simple Parallel Execution

Demonstrates basic parallel task execution using the Swarm pattern.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure API key is set
if not os.getenv("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY not set. Please set it in your .env file.")
    exit(1)


async def simple_parallel_example():
    """Run simple parallel tasks."""
    from claude_agent_swarm import SwarmOrchestrator
    from claude_agent_swarm.patterns import SwarmPattern
    
    print("=" * 60)
    print("Example 1: Simple Parallel Execution")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = SwarmOrchestrator()
    
    # Create swarm pattern
    swarm = SwarmPattern(orchestrator, max_agents=5)
    
    # Define parallel tasks
    tasks = [
        "What are the key benefits of Python async/await?",
        "Explain the difference between concurrency and parallelism",
        "What are Python coroutines and how do they work?",
        "Describe Python's asyncio event loop",
        "What are the best practices for async Python code?"
    ]
    
    print(f"\nExecuting {len(tasks)} tasks in parallel...\n")
    
    # Execute tasks in parallel
    results = await swarm.execute_parallel(
        tasks,
        agent_template={
            "model": "claude-3-7-sonnet-20250219",
            "system_prompt": "You are a Python expert. Provide concise, accurate answers."
        }
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    
    for i, (task, result) in enumerate(zip(tasks, results), 1):
        print(f"\n--- Task {i}: {task[:50]}... ---")
        if result.success:
            print(f"✓ Success (Agent: {result.agent_id})")
            print(f"Response: {str(result.data)[:200]}...")
        else:
            print(f"✗ Failed: {result.error}")
    
    # Print summary
    successful = sum(1 for r in results if r.success)
    print(f"\n{'=' * 60}")
    print(f"Summary: {successful}/{len(results)} tasks successful")
    print(f"Execution time: {sum(r.execution_time_ms for r in results)/1000:.2f}s total")
    
    # Cleanup
    await orchestrator.terminate_all()
    
    return results


async def main():
    """Main entry point."""
    try:
        results = await simple_parallel_example()
        print("\n✓ Example completed successfully!")
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
