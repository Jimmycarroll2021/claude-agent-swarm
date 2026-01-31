#!/usr/bin/env python3
"""
Example 3: Code Generation Pipeline

Demonstrates a sequential pipeline for code generation, review, and testing.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY not set.")
    exit(1)


async def code_pipeline_example():
    """Run a code generation pipeline."""
    from claude_agent_swarm import SwarmOrchestrator
    from claude_agent_swarm.patterns import PipelinePattern
    
    print("=" * 70)
    print("Example 3: Code Generation Pipeline")
    print("=" * 70)
    
    orchestrator = SwarmOrchestrator()
    
    # Define the code generation task
    project_description = """
Create a Python class `TaskQueue` that:
1. Manages a queue of tasks with priorities
2. Supports adding, removing, and retrieving tasks
3. Is thread-safe using asyncio
4. Has methods: add_task(), get_next(), peek(), is_empty(), size()
5. Includes proper error handling and type hints
"""
    
    print(f"\nProject: Task Queue Implementation")
    print("Pipeline: Design → Implement → Review → Test\n")
    
    # Create pipeline
    pipeline = PipelinePattern(orchestrator)
    
    # Stage 1: Design
    print("Stage 1: Design")
    print("-" * 40)
    
    design_result = await pipeline.execute_stage(
        "design",
        f"Design the architecture for: {project_description}",
        agent_config={
            "model": "claude-3-7-sonnet-20250219",
            "system_prompt": "You are a software architect. Design clean, well-structured solutions with clear interfaces."
        }
    )
    
    if not design_result.success:
        print(f"Design failed: {design_result.error}")
        return
    
    design = design_result.data
    print(f"✓ Design completed\n")
    
    # Stage 2: Implementation (parallel - 3 implementations)
    print("Stage 2: Implementation (3 parallel implementations)")
    print("-" * 40)
    
    implementations = []
    for i in range(3):
        impl_result = await pipeline.execute_stage(
            f"implement_{i}",
            f"Implement the TaskQueue class based on this design:\n{design}",
            agent_config={
                "model": "claude-3-7-sonnet-20250219",
                "system_prompt": "You are an expert Python developer. Write clean, efficient, well-documented code."
            }
        )
        if impl_result.success:
            implementations.append(impl_result.data)
            print(f"✓ Implementation {i+1} completed")
        else:
            print(f"✗ Implementation {i+1} failed: {impl_result.error}")
    
    if not implementations:
        print("All implementations failed")
        return
    
    # Stage 3: Review
    print("\nStage 3: Review")
    print("-" * 40)
    
    review_tasks = [
        f"Review this implementation for correctness and best practices:\n{impl}"
        for impl in implementations
    ]
    
    from claude_agent_swarm.patterns import SwarmPattern
    swarm = SwarmPattern(orchestrator, max_agents=3)
    
    review_results = await swarm.execute_parallel(
        review_tasks,
        agent_template={
            "model": "claude-3-7-sonnet-20250219",
            "system_prompt": "You are a code reviewer. Check for correctness, efficiency, and best practices."
        }
    )
    
    # Select best implementation
    best_impl = None
    best_score = -1
    
    for i, (impl, review) in enumerate(zip(implementations, review_results)):
        if review.success:
            # Simple scoring based on review positivity
            score = review.data.lower().count("good") + review.data.lower().count("correct")
            print(f"Implementation {i+1} score: {score}")
            if score > best_score:
                best_score = score
                best_impl = impl
    
    if not best_impl:
        best_impl = implementations[0]
    
    print(f"✓ Best implementation selected\n")
    
    # Stage 4: Testing
    print("Stage 4: Testing")
    print("-" * 40)
    
    test_result = await pipeline.execute_stage(
        "test",
        f"Write comprehensive unit tests for this implementation:\n{best_impl}",
        agent_config={
            "model": "claude-3-7-sonnet-20250219",
            "system_prompt": "You are a QA engineer. Write thorough unit tests covering edge cases."
        }
    )
    
    if test_result.success:
        print(f"✓ Tests generated\n")
    else:
        print(f"✗ Test generation failed: {test_result.error}\n")
    
    # Display final output
    print("=" * 70)
    print("FINAL OUTPUT")
    print("=" * 70)
    print("\n--- Design ---")
    print(design[:500] + "..." if len(design) > 500 else design)
    print("\n--- Implementation ---")
    print(best_impl[:800] + "..." if len(str(best_impl)) > 800 else best_impl)
    print("\n--- Tests ---")
    if test_result.success:
        print(test_result.data[:500] + "..." if len(test_result.data) > 500 else test_result.data)
    
    await orchestrator.terminate_all()


async def main():
    """Main entry point."""
    try:
        await code_pipeline_example()
        print("\n✓ Code pipeline completed!")
    except Exception as e:
        print(f"\n✗ Code pipeline failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
