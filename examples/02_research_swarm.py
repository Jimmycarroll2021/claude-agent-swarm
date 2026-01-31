#!/usr/bin/env python3
"""
Example 2: Research Swarm

Demonstrates multi-source research using the Swarm pattern with result synthesis.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY not set.")
    exit(1)


async def research_swarm_example():
    """Run a research swarm."""
    from claude_agent_swarm import SwarmOrchestrator
    from claude_agent_swarm.patterns import SwarmPattern, LeaderPattern
    
    print("=" * 70)
    print("Example 2: Multi-Source Research Swarm")
    print("=" * 70)
    
    orchestrator = SwarmOrchestrator()
    
    # Define research topics
    research_topics = [
        "Latest breakthroughs in quantum computing hardware (2024-2025)",
        "Recent advances in quantum error correction",
        "Commercial applications of quantum computing",
        "Major players and investments in quantum computing",
        "Timeline predictions for quantum advantage"
    ]
    
    print(f"\nResearching: Quantum Computing")
    print(f"Topics: {len(research_topics)}")
    print(f"Agents: 1 per topic\n")
    
    # Phase 1: Parallel research
    print("Phase 1: Parallel Research")
    print("-" * 40)
    
    swarm = SwarmPattern(orchestrator, max_agents=len(research_topics))
    
    research_results = await swarm.execute_parallel(
        research_topics,
        agent_template={
            "model": "claude-3-7-sonnet-20250219",
            "system_prompt": """You are a research specialist. Your task is to:
1. Research the given topic thoroughly
2. Provide key findings, facts, and insights
3. Cite specific examples and data points
4. Keep your response structured and factual
5. Limit response to 300-500 words""",
            "tools": ["web_search"]
        },
        timeout=120
    )
    
    # Collect successful research
    research_findings = []
    for topic, result in zip(research_topics, research_results):
        if result.success:
            research_findings.append({
                "topic": topic,
                "findings": result.data
            })
            print(f"✓ {topic[:50]}...")
        else:
            print(f"✗ {topic[:50]}... - {result.error}")
    
    # Phase 2: Synthesis
    print("\nPhase 2: Synthesis")
    print("-" * 40)
    
    leader = LeaderPattern(orchestrator)
    
    synthesis_task = f"""Synthesize the following research findings into a comprehensive overview:

{chr(10).join(f"### {f['topic']}\n{f['findings']}\n" for f in research_findings)}

Create a well-structured report covering:
1. Executive Summary
2. Key Findings by Area
3. Trends and Implications
4. Conclusions
"""
    
    synthesis_result = await leader.delegate(
        synthesis_task,
        specialist_type="synthesizer",
        agent_config={
            "model": "claude-3-7-sonnet-20250219",
            "system_prompt": "You are an expert analyst. Synthesize research findings into clear, actionable insights."
        }
    )
    
    # Display final report
    print("\n" + "=" * 70)
    print("FINAL RESEARCH REPORT")
    print("=" * 70)
    
    if synthesis_result.success:
        print(synthesis_result.data)
    else:
        print(f"Synthesis failed: {synthesis_result.error}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Research agents: {len(research_topics)}")
    print(f"Successful research tasks: {len(research_findings)}")
    print(f"Total execution time: {sum(r.execution_time_ms for r in research_results)/1000:.2f}s")
    
    await orchestrator.terminate_all()
    
    return synthesis_result


async def main():
    """Main entry point."""
    try:
        result = await research_swarm_example()
        print("\n✓ Research swarm completed!")
    except Exception as e:
        print(f"\n✗ Research swarm failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
