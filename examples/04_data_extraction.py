#!/usr/bin/env python3
"""
Example 4: Data Extraction from Multiple Sources

Demonstrates extracting and consolidating data from multiple sources in parallel.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY not set.")
    exit(1)


async def data_extraction_example():
    """Run data extraction from multiple sources."""
    from claude_agent_swarm import SwarmOrchestrator
    from claude_agent_swarm.patterns import SwarmPattern, CouncilPattern
    
    print("=" * 70)
    print("Example 4: Data Extraction from Multiple Sources")
    print("=" * 70)
    
    orchestrator = SwarmOrchestrator()
    
    # Define extraction tasks from different sources
    extraction_tasks = [
        {
            "source": "Company Website",
            "task": "Extract company information for Anthropic: founding date, founders, funding rounds, key products, employee count"
        },
        {
            "source": "LinkedIn/TechCrunch",
            "task": "Extract recent news and milestones for Anthropic from 2024-2025"
        },
        {
            "source": "Product Documentation",
            "task": "Extract key features and capabilities of Claude AI models"
        },
        {
            "source": "Industry Analysis",
            "task": "Extract competitive positioning of Anthropic vs OpenAI, Google"
        },
        {
            "source": "Research Papers",
            "task": "Extract key technical innovations from Anthropic's research"
        }
    ]
    
    print(f"\nExtracting data from {len(extraction_tasks)} sources...\n")
    
    # Phase 1: Parallel extraction
    print("Phase 1: Parallel Data Extraction")
    print("-" * 40)
    
    swarm = SwarmPattern(orchestrator, max_agents=len(extraction_tasks))
    
    extraction_results = await swarm.execute_parallel(
        [t["task"] for t in extraction_tasks],
        agent_template={
            "model": "claude-3-7-sonnet-20250219",
            "system_prompt": """You are a data extraction specialist. Extract structured information:
1. Use clear headings and bullet points
2. Include specific numbers, dates, and facts
3. Note uncertainty with [uncertain] tag
4. Format as structured data
5. Be concise but comprehensive""",
            "tools": ["web_search"]
        },
        timeout=90
    )
    
    # Collect extracted data
    extracted_data = []
    for task_info, result in zip(extraction_tasks, extraction_results):
        if result.success:
            extracted_data.append({
                "source": task_info["source"],
                "data": result.data
            })
            print(f"✓ {task_info['source']}")
        else:
            print(f"✗ {task_info['source']} - {result.error}")
    
    # Phase 2: Consolidation via Council
    print("\nPhase 2: Data Consolidation (Council Pattern)")
    print("-" * 40)
    
    council = CouncilPattern(orchestrator)
    
    # Add consolidation perspectives
    consolidation_task = f"""Consolidate the following data into a unified company profile:

{chr(10).join(f"### Source: {d['source']}\n{d['data']}\n" for d in extracted_data)}

Create a structured profile with:
1. Company Overview
2. Key Metrics (founded, employees, funding)
3. Product Portfolio
4. Recent Developments
5. Competitive Position
6. Data Confidence Score
"""
    
    # Add different consolidation strategies
    council.add_perspective(
        "structural",
        agent_config={
            "model": "claude-3-7-sonnet-20250219",
            "system_prompt": "Focus on organizing data into clear categories and hierarchies."
        },
        perspective_description="Structural organization"
    )
    
    council.add_perspective(
        "analytical",
        agent_config={
            "model": "claude-3-7-sonnet-20250219",
            "system_prompt": "Focus on insights, trends, and implications from the data."
        },
        perspective_description="Analytical insights"
    )
    
    council.add_perspective(
        "validation",
        agent_config={
            "model": "claude-3-7-sonnet-20250219",
            "system_prompt": "Focus on data consistency, conflicts, and confidence levels."
        },
        perspective_description="Data validation"
    )
    
    consolidation_result = await council.convene_council(consolidation_task)
    
    # Display final profile
    print("\n" + "=" * 70)
    print("CONSOLIDATED COMPANY PROFILE")
    print("=" * 70)
    
    if consolidation_result.success:
        print(consolidation_result.data)
    else:
        print(f"Consolidation failed: {consolidation_result.error}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Sources attempted: {len(extraction_tasks)}")
    print(f"Sources successful: {len(extracted_data)}")
    print(f"Success rate: {len(extracted_data)/len(extraction_tasks)*100:.1f}%")
    print(f"Total extraction time: {sum(r.execution_time_ms for r in extraction_results)/1000:.2f}s")
    
    await orchestrator.terminate_all()
    
    return consolidation_result


async def main():
    """Main entry point."""
    try:
        result = await data_extraction_example()
        print("\n✓ Data extraction completed!")
    except Exception as e:
        print(f"\n✗ Data extraction failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
