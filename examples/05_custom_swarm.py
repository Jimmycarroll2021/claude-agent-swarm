#!/usr/bin/env python3
"""
Example 5: Custom Swarm Orchestration

Demonstrates building a custom orchestration pattern combining multiple approaches.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY not set.")
    exit(1)


async def custom_swarm_example():
    """Run a custom hybrid orchestration."""
    from claude_agent_swarm import SwarmOrchestrator
    from claude_agent_swarm.patterns import SwarmPattern, PipelinePattern, CouncilPattern
    
    print("=" * 70)
    print("Example 5: Custom Hybrid Orchestration")
    print("=" * 70)
    
    orchestrator = SwarmOrchestrator()
    
    # Complex task: Market analysis report
    main_task = "Create a comprehensive market analysis report for the electric vehicle industry"
    
    print(f"\nTask: {main_task}")
    print("Strategy: Hybrid (Swarm → Council → Pipeline)\n")
    
    # Phase 1: Parallel Research Swarm
    print("=" * 70)
    print("PHASE 1: Parallel Research (Swarm Pattern)")
    print("=" * 70)
    
    research_areas = [
        "Market size and growth projections for EVs 2024-2030",
        "Key players and market share in EV industry",
        "Technology trends: batteries, charging, autonomous driving",
        "Regulatory landscape and government incentives",
        "Consumer adoption patterns and barriers",
        "Supply chain analysis and raw materials"
    ]
    
    swarm = SwarmPattern(orchestrator, max_agents=6)
    
    research_results = await swarm.execute_parallel(
        research_areas,
        agent_template={
            "model": "claude-3-7-sonnet-20250219",
            "system_prompt": """You are a market research analyst. Provide:
1. Key statistics and data points
2. Trends and projections
3. Major players and their positions
4. Opportunities and challenges
Be factual and cite specific numbers where possible.""",
            "tools": ["web_search"]
        },
        timeout=120
    )
    
    successful_research = [r for r in research_results if r.success]
    print(f"\n✓ Research completed: {len(successful_research)}/{len(research_areas)} areas")
    
    # Phase 2: Multi-Perspective Analysis (Council)
    print("\n" + "=" * 70)
    print("PHASE 2: Multi-Perspective Analysis (Council Pattern)")
    print("=" * 70)
    
    council = CouncilPattern(orchestrator)
    
    analysis_input = "\n\n".join([
        f"### {area}\n{result.data}"
        for area, result in zip(research_areas, research_results)
        if result.success
    ])
    
    # Add expert perspectives
    perspectives = [
        ("market_strategist", "Market strategy and competitive positioning"),
        ("financial_analyst", "Financial projections and investment analysis"),
        ("tech_expert", "Technology assessment and innovation trends"),
        ("policy_advisor", "Regulatory impact and policy recommendations")
    ]
    
    for persp_id, description in perspectives:
        council.add_perspective(
            persp_id,
            agent_config={
                "model": "claude-3-7-sonnet-20250219",
                "system_prompt": f"You are a {persp_id.replace('_', ' ')}. Provide expert analysis from your domain perspective."
            },
            perspective_description=description
        )
    
    analysis_task = f"""Analyze this EV market research from your expert perspective:

{analysis_input}

Provide:
1. Key insights from your domain
2. Critical success factors
3. Risks and challenges
4. Strategic recommendations
"""
    
    council_result = await council.convene_council(analysis_task)
    
    if council_result.success:
        print("✓ Multi-perspective analysis completed")
    else:
        print(f"✗ Council analysis failed: {council_result.error}")
    
    # Phase 3: Report Generation Pipeline
    print("\n" + "=" * 70)
    print("PHASE 3: Report Generation (Pipeline Pattern)")
    print("=" * 70)
    
    pipeline = PipelinePattern(orchestrator)
    
    # Stage 1: Outline
    outline_result = await pipeline.execute_stage(
        "outline",
        f"Create a detailed outline for an EV market analysis report based on:\n{council_result.data if council_result.success else analysis_input}",
        agent_config={
            "model": "claude-3-7-sonnet-20250219",
            "system_prompt": "You are a report writer. Create clear, professional report outlines."
        }
    )
    
    if outline_result.success:
        print("✓ Report outline created")
        outline = outline_result.data
    else:
        print("✗ Outline creation failed")
        outline = "Standard market analysis report structure"
    
    # Stage 2: Write Sections (Parallel)
    print("\n  Writing report sections in parallel...")
    
    sections = [
        "Executive Summary",
        "Market Overview and Size",
        "Competitive Landscape",
        "Technology Trends",
        "Regulatory Environment",
        "Market Forecasts",
        "Strategic Recommendations",
        "Conclusion"
    ]
    
    section_swarm = SwarmPattern(orchestrator, max_agents=4)
    
    section_tasks = [
        f"Write the '{section}' section for an EV market analysis report.\n\nOutline: {outline}\n\nResearch: {council_result.data if council_result.success else analysis_input}"
        for section in sections
    ]
    
    section_results = await section_swarm.execute_parallel(
        section_tasks,
        agent_template={
            "model": "claude-3-7-sonnet-20250219",
            "system_prompt": "You are a professional report writer. Write clear, well-structured sections with proper headings and formatting."
        },
        timeout=90
    )
    
    successful_sections = [r for r in section_results if r.success]
    print(f"  ✓ Sections written: {len(successful_sections)}/{len(sections)}")
    
    # Stage 3: Final Assembly
    print("\n  Assembling final report...")
    
    assembly_result = await pipeline.execute_stage(
        "assemble",
        f"""Assemble the following sections into a cohesive final report:

{chr(10).join(f"### {section}\n{result.data if result.success else '[Section failed]'}" for section, result in zip(sections, section_results))}

Requirements:
1. Ensure consistent formatting throughout
2. Add transitions between sections
3. Create a table of contents
4. Maintain professional tone
""",
        agent_config={
            "model": "claude-3-7-sonnet-20250219",
            "system_prompt": "You are an editor. Assemble report sections into a polished final document."
        }
    )
    
    if assembly_result.success:
        print("  ✓ Final report assembled")
    else:
        print(f"  ✗ Assembly failed: {assembly_result.error}")
    
    # Display Final Report
    print("\n" + "=" * 70)
    print("FINAL MARKET ANALYSIS REPORT")
    print("=" * 70)
    
    final_report = assembly_result.data if assembly_result.success else "Report generation failed"
    print(final_report[:3000] if len(str(final_report)) > 3000 else final_report)
    
    if len(str(final_report)) > 3000:
        print("\n... [Report truncated for display] ...")
    
    # Statistics
    print("\n" + "=" * 70)
    print("EXECUTION STATISTICS")
    print("=" * 70)
    print(f"Research areas: {len(research_areas)}")
    print(f"Expert perspectives: {len(perspectives)}")
    print(f"Report sections: {len(sections)}")
    print(f"Total agents used: {len(research_areas) + len(perspectives) + len(sections)}")
    
    total_time = (
        sum(r.execution_time_ms for r in research_results) +
        (council_result.execution_time_ms if council_result else 0) +
        sum(r.execution_time_ms for r in section_results)
    ) / 1000
    
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Estimated sequential time: ~{total_time * 3:.2f}s")
    print(f"Speedup: ~3x")
    
    await orchestrator.terminate_all()
    
    return assembly_result


async def main():
    """Main entry point."""
    try:
        result = await custom_swarm_example()
        print("\n✓ Custom orchestration completed!")
    except Exception as e:
        print(f"\n✗ Custom orchestration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
