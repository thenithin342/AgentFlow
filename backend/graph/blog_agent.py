"""
Blog Writer agent node — Phase 9E.

Produces a structured, SEO-friendly blog post given a topic from the user.
The agent:
    1. Researches the topic with tavily_search (2-3 targeted queries)
    2. Optionally pulls from uploaded documents via retrieve_documents
    3. Generates a structured JSON blog post: {title, meta_description,
       tags, sections: [{heading, content}]}
    4. Writes the structured data into `state["blog_output"]`
    5. Also writes a plain-text `agent_output` for the synthesizer / chat rail

The blog agent bypasses the synthesizer (similar to chat_agent) — it sets
`final_response` to a short summary text so the synthesizer is skipped, and
the rich blog_output is consumed directly by the frontend Blog tab.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.runnables import RunnableConfig

from backend.graph.messages import (
    content_to_str,
    extract_sources,
    get_msg_content,
    is_ai_message,
)
from backend.graph.state import AgentState
from backend.graph.tools import make_retrieve_documents_tool, tavily_search

logger = logging.getLogger("agentflow.blog")


# ---------------------------------------------------------------------------
# Blog writer system prompt
# ---------------------------------------------------------------------------

BLOG_WRITER_PROMPT = """\
You are AgentFlow's Blog Writer — a skilled content strategist and writer who \
produces structured, SEO-optimised blog posts.

## Your workflow
1. Use `tavily_search` to research the topic thoroughly (2-3 targeted queries).
2. Use `retrieve_documents` if the user has uploaded relevant files.
3. After gathering information, write the full blog post.

## Output format — CRITICAL
After your research, output EXACTLY this JSON structure (no markdown fences, \
no extra text before or after):

{
  "title": "Your engaging, SEO-optimised title (max 70 chars)",
  "meta_description": "A compelling 150-160 char meta description for search engines",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
  "sections": [
    {
      "heading": "Introduction",
      "content": "2-3 paragraph introduction that hooks the reader..."
    },
    {
      "heading": "Section Heading",
      "content": "Full markdown content for this section with **bold**, lists, etc."
    }
  ]
}

## Writing standards
- 800-1500 words total across all sections
- 5-7 sections minimum: Introduction, 3-4 body sections, Conclusion, Call-to-Action
- Use H2-level headings (section.heading is an H2, never H1)
- Each section content should be 150-300 words
- Use markdown within section content: **bold**, *italic*, bullet lists, numbered lists
- Cite sources inline as [Source](url) when referencing web results
- SEO: include the main keyword in the title, first paragraph, and 2-3 headings
- Tone: informative, engaging, professional — avoid corporate jargon

## Research strategy
- Run targeted queries like "topic + key concepts 2024" or "topic + best practices"
- Look for: statistics, expert opinions, real-world examples, case studies
- Synthesise — don't dump raw search results into the post
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_blog_json(text: str) -> dict | None:
    """Extract and parse the JSON blog structure from the LLM output.

    The LLM may wrap it in ```json ... ``` fences or emit it bare.
    We try progressively looser parsing strategies.
    """
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    # Find the outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None

    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Try to fix common LLM JSON mistakes: trailing commas
        candidate_fixed = re.sub(r",(\s*[}\]])", r"\1", candidate)
        try:
            parsed = json.loads(candidate_fixed)
        except json.JSONDecodeError:
            return None

    # Validate required fields
    if not isinstance(parsed, dict):
        return None
    if "title" not in parsed or "sections" not in parsed:
        return None

    return parsed


def _blog_to_markdown(blog: dict) -> str:
    """Convert the structured blog dict to plain markdown for agent_output."""
    parts: list[str] = []
    parts.append(f"# {blog.get('title', 'Blog Post')}")
    meta = blog.get("meta_description", "")
    if meta:
        parts.append(f"*{meta}*")
    tags = blog.get("tags") or []
    if tags:
        parts.append(f"**Tags:** {', '.join(tags)}")
    parts.append("")
    for section in blog.get("sections") or []:
        heading = section.get("heading", "")
        content = section.get("content", "")
        if heading:
            parts.append(f"## {heading}")
        if content:
            parts.append(content)
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def blog_writer_node(state: AgentState, config: RunnableConfig) -> dict:
    """ReAct blog writer agent with tavily_search + retrieve_documents.

    Returns:
        blog_output: structured dict {title, meta_description, tags, sections}
        agent_output: plain markdown of the blog (for synthesizer / fallback)
        final_response: short summary text that skips synthesizer
        sources: URLs collected during research
    """
    from backend.graph.agents import _get_cached_agent, _thread_id_from_config

    thread_id = _thread_id_from_config(config)
    rag_tool = make_retrieve_documents_tool(thread_id)

    from backend.llm import llm_smart
    agent = _get_cached_agent(
        [tavily_search, rag_tool],
        llm_smart,
        prompt=BLOG_WRITER_PROMPT,
        thread_id=f"{thread_id}:blog",  # separate cache key from research agent
    )

    try:
        result = agent.invoke({"messages": state["messages"]}, config=config)
    except Exception:
        logger.warning("[blog_writer] agent invocation failed", exc_info=True)
        return {
            "agent_output": "Blog generation failed. Please try again.",
            "final_response": "Blog generation failed. Please try again.",
            "blog_output": None,
            "sources": [],
        }

    result_messages = result.get("messages") or []
    sources = extract_sources(result_messages)

    # Extract the last AI message (the blog JSON)
    blog_text = ""
    for m in reversed(result_messages):
        if is_ai_message(m):
            blog_text = content_to_str(get_msg_content(m))
            break

    blog_dict = _parse_blog_json(blog_text)

    if blog_dict:
        markdown = _blog_to_markdown(blog_dict)
        summary = (
            f"✍️ Blog post created: **{blog_dict.get('title', 'Untitled')}** "
            f"({len(blog_dict.get('sections', []))} sections). "
            "View it in the Blog tab."
        )
    else:
        # Fallback: treat the raw output as plain markdown
        logger.warning("[blog_writer] failed to parse structured JSON from output")
        markdown = blog_text
        summary = "Blog post generated. View it in the Blog tab."
        # Build a minimal blog_dict so the frontend still has something to render
        blog_dict = {
            "title": "Blog Post",
            "meta_description": "",
            "tags": [],
            "sections": [{"heading": "Content", "content": blog_text}],
        }

    return {
        "agent_output": markdown,
        "final_response": summary,
        "blog_output": blog_dict,
        "sources": sources,
    }
