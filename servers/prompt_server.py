"""
MCP Server with Prompts and Tools
This server provides prompts for analysis and historical reports,
along with tools to execute these prompts.
"""

from fastmcp import FastMCP

mcp = FastMCP("Prompt")


# =====================================================
# PROMPTS
# =====================================================

@mcp.prompt
def get_prompt(topic: str) -> str:
    """
    Returns a prompt that will do a detailed analysis on a topic
    Args:
        topic: the topic to do research on
    """
    return f"Do a detailed analysis on the following topic: {topic}"


@mcp.prompt
def write_detailed_historical_report(topic: str, number_of_paragraphs: int) -> str:
    """
    Writes a detailed historical report
    Args:
        topic: the topic to do research on
        number_of_paragraphs: the number of paragraphs that the main body should be 
    """

    prompt = """
    Create a concise research report on the history of {topic}. 
    The report should contain 3 sections: INTRODUCTION, MAIN, and CONCLUSION.
    The MAIN section should be {number_of_paragraphs} paragraphs long. 
    Include a timeline of key events.
    The conclusion should be in bullet points format. 
    """

    prompt = prompt.format(topic=topic, number_of_paragraphs=number_of_paragraphs)

    return prompt


# =====================================================
# TOOLS
# =====================================================

@mcp.tool
def analyze_topic(topic: str) -> str:
    """
    Executes a detailed analysis on a given topic using the get_prompt function.
    
    Args:
        topic: The topic to analyze
        
    Returns:
        The analysis prompt for the topic
    """
    prompt = get_prompt(topic)
    return f"Analysis Prompt Generated:\n{prompt}\n\nUse this prompt with an LLM to get detailed analysis."


@mcp.tool
def generate_historical_report(topic: str, paragraphs: int = 5) -> str:
    """
    Generates a detailed historical report prompt for a given topic.
    
    Args:
        topic: The historical topic to research
        paragraphs: Number of paragraphs for the main section (default: 5)
        
    Returns:
        The historical report prompt
    """
    """prompt =
    Create a concise research report on the history of {topic}. 
    The report should contain 3 sections: INTRODUCTION, MAIN, and CONCLUSION.
    The MAIN section should be {paragraphs} paragraphs long. 
    Include a timeline of key events.
    The conclusion should be in bullet points format. 
    """

    if paragraphs < 1:
        return "Error: Number of paragraphs must be at least 1"
    
    if paragraphs > 20:
        return "Warning: Consider using fewer paragraphs for readability (max recommended: 20)"
    
    prompt = write_detailed_historical_report(topic, paragraphs)
    return f"Historical Report Prompt Generated:\n{prompt}\n\nUse this prompt with an LLM to generate the report."


@mcp.tool
def get_available_prompts() -> str:
    """
    Returns a list of all available prompts and their descriptions.
    
    Returns:
        A formatted string listing all available prompts
    """
    prompts_info = """
    Available Prompts:
    
    1. get_prompt(topic: str)
       - Generates a prompt for detailed analysis on a topic
       - Returns a prompt that instructs an LLM to analyze the topic
       
    2. write_detailed_historical_report(topic: str, number_of_paragraphs: int)
       - Generates a prompt for creating a detailed historical report
       - Returns a prompt structured with INTRODUCTION, MAIN (configurable paragraphs), and CONCLUSION sections
       - Includes a timeline of key events
    
    Available Tools:
    
    1. analyze_topic(topic: str)
       - Executes the get_prompt function and returns the analysis prompt
       
    2. generate_historical_report(topic: str, paragraphs: int = 5)
       - Executes the write_detailed_historical_report function
       - Validates paragraph count and returns the report prompt
       
    3. get_available_prompts()
       - Returns this list of available prompts and tools
    """
    return prompts_info


@mcp.tool
def format_analysis_request(topic: str, depth: str = "comprehensive") -> str:
    """
    Formats an analysis request with customizable depth levels.
    
    Args:
        topic: The topic to analyze
        depth: Level of analysis - "quick", "standard", or "comprehensive" (default: "comprehensive")
        
    Returns:
        A formatted analysis prompt with the specified depth
    """
    depth_instructions = {
        "quick": "Provide a brief overview (2-3 paragraphs)",
        "standard": "Provide a moderate analysis (4-5 paragraphs)",
        "comprehensive": "Provide a detailed analysis covering all aspects"
    }
    
    instruction = depth_instructions.get(depth, depth_instructions["standard"])
    prompt = get_prompt(topic)
    
    return f"{prompt}\n\nDepth Level: {instruction}"


@mcp.tool
def batch_report_generator(topics: list, paragraphs_per_topic: int = 3) -> str:
    """
    Generates historical report prompts for multiple topics at once.
    
    Args:
        topics: A list of topics to generate reports for
        paragraphs_per_topic: Number of paragraphs per report (default: 3)
        
    Returns:
        A formatted string with prompts for all topics
    """
    if not topics:
        return "Error: Please provide at least one topic"
    
    if paragraphs_per_topic < 1:
        return "Error: Paragraphs per topic must be at least 1"
    
    batch_prompts = f"Batch Historical Report Generation ({len(topics)} topics)\n"
    batch_prompts += "=" * 50 + "\n\n"
    
    for i, topic in enumerate(topics, 1):
        prompt = write_detailed_historical_report(topic, paragraphs_per_topic)
        batch_prompts += f"Report #{i}: {topic.upper()}\n"
        batch_prompts += "-" * 30 + "\n"
        batch_prompts += prompt + "\n\n"
    
    return batch_prompts


if __name__ == "__main__":
    mcp.run()
