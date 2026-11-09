import os
import sys
import json
from pathlib import Path
from typing import Any
import certifi
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq


os.environ["SSL_CERT_FILE"]=certifi.where()
os.environ["REQUESTS_CA_BUNDLE"]=certifi.where()

load_dotenv()


TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY=os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY=os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")





# Automatically find the current project folder.
# This replaces the hard-coded Windows paths.
PROJECT_DIR = Path(__file__).resolve().parent
WEATHER_SERVER_PATH = (
    PROJECT_DIR / "weather_mcp_server.py"
    if (PROJECT_DIR / "weather_mcp_server.py").exists()
    else PROJECT_DIR / "custom_weather_mcp_server.py"
)



# Preserve the complete Windows environment when starting
# local stdio MCP servers.
AVIATION_ENV = os.environ.copy()
AVIATION_ENV["AVIATION_STACK_API_KEY"] = (
    AVIATION_STACK_API_KEY or ""
)

WEATHER_ENV = os.environ.copy()
WEATHER_ENV["OPENWEATHER_API_KEY"] = (
    OPENWEATHER_API_KEY or ""
)




# ==========================================
# LLM
# ==========================================

llm=ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY
)




# ==========================================
# MCP client configuration
# ==========================================
client=MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": (
                "https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={TAVILY_API_KEY}"
            )
        },
        
        "Aviationstack": {
            "transport":"stdio",
            "command": "uvx",
            "args": [
                "aviationstack-mcp"
            ],
            "env":AVIATION_ENV
        },

        # Start the local weather MCP server as a child process. Using the
        # current interpreter keeps the connection inside this project's
        # virtual environment and avoids relying on a global `python`.
        "weather": {
            "transport": "stdio",
            "command": sys.executable,
             # Automatically use custom_weather_mcp_server.py
            # from the current project directory.
            "args": [
                str(WEATHER_SERVER_PATH)
            ],

            "env": WEATHER_ENV
        }
        
    }
)





# ==========================================
# Diagnostic function
# ==========================================
async def get_all_tools():
    """
    Load each MCP server separately.

    A broken server will no longer prevent the other
    working servers from loading.
    """

    all_tools = []

    for server_name in (
        "tavily",
        "Aviationstack",
        "weather"
    ):
        try:
            tools = await client.get_tools(
                server_name=server_name
            )

            all_tools.extend(tools)

            print(
                f"\nAvailable tools from "
                f"{server_name} MCP:\n"
            )

            for tool in tools:
                print(tool.name)

        except Exception as error:
            print(
                f"\nCould not connect to "
                f"{server_name} MCP:\n{error}\n"
            )

    return all_tools




# ==========================================
# Tavily MCP tool
# ==========================================

search_tool = None


async def initialize_mcp():
    """
    Initialize only Tavily.

    Previously this function initialized all MCP servers,
    so an AviationStack or Weather failure also caused
    Tavily hotel search to fail.
    """

    global search_tool

    if search_tool is not None:
        return

    tools = await client.get_tools(
        server_name="tavily"
    )

    tools_by_name = {
        tool.name: tool
        for tool in tools
    }

    search_tool = tools_by_name.get(
        "tavily_search"
    )

    if search_tool is None:
        available_tools = ", ".join(
            tools_by_name.keys()
        )

        raise RuntimeError(
            "Tavily MCP connected, but the "
            "'tavily_search' tool was not found. "
            f"Available tools: "
            f"{available_tools or 'none'}"
        )
        
        
        
def slice_hotel_results(
    raw_result: Any,
    max_items: int = 5,
    max_snippet_len: int = 250,
) -> str:
    """
    Slice and format hotel/Tavily search results to the top 3-5 items to stay well
    within LLM token limits (e.g. 8,000 token limit) rather than dumping full web responses.
    """
    if not raw_result:
        return "No hotel information available."

    # Handle LangChain MCP content block lists (e.g. [{'type': 'text', 'text': ...}])
    if isinstance(raw_result, list):
        if len(raw_result) > 0 and isinstance(raw_result[0], dict) and "text" in raw_result[0]:
            raw_result = "\n".join(
                str(block["text"])
                for block in raw_result
                if isinstance(block, dict) and "text" in block
            )
        elif len(raw_result) > 0 and hasattr(raw_result[0], "text"):
            raw_result = "\n".join(
                str(getattr(block, "text", ""))
                for block in raw_result
            )

    # Parse JSON if raw_result is a JSON string
    parsed = None
    if isinstance(raw_result, str):
        text = raw_result.strip()
        start_brace = text.find("{")
        start_bracket = text.find("[")
        if start_brace != -1 or start_bracket != -1:
            try:
                parsed = json.loads(text)
            except Exception:
                try:
                    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
                        end_brace = text.rfind("}")
                        if end_brace > start_brace:
                            parsed = json.loads(text[start_brace : end_brace + 1])
                    elif start_bracket != -1:
                        end_bracket = text.rfind("]")
                        if end_bracket > start_bracket:
                            parsed = json.loads(text[start_bracket : end_bracket + 1])
                except Exception:
                    pass
    elif isinstance(raw_result, (dict, list)):
        parsed = raw_result

    items = []
    direct_answer = ""

    if isinstance(parsed, dict):
        direct_answer = parsed.get("answer", "") or ""
        if "results" in parsed and isinstance(parsed["results"], list):
            items = parsed["results"]
        elif "data" in parsed and isinstance(parsed["data"], list):
            items = parsed["data"]
        elif "hotels" in parsed and isinstance(parsed["hotels"], list):
            items = parsed["hotels"]
    elif isinstance(parsed, list):
        items = parsed

    if items:
        sliced_items = items[:max_items]
        formatted = []
        if direct_answer:
            summary = direct_answer.strip()
            if len(summary) > 250:
                summary = summary[:250].rsplit(" ", 1)[0] + "..."
            formatted.append(f"Search Summary: {summary}\n")

        for idx, item in enumerate(sliced_items, 1):
            if isinstance(item, dict):
                title = item.get("title") or item.get("name") or f"Hotel Option {idx}"
                url = item.get("url") or ""
                snippet = (
                    item.get("content")
                    or item.get("snippet")
                    or item.get("description")
                    or ""
                )
                snippet = str(snippet).strip().replace("\n", " ")
                if len(snippet) > max_snippet_len:
                    snippet = snippet[:max_snippet_len].rsplit(" ", 1)[0] + "..."

                entry = f"{idx}. {title}"
                if url:
                    entry += f"\n   Website: {url}"
                if snippet:
                    entry += f"\n   Details: {snippet}"
                formatted.append(entry)
            else:
                formatted.append(f"{idx}. {str(item)[:max_snippet_len]}")
        return "\n\n".join(formatted)

    # Fallback for plain text: cap to reasonable length to avoid blowing 8k token context
    text_res = str(raw_result).strip()
    if len(text_res) > 1800:
        return text_res[:1800].rsplit(" ", 1)[0] + "... [trimmed for context limits]"
    return text_res


def slice_tool_output(
    data: Any,
    max_items: int = 5,
    max_chars: int = 1200,
) -> str:
    """
    Cap and slice general tool outputs (airports, airlines, weather, etc.) to the top 3-5 items
    and a safe character length to prevent context explosion.
    """
    if not data:
        return "No data available."

    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict) and "text" in data[0]:
            data = "\n".join(
                str(block["text"])
                for block in data
                if isinstance(block, dict) and "text" in block
            )
        elif len(data) > 0 and hasattr(data[0], "text"):
            data = "\n".join(str(getattr(block, "text", "")) for block in data)

    parsed = None
    if isinstance(data, str):
        text = data.strip()
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            try:
                parsed = json.loads(text)
            except Exception:
                pass
    elif isinstance(data, (dict, list)):
        parsed = data

    if isinstance(parsed, dict):
        if "data" in parsed and isinstance(parsed["data"], list):
            items = parsed["data"][:max_items]
            return json.dumps(items, indent=2)[:max_chars]
        if "results" in parsed and isinstance(parsed["results"], list):
            items = parsed["results"][:max_items]
            return json.dumps(items, indent=2)[:max_chars]
        return json.dumps(parsed, indent=2)[:max_chars]
    elif isinstance(parsed, list):
        items = parsed[:max_items]
        return json.dumps(items, indent=2)[:max_chars]

    text_res = str(data).strip()
    if len(text_res) > max_chars:
        return text_res[:max_chars].rsplit(" ", 1)[0] + "... [trimmed]"
    return text_res


async def tavily_mcp_search(query: str, max_items: int = 5):
    await initialize_mcp()

    tool_args = {"query": query}
    # Pass max_results to Tavily MCP if supported by tool schema
    if hasattr(search_tool, "args") and isinstance(search_tool.args, dict) and "max_results" in search_tool.args:
        tool_args["max_results"] = max_items
    elif hasattr(search_tool, "args_schema") and search_tool.args_schema:
        fields = getattr(search_tool.args_schema, "model_fields", {})
        if "max_results" in fields:
            tool_args["max_results"] = max_items

    result = await search_tool.ainvoke(tool_args)

    # Slice output to top 3-5 items rather than dumping full web responses
    return slice_hotel_results(result, max_items=max_items)




# ==========================================
# AviationStack MCP tools
# ==========================================

aviation_tools = {}


async def initialize_aviation_tools():
    global aviation_tools

    if aviation_tools:
        return

    # Load only AviationStack.
    # Tavily and Weather are initialized independently.
    tools = await client.get_tools(
        server_name="Aviationstack"
    )

    aviation_tools = {
        tool.name: tool
        for tool in tools
    }

    if not aviation_tools:
        raise RuntimeError(
            "AviationStack MCP connected but "
            "returned no tools."
        )
        
        
        

async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict = None
):
    await initialize_aviation_tools()

    tool = aviation_tools.get(tool_name)

    if tool is None:
        available_tools = ", ".join(
            sorted(aviation_tools.keys())
        )

        raise ValueError(
            f"AviationStack tool '{tool_name}' "
            "was not found. "
            f"Available tools: "
            f"{available_tools or 'none'}"
        )

    result = await tool.ainvoke(
        tool_args or {}
    )

    return result


# ==========================================
# Weather MCP tools
# ==========================================

weather_tool = None
forecast_tool = None


async def initialize_weather_tools():
    global weather_tool
    global forecast_tool

    if (
        weather_tool is not None
        and forecast_tool is not None
    ):
        return

    if not WEATHER_SERVER_PATH.exists():
        raise FileNotFoundError(
            "Weather MCP server file was not found: "
            f"{WEATHER_SERVER_PATH}"
        )

    # Load only Weather.
    # Tavily and AviationStack will not be started.
    tools = await client.get_tools(
        server_name="weather"
    )

    tools_by_name = {
        tool.name: tool
        for tool in tools
    }

    weather_tool = tools_by_name.get(
        "get_current_weather"
    )

    forecast_tool = tools_by_name.get(
        "get_forecast"
    )

    missing_tools = []

    if weather_tool is None:
        missing_tools.append(
            "get_current_weather"
        )

    if forecast_tool is None:
        missing_tools.append(
            "get_forecast"
        )

    if missing_tools:
        available_tools = ", ".join(
            tools_by_name.keys()
        )

        raise RuntimeError(
            "Missing Weather MCP tools: "
            f"{', '.join(missing_tools)}. "
            f"Available tools: "
            f"{available_tools or 'none'}"
        )


# async def weather_mcp_call(
#     tool_name: str,
#     tool_args: dict = None
# ):
#     """Invoke a tool from the local weather MCP server."""

#     await initialize_weather_tools()

#     tool = weather_tools.get(tool_name)

#     if tool is None:
#         available_tools = ", ".join(sorted(weather_tools.keys()))
#         raise ValueError(
#             f"Weather tool '{tool_name}' was not found. "
#             f"Available tools: {available_tools or 'none'}"
#         )

#     return await tool.ainvoke(tool_args or {})



async def weather_mcp_search(city: str):
    await initialize_weather_tools()

    result = await weather_tool.ainvoke(
        {
            "city": city
        }
    )

    return result


async def forecast_mcp_search(city: str):
    await initialize_weather_tools()

    result = await forecast_tool.ainvoke(
        {
            "city": city
        }
    )

    return result



# ==========================================
# Destination extractor
# ==========================================

def extract_destination(query: str):
    prompt = f"""
    Extract only the destination city or country.

    Query:
    {query}

    Return only destination name.
    """

    response = llm.invoke(prompt)

    return response.content.strip()
