# streamlit_mcp_app.py - With write_file → download interception and detailed connection feedback
import os
import re
import asyncio
import json
import yaml
import logging
import threading
import queue
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from contextlib import AsyncExitStack

import streamlit as st
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------
@dataclass
class ServerConnection:
    name: str
    session: ClientSession
    exit_stack: AsyncExitStack
    tools: List[Dict] = field(default_factory=list)
    prompts: List[Dict] = field(default_factory=list)
    resources: List[Dict] = field(default_factory=list)

    async def cleanup(self):
        await self.exit_stack.aclose()


class ConversationMemory:
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages: List[Dict] = []
        self._cache_key = None
        self._cached_messages = None

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._prune()
        self._invalidate_cache()

    def add_assistant_tool_calls(self, tool_calls: List[Dict]):
        self.messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        self._invalidate_cache()

    def add_tool_response(self, tool_call_id: str, content: str):
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})
        self._invalidate_cache()

    def _prune(self):
        if len(self.messages) > self.max_messages:
            if self.messages and self.messages[0]["role"] == "system":
                self.messages = [self.messages[0]] + self.messages[-(self.max_messages - 1):]
            else:
                self.messages = self.messages[-self.max_messages:]

    def get_messages(self) -> List[Dict]:
        current_key = hash(tuple(json.dumps(msg, sort_keys=True) for msg in self.messages))
        if self._cache_key == current_key and self._cached_messages is not None:
            return self._cached_messages

        validated = self._validate_conversation(self.messages.copy())
        self._cache_key = current_key
        self._cached_messages = validated
        return validated

    def _validate_conversation(self, messages):
        i = 0
        while i < len(messages):
            if messages[i]["role"] == "tool":
                found = False
                for j in range(i-1, -1, -1):
                    if messages[j]["role"] == "assistant" and messages[j].get("tool_calls"):
                        for tc in messages[j]["tool_calls"]:
                            if tc["id"] == messages[i]["tool_call_id"]:
                                found = True
                                break
                        if found:
                            break
                    if messages[j]["role"] == "user":
                        break
                if not found:
                    del messages[i]
                    continue
            i += 1
        return messages

    def _invalidate_cache(self):
        self._cache_key = None
        self._cached_messages = None

    def clear(self):
        self.messages = []
        self._invalidate_cache()


# -----------------------------------------------------------------------------
# MCP Client (async background thread)
# -----------------------------------------------------------------------------
class MCPClient:
    def __init__(self):
        self.servers: Dict[str, ServerConnection] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.request_queue = queue.Queue(maxsize=100)
        self.response_queue = queue.Queue(maxsize=100)
        self.running = False
        self._tools_schema_cache = None
        self._tools_schema_cache_time = 0
        self._cache_ttl = 30

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        for _ in range(100):
            if self.loop is not None:
                break
            time.sleep(0.01)

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._process_requests())
        self.loop.close()

    async def _process_requests(self):
        while self.running:
            try:
                req = self.request_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            try:
                if req["type"] == "connect":
                    result = await self._connect_all_servers(req["config_path"])
                    self.response_queue.put(("connect_result", result))
                    self._invalidate_cache()
                elif req["type"] == "execute_tool":
                    result = await self.execute_tool(req["server"], req["tool"], req["args"])
                    self.response_queue.put(("tool_result", req["call_id"], result))
                elif req["type"] == "cleanup":
                    await self.cleanup()
                    self.response_queue.put(("cleanup_done", None))
                    self._invalidate_cache()
                elif req["type"] == "get_tools_schema":
                    now = time.time()
                    if self._tools_schema_cache is not None and now - self._tools_schema_cache_time < self._cache_ttl:
                        schema = self._tools_schema_cache
                    else:
                        schema = self.build_tools_schema()
                        self._tools_schema_cache = schema
                        self._tools_schema_cache_time = now
                    self.response_queue.put(("tools_schema", schema))
                elif req["type"] == "list_prompts":
                    prompts = await self.list_all_prompts()
                    self.response_queue.put(("prompts_list", prompts))
                elif req["type"] == "get_prompt":
                    messages = await self.get_prompt(req["name"], req.get("arguments", {}))
                    self.response_queue.put(("prompt_messages", req["name"], messages))
                elif req["type"] == "list_resources":
                    resources = await self.list_all_resources()
                    self.response_queue.put(("resources_list", resources))
                elif req["type"] == "read_resource":
                    content = await self.read_resource(req["uri"])
                    self.response_queue.put(("resource_content", req["uri"], content))
                elif req["type"] == "get_servers":
                    servers_info = self._get_servers_info()
                    self.response_queue.put(("servers_list", servers_info))
            except Exception as e:
                logger.exception("Error in background loop")
                self.response_queue.put(("error", str(e)))

    def _invalidate_cache(self):
        self._tools_schema_cache = None
        self._tools_schema_cache_time = 0

    def _get_servers_info(self) -> Dict:
        return {
            name: {
                "name": srv.name,
                "tools": srv.tools,
                "prompts": srv.prompts,
                "resources": srv.resources,
            }
            for name, srv in self.servers.items()
        }

    async def _connect_all_servers(self, config_path: str) -> Dict[str, Any]:
        """Connect to all servers and return detailed results."""
        config = await self._load_config(config_path)
        tasks = []
        server_names = []
        for srv in config.get("servers", {}).get("local", []):
            tasks.append(self._connect_stdio(srv))
            server_names.append(srv.get("name", "unnamed"))
        for srv in config.get("servers", {}).get("remote", []):
            tasks.append(self._connect_sse(srv))
            server_names.append(srv.get("name", "unnamed"))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = 0
        details = []
        for i, result in enumerate(results):
            name = server_names[i]
            if isinstance(result, Exception):
                error_msg = str(result)
                logger.error(f"Connection to '{name}' failed: {error_msg}")
                details.append({"name": name, "success": False, "error": error_msg})
            else:
                self.servers[result.name] = result
                successful += 1
                details.append({"name": result.name, "success": True, "error": None})

        return {"successful": successful, "details": details}

    async def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    async def _connect_stdio(self, cfg: dict) -> ServerConnection:
        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg.get("args", []),
            env=cfg.get("env", {})
        )
        exit_stack = AsyncExitStack()
        try:
            transport = await exit_stack.enter_async_context(stdio_client(params))
            read, write = transport
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            tools = await self._safe_list_tools(session)
            prompts = await self._safe_list_prompts(session)
            resources = await self._safe_list_resources(session)
            return ServerConnection(
                name=cfg["name"],
                session=session,
                exit_stack=exit_stack,
                tools=tools,
                prompts=prompts,
                resources=resources,
            )
        except Exception:
            await exit_stack.aclose()
            raise

    async def _connect_sse(self, cfg: dict) -> ServerConnection:
        exit_stack = AsyncExitStack()
        try:
            transport = await exit_stack.enter_async_context(sse_client(cfg["url"]))
            read, write = transport
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            tools = await self._safe_list_tools(session)
            prompts = await self._safe_list_prompts(session)
            resources = await self._safe_list_resources(session)
            return ServerConnection(
                name=cfg["name"],
                session=session,
                exit_stack=exit_stack,
                tools=tools,
                prompts=prompts,
                resources=resources,
            )
        except Exception:
            await exit_stack.aclose()
            raise

    async def _safe_list_tools(self, session):
        try:
            result = await session.list_tools()
            return [t.model_dump() for t in result.tools]
        except Exception:
            return []

    async def _safe_list_prompts(self, session):
        try:
            result = await session.list_prompts()
            return [p.model_dump() for p in result.prompts]
        except Exception:
            return []

    async def _safe_list_resources(self, session):
        try:
            result = await session.list_resources()
            return [r.model_dump() for r in result.resources]
        except Exception:
            return []

    def _sanitize_name(self, name: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

    def build_tools_schema(self) -> List[dict]:
        tools = []
        for srv_name, srv in self.servers.items():
            safe_srv = self._sanitize_name(srv_name)
            for tool in srv.tools:
                safe_tool = self._sanitize_name(tool["name"])
                tools.append({
                    "type": "function",
                    "function": {
                        "name": f"{safe_srv}__{safe_tool}",
                        "description": tool.get("description", f"Tool from {srv_name}"),
                        "parameters": tool.get("inputSchema", {"type": "object", "properties": {}})
                    }
                })
        return tools

    async def execute_tool(self, server_name: str, tool_name: str, args: dict) -> str:
        if server_name not in self.servers:
            return f"Error: server '{server_name}' not connected"
        try:
            result = await self.servers[server_name].session.call_tool(tool_name, args)
            if hasattr(result, 'content') and result.content:
                return result.content[0].text if result.content else "No output"
            return str(result)
        except Exception as e:
            return f"Tool error: {e}"

    async def list_all_prompts(self) -> List[Dict]:
        return [{"server": srv.name, **p} for srv in self.servers.values() for p in srv.prompts]

    async def get_prompt(self, prompt_name: str, arguments: dict = None) -> List[Dict]:
        for srv in self.servers.values():
            for p in srv.prompts:
                if p["name"] == prompt_name:
                    result = await srv.session.get_prompt(prompt_name, arguments=arguments or {})
                    return [msg.model_dump() for msg in result.messages]
        return []

    async def list_all_resources(self) -> List[Dict]:
        return [{"server": srv.name, **r} for srv in self.servers.values() for r in srv.resources]

    async def read_resource(self, uri: str) -> str:
        for srv in self.servers.values():
            for r in srv.resources:
                if r["uri"] == uri:
                    result = await srv.session.read_resource(uri)
                    if result.contents:
                        return result.contents[0].text if hasattr(result.contents[0], 'text') else str(result.contents[0])
                    return "Resource empty"
        return "Resource not found"

    async def cleanup(self):
        for srv in self.servers.values():
            await srv.cleanup()
        self.servers.clear()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)


# -----------------------------------------------------------------------------
# Synchronous wrapper with caching
# -----------------------------------------------------------------------------
class SyncMCPClient:
    def __init__(self):
        self._client = MCPClient()
        self._client.start()
        self._next_call_id = 0
        self._response_cache = {}
        self._cache_timeout = 5

    def connect(self, config_path: str, timeout=400) -> Dict[str, Any]:
        self._client.request_queue.put({"type": "connect", "config_path": config_path})
        return self._wait_for_result("connect_result", timeout)

    def get_tools_schema(self, timeout=5) -> List[dict]:
        cache_key = "tools_schema"
        if cache_key in self._response_cache:
            cached_time, cached_value = self._response_cache[cache_key]
            if time.time() - cached_time < self._cache_timeout:
                return cached_value
        self._client.request_queue.put({"type": "get_tools_schema"})
        result = self._wait_for_result("tools_schema", timeout)
        self._response_cache[cache_key] = (time.time(), result)
        return result

    def get_servers(self, timeout=30) -> Dict:
        cache_key = "servers_list"
        if cache_key in self._response_cache:
            cached_time, cached_value = self._response_cache[cache_key]
            if time.time() - cached_time < self._cache_timeout:
                return cached_value
        self._client.request_queue.put({"type": "get_servers"})
        result = self._wait_for_result("servers_list", timeout)
        self._response_cache[cache_key] = (time.time(), result)
        return result

    def execute_tool(self, server_name: str, tool_name: str, args: dict, timeout=30) -> str:
        call_id = self._next_call_id
        self._next_call_id += 1
        self._client.request_queue.put({
            "type": "execute_tool",
            "server": server_name,
            "tool": tool_name,
            "args": args,
            "call_id": call_id
        })
        start = time.time()
        while time.time() - start < timeout:
            try:
                typ, cid, val = self._client.response_queue.get(timeout=0.5)
                if typ == "tool_result" and cid == call_id:
                    return val
            except queue.Empty:
                continue
            except ValueError:
                pass
        return "Error: Tool execution timeout"

    def list_prompts(self, timeout=5) -> List[Dict]:
        cache_key = "prompts_list"
        if cache_key in self._response_cache:
            cached_time, cached_value = self._response_cache[cache_key]
            if time.time() - cached_time < self._cache_timeout:
                return cached_value
        self._client.request_queue.put({"type": "list_prompts"})
        result = self._wait_for_result("prompts_list", timeout)
        self._response_cache[cache_key] = (time.time(), result)
        return result

    def get_prompt(self, prompt_name: str, arguments: dict = None, timeout=5) -> List[Dict]:
        cache_key = f"prompt_{prompt_name}_{hash(str(arguments))}"
        if cache_key in self._response_cache:
            cached_time, cached_value = self._response_cache[cache_key]
            if time.time() - cached_time < self._cache_timeout:
                return cached_value
        self._client.request_queue.put({
            "type": "get_prompt",
            "name": prompt_name,
            "arguments": arguments or {}
        })
        start = time.time()
        while time.time() - start < timeout:
            try:
                typ, name, msgs = self._client.response_queue.get(timeout=0.5)
                if typ == "prompt_messages" and name == prompt_name:
                    self._response_cache[cache_key] = (time.time(), msgs)
                    return msgs
            except queue.Empty:
                continue
            except ValueError:
                pass
        return []

    def list_resources(self, timeout=5) -> List[Dict]:
        cache_key = "resources_list"
        if cache_key in self._response_cache:
            cached_time, cached_value = self._response_cache[cache_key]
            if time.time() - cached_time < self._cache_timeout:
                return cached_value
        self._client.request_queue.put({"type": "list_resources"})
        result = self._wait_for_result("resources_list", timeout)
        self._response_cache[cache_key] = (time.time(), result)
        return result

    def read_resource(self, uri: str, timeout=5) -> str:
        cache_key = f"resource_{uri}"
        if cache_key in self._response_cache:
            cached_time, cached_value = self._response_cache[cache_key]
            if time.time() - cached_time < self._cache_timeout:
                return cached_value
        self._client.request_queue.put({
            "type": "read_resource",
            "uri": uri
        })
        start = time.time()
        while time.time() - start < timeout:
            try:
                typ, u, content = self._client.response_queue.get(timeout=0.5)
                if typ == "resource_content" and u == uri:
                    self._response_cache[cache_key] = (time.time(), content)
                    return content
            except queue.Empty:
                continue
            except ValueError:
                pass
        return "Error: Resource read timeout"

    def cleanup(self, timeout=10):
        self._client.request_queue.put({"type": "cleanup"})
        self._wait_for_result("cleanup_done", timeout)
        self._client.stop()
        self._response_cache.clear()

    def _wait_for_result(self, expected_type, timeout):
        start = time.time()
        while time.time() - start < timeout:
            try:
                result = self._client.response_queue.get(timeout=0.5)
                if result[0] == expected_type:
                    return result[1]
                elif result[0] == "error":
                    raise Exception(result[1])
            except queue.Empty:
                continue
        raise TimeoutError(f"Timeout waiting for {expected_type}")


# -----------------------------------------------------------------------------
# DeepSeek Engine with write_file interception
# -----------------------------------------------------------------------------
class DeepSeekEngine:
    def __init__(self, mcp_client: Optional[SyncMCPClient] = None):
        self.mcp = mcp_client
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.search_stats = {
            'count': 0,
            'engines': set(),
            'results': 0
        }

    def _parse_tool_name(self, full_name: str) -> tuple:
        if "__" in full_name:
            return full_name.split("__", 1)
        return "unknown", full_name

    def _is_write_file_tool(self, tool_name: str) -> bool:
        base_name = tool_name.split("__")[-1] if "__" in tool_name else tool_name
        return base_name == "write_file"

    def _track_search(self, tool_name: str, args: dict, result: str):
        if "search" in tool_name.lower():
            self.search_stats['count'] += 1
            engines = args.get('engines', ['default'])
            if isinstance(engines, list):
                self.search_stats['engines'].update(engines)
            else:
                self.search_stats['engines'].add(str(engines))
            try:
                result_data = json.loads(result)
                if 'totalResults' in result_data:
                    self.search_stats['results'] += result_data['totalResults']
                elif 'results' in result_data and isinstance(result_data['results'], list):
                    self.search_stats['results'] += len(result_data['results'])
            except:
                limit = args.get('limit', 0)
                if isinstance(limit, (int, float)) and limit > 0:
                    self.search_stats['results'] += limit

    def reset_search_stats(self):
        self.search_stats = {
            'count': 0,
            'engines': set(),
            'results': 0
        }

    def get_search_stats_dict(self):
        return {
            'count': self.search_stats['count'],
            'engines': sorted(self.search_stats['engines']),
            'results': self.search_stats['results']
        }

    def process_conversation(self, messages: List[Dict], manual_limit: int = 0) -> Tuple[str, List[Dict], List[Dict]]:
        tools_schema = self.mcp.get_tools_schema() if self.mcp else []
        current_messages = messages.copy()
        pending_downloads = []

        while True:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=current_messages,
                tools=tools_schema if tools_schema else None,
                tool_choice="auto" if tools_schema else None,
                temperature=0.7,
                max_tokens=2000
            )
            msg = response.choices[0].message

            if not (self.mcp and hasattr(msg, 'tool_calls') and msg.tool_calls):
                if msg.content:
                    current_messages.append({"role": "assistant", "content": msg.content})
                    return msg.content, current_messages, pending_downloads
                return "", current_messages, pending_downloads

            tool_calls_data = [tc.model_dump() for tc in msg.tool_calls]
            current_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls_data
            })

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                srv_name, tool_name = self._parse_tool_name(tc.function.name)

                if self._is_write_file_tool(tool_name):
                    content = args.get("content", "")
                    filename = args.get("filename", "output.txt")
                    filename = os.path.basename(filename)
                    if not filename:
                        filename = "output.txt"
                    if "." not in filename:
                        filename += ".txt"

                    pending_downloads.append({
                        "content": content,
                        "filename": filename,
                        "tool_call_id": tc.id
                    })

                    tool_response = (
                        f"File '{filename}' prepared for download. "
                        f"Since the app is hosted, the file will be saved to your local computer "
                        f"via a download button. Click the button below to save it."
                    )
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_response
                    })
                    logger.info(f"Intercepted write_file: {filename} (will offer download)")
                    continue

                if "search" in tool_name.lower() and manual_limit > 0:
                    if tool_name.lower() != 'perform_websearch':
                        args["limit"] = manual_limit
                        logger.info(f"Manual override: limit={manual_limit}")

                logger.info(f"Executing {srv_name}.{tool_name} with args {json.dumps(args)}")
                result = self.mcp.execute_tool(srv_name, tool_name, args)
                logger.info(f"Tool result length: {len(result)}")

                self._track_search(tool_name, args, result)

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def init_state():
    if "mcp_client" not in st.session_state:
        st.session_state.mcp_client = None
    if "engine" not in st.session_state:
        st.session_state.engine = DeepSeekEngine(None)
    if "conv" not in st.session_state:
        st.session_state.conv = ConversationMemory()
        st.session_state.conv.add_message("system", "You are a helpful AI assistant.")
    if "messages_ui" not in st.session_state:
        st.session_state.messages_ui = []
    if "mcp_connected" not in st.session_state:
        st.session_state.mcp_connected = False
    if "pending_downloads" not in st.session_state:
        st.session_state.pending_downloads = []
    if "search_stats" not in st.session_state:
        st.session_state.search_stats = {
            'count': 0,
            'engines': [],
            'results': 0
        }


def connect_mcp(config_path: str):
    """Connect to MCP servers and display detailed progress/errors."""
    progress_placeholder = st.empty()
    progress_placeholder.info("⏳ Connecting to MCP servers...")

    try:
        os.environ["FASTMCP_OAUTH_REDIRECT_URI"] = "https://mcp-st-client-test.onrender.com:8471/oauth/callback"
        mcp = SyncMCPClient()
        result = mcp.connect(config_path)

        successful = result["successful"]
        details = result["details"]

        # Build detailed feedback
        success_list = []
        failure_list = []
        for d in details:
            if d["success"]:
                success_list.append(f"✅ {d['name']} connected successfully")
            else:
                failure_list.append(f"❌ {d['name']} failed: {d['error']}")

        if successful == len(details):
            progress_placeholder.success(f"🎉 All {successful} server(s) connected!")
            for msg in success_list:
                st.success(msg)
        else:
            progress_placeholder.warning(f"⚠️ Connected to {successful}/{len(details)} servers")
            for msg in success_list:
                st.success(msg)
            for msg in failure_list:
                st.error(msg)

        # Initialize session state with the client
        st.session_state.mcp_client = mcp
        st.session_state.engine = DeepSeekEngine(mcp)
        st.session_state.mcp_connected = True
        st.session_state.conv.clear()
        st.session_state.conv.add_message(
            "system",
            "You have access to MCP tools, including a 'search' tool for web search.\n"
            "When using the search tool, you can specify which search engine(s) to use via the 'engines' parameter.\n"
            "Recommended engine selection:\n"
            "- General knowledge, news, casual queries: engines=['duckduckgo']\n"
            "- Technical, programming, science: engines=['bing','brave']\n"
            "- Privacy-focused or unrestricted content: engines=['brave']\n"
            "- Academic, research papers, scholarly: engines=['exa']\n"
            "- Comprehensive results: engines=['duckduckgo','bing','brave']\n"
            "Example: search(query='machine learning trends', limit=5, engines=['bing','exa'])\n"
            "If you don't specify engines, the default will be used. Always provide a clear query."

            "When using the 'search' tool, you can specify:"
            "- 'query': The search terms."
            "- 'engine': (Optional) 'duckduckgo', 'bing', 'brave', or 'exa'."
            "- 'limit': (Optional) The number of results to return (e.g., 1 to 10)."

            "If no limit is set use a higher 'limit' if the user asks for a comprehensive list, and a lower limit (1-3) for quick fact-checks to 	save tokens."

            "\n\nIMPORTANT: After each response, the system will automatically append search statistics showing:\n"
            "- Number of searches performed\n"
            "- Engines used\n"
            "- Total results obtained\n"
            "Mention these statistics in your response - they will be added automatically."
        )
        st.session_state.messages_ui = []
        st.session_state.search_stats = st.session_state.engine.get_search_stats_dict()

        return successful

    except Exception as e:
        progress_placeholder.error(f"❌ Connection failed: {str(e)}")
        raise


def disconnect_mcp():
    if st.session_state.mcp_client:
        st.session_state.mcp_client.cleanup()
    st.session_state.mcp_client = None
    st.session_state.engine = DeepSeekEngine(None)
    st.session_state.mcp_connected = False
    st.session_state.conv.clear()
    st.session_state.conv.add_message("system", "You are a helpful AI assistant. No external tools.")
    st.session_state.messages_ui = []
    st.session_state.pending_downloads = []
    st.session_state.search_stats = {'count': 0, 'engines': [], 'results': 0}


def main():
    st.set_page_config(page_title="DeepSeek + MCP", page_icon="🤖", layout="wide")
    st.title("🤖 DeepSeek Chat with MCP")
    st.markdown("Connect MCP servers to use tools, prompts, and resources. Files that would be written are offered as downloads.")
    init_state()

    with st.sidebar:
        st.header("🔌 MCP Connection")
        if not st.session_state.mcp_connected:
            cfg = st.text_input("Config file", value="servers.yaml", key="config_file_input")
            if st.button("Connect MCP Servers", use_container_width=True):
                try:
                    n = connect_mcp(cfg)
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to connect: {e}")
        else:
            st.success("MCP active")

            st.subheader("⚙️ Tool Settings")
            search_limit = st.slider(
                "Manual Search Limit (0 = AI Decide)",
                min_value=0, max_value=10, value=2,
                help="If set to 0, the AI follows its system prompt. 1-10 overrides the AI.",
                key="search_limit_slider"
            )

            st.divider()
            st.subheader("📊 Cumulative Search Statistics")
            stats = st.session_state.search_stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Searches", stats['count'])
            with col2:
                st.metric("Total Results", stats['results'])
            with col3:
                if stats['engines']:
                    st.write("Engines:", ", ".join(stats['engines']))
                else:
                    st.write("No searches yet")
            if stats['count'] > 0:
                with st.expander("📋 Detailed Search Log"):
                    st.markdown(f"**Searches performed:** {stats['count']}")
                    st.markdown(f"**Engines used:** {', '.join(stats['engines'])}")
                    st.markdown(f"**Total results retrieved:** {stats['results']}")
            if st.button("Reset Statistics", use_container_width=True):
                st.session_state.engine.reset_search_stats()
                st.session_state.search_stats = st.session_state.engine.get_search_stats_dict()
                st.rerun()

            st.divider()

            if st.button("Disconnect MCP", use_container_width=True):
                disconnect_mcp()
                st.rerun()

            with st.expander("🔧 Available Tools"):
                try:
                    time.sleep(0.3)
                    servers = st.session_state.mcp_client.get_servers()
                    for srv_name, srv_info in servers.items():
                        st.markdown(f"**{srv_name}**")
                        for tool in srv_info.get("tools", []):
                            st.markdown(f"- `{tool['name']}`: {tool.get('description', 'No description')[:100]}")
                except TimeoutError:
                    st.error("⏰ Timed out waiting for tools. The MCP server may be slow. Try disconnecting and reconnecting.")
                except Exception as e:
                    st.error(f"Error loading tools: {e}")

            with st.expander("📋 Available Prompts"):
                try:
                    prompts = st.session_state.mcp_client.list_prompts()
                    for prompt in prompts:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{prompt['name']}**")
                            if prompt.get("description"):
                                st.caption(prompt["description"])
                        with col2:
                            if st.button("Use", key=f"prompt_{prompt['name']}"):
                                messages = st.session_state.mcp_client.get_prompt(prompt["name"])
                                for msg in messages:
                                    role = msg.get("role", "user")
                                    content = msg.get("content", {}).get("text", "")
                                    if content:
                                        st.session_state.conv.add_message(role, content)
                                        st.session_state.messages_ui.append({"role": role, "content": content})
                                st.rerun()
                except TimeoutError:
                    st.error("⏰ Timed out loading prompts.")
                except Exception as e:
                    st.error(f"Error loading prompts: {e}")

            with st.expander("📁 Available Resources"):
                try:
                    resources = st.session_state.mcp_client.list_resources()
                    for resource in resources:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{resource.get('name', resource['uri'])}**")
                            st.caption(f"URI: {resource['uri']}")
                            if resource.get("description"):
                                st.caption(resource["description"])
                        with col2:
                            if st.button("Read", key=f"resource_{resource['uri']}"):
                                content = st.session_state.mcp_client.read_resource(resource["uri"])
                                prompt_text = f"Content from resource {resource['uri']}:\n\n{content}"
                                st.session_state.conv.add_message("user", prompt_text)
                                st.session_state.messages_ui.append({"role": "user", "content": prompt_text})
                                st.rerun()
                except TimeoutError:
                    st.error("⏰ Timed out loading resources.")
                except Exception as e:
                    st.error(f"Error loading resources: {e}")

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.conv.clear()
            sys_msg = "You have access to MCP tools, prompts, and resources." if st.session_state.mcp_connected else "You are a helpful AI assistant."
            st.session_state.conv.add_message("system", sys_msg)
            st.session_state.messages_ui = []
            st.session_state.pending_downloads = []
            st.rerun()

    # Chat display
    for i, msg in enumerate(st.session_state.messages_ui):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Your message"):
        st.session_state.messages_ui.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.conv.add_message("user", prompt)

        with st.spinner("Thinking..."):
            try:
                answer, new_conv, downloads = st.session_state.engine.process_conversation(
                    st.session_state.conv.get_messages(),
                    manual_limit=search_limit if st.session_state.mcp_connected else 0
                )
                st.session_state.conv.messages = new_conv
                st.session_state.messages_ui.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)

                st.session_state.search_stats = st.session_state.engine.get_search_stats_dict()

                if downloads:
                    st.session_state.pending_downloads = downloads
                    for dl in downloads:
                        st.download_button(
                            label=f"📥 Download {dl['filename']}",
                            data=dl['content'],
                            file_name=dl['filename'],
                            mime="text/plain",
                            key=f"download_{dl['tool_call_id']}_{time.time()}"
                        )
                    st.session_state.pending_downloads = []

                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")
                logger.exception("Processing error")

    st.caption("Powered by DeepSeek. MCP servers provide tools, prompts, and resources. Write operations are converted to downloads.")


if __name__ == "__main__":
    main()