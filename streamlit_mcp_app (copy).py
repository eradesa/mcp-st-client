# streamlit_mcp_app.py
import os
import re
import asyncio
import json
import yaml
import logging
import threading
import queue
import time
from typing import Dict, List, Optional, Any
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
# Data classes (unchanged)
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

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._prune()

    def add_assistant_tool_calls(self, tool_calls: List[Dict]):
        self.messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

    def add_tool_response(self, tool_call_id: str, content: str):
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def _prune(self):
        if len(self.messages) > self.max_messages:
            if self.messages and self.messages[0]["role"] == "system":
                self.messages = [self.messages[0]] + self.messages[-(self.max_messages - 1):]
            else:
                self.messages = self.messages[-self.max_messages:]

    def get_messages(self) -> List[Dict]:
        # Validate conversation before sending (remove orphaned tool messages)
        return self._validate_conversation(self.messages.copy())

    def _validate_conversation(self, messages):
        """Remove tool messages without a matching assistant tool_calls parent."""
        i = 0
        while i < len(messages):
            if messages[i]["role"] == "tool":
                found = False
                # Look backward for an assistant message with tool_calls containing this id
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
                    # Remove orphaned tool message
                    del messages[i]
                    continue
            i += 1
        return messages

    def clear(self):
        self.messages = []


# -----------------------------------------------------------------------------
# MCP Client (async) – runs in a background thread
# -----------------------------------------------------------------------------
class MCPClient:
    def __init__(self):
        self.servers: Dict[str, ServerConnection] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        while self.loop is None:
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
                    num = await self._connect_all_servers(req["config_path"])
                    self.response_queue.put(("connect_result", num))
                elif req["type"] == "execute_tool":
                    result = await self.execute_tool(req["server"], req["tool"], req["args"])
                    self.response_queue.put(("tool_result", req["call_id"], result))
                elif req["type"] == "cleanup":
                    await self.cleanup()
                    self.response_queue.put(("cleanup_done", None))
                elif req["type"] == "get_tools_schema":
                    schema = self.build_tools_schema()
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
                    # Return a serializable copy
                    servers_info = {}
                    for name, srv in self.servers.items():
                        servers_info[name] = {
                            "name": srv.name,
                            "tools": srv.tools,
                            "prompts": srv.prompts,
                            "resources": srv.resources,
                        }
                    self.response_queue.put(("servers_list", servers_info))
            except Exception as e:
                logger.exception("Error in background loop")
                self.response_queue.put(("error", str(e)))

    # -------------------------------------------------------------------------
    # Connection and server discovery (existing, extended to fetch prompts/resources)
    # -------------------------------------------------------------------------
    async def _connect_all_servers(self, config_path: str) -> int:
        config = await self._load_config(config_path)
        for srv in config.get("servers", {}).get("local", []):
            try:
                conn = await self._connect_stdio(srv)
                self.servers[conn.name] = conn
            except Exception as e:
                logger.error(f"Skip {srv['name']}: {e}")
        for srv in config.get("servers", {}).get("remote", []):
            try:
                conn = await self._connect_sse(srv)
                self.servers[conn.name] = conn
            except Exception as e:
                logger.error(f"Skip {srv['name']}: {e}")
        return len(self.servers)

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

            # Initialize empty containers for capabilities
            tools, prompts, resources = [], [], []

            # 1. Safely fetch tools
            try:
                tools_result = await session.list_tools()
                tools = [t.model_dump() for t in tools_result.tools]
            except Exception as e:
                logger.warning(f"Server {cfg['name']} failed to list tools: {e}")

            # 2. Safely fetch prompts
            try:
                prompts_result = await session.list_prompts()
                prompts = [p.model_dump() for p in prompts_result.prompts]
            except Exception as e:
                logger.info(f"Server {cfg['name']} does not support prompts (Method not found)")

            # 3. Safely fetch resources
            try:
                resources_result = await session.list_resources()
                resources = [r.model_dump() for r in resources_result.resources]
            except Exception as e:
                logger.info(f"Server {cfg['name']} does not support resources (Method not found)")

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

            # Initialize empty containers for capabilities
            tools, prompts, resources = [], [], []

            # 1. Safely fetch tools
            try:
                tools_result = await session.list_tools()
                tools = [t.model_dump() for t in tools_result.tools]
            except Exception as e:
                logger.warning(f"Remote server {cfg['name']} failed to list tools: {e}")

            # 2. Safely fetch prompts
            try:
                prompts_result = await session.list_prompts()
                prompts = [p.model_dump() for p in prompts_result.prompts]
            except Exception as e:
                logger.info(f"Remote server {cfg['name']} does not support prompts")

            # 3. Safely fetch resources
            try:
                resources_result = await session.list_resources()
                resources = [r.model_dump() for r in resources_result.resources]
            except Exception as e:
                logger.info(f"Remote server {cfg['name']} does not support resources")

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
    # -------------------------------------------------------------------------
    # Tool methods (existing, with sanitized names)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Prompt methods (new)
    # -------------------------------------------------------------------------
    async def list_all_prompts(self) -> List[Dict]:
        all_prompts = []
        for srv in self.servers.values():
            for p in srv.prompts:
                all_prompts.append({
                    "server": srv.name,
                    **p
                })
        return all_prompts

    async def get_prompt(self, prompt_name: str, arguments: dict = None) -> List[Dict]:
        # Find which server has this prompt
        for srv in self.servers.values():
            for p in srv.prompts:
                if p["name"] == prompt_name:
                    result = await srv.session.get_prompt(prompt_name, arguments=arguments or {})
                    # result.messages is a list of PromptMessage objects
                    return [msg.model_dump() for msg in result.messages]
        return []

    # -------------------------------------------------------------------------
    # Resource methods (new)
    # -------------------------------------------------------------------------
    async def list_all_resources(self) -> List[Dict]:
        all_resources = []
        for srv in self.servers.values():
            for r in srv.resources:
                all_resources.append({
                    "server": srv.name,
                    **r
                })
        return all_resources

    async def read_resource(self, uri: str) -> str:
        # Find which server holds this resource
        for srv in self.servers.values():
            for r in srv.resources:
                if r["uri"] == uri:
                    result = await srv.session.read_resource(uri)
                    # result.contents is a list of TextResourceContents or BlobResourceContents
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
# Synchronous wrapper for the background MCP client
# -----------------------------------------------------------------------------
class SyncMCPClient:
    def __init__(self):
        self._client = MCPClient()
        self._client.start()
        self._next_call_id = 0

    def connect(self, config_path: str, timeout=60) -> int:
        self._client.request_queue.put({"type": "connect", "config_path": config_path})
        return self._wait_for_result("connect_result", timeout)

    def get_tools_schema(self, timeout=5) -> List[dict]:
        self._client.request_queue.put({"type": "get_tools_schema"})
        return self._wait_for_result("tools_schema", timeout)

    def get_servers(self, timeout=5) -> Dict:
        self._client.request_queue.put({"type": "get_servers"})
        return self._wait_for_result("servers_list", timeout)

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
        self._client.request_queue.put({"type": "list_prompts"})
        return self._wait_for_result("prompts_list", timeout)

    def get_prompt(self, prompt_name: str, arguments: dict = None, timeout=5) -> List[Dict]:
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
                    return msgs
            except queue.Empty:
                continue
            except ValueError:
                pass
        return []

    def list_resources(self, timeout=5) -> List[Dict]:
        self._client.request_queue.put({"type": "list_resources"})
        return self._wait_for_result("resources_list", timeout)

    def read_resource(self, uri: str, timeout=5) -> str:
        self._client.request_queue.put({
            "type": "read_resource",
            "uri": uri
        })
        start = time.time()
        while time.time() - start < timeout:
            try:
                typ, u, content = self._client.response_queue.get(timeout=0.5)
                if typ == "resource_content" and u == uri:
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
# DeepSeek Engine (sync, uses SyncMCPClient)
# -----------------------------------------------------------------------------
class DeepSeekEngine:
    def __init__(self, mcp_client: Optional[SyncMCPClient] = None):
        self.mcp = mcp_client
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

    def _parse_tool_name(self, full_name: str) -> tuple:
        if "__" in full_name:
            return full_name.split("__", 1)
        return "unknown", full_name

    def process_conversation(self, messages: List[Dict], manual_limit: int = 0) -> tuple:
        tools_schema = self.mcp.get_tools_schema() if self.mcp else []
        current_messages = messages.copy()

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
                    return msg.content, current_messages
                return "", current_messages

            current_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls]
            })

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                srv_name, tool_name = self._parse_tool_name(tc.function.name)
                
                # --- Dynamic Limit Logic ---
                # Override the AI's 'limit' argument if the manual slider is set > 0
                if "search" in tool_name.lower() and manual_limit > 0:
                    args["limit"] = manual_limit
                    logger.info(f"Manual override: Setting {tool_name} limit to {manual_limit}")
                
                # --- Log the final JSON string to the console ---
                # This shows exactly what is being sent to the MCP server
                json_string = json.dumps(args)
                print(f"DEBUG - Tool: {srv_name}.{tool_name} | Args JSON: {json_string}")
                logger.info(f"Executing {srv_name}.{tool_name} with args {json_string}")

                result = self.mcp.execute_tool(srv_name, tool_name, args)
                logger.info(f"Tool result received (length {len(result)})")
                
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })
            # Continue loop to let DeepSeek see tool results


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


def connect_mcp(config_path: str):
    with st.spinner("Connecting to MCP servers..."):
        mcp = SyncMCPClient()
        num = mcp.connect(config_path)
        st.session_state.mcp_client = mcp
        st.session_state.engine = DeepSeekEngine(mcp)
        st.session_state.mcp_connected = True
        st.session_state.conv.clear()
        # Updated system prompt with dynamic engine selection guidance
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

            "If default limit=0 use a higher 'limit' if the user asks for a comprehensive list, and a lower limit (1-3) for quick fact-checks to save tokens."
        )
        st.session_state.messages_ui = []
    return num


def disconnect_mcp():
    if st.session_state.mcp_client:
        st.session_state.mcp_client.cleanup()
    st.session_state.mcp_client = None
    st.session_state.engine = DeepSeekEngine(None)
    st.session_state.mcp_connected = False
    st.session_state.conv.clear()
    st.session_state.conv.add_message("system", "You are a helpful AI assistant. No external tools.")
    st.session_state.messages_ui = []


def main():
    st.set_page_config(page_title="DeepSeek + MCP (Tools, Prompts, Resources)", page_icon="🤖", layout="wide")
    st.title("🤖 DeepSeek Chat with MCP")
    st.markdown("Connect MCP servers to use tools, prompts, and resources.")
    init_state()

    with st.sidebar:
        st.header("🔌 MCP Connection")
        if not st.session_state.mcp_connected:
            cfg = st.text_input("Config file", value="servers.yaml")
            if st.button("Connect MCP Servers", use_container_width=True):
                try:
                    n = connect_mcp(cfg)
                    st.success(f"Connected to {n} server(s)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        else:
            st.success("MCP active")          
            
            # --- New Slider Section ---
            st.subheader("⚙️ Tool Settings")
            search_limit = st.slider(
                "Manual Search Limit (0 = AI Decide)", 
                min_value=0, 
                max_value=10, 
                value=0,
                help="If set to 0, the AI follows its system prompt instructions. Values 1-10 will override the AI."
            )
            st.divider()
            # --------------------------

            #if st.button("Disconnect MCP", use_container_width=True):
            if st.button("Disconnect MCP", use_container_width=True):
                disconnect_mcp()
                st.rerun()

            # Display tools
            with st.expander("🔧 Available Tools"):
                servers = st.session_state.mcp_client.get_servers()
                for srv_name, srv_info in servers.items():
                    st.markdown(f"**{srv_name}**")
                    for tool in srv_info.get("tools", []):
                        st.markdown(f"- `{tool['name']}`: {tool.get('description', 'No description')[:100]}")

            # Display prompts with "Use" buttons
            with st.expander("📋 Available Prompts"):
                prompts = st.session_state.mcp_client.list_prompts()
                for prompt in prompts:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{prompt['name']}**")
                        if prompt.get("description"):
                            st.caption(prompt["description"])
                    with col2:
                        if st.button("Use", key=f"prompt_{prompt['name']}"):
                            # Fetch prompt messages and add to conversation
                            messages = st.session_state.mcp_client.get_prompt(prompt["name"])
                            for msg in messages:
                                role = msg.get("role", "user")
                                content = msg.get("content", {}).get("text", "")
                                if content:
                                    st.session_state.conv.add_message(role, content)
                                    st.session_state.messages_ui.append({"role": role, "content": content})
                            st.rerun()

            # Display resources with "Read" buttons
            with st.expander("📁 Available Resources"):
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
                            # Add resource content as a user message
                            prompt_text = f"Content from resource {resource['uri']}:\n\n{content}"
                            st.session_state.conv.add_message("user", prompt_text)
                            st.session_state.messages_ui.append({"role": "user", "content": prompt_text})
                            st.rerun()

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.conv.clear()
            sys_msg = "You have access to MCP tools, prompts, and resources." if st.session_state.mcp_connected else "You are a helpful AI assistant."
            st.session_state.conv.add_message("system", sys_msg)
            st.session_state.messages_ui = []
            st.rerun()

    # Chat display
    for msg in st.session_state.messages_ui:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Your message"):
        st.session_state.messages_ui.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.conv.add_message("user", prompt)

        with st.spinner("Thinking..."):
            try:
                # Pass 'search_limit' (the value from the slider) to the engine
                answer, new_conv = st.session_state.engine.process_conversation(
                    st.session_state.conv.get_messages(),
                    manual_limit=search_limit                
                )
                st.session_state.conv.messages = new_conv
                st.session_state.messages_ui.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
            except Exception as e:
                st.error(f"Error: {e}")
                logger.exception("Processing error")

    st.caption("Powered by DeepSeek. MCP servers provide tools, prompts, and resources.")


if __name__ == "__main__":
    main()
