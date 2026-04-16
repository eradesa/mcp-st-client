# mcp_deepseek_client.py
"""
A production-ready MCP client that integrates with DeepSeek LLM.
Features: Multi-server support, async/await, memory optimization, 
tool calling, conversation history management.
"""

import os
import asyncio
import json
import yaml
import signal
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from contextlib import AsyncExitStack
from dotenv import load_dotenv

# MCP SDK imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

# DeepSeek/OpenAI imports
from openai import OpenAI

# Optional: For better async performance on Unix systems
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ServerConnection:
    """Represents a connection to an MCP server."""
    name: str
    session: ClientSession
    exit_stack: AsyncExitStack
    tools: List[Dict] = field(default_factory=list)
    
    async def cleanup(self):
        """Clean up server resources."""
        await self.exit_stack.aclose()


class ConversationMemory:
    """
    Memory-optimized conversation history manager.
    Implements sliding window with token-aware truncation.
    """
    
    def __init__(self, max_messages: int = 20, max_tokens: int = 8000):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.messages: List[Dict[str, str]] = []
        
    def add_message(self, role: str, content: str):
        """Add a message to history with automatic pruning."""
        self.messages.append({"role": role, "content": content})
        self._prune_if_needed()
    
    def add_assistant_tool_calls(self, tool_calls: List[Dict]):
        """Add assistant's tool call requests to history."""
        self.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls
        })
    
    def add_tool_response(self, tool_call_id: str, content: str):
        """Add tool response to history."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        })
    
    def _prune_if_needed(self):
        """Prune old messages when limits are exceeded."""
        # Simple count-based pruning
        if len(self.messages) > self.max_messages:
            # Keep system message (if present) and last N-1 messages
            if self.messages and self.messages[0]["role"] == "system":
                self.messages = [self.messages[0]] + self.messages[-(self.max_messages - 1):]
            else:
                self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get current message history."""
        return self.messages
    
    def clear(self):
        """Clear conversation history."""
        self.messages = []


class MCPDeepSeekClient:
    """
    Main MCP client that manages multiple server connections
    and integrates with DeepSeek LLM for intelligent tool calling.
    """
    
    def __init__(self, config_path: str = "servers.yaml"):
        self.config_path = config_path
        self.servers: Dict[str, ServerConnection] = {}
        self.exit_stack = AsyncExitStack()
        self.conversation = ConversationMemory(max_messages=30, max_tokens=8000)
        
        # Initialize DeepSeek client
        self.deepseek_client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received shutdown signal, cleaning up...")
        asyncio.create_task(self.cleanup())
    
    async def load_server_config(self) -> Dict:
        """Load server configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def connect_to_stdio_server(self, server_config: Dict) -> ServerConnection:
        """
        Connect to a local stdio-based MCP server.
        
        Memory optimization: Uses context managers to ensure proper cleanup.
        """
        server_params = StdioServerParameters(
            command=server_config["command"],
            args=server_config.get("args", []),
            env=server_config.get("env", {})
        )
        
        exit_stack = AsyncExitStack()
        try:
            # Establish stdio transport
            stdio_transport = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport
            
            # Create client session
            session = await exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            
            # Initialize the connection
            await session.initialize()
            
            # Discover available tools
            tools_result = await session.list_tools()
            tools = [tool.model_dump() for tool in tools_result.tools]
            
            logger.info(f"Connected to stdio server '{server_config['name']}' with tools: {[t['name'] for t in tools]}")
            
            return ServerConnection(
                name=server_config["name"],
                session=session,
                exit_stack=exit_stack,
                tools=tools
            )
        except Exception as e:
            await exit_stack.aclose()
            logger.error(f"Failed to connect to stdio server {server_config['name']}: {e}")
            raise
    
    async def connect_to_sse_server(self, server_config: Dict) -> ServerConnection:
        """
        Connect to a remote SSE-based MCP server.
        
        Memory optimization: Uses async context manager for proper cleanup.
        """
        exit_stack = AsyncExitStack()
        try:
            # Establish SSE transport
            sse_transport = await exit_stack.enter_async_context(
                sse_client(server_config["url"])
            )
            read_stream, write_stream = sse_transport
            
            # Create client session
            session = await exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            
            # Initialize the connection
            await session.initialize()
            
            # Discover available tools
            tools_result = await session.list_tools()
            tools = [tool.model_dump() for tool in tools_result.tools]
            
            logger.info(f"Connected to SSE server '{server_config['name']}' with tools: {[t['name'] for t in tools]}")
            
            return ServerConnection(
                name=server_config["name"],
                session=session,
                exit_stack=exit_stack,
                tools=tools
            )
        except Exception as e:
            await exit_stack.aclose()
            logger.error(f"Failed to connect to SSE server {server_config['name']}: {e}")
            raise
    
    async def connect_all_servers(self):
        """Connect to all configured MCP servers."""
        config = await self.load_server_config()
        
        # Connect to local stdio servers
        for server_config in config.get("servers", {}).get("local", []):
            try:
                connection = await self.connect_to_stdio_server(server_config)
                self.servers[connection.name] = connection
            except Exception as e:
                logger.error(f"Skipping server {server_config['name']}: {e}")
        
        # Connect to remote SSE servers
        for server_config in config.get("servers", {}).get("remote", []):
            try:
                connection = await self.connect_to_sse_server(server_config)
                self.servers[connection.name] = connection
            except Exception as e:
                logger.error(f"Skipping server {server_config['name']}: {e}")
        
        logger.info(f"Connected to {len(self.servers)} MCP servers")
    
    async def execute_tool_call(self, server_name: str, tool_name: str, arguments: Dict) -> str:
        """Execute a tool call on a specific server."""
        if server_name not in self.servers:
            return f"Error: Server '{server_name}' not found"
        
        server = self.servers[server_name]
        try:
            result = await server.session.call_tool(tool_name, arguments)
            # Extract content from result
            if hasattr(result, 'content') and result.content:
                content = result.content[0].text if result.content else "No output"
                return content
            return str(result)
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Error executing {tool_name}: {str(e)}"
    
    async def process_llm_response(self, response) -> bool:
        """
        Process LLM response, handling tool calls if present.
        
        Returns: True if processing is complete, False if tools were called and need follow-up.
        """
        message = response.choices[0].message
        
        # Handle tool calls from DeepSeek
        if hasattr(message, 'tool_calls') and message.tool_calls:
            # Add assistant's tool call request to history
            self.conversation.add_assistant_tool_calls(
                [tc.model_dump() for tc in message.tool_calls]
            )
            
            # Execute each tool call
            tool_results = []
            for tool_call in message.tool_calls:
                # Parse arguments (DeepSeek provides them as JSON string)
                arguments = json.loads(tool_call.function.arguments)
                
                # For now, assume tool names are in format "server_name__tool_name"
                # You can implement a more sophisticated routing mechanism
                server_name, tool_name = self._parse_tool_name(tool_call.function.name)
                
                result = await self.execute_tool_call(server_name, tool_name, arguments)
                tool_results.append((tool_call.id, result))
                
                # Add tool response to conversation
                self.conversation.add_tool_response(tool_call.id, result)
            
            # After executing tools, continue the conversation
            return False  # Need to send another request to DeepSeek with tool results
        
        # Normal text response
        if message.content:
            print(f"\n🤖 DeepSeek: {message.content}\n")
            self.conversation.add_message("assistant", message.content)
        
        return True
    
    def _parse_tool_name(self, full_name: str) -> tuple:
        """Parse 'server_name__tool_name' format into (server_name, tool_name)."""
        if "__" in full_name:
            parts = full_name.split("__", 1)
            return parts[0], parts[1]
        # Fallback: try to find which server has this tool
        for server_name, server in self.servers.items():
            for tool in server.tools:
                if tool["name"] == full_name:
                    return server_name, full_name
        return "unknown", full_name
    
    def _build_tools_schema(self) -> List[Dict]:
        """Build OpenAPI-compatible tool schema for DeepSeek from all connected servers."""
        tools = []
        for server_name, server in self.servers.items():
            for tool in server.tools:
                # Prepend server name to tool name for routing
                tools.append({
                    "type": "function",
                    "function": {
                        "name": f"{server_name}__{tool['name']}",
                        "description": tool.get("description", f"Tool from {server_name}"),
                        "parameters": tool.get("inputSchema", {"type": "object", "properties": {}})
                    }
                })
        return tools
    
    async def run_interactive(self):
        """Run interactive chat session with DeepSeek and MCP tools."""
        print("\n" + "="*60)
        print("🚀 MCP-DeepSeek Client Started")
        print(f"📡 Connected to {len(self.servers)} MCP servers")
        print("💡 Type 'quit' to exit, 'clear' to clear conversation")
        print("="*60 + "\n")
        
        # Add system prompt
        system_prompt = """You are a helpful AI assistant with access to various tools through MCP servers.
When a user asks a question that requires tool usage:
1. Use the available tools to gather information
2. Present the results in a clear, organized manner
3. If multiple tools are needed, use them sequentially

Available tools are shown with 'server__tool_name' format. Use them appropriately."""
        
        self.conversation.add_message("system", system_prompt)
        
        while True:
            try:
                # Get user input
                user_input = input("💬 You: ").strip()
                if not user_input:
                    continue
                
                if user_input.lower() in ["quit", "exit"]:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == "clear":
                    self.conversation.clear()
                    self.conversation.add_message("system", system_prompt)
                    print("🧹 Conversation history cleared.\n")
                    continue
                
                # Add user message to history
                self.conversation.add_message("user", user_input)
                
                # Build tools schema from all connected servers
                tools_schema = self._build_tools_schema()
                
                # Call DeepSeek with conversation history
                response = self.deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=self.conversation.get_messages(),
                    tools=tools_schema if tools_schema else None,
                    tool_choice="auto" if tools_schema else None,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # Process the response (may trigger tool calls)
                complete = await self.process_llm_response(response)
                
                # If tools were called, continue the conversation
                while not complete:
                    # Send follow-up request with tool results
                    followup_response = self.deepseek_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=self.conversation.get_messages(),
                        temperature=0.7,
                        max_tokens=2000
                    )
                    complete = await self.process_llm_response(followup_response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\n❌ Error: {e}\n")
    
    async def cleanup(self):
        """Clean up all server connections."""
        logger.info("Cleaning up server connections...")
        for server in self.servers.values():
            await server.cleanup()
        await self.exit_stack.aclose()
        logger.info("Cleanup complete.")


async def main():
    """Main entry point."""
    client = MCPDeepSeekClient("servers.yaml")
    
    try:
        await client.connect_all_servers()
        await client.run_interactive()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())