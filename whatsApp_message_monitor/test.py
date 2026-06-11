# test_mcp_call.py
import sys
sys.path.insert(0, '../')  # adjust path to streamlit_mcp_app
from streamlit_mcp_app import SyncMCPClient

mcp = SyncMCPClient()
mcp.connect("../servers.yaml")   # use your actual path
news = mcp.execute_tool("sri_lanka_news", "get_headlines", {"limit": 30})
print("RAW NEWS OUTPUT:")
print(news)
mcp.cleanup()