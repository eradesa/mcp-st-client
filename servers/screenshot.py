"""
FastMCP Screenshot Server
A simple MCP server that provides screenshot capture functionality
"""

from fastmcp import FastMCP
import base64
import subprocess
import os
from pathlib import Path

# Create the FastMCP server instance
mcp = FastMCP("Screenshot Server 📸")


@mcp.tool
def take_screenshot(filename: str = "screenshot.png") -> dict:
    """
    Take a screenshot of the current screen and save it to a file.
    
    Args:
        filename: The filename to save the screenshot as (default: screenshot.png)
    
    Returns:
        A dictionary with success status, file path, and base64-encoded image data
    """
    try:
        # Use gnome-screenshot for Linux or scrot as fallback
        output_path = Path.home() / "Pictures" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try using gnome-screenshot first
        try:
            subprocess.run(
                ["gnome-screenshot", "-f", str(output_path)],
                check=True,
                capture_output=True,
                timeout=5
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            # Fallback to scrot if gnome-screenshot is not available
            subprocess.run(
                ["scrot", str(output_path)],
                check=True,
                capture_output=True,
                timeout=5
            )
        
        # Read and encode the image as base64
        with open(output_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        return {
            "success": True,
            "message": f"Screenshot saved to {output_path}",
            "file_path": str(output_path),
            "image_base64": image_data,
            "file_size": output_path.stat().st_size
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to take screenshot: {str(e)}",
            "error": str(e)
        }


@mcp.tool
def capture_region(x: int, y: int, width: int, height: int, filename: str = "region.png") -> dict:
    """
    Capture a specific region of the screen.
    
    Args:
        x: X coordinate of the top-left corner
        y: Y coordinate of the top-left corner
        width: Width of the region to capture
        height: Height of the region to capture
        filename: The filename to save the screenshot as (default: region.png)
    
    Returns:
        A dictionary with success status, file path, and image information
    """
    try:
        output_path = Path.home() / "Pictures" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use scrot to capture a specific region
        subprocess.run(
            ["scrot", "-a", f"{x},{y},{width},{height}", str(output_path)],
            check=True,
            capture_output=True,
            timeout=5
        )
        
        # Read and encode the image as base64
        with open(output_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        return {
            "success": True,
            "message": f"Region screenshot saved to {output_path}",
            "file_path": str(output_path),
            "image_base64": image_data,
            "region": {"x": x, "y": y, "width": width, "height": height},
            "file_size": output_path.stat().st_size
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to capture region: {str(e)}",
            "error": str(e)
        }


@mcp.tool
def get_screen_info() -> dict:
    """
    Get information about the current screen(s).
    
    Returns:
        A dictionary with screen resolution and layout information
    """
    try:
        # Use xrandr to get screen information
        result = subprocess.run(
            ["xrandr"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        screens = []
        for line in result.stdout.split('\n'):
            if ' connected ' in line:
                parts = line.split()
                screens.append({
                    "name": parts[0],
                    "connected": True,
                    "primary": "primary" in line
                })
            elif ' disconnected ' in line:
                parts = line.split()
                screens.append({
                    "name": parts[0],
                    "connected": False
                })
        
        return {
            "success": True,
            "screens": screens,
            "raw_output": result.stdout
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to get screen info: {str(e)}",
            "error": str(e)
        }


if __name__ == "__main__":
    print("🚀 Starting FastMCP Screenshot Server...")
    print("Available tools:")
    print("  - take_screenshot(filename): Capture full screen")
    print("  - capture_region(x, y, width, height, filename): Capture screen region")
    print("  - get_screen_info(): Get screen information")
    mcp.run()
