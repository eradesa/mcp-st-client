from fastmcp import FastMCP

mcp = FastMCP("Resources")

# -------------------------------------------------------------------------
# Tools (callable by LLM)
# -------------------------------------------------------------------------
@mcp.tool
def get_inventory_overviews() -> str:
    """Get inventory overview from get_inventory_overview"""
    # Call the resource function and return its result
    return get_inventory_overview()

@mcp.tool 
def get_inventory_price(item_name: str) -> str:
    """Get price for a specific inventory item"""
    # First get the ID from the name
    item_id = get_inventory_id_from_inventory_name(item_name)
    # Then get the price using the ID
    return get_inventory_price_from_inventory_id(item_id)

# -------------------------------------------------------------------------
# Resources (internal data endpoints)
# -------------------------------------------------------------------------
@mcp.resource("inventory://overview")
def get_inventory_overview() -> str:
    """
    Returns overview of inventory
    """
    overview = """
    Inventory Overview:
    - Coffee
    - Tea
    - Cookies
    """
    return overview.strip()

# Data mappings
inventory_id_to_price = {
    "123": "6.99",
    "456": "17.99",
    "789": "84.99"
}

inventory_name_to_id = {
    "Coffee": "123",
    "Tea": "456",
    "Cookies": "789"
}

@mcp.resource("inventory://{inventory_id}/price")
def get_inventory_price_from_inventory_id(inventory_id: str) -> str:
    """
    Returns price from inventory id
    """
    # Add error handling
    if inventory_id not in inventory_id_to_price:
        return f"Error: No price found for ID '{inventory_id}'"
    return inventory_id_to_price[inventory_id]

@mcp.resource("inventory://{inventory_name}/id")
def get_inventory_id_from_inventory_name(inventory_name: str) -> str:
    """
    Returns id from inventory name
    """
    # Add error handling
    if inventory_name not in inventory_name_to_id:
        return f"Error: No ID found for item '{inventory_name}'"
    return inventory_name_to_id[inventory_name]

if __name__ == "__main__":
    mcp.run()