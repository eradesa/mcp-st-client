from fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import List

# Create server
mcp = FastMCP("Other Inputs")

class Person(BaseModel):
    first_name: str = Field(..., description="The person's first name")
    last_name: str = Field(..., description="The person's last name")
    years_of_experience: int = Field(..., description="Number of years of experience")
    previous_addresses: List[str] = Field(default_factory=list, description="List of previous addresses")


@mcp.tool()
def add_person_to_member_database(person: Person) -> str:
    """
    Logs the personal details of the given person to the database.
    Args:
        person (Person): An instance of the Person class containing the following personal details:
            - first_name (str): The person's given name.
            - last_name (str): The person's family name.
            - years_of_experience (int): Number of years of experienceh.
            - previous_addresses (List[str]): A list of the person's previous residential addresses.

    Returns:
        str: A confirmation message indicating that the data has been logged.

    """

    with open("log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"First Name: {person.first_name}\n")
        log_file.write(f"Last Name: {person.last_name}\n")
        log_file.write(f"Years of Experience: {person.years_of_experience}\n")
        log_file.write("Previous Addresses:\n")
        for idx, address in enumerate(person.previous_addresses, 1):
            log_file.write(f"  {idx}. {address}\n")
        log_file.write("\n")

    return "Data has been logged"

@mcp.tool()
def read_full_log() -> str:
    """
    Reads and returns the complete contents of the log.txt file.
    
    Returns:
        str: The full contents of the log.txt file, or a message if the file doesn't exist.
    """
    try:
        with open("log.txt", "r", encoding="utf-8") as log_file:
            content = log_file.read()
        if not content:
            return "The log.txt file is empty."
        return content
    except FileNotFoundError:
        return "The log.txt file does not exist yet."


@mcp.tool()
def search_person_by_name(name: str) -> str:
    """
    Searches the log.txt file for a person by their first or last name and returns their details.
    
    Args:
        name (str): The first name or last name of the person to search for.
    
    Returns:
        str: The details of the person if found, or a message indicating no match was found.
    """
    try:
        with open("log.txt", "r", encoding="utf-8") as log_file:
            content = log_file.read()
    except FileNotFoundError:
        return "The log.txt file does not exist yet."
    
    # Split the content into individual person records (separated by double newlines)
    records = content.split("\n\n")
    search_name = name.lower()
    matches = []
    
    for record in records:
        if record.strip() and search_name in record.lower():
            matches.append(record.strip())
    
    if matches:
        result = f"Found {len(matches)} match(es) for '{name}':\n\n"
        for match in matches:
            result += match + "\n" + "="*50 + "\n"
        return result
    else:
        return f"No records found for '{name}'."


if __name__ == "__main__":
    mcp.run()
