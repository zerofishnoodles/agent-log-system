"""
Simple tools for MAS Reliability Framework

Provides basic tools for testing the framework.
"""

import json
import re
from typing import Dict, Any, Optional


class Calculator:
    """Simple calculator tool for testing tool usage."""
    
    def __init__(self):
        self.name = "calculator"
        self.description = "Performs basic arithmetic calculations"
    
    def run(self, expression: str) -> str:
        """
        Evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression as string
            
        Returns:
            Result of the calculation or error message
        """
        try:
            # Simple safety check - only allow basic operations
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Only basic arithmetic operations allowed"
            
            # Evaluate the expression
            result = eval(expression)
            return f"Result: {result}"
            
        except Exception as e:
            return f"Error: {str(e)}"


class Echo:
    """Simple echo tool for testing tool input/output."""
    
    def __init__(self):
        self.name = "echo"
        self.description = "Echoes back the input message"
    
    def run(self, message: str) -> str:
        """
        Echo back the input message.
        
        Args:
            message: Message to echo back
            
        Returns:
            The same message with echo prefix
        """
        return f"Echo: {message}"


class CodeValidator:
    """Simple code validator for testing verification scenarios."""
    
    def __init__(self):
        self.name = "code_validator"
        self.description = "Validates Python code syntax"
    
    def run(self, code: str) -> str:
        """
        Validate Python code syntax.
        
        Args:
            code: Python code to validate
            
        Returns:
            Validation result
        """
        try:
            # Try to compile the code
            compile(code, '<string>', 'exec')
            return "VALIDATION: Code syntax is valid"
        except SyntaxError as e:
            return f"VALIDATION: Syntax error - {str(e)}"
        except Exception as e:
            return f"VALIDATION: Error - {str(e)}"


class FileWriter:
    """Simple file writer for testing file operations."""
    
    def __init__(self):
        self.name = "file_writer"
        self.description = "Writes content to a file"
    
    def run(self, content: str, filename: str = "output.txt") -> str:
        """
        Write content to a file.
        
        Args:
            content: Content to write
            filename: Name of the file to write to
            
        Returns:
            Success or error message
        """
        try:
            # Simple safety check for filename
            if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
                return "Error: Invalid filename"
            
            with open(filename, 'w') as f:
                f.write(content)
            
            return f"Successfully wrote {len(content)} characters to {filename}"
            
        except Exception as e:
            return f"Error writing file: {str(e)}"


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self.tools = {
            "calculator": Calculator(),
            "echo": Echo(),
            "code_validator": CodeValidator(),
            "file_writer": FileWriter()
        }
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get a tool by name."""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> Dict[str, str]:
        """List all available tools and their descriptions."""
        return {name: tool.description for name, tool in self.tools.items()}
    
    def run_tool(self, tool_name: str, input_data: str) -> str:
        """Run a tool with input data."""
        tool = self.get_tool(tool_name)
        if tool:
            return tool.run(input_data)
        else:
            return f"Error: Tool '{tool_name}' not found"


# Global tool registry instance
tool_registry = ToolRegistry()

