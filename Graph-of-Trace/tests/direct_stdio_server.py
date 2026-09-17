"""Test harness: real server and real direct writer, with networking forbidden."""
import socket
import sys


def no_network(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto"):
        raise AssertionError(f"Network operation forbidden in direct MCP test: {event}")
    if event == "socket.__new__" and args[1] in (socket.AF_INET, socket.AF_INET6):
        raise AssertionError("Internet sockets forbidden in direct MCP test")
    if event == "import" and (args[0] == "Monitor.steps_llm" or args[0].startswith("Monitor.adapter")):
        raise AssertionError("LLM module imported in direct MCP test")


sys.addaudithook(no_network)
from server import main

if __name__ == "__main__":
    main()
