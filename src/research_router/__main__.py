import asyncio
import os

from research_router.server import create_mcp_server


async def main():
    mcp = create_mcp_server()

    transport = os.getenv("MCP_TRANSPORT", "stdio")

    if transport == "http":
        port = int(os.getenv("PORT", "10000"))

        await mcp.run_streamable_http_async(
            host="0.0.0.0",
            port=port,
            streamable_http_path="/mcp",
        )
    else:
        await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())