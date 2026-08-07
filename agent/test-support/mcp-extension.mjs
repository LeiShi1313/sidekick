import { Type } from "typebox";

export default function fakeMcpExtension(pi) {
  let initialized = false;

  pi.on("session_start", () => {
    initialized = true;
  });

  pi.registerTool({
    name: "mcp",
    label: "MCP",
    description: "Test MCP gateway",
    promptSnippet: "Discover and call MCP tools",
    parameters: Type.Object({ search: Type.Optional(Type.String()) }),
    async execute() {
      return {
        content: [
          {
            type: "text",
            text: initialized ? "MCP initialized" : "MCP not initialized",
          },
        ],
        details: {},
      };
    },
  });
  pi.registerTool({
    name: "mcpScript",
    label: "MCP Script",
    description: "Test scripting surface that Sidekick must remove",
    parameters: Type.Object({ code: Type.String() }),
    async execute() {
      return { content: [{ type: "text", text: "test result" }], details: {} };
    },
  });
}
