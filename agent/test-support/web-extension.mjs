import { Type } from "typebox";

export default function fakeWebExtension(pi) {
  pi.registerTool({
    name: "web_search",
    label: "Web search",
    description: "Test web search",
    parameters: Type.Object({ query: Type.String() }),
    async execute() {
      return { content: [{ type: "text", text: "Search complete" }], details: {} };
    },
  });
  pi.registerTool({
    name: "fetch_content",
    label: "Fetch content",
    description: "Test page fetch",
    parameters: Type.Object({ url: Type.String() }),
    async execute() {
      return { content: [{ type: "text", text: "Page fetched" }], details: {} };
    },
  });
}
