import { createMcpAdapter } from "pi-mcp-adapter";

import { createTaibuMcpConfig } from "./taibu-mcp-config.mjs";

export default createMcpAdapter({ config: createTaibuMcpConfig() });
