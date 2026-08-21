import express from "express";
import helmet from "helmet";
import cors from "cors";
import rateLimit from "express-rate-limit";
import { config } from "./config.js";
import { requireApiKey } from "./middleware/auth.js";
import { errorHandler } from "./middleware/errorHandler.js";
import { createChatHandler } from "./router/chat.js";
import { ProviderRegistry } from "./providers/registry.js";
import { createOpenRouterProvider } from "./providers/openrouter.js";

export function createServer(): express.Express {
  const registry = new ProviderRegistry();
  if (config.openRouterApiKey) {
    registry.register(createOpenRouterProvider(config.openRouterApiKey));
  }

  const app = express();

  app.use(helmet());
  app.use(
    cors({
      origin: config.allowedOrigins.length > 0 ? config.allowedOrigins : false,
    })
  );
  app.use(express.json({ limit: "100kb" }));
  app.use(
    rateLimit({
      windowMs: 60_000,
      limit: 30,
      standardHeaders: true,
      legacyHeaders: false,
    })
  );

  app.get("/health", (_req, res) => {
    res.json({ status: "ok" });
  });

  app.use(requireApiKey);
  app.post("/v1/chat", createChatHandler(registry));

  app.use(errorHandler);

  return app;
}
