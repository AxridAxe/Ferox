import type { Request, Response } from "express";
import { z } from "zod";
import type { ProviderRegistry } from "../providers/registry.js";
import { FEROX_SYSTEM_PROMPT } from "../persona/systemPrompt.js";

const chatRequestSchema = z.object({
  messages: z
    .array(
      z.object({
        role: z.enum(["user", "assistant"]),
        content: z.string().min(1).max(8000),
      })
    )
    .min(1)
    .max(50),
});

export function createChatHandler(registry: ProviderRegistry) {
  return async function chatHandler(req: Request, res: Response): Promise<void> {
    const parsed = chatRequestSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "Invalid request body" });
      return;
    }

    const messages = [
      { role: "system" as const, content: FEROX_SYSTEM_PROMPT },
      ...parsed.data.messages,
    ];

    try {
      const { content } = await registry.chatWithFailover(messages);
      res.json({ reply: content });
    } catch (err) {
      console.error("Chat routing failed:", err);
      res.status(503).json({ error: "No model was able to handle this request" });
    }
  };
}
