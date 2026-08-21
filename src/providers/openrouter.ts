import type { ChatMessage, FreeModel, Provider } from "./types.js";

const BASE_URL = "https://openrouter.ai/api/v1";

interface OpenRouterModel {
  id: string;
  pricing?: { prompt?: string; completion?: string };
}

export function createOpenRouterProvider(apiKey: string): Provider {
  return {
    name: "openrouter",

    async listFreeModels(): Promise<FreeModel[]> {
      const res = await fetch(`${BASE_URL}/models`);
      if (!res.ok) {
        throw new Error(`OpenRouter model list failed: ${res.status}`);
      }
      const data = (await res.json()) as { data: OpenRouterModel[] };
      return data.data
        .filter((m) => m.pricing?.prompt === "0" && m.pricing?.completion === "0")
        .map((m) => ({ id: m.id, provider: "openrouter" }));
    },

    async chat(modelId: string, messages: ChatMessage[]): Promise<string> {
      const res = await fetch(`${BASE_URL}/chat/completions`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model: modelId, messages }),
      });
      if (!res.ok) {
        throw new Error(`OpenRouter chat failed: ${res.status}`);
      }
      const data = (await res.json()) as {
        choices: { message: { content: string } }[];
      };
      const content = data.choices?.[0]?.message?.content;
      if (!content) {
        throw new Error("OpenRouter returned no content");
      }
      return content;
    },
  };
}
