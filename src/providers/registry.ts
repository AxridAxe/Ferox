import type { ChatMessage, FreeModel, Provider } from "./types.js";

const CACHE_TTL_MS = 5 * 60 * 1000;

export class ProviderRegistry {
  private providers: Provider[] = [];
  private cache: { models: FreeModel[]; fetchedAt: number } | null = null;

  register(provider: Provider): void {
    this.providers.push(provider);
  }

  private providerByName(name: string): Provider {
    const provider = this.providers.find((p) => p.name === name);
    if (!provider) {
      throw new Error(`Unknown provider: ${name}`);
    }
    return provider;
  }

  async listFreeModels(): Promise<FreeModel[]> {
    if (this.cache && Date.now() - this.cache.fetchedAt < CACHE_TTL_MS) {
      return this.cache.models;
    }
    const lists = await Promise.all(
      this.providers.map((p) =>
        p.listFreeModels().catch(() => [] as FreeModel[])
      )
    );
    const models = lists.flat();
    this.cache = { models, fetchedAt: Date.now() };
    return models;
  }

  /**
   * Tries free models in order until one responds successfully, since free
   * tiers are frequently rate-limited or temporarily unavailable.
   */
  async chatWithFailover(
    messages: ChatMessage[],
    maxAttempts = 3
  ): Promise<{ modelId: string; content: string }> {
    const models = await this.listFreeModels();
    if (models.length === 0) {
      throw new Error("No free models currently available");
    }

    const shuffled = [...models].sort(() => Math.random() - 0.5);
    const attempts = shuffled.slice(0, Math.max(maxAttempts, 1));

    let lastError: unknown;
    for (const model of attempts) {
      try {
        const provider = this.providerByName(model.provider);
        const content = await provider.chat(model.id, messages);
        return { modelId: model.id, content };
      } catch (err) {
        lastError = err;
      }
    }
    throw new Error(
      `All model attempts failed: ${
        lastError instanceof Error ? lastError.message : String(lastError)
      }`
    );
  }
}
