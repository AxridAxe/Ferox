export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface FreeModel {
  id: string;
  provider: string;
}

export interface Provider {
  name: string;
  listFreeModels(): Promise<FreeModel[]>;
  chat(modelId: string, messages: ChatMessage[]): Promise<string>;
}
