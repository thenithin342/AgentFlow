export function streamAgentMeta(agent) {
  if (agent === "chat_agent") return "chat";
  if (agent === "research_agent") return "research";
  if (agent === "analysis_agent") return "analysis";
  if (agent === "blog_writer") return "blog";
  if (agent === "memory_reader") return "ltm read";
  if (agent === "memory_writer") return "ltm write";
  if (agent === "stm_compressor") return "stm compress";
  return agent;
}

export function agentLabelFromRoute(route) {
  if (route === "chat") return "chat_agent";
  if (route === "research") return "research_agent";
  if (route === "analysis") return "analysis_agent";
  if (route === "blog") return "blog_writer";
  return "synthesizer";
}

export function formatLastSeen(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isFinite(d.getTime()) ? d.toLocaleString() : "—";
}

export const uuid = () => {
  const webCrypto = globalThis.crypto;
  if (webCrypto?.randomUUID) return webCrypto.randomUUID();
  if (!webCrypto?.getRandomValues) {
    throw new Error("Web Crypto API is required to generate thread IDs");
  }
  return ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, c =>
    (c ^ (webCrypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (c / 4)))).toString(16)
  );
};

export function parseCitations(text) {
  if (!text) return [];
  const out = [];
  let m;
  const re = /\[(\d+)\]\s+(https?:\/\/\S+)/g;
  try {
    while ((m = re.exec(text)) !== null) {
      let host = m[2];
      try {
        host = new URL(m[2]).hostname.replace(/^www\./, "");
      } catch {
        // Malformed URL — fall back to the raw match
      }
      out.push({ n: Number(m[1]), url: m[2], host });
    }
  } catch {
    return out;
  }
  return out;
}

export function now() {
  return new Date().toLocaleTimeString("en-GB", { hour12: false });
}
