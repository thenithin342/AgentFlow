import { renderHook, act } from "@testing-library/react";
import useSSE from "../src/hooks/useSSE";
import { expect, test, describe, vi } from "vitest";
import { apiFetch } from "../src/api/client";

// Mock the API client
vi.mock("../src/api/client", () => ({
  apiFetch: vi.fn(),
}));

describe("useSSE", () => {
  test("uses the authoritative final answer after a draft", async () => {
    const wire = 'data: preliminary draft\n\ndata: [FINAL:"verified answer"]\n\ndata: [DONE]\n\n';
    apiFetch.mockResolvedValueOnce({
      ok: true,
      body: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(wire));
          controller.close();
        },
      }),
    });
    const { result } = renderHook(() => useSSE({ threadId: "final-test", showError: vi.fn() }));
    await act(async () => { await result.current.sendMessage("hello"); });
    expect(result.current.messages.at(-1).text).toBe("verified answer");
    expect(result.current.messages.at(-1).streaming).toBe(false);
  });

  test("initializes with default state", () => {
    const { result } = renderHook(() =>
      useSSE({
        threadId: "test-thread",
        showError: vi.fn(),
        reviewRequired: false,
        setReviewRequired: vi.fn(),
        setEditingReview: vi.fn(),
        activeTab: "chat",
      })
    );

    expect(result.current.messages).toEqual([]);
    expect(result.current.trace).toEqual([]);
    expect(result.current.isStreaming).toBe(false);
  });

  test("resetStreamState clears state and aborts", () => {
    const { result } = renderHook(() =>
      useSSE({
        threadId: "test-thread",
        showError: vi.fn(),
      })
    );
    
    act(() => {
      result.current.setMessages([{ role: "user", text: "hello" }]);
      result.current.setTrace([{ node: "router", active: true }]);
      result.current.setIsStreaming(true);
    });
    
    expect(result.current.messages.length).toBe(1);
    
    act(() => {
      result.current.resetStreamState();
    });
    
    expect(result.current.messages).toEqual([]);
    expect(result.current.trace).toEqual([]);
    expect(result.current.isStreaming).toBe(false);
  });
});
