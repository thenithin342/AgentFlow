import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import ChatInput from "../src/components/Chat/ChatInput";
import { expect, test, describe, vi } from "vitest";

describe("ChatInput", () => {
  test("renders text area and buttons", () => {
    const handleSend = vi.fn();
    const setInput = vi.fn();
    const setReviewRequired = vi.fn();

    render(
      <ChatInput
        input=""
        setInput={setInput}
        isStreaming={false}
        isUploading={false}
        reviewRequired={false}
        setReviewRequired={setReviewRequired}
        handleSend={handleSend}
        handleFileChange={vi.fn()}
      />
    );

    expect(screen.getByLabelText("Chat message")).toBeInTheDocument();
    expect(screen.getByLabelText("Send message")).toBeInTheDocument();
    expect(screen.getByLabelText("Attach PDF")).toBeInTheDocument();
    expect(screen.getByLabelText("Toggle review mode")).toBeInTheDocument();
  });

  test("disables input when streaming", () => {
    render(
      <ChatInput
        input="hello"
        setInput={vi.fn()}
        isStreaming={true}
        isUploading={false}
        reviewRequired={false}
        setReviewRequired={vi.fn()}
        handleSend={vi.fn()}
        handleFileChange={vi.fn()}
      />
    );

    expect(screen.getByLabelText("Chat message")).toHaveAttribute("readonly");
    expect(screen.getByLabelText("Send message")).toBeDisabled();
    expect(screen.getByLabelText("Attach PDF")).toBeDisabled();
  });

  test("triggers send on enter key", () => {
    const handleSend = vi.fn();
    render(
      <ChatInput
        input="hello"
        setInput={vi.fn()}
        isStreaming={false}
        isUploading={false}
        reviewRequired={false}
        setReviewRequired={vi.fn()}
        handleSend={handleSend}
        handleFileChange={vi.fn()}
      />
    );

    const inputArea = screen.getByLabelText("Chat message");
    fireEvent.keyDown(inputArea, { key: "Enter", code: "Enter", shiftKey: false });
    expect(handleSend).toHaveBeenCalledOnce();
  });

  test("disables send if input exceeds max chars", () => {
    // 16000 max chars
    const longString = "a".repeat(16001);
    render(
      <ChatInput
        input={longString}
        setInput={vi.fn()}
        isStreaming={false}
        isUploading={false}
        reviewRequired={false}
        setReviewRequired={vi.fn()}
        handleSend={vi.fn()}
        handleFileChange={vi.fn()}
      />
    );
    expect(screen.getByLabelText("Send message")).toBeDisabled();
  });
});
