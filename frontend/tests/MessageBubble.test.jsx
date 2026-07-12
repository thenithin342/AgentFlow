import React from "react";
import { render, screen } from "@testing-library/react";
import MessageBubble from "../src/components/Chat/MessageBubble";
import { expect, test, describe } from "vitest";

describe("MessageBubble", () => {
  test("renders user message correctly", () => {
    const msg = { role: "user", text: "Hello AI!" };
    render(<MessageBubble msg={msg} />);
    expect(screen.getByText("Hello AI!")).toBeInTheDocument();
  });

  test("renders agent message and markdown", () => {
    const msg = { role: "agent", agent: "chat_agent", text: "**Bold text**" };
    render(<MessageBubble msg={msg} />);
    expect(screen.getByText("Bold text")).toBeInTheDocument();
    expect(screen.getByText("Bold text").tagName).toBe("STRONG");
  });

  test("renders typing indicator when streaming without text", () => {
    const msg = { role: "agent", streaming: true };
    const { container } = render(<MessageBubble msg={msg} />);
    expect(container.querySelector(".af-typing-indicator")).toBeInTheDocument();
  });

  test("shows aborted state", () => {
    const msg = { role: "agent", aborted: true };
    render(<MessageBubble msg={msg} />);
    expect(screen.getAllByText(/aborted/).length).toBeGreaterThan(0);
  });
});
