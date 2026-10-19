import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import MessageBubble from "../src/components/Chat/MessageBubble";
import { expect, test, describe, vi } from "vitest";

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

  test("calls onFeedback once with the rating when a thumb is clicked", () => {
    const onFeedback = vi.fn();
    const msg = { role: "agent", agent: "chat_agent", text: "hello" };
    render(<MessageBubble msg={msg} onFeedback={onFeedback} />);
    fireEvent.click(screen.getByLabelText("Good response"));
    expect(onFeedback).toHaveBeenCalledTimes(1);
    expect(onFeedback).toHaveBeenCalledWith(msg, "up");
  });

  test("locks feedback buttons after a rating is submitted", () => {
    const onFeedback = vi.fn();
    const msg = { role: "agent", agent: "chat_agent", text: "hello" };
    render(<MessageBubble msg={msg} onFeedback={onFeedback} />);
    fireEvent.click(screen.getByLabelText("Bad response"));
    fireEvent.click(screen.getByLabelText("Bad response"));
    expect(onFeedback).toHaveBeenCalledTimes(1);
  });

  test("keeps the rating locked when the handler resolves success", async () => {
    const onFeedback = vi.fn().mockResolvedValue(true);
    const msg = { role: "agent", agent: "chat_agent", text: "hello" };
    render(<MessageBubble msg={msg} onFeedback={onFeedback} />);
    fireEvent.click(screen.getByLabelText("Good response"));
    await waitFor(() => expect(screen.getByText("Thanks for the feedback")).toBeInTheDocument());
    expect(screen.getByLabelText("Good response")).toBeDisabled();
  });

  test("reverts the optimistic state when the handler resolves false", async () => {
    const onFeedback = vi.fn().mockResolvedValue(false);
    const msg = { role: "agent", agent: "chat_agent", text: "hello" };
    render(<MessageBubble msg={msg} onFeedback={onFeedback} />);
    fireEvent.click(screen.getByLabelText("Good response"));
    await waitFor(() => expect(screen.getByLabelText("Good response")).not.toBeDisabled());
    expect(screen.queryByText("Thanks for the feedback")).toBeNull();
    fireEvent.click(screen.getByLabelText("Good response"));
    expect(onFeedback).toHaveBeenCalledTimes(2);
  });

  test("does not render feedback buttons while streaming", () => {
    const msg = { role: "agent", agent: "chat_agent", text: "hi", streaming: true };
    render(<MessageBubble msg={msg} onFeedback={() => {}} />);
    expect(screen.queryByLabelText("Good response")).toBeNull();
  });

  test("does not render feedback buttons without an onFeedback handler", () => {
    const msg = { role: "agent", agent: "chat_agent", text: "hello" };
    render(<MessageBubble msg={msg} />);
    expect(screen.queryByLabelText("Good response")).toBeNull();
  });
});
