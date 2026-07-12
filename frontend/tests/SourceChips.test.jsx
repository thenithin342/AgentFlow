import React from "react";
import { render, screen } from "@testing-library/react";
import SourceChips from "../src/components/Chat/SourceChips";
import { expect, test, describe } from "vitest";

describe("SourceChips", () => {
  test("returns null if no citations", () => {
    const { container } = render(<SourceChips citations={[]} />);
    expect(container.firstChild).toBeNull();
  });

  test("renders citations correctly", () => {
    const citations = [
      { n: 1, host: "wikipedia.org", url: "https://en.wikipedia.org" },
      { n: 2, host: "github.com", url: "https://github.com" },
    ];
    render(<SourceChips citations={citations} />);
    
    expect(screen.getByText("Sources:")).toBeInTheDocument();
    expect(screen.getByText("[1] wikipedia.org")).toHaveAttribute("href", "https://en.wikipedia.org");
    expect(screen.getByText("[2] github.com")).toHaveAttribute("href", "https://github.com");
  });

  test("truncates at 8 citations", () => {
    const citations = Array.from({ length: 10 }, (_, i) => ({
      n: i + 1,
      host: `site${i}.com`,
      url: `https://site${i}.com`
    }));
    
    render(<SourceChips citations={citations} />);
    
    expect(screen.getByText("[8] site7.com")).toBeInTheDocument();
    expect(screen.queryByText("[9] site8.com")).not.toBeInTheDocument();
    expect(screen.getByText("…")).toBeInTheDocument();
  });
});
