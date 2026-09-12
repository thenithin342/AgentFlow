import { describe, expect, test } from "vitest";
import { parseCitations } from "../src/utils";

describe("parseCitations", () => {
  test("returns empty array for empty or falsy input", () => {
    expect(parseCitations("")).toEqual([]);
    expect(parseCitations(null)).toEqual([]);
    expect(parseCitations(undefined)).toEqual([]);
  });

  test("parses standard citation URL correctly", () => {
    const text = "[1] https://example.com/article";
    expect(parseCitations(text)).toEqual([
      { n: 1, url: "https://example.com/article", host: "example.com" }
    ]);
  });

  test("preserves balanced parentheses in Wikipedia URLs", () => {
    const text = "[1] https://en.wikipedia.org/wiki/Foo_(bar)";
    expect(parseCitations(text)).toEqual([
      {
        n: 1,
        url: "https://en.wikipedia.org/wiki/Foo_(bar)",
        host: "en.wikipedia.org"
      }
    ]);
  });

  test("trims trailing prose punctuation and unmatched closing parenthesis", () => {
    const text = "[1] https://en.wikipedia.org/wiki/Foo_(bar).";
    expect(parseCitations(text)).toEqual([
      {
        n: 1,
        url: "https://en.wikipedia.org/wiki/Foo_(bar)",
        host: "en.wikipedia.org"
      }
    ]);

    const textWithUnmatchedParen = "[2] https://example.com/page).";
    expect(parseCitations(textWithUnmatchedParen)).toEqual([
      {
        n: 2,
        url: "https://example.com/page",
        host: "example.com"
      }
    ]);
  });

  test("parses multiple citations", () => {
    const text = "[1] https://example.org/a [2] https://example.com/b";
    expect(parseCitations(text)).toEqual([
      { n: 1, url: "https://example.org/a", host: "example.org" },
      { n: 2, url: "https://example.com/b", host: "example.com" }
    ]);
  });
});
