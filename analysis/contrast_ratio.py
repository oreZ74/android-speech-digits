#!/usr/bin/env python3
"""
WCAG Contrast Ratio Calculator

Calculates contrast ratios between two colors according to WCAG 2.1 guidelines.
Optionally cross-checks with WebAIM API.

Output:
- results/contrast_check.json (offline calculation)
- results/contrast_webaim_api.json (optional, WebAIM API response)

WCAG Requirements:
- AA Normal Text: >= 4.5:1
- AA Large Text: >= 3:1
- AAA Normal Text: >= 7:1
- AAA Large Text: >= 4.5:1
"""

import argparse
import json
import os
from typing import Tuple
import urllib.request
import urllib.parse


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def relative_luminance(rgb: Tuple[int, int, int]) -> float:
    """
    Calculate relative luminance of a color.
    Formula: https://www.w3.org/WAI/GL/wiki/Relative_luminance
    """
    def channel_luminance(c: int) -> float:
        c_srgb = c / 255
        if c_srgb <= 0.03928:
            return c_srgb / 12.92
        return ((c_srgb + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return (
        0.2126 * channel_luminance(r)
        + 0.7152 * channel_luminance(g)
        + 0.0722 * channel_luminance(b)
    )


def contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    """
    Calculate WCAG contrast ratio between foreground and background colors.
    Formula: (L1 + 0.05) / (L2 + 0.05) where L1 is the lighter luminance.
    """
    l1 = relative_luminance(hex_to_rgb(fg_hex))
    l2 = relative_luminance(hex_to_rgb(bg_hex))

    if l1 < l2:
        l1, l2 = l2, l1

    return (l1 + 0.05) / (l2 + 0.05)


def evaluate_wcag(ratio: float) -> dict:
    """Evaluate WCAG conformance levels."""
    return {
        "AA_normal_text": ratio >= 4.5,
        "AA_large_text": ratio >= 3.0,
        "AAA_normal_text": ratio >= 7.0,
        "AAA_large_text": ratio >= 4.5,
    }


def calculate_contrast(fg: str, bg: str, name: str = None) -> dict:
    """Calculate contrast ratio and WCAG conformance."""
    ratio = contrast_ratio(fg, bg)
    wcag = evaluate_wcag(ratio)

    return {
        "name": name or f"{fg} on {bg}",
        "foreground": fg.upper(),
        "background": bg.upper(),
        "ratio": round(ratio, 2),
        "ratio_display": f"{ratio:.2f}:1",
        "wcag": wcag,
        "passes_aa_normal": wcag["AA_normal_text"],
        "passes_aaa_normal": wcag["AAA_normal_text"],
    }


def fetch_webaim_api(fg: str, bg: str) -> dict | None:
    """
    Fetch contrast check from WebAIM API.
    Returns None on failure.
    """
    fg_clean = fg.lstrip("#")
    bg_clean = bg.lstrip("#")

    url = f"https://webaim.org/resources/contrastchecker/?fcolor={fg_clean}&bcolor={bg_clean}&api"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = response.read().decode("utf-8")
            return json.loads(data)
    except Exception as e:
        print(f"WebAIM API request failed: {e}")
        return None


# Color combinations used in the app
APP_COLORS = [
    {
        "name": "Success Text (Green on Dark)",
        "fg": "#70E586",
        "bg": "#1E1E27",
    },
    {
        "name": "Error Text (Red on Dark)",
        "fg": "#FF6B6B",
        "bg": "#1E1E27",
    },
    {
        "name": "Primary Text (White on Dark)",
        "fg": "#FFFFFF",
        "bg": "#1E1E27",
    },
    {
        "name": "Secondary Text (Light Gray on Dark)",
        "fg": "#E0E0E0",
        "bg": "#1E1E27",
    },
    {
        "name": "PIN Digits (White on Dark)",
        "fg": "#FFFFFF",
        "bg": "#121212",
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="Calculate WCAG contrast ratios"
    )
    parser.add_argument(
        "--fg", type=str, help="Foreground color (hex, e.g., #70E586)"
    )
    parser.add_argument(
        "--bg", type=str, help="Background color (hex, e.g., #1E1E27)"
    )
    parser.add_argument(
        "--webaim", action="store_true", help="Cross-check with WebAIM API"
    )
    parser.add_argument(
        "--all-app-colors", action="store_true",
        help="Check all app color combinations"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for JSON (default: results)"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    webaim_results = []

    if args.all_app_colors:
        # Check all app colors
        for combo in APP_COLORS:
            result = calculate_contrast(combo["fg"], combo["bg"], combo["name"])
            results.append(result)
            print(f"[{'PASS' if result['passes_aa_normal'] else 'FAIL'}] "
                  f"{result['name']}: {result['ratio_display']} "
                  f"(AA: {'PASS' if result['passes_aa_normal'] else 'FAIL'}, "
                  f"AAA: {'PASS' if result['passes_aaa_normal'] else 'FAIL'})")

            if args.webaim:
                api_result = fetch_webaim_api(combo["fg"], combo["bg"])
                if api_result:
                    webaim_results.append({
                        "name": combo["name"],
                        "api_response": api_result
                    })

    elif args.fg and args.bg:
        # Check single color pair
        result = calculate_contrast(args.fg, args.bg)
        results.append(result)
        print(f"Contrast Ratio: {result['ratio_display']}")
        print(f"WCAG AA (Normal Text): {'PASS' if result['passes_aa_normal'] else 'FAIL'}")
        print(f"WCAG AAA (Normal Text): {'PASS' if result['passes_aaa_normal'] else 'FAIL'}")

        if args.webaim:
            api_result = fetch_webaim_api(args.fg, args.bg)
            if api_result:
                webaim_results.append({"api_response": api_result})

    else:
        # Default: check primary success color
        result = calculate_contrast("#70E586", "#1E1E27", "Success Text (Default)")
        results.append(result)
        print(f"Default check: {result['ratio_display']} - "
              f"AA: {'PASS' if result['passes_aa_normal'] else 'FAIL'}")

    # Save results
    output = {
        "tool": "WCAG Contrast Ratio Calculator",
        "wcag_version": "2.1",
        "requirements": {
            "AA_normal_text": "4.5:1",
            "AA_large_text": "3:1",
            "AAA_normal_text": "7:1",
            "AAA_large_text": "4.5:1",
        },
        "results": results,
        "all_pass_aa": all(r["passes_aa_normal"] for r in results),
    }

    json_path = os.path.join(args.output_dir, "contrast_check.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {json_path}")

    if webaim_results:
        webaim_path = os.path.join(args.output_dir, "contrast_webaim_api.json")
        with open(webaim_path, "w", encoding="utf-8") as f:
            json.dump(webaim_results, f, indent=2, ensure_ascii=False)
        print(f"Saved: {webaim_path}")


if __name__ == "__main__":
    main()
