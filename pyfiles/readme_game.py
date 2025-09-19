#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
My README Game Builder (Pastel + Animated)
Author: Cazzy Aporbo
Run: python readme_game.py

What it does:
- Asks questions ("game style") and assembles a README in the same pastel + animated theme I have.
- Features:
  - Animated wave headers (capsule-render) and typing subtitle (readme-typing-svg)
  - Pastel badges (shields.io) + optional GitHub stats cards
  - Section dividers with animated gradient bars
  - "Bases" (inline pastel gradient pills)
  - Tables (interactive builder)
  - Mermaid diagrams (flowchart/sequence/ER/graph), guided ER builder included
  - Repository counter badge (pastel) and/or GitHub stats
- Never uses emojis (per your request).
- Output: README_generated.md (or a path you choose)
"""

from urllib.parse import quote_plus
from textwrap import dedent
import os
import sys

# ---------- Theme presets ----------

PASTEL_PALETTES = {
    "lavender-pink-mint": ["FFE0F5", "E6E0FF", "D4FFE4", "FDE1C9", "E0FFFF", "FFF8FD"],
    "soft-ocean": ["E6F0FF", "D9FBFF", "E6FFF5", "FFF5E6", "F0E6FF", "FFE6F4"],
    "rose-lilac-ice": ["FFF0F6", "F3E8FF", "E6FAFF", "FDF6E4", "EAFBF1", "F1F2FF"],
}

CAPSULE_COLORLIST_DEFAULT = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"


def ask(prompt, default=None, allow_empty=False):
    sfx = f" [{default}]" if default else ""
    while True:
        val = input(f"{prompt}{sfx}: ").strip()
        if val:
            return val
        if default is not None and (val == "" or val is None):
            return default
        if allow_empty:
            return ""
        print("Please enter a value.")


def ask_yesno(prompt, default=True):
    d = "Y/n" if default else "y/N"
    while True:
        v = input(f"{prompt} ({d}): ").strip().lower()
        if not v:
            return default
        if v in ("y", "yes"):
            return True
        if v in ("n", "no"):
            return False
        print("Please answer y or n.")


def choose(prompt, options, default_index=0):
    print(prompt)
    for i, opt in enumerate(options):
        print(f"  {i+1}. {opt}")
    while True:
        v = input(f"Select [1-{len(options)}] (default {default_index+1}): ").strip()
        if not v:
            return options[default_index]
        if v.isdigit():
            n = int(v)
            if 1 <= n <= len(options):
                return options[n - 1]
        print("Invalid choice.")


# ---------- Builders ----------

def capsule_wave_header(title, desc, font_size=88, height=300, colors=CAPSULE_COLORLIST_DEFAULT):
    return dedent(f"""
    <div align="center">

    <picture>
      <img width="100%" alt="Header" src="https://capsule-render.vercel.app/api?type=waving&amp;color=gradient&amp;customColorList={quote_plus(colors)}&amp;height={height}&amp;section=header&amp;text={quote_plus(title)}&amp;fontSize={font_size}&amp;animation=fadeIn&amp;fontAlignY=36&amp;desc={quote_plus(desc)}&amp;descAlignY=62&amp;descSize=20&amp;fontColor=FFF8FD" />
    </picture>
    """).strip() + "\n"


def typing_subtitle(lines, width=980, height=70, color="E6C6FF"):
    # lines param expects semicolon-separated; we handle multiple lines in URL
    joined = ";".join([l.strip() for l in lines if l.strip()])
    return dedent(f"""
    <picture>
      <img alt="Typing subtitle" src="https://readme-typing-svg.demolab.com?font=Fira+Code&amp;weight=600&amp;size=20&amp;duration=3400&amp;pause=900&amp;color={quote_plus(color)}&amp;center=true&amp;vCenter=true&amp;multiline=true&amp;width={width}&amp;height={height}&amp;lines={quote_plus(joined)}" />
    </picture>
    """).strip() + "\n"


def badges_block(badges):
    """
    badges: list of dicts with keys: label, message, color, labelColor, logo(optional), link(optional), style
    """
    imgs = []
    for b in badges:
        style = b.get("style", "for-the-badge")
        logo = f"&amp;logo={quote_plus(b['logo'])}" if b.get("logo") else ""
        url = (
            f"https://img.shields.io/badge/{quote_plus(b['label'])}-"
            f"{quote_plus(b['message'])}-{quote_plus(b['color'])}"
            f"?style={quote_plus(style)}&amp;labelColor={quote_plus(b['labelColor'])}{logo}"
        )
        img = f'<img src="{url}" alt="{b["label"]}">'
        if b.get("link"):
            imgs.append(f'<a href="{b["link"]}" target="_blank">{img}</a>')
        else:
            imgs.append(img)
    return "<p>\n  " + "\n  ".join(imgs) + "\n</p>\n"


def divider(colors=CAPSULE_COLORLIST_DEFAULT, height=4):
    return dedent(f"""
    <picture>
      <img alt="Divider" width="100%" src="https://capsule-render.vercel.app/api?type=rect&amp;color=gradient&amp;customColorList={quote_plus(colors)}&amp;height={height}" />
    </picture>
    """).strip() + "\n"


def section_banner(text, colors=CAPSULE_COLORLIST_DEFAULT):
    return dedent(f"""
    <div align="center">
    <picture>
      <img alt="Section Banner" width="100%" src="https://capsule-render.vercel.app/api?type=soft&amp;color=gradient&amp;customColorList={quote_plus(colors)}&amp;height=120&amp;text={quote_plus(text)}&amp;fontSize=30&amp;fontColor=4A4A4A" />
    </picture>
    </div>
    """).strip() + "\n"


def pastel_bases(bases, palette_key="lavender-pink-mint"):
    palette = PASTEL_PALETTES.get(palette_key, PASTEL_PALETTES["lavender-pink-mint"])
    html = ['<div align="center">']
    for i, text in enumerate(bases):
        c1 = palette[i % len(palette)]
        c2 = palette[(i+1) % len(palette)]
        pill = (
            f'<div style="background: linear-gradient(135deg, #{c1}, #{c2}); '
            f'padding: 12px 18px; border-radius: 14px; margin: 6px; '
            f'display: inline-block; font-weight: 600;">{text}</div>'
        )
        html.append(pill)
    html.append("</div>\n")
    return "\n".join(html)


def html_text_block(title_md, body_md, frame=True, palette_key="lavender-pink-mint"):
    palette = PASTEL_PALETTES.get(palette_key, PASTEL_PALETTES["lavender-pink-mint"])
    bg = f"linear-gradient(135deg, #{palette[0]} 0%, #{palette[1]} 100%)"
    if not frame:
        return f"\n\n# {title_md}\n\n{body_md}\n\n"
    return dedent(f"""
    <div style="background: {bg}; padding: 1.25rem; border-radius: 12px; margin: 1rem 0;">
      <h2 style="color: #4A4A4A; margin-top: 0;">{title_md}</h2>
      <div style="color: #3E3E3E; line-height: 1.7;">
      {body_md}
      </div>
    </div>
    """).strip() + "\n"


def make_table(headers, rows):
    out = ["", "|" + "|".join(headers) + "|", "|" + "|".join([":---:" for _ in headers]) + "|"]
    for r in rows:
        out.append("|" + "|".join(r) + "|")
    out.append("")
    return "\n".join(out) + "\n"


def mermaid_block(kind, content=None, guided_er=None):
    if kind == "erDiagram" and guided_er:
        # guided_er is dict with entities {name: [(type, attr), ...]} and relations list of tuples
        lines = ["erDiagram"]
        for name, attrs in guided_er["entities"].items():
            lines.append(f"  {name} {{")
            for t, a in attrs:
                lines.append(f"    {t} {a}")
            lines.append("  }")
        for (lhs, card, rhs, label) in guided_er.get("relations", []):
            # Example: USER ||--o{ ORDER : places
            lines.append(f"  {lhs} {card} {rhs} : {label}")
        code = "\n".join(lines)
    else:
        code = content or "graph TD\n  A[Start] --> B[Next]\n  B --> C[Done]"
    return f"\n```mermaid\n{code}\n```\n"


def github_stats(username, theme_bg="FFF8FD", text_color="4A4A4A", title_color="4A4A4A"):
    stats = (
        f'<img alt="GitHub stats" '
        f'src="https://github-readme-stats.vercel.app/api?username={quote_plus(username)}'
        f'&amp;show_icons=false&amp;hide_title=false&amp;hide_rank=false&amp;include_all_commits=true'
        f'&amp;count_private=true&amp;title_color={title_color}&amp;text_color={text_color}&amp;bg_color={theme_bg}" />'
    )
    return f'<div align="center">\n  {stats}\n</div>\n'


def repo_count_badge(user, count_text="Repos", color="E6E0FF", labelColor="FFE0F5", link=None):
    # This is a static badge unless user provides a count string. You can edit the message manually.
    message = ask(f"Enter repo count text for badge (e.g., 42) for {user}", default="N/A")
    url = (
        f"https://img.shields.io/badge/{quote_plus(count_text)}-"
        f"{quote_plus(message)}-{quote_plus(color)}?style=for-the-badge&amp;labelColor={quote_plus(labelColor)}"
    )
    img = f'<img src="{url}" alt="Repository Count">'
    return f'<p align="center">{img}</p>\n'


# ---------- Game Flow ----------

def build_badges_interactively():
    badges = []
    add = ask_yesno("Add pastel badges?")
    if not add:
        return badges
    print("\nEnter badges (press Enter at label to stop). Examples:\n"
          "  label=Python, message=3.10+, color=FFE0F5, labelColor=E6E0FF, logo=python\n"
          "  label=License, message=MIT, color=D4FFE4, labelColor=E6E6FA\n")
    while True:
        label = ask("Badge label (blank to finish)", default="", allow_empty=True)
        if not label:
            break
        message = ask("Badge message", default="Ready")
        color = ask("Badge color hex (no #)", default="FFE0F5")
        labelColor = ask("Label color hex (no #)", default="E6E0FF")
        logo = ask("Logo (optional shields.io name)", default="", allow_empty=True)
        link = ask("Link (optional)", default="", allow_empty=True)
        badges.append({
            "label": label, "message": message, "color": color, "labelColor": labelColor,
            "logo": logo, "link": link, "style": "for-the-badge"
        })
    return badges


def build_bases_interactively():
    if not ask_yesno("Add pastel 'bases' (gradient pills)?"):
        return ""
    palette_key = choose("Palette for bases:", list(PASTEL_PALETTES.keys()), 0)
    bases = []
    print("Enter base texts. Leave blank to finish.")
    while True:
        t = ask("Base text", default="", allow_empty=True)
        if not t:
            break
        bases.append(t)
    return pastel_bases(bases, palette_key=palette_key)


def build_tables_interactively():
    blocks = []
    if not ask_yesno("Add table(s)?"):
        return blocks
    while True:
        headers = []
        print("Enter headers (blank to finish headers).")
        while True:
            h = ask("Header", default="", allow_empty=True)
            if not h:
                break
            headers.append(h)
        if not headers:
            print("No headers entered; skipping table.")
        else:
            rows = []
            print("Enter rows; blank in first column to finish rows.")
            while True:
                row = []
                for i, h in enumerate(headers):
                    val = ask(f"Row value for '{h}' (blank to finish rows)" if i == 0 else f"  value for '{h}'", default="", allow_empty=True)
                    if i == 0 and val == "":
                        row = None
                        break
                    row.append(val if val else "")
                if row is None:
                    break
                if len(row) < len(headers):
                    row += [""] * (len(headers) - len(row))
                rows.append(row)
            blocks.append(make_table(headers, rows))
        if not ask_yesno("Add another table?", default=False):
            break
    return blocks


def build_mermaid_interactively():
    blocks = []
    if not ask_yesno("Add Mermaid diagram(s)?"):
        return blocks
    while True:
        kind = choose("Diagram type", ["flowchart", "sequenceDiagram", "erDiagram", "graph", "pie"], 0)
        guided = False
        guided_data = None
        content = None
        if kind == "erDiagram" and ask_yesno("Use guided ER builder?"):
            guided = True
            entities = {}
            print("Define entities. For each entity, add attributes (type then name). Leave name blank to finish.")
            while True:
                en = ask("Entity name", default="", allow_empty=True)
                if not en:
                    break
                attrs = []
                print(f"Attributes for {en} (e.g., 'int id', 'string name'). Blank to finish.")
                while True:
                    at = ask("  type", default="", allow_empty=True)
                    if not at:
                        break
                    an = ask("  name", default="field", allow_empty=False)
                    attrs.append((at, an))
                entities[en] = attrs
            rels = []
            if ask_yesno("Add relationships?"):
                print("Relationship format: LHS  CARD  RHS  LABEL  (e.g., USER  ||--o{  ORDER  places)")
                while True:
                    lhs = ask("  LHS entity (blank to finish)", default="", allow_empty=True)
                    if not lhs:
                        break
                    card = ask("  cardinality token (e.g., ||--o{)", default="||--o{")
                    rhs = ask("  RHS entity", default="")
                    lab = ask("  label", default="relates")
                    rels.append((lhs, card, rhs, lab))
            guided_data = {"entities": entities, "relations": rels}
        else:
            print(f"Enter Mermaid code for {kind}. Finish with a single line containing only END.")
            lines = []
            while True:
                l = input()
                if l.strip() == "END":
                    break
                lines.append(l)
            base = "\n".join(lines).strip()
            if not base:
                if kind == "flowchart":
                    base = "flowchart TD\n  A[Start] --> B[Next]\n  B --> C[Done]"
                elif kind == "sequenceDiagram":
                    base = "sequenceDiagram\n  participant A\n  participant B\n  A->>B: Request\n  B-->>A: Response"
                elif kind == "graph":
                    base = "graph LR\n  A-->B\n  B-->C\n  C-->A"
                elif kind == "pie":
                    base = "pie title Example\n  \"One\" : 40\n  \"Two\" : 60"
            content = f"{kind}\n{base}" if not base.startswith(kind) else base
        blocks.append(mermaid_block(kind, content=content, guided_er=guided_data if guided else None))
        if not ask_yesno("Add another diagram?", default=False):
            break
    return blocks


def build_text_sections():
    blocks = []
    while ask_yesno("Add a pastel-framed text section?"):
        title = ask("Section title", default="Section")
        print("Enter Markdown body; finish with a single line containing only END.")
        lines = []
        while True:
            l = input()
            if l.strip() == "END":
                break
            lines.append(l)
        body = "\n".join(lines)
        framed = ask_yesno("Use pastel frame around this section?", default=True)
        blocks.append(html_text_block(title, body, frame=framed))
    return blocks


def main():
    print("\n=== CLoudHelix README Game Builder ===\n")
    out_path = ask("Output file path", default="README_generated.md")
    title = ask("Main title text", default="Project Title")
    desc = ask("One-line description under header", default="A pastel + animated README")
    author = ask("Author name to show (optional)", default="Cazzy Aporbo, MS", allow_empty=True)
    colors = CAPSULE_COLORLIST_DEFAULT  # keep gradient sweep

    # Header section
    pieces = []
    pieces.append(capsule_wave_header(title, desc, font_size=88, height=300, colors=colors))

    # Typing subtitle
    if ask_yesno("Add animated typing subtitle?"):
        print("Enter one or more typing lines (each shown as an animated line). Leave blank to finish.")
        lines = []
        while True:
            t = ask("Typing line", default="", allow_empty=True)
            if not t:
                break
            lines.append(t)
        if lines:
            pieces.append(typing_subtitle(lines))
    # Badges
    badges = build_badges_interactively()
    if badges:
        pieces.append(badges_block(badges))

    pieces.append(divider(colors=colors, height=4))

    # Optional author line
    if author:
        pieces.append(f"\n*Author: **{author}***\n")

    # Bases
    bases_html = build_bases_interactively()
    if bases_html:
        pieces.append(bases_html)

    # Section Banner
    if ask_yesno("Insert a section banner?"):
        banner_text = ask("Banner text", default="Overview")
        pieces.append(section_banner(banner_text, colors=colors))

    # Text sections
    pieces.extend(build_text_sections())

    # Tables
    pieces.extend(build_tables_interactively())

    # Diagrams
    pieces.extend(build_mermaid_interactively())

    # GitHub stats / repo badge
    if ask_yesno("Add GitHub stats card or repo count badge?"):
        user = ask("GitHub username", default="your-username")
        mode = choose("Pick one", ["Stats Card", "Repo Count Badge", "Both"], 0)
        if mode in ("Stats Card", "Both"):
            pieces.append(github_stats(user))
        if mode in ("Repo Count Badge", "Both"):
            pieces.append(repo_count_badge(user))

    # Final divider + footer line
    pieces.append(divider(colors=colors, height=3))
    footer_txt = ask("Footer tagline (optional)", default="Decisions you can explain. Designs you can ship. Systems you can trust.", allow_empty=True)
    if footer_txt:
        pieces.append(f'<div align="center"><i>{footer_txt}</i></div>\n')

    # Write out
    content = "\n".join(pieces).strip() + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nDone. Wrote: {os.path.abspath(out_path)}")
    print("Open it in a Markdown preview or paste into GitHub to see the animations (wave headers, typing, etc.).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
