import ast
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_docstring(doc: str) -> Dict[str, Any]:
    """Parse a simple Google style docstring.

    This parser is purposely lightweight and does not require external
    dependencies. It extracts parameter descriptions, return information and
    raised exceptions if they are present in the docstring. The returned
    structure is compatible with the template used for the API documentation.
    """

    info: Dict[str, Any] = {
        "summary": "",
        "params": {},
        "returns": "",
        "raises": [],
        "examples": [],
    }

    if not doc:
        return info

    lines = [line.rstrip() for line in doc.splitlines()]
    if not lines:
        return info

    info["summary"] = lines[0].strip()

    section = None
    buf: List[str] = []

    def _flush():
        nonlocal buf
        data = buf
        buf = []
        return data

    for line in lines[1:]:
        stripped = line.strip()
        lower = stripped.lower()
        if lower in {"args:", "arguments:", "parameters:"}:
            section = "args"
            _flush()
            continue
        if lower == "returns:":
            if section == "args":
                _flush()
            section = "returns"
            continue
        if lower == "raises:":
            if section in {"args", "returns"}:
                _flush()
            section = "raises"
            continue
        if lower == "examples:":
            if section in {"args", "returns", "raises"}:
                _flush()
            section = "examples"
            continue

        if not stripped:
            if section == "examples":
                buf.append("")
            continue

        if section == "args":
            if ":" in stripped:
                name_part, desc = stripped.split(":", 1)
                name_part = name_part.strip()
                desc = desc.strip()
                if "(" in name_part and ")" in name_part:
                    name = name_part.split("(")[0].strip()
                    typ = name_part[
                        name_part.find("(") + 1 : name_part.find(")")
                    ].strip()
                else:
                    pieces = name_part.split()
                    name = pieces[0]
                    typ = pieces[1] if len(pieces) > 1 else "Any"
                info["params"][name] = {"type": typ, "desc": desc}
        elif section == "returns":
            buf.append(stripped)
        elif section == "raises":
            if ":" in stripped:
                exc, desc = stripped.split(":", 1)
                info["raises"].append((exc.strip(), desc.strip()))
            else:
                info["raises"].append((stripped, ""))
        elif section == "examples":
            buf.append(line.rstrip())

    if section == "returns" and buf:
        info["returns"] = " ".join(buf).strip()
    if section == "examples" and buf:
        info["examples"] = buf

    return info


def parse_functions(file_path: Path) -> List[Dict[str, Any]]:
    """Return metadata for all functions in ``file_path``."""

    with open(file_path, "r", encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src)
    funcs = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args: List[Tuple[str, str]] = []
            for arg in node.args.args:
                if arg.arg == "self":
                    continue
                if arg.annotation:
                    annotation = ast.unparse(arg.annotation)
                else:
                    annotation = "Any"
                args.append((arg.arg, annotation))

            sig = ", ".join(f"{name}: {ann}" for name, ann in args)
            returns = ast.unparse(node.returns) if node.returns else "Any"
            doc = ast.get_docstring(node) or ""
            doc_info = _parse_docstring(doc)

            funcs.append(
                {
                    "name": node.name,
                    "sig": sig,
                    "returns": returns,
                    "doc": doc_info,
                }
            )

    return funcs


def walk_modules(root_dir):
    all_funcs = {}
    for path in Path(root_dir).rglob("*.py"):
        rel = path.relative_to(root_dir)
        module = ".".join(rel.with_suffix("").parts)
        funcs = parse_functions(path)
        if funcs:
            all_funcs[module] = funcs
    return all_funcs


if __name__ == "__main__":
    root = Path("src/araras")
    all_funcs = walk_modules(root)
    out_lines = ["# API Documentation"]
    for module, funcs in sorted(all_funcs.items()):
        out_lines.append(f"\n## {module}")
        for func in funcs:
            doc = func["doc"]
            out_lines.append(f"\n### {func['name']}")
            out_lines.append("```python")
            out_lines.append(f"{func['name']}({func['sig']})  [source]")
            out_lines.append("```")
            out_lines.append(doc.get("summary", ""))

            # Optional extra notes for selected functions
            if func["name"] == "send_email":
                out_lines.append(
                    "> [!CAUTION]\n> Requires valid SMTP credentials and network access."
                )
            if func["name"] == "run_auto_restart":
                out_lines.append(
                    "> [!WARNING]\n> Automatically restarts the target process when it crashes."
                )
            if func["name"] == "log_resources":
                out_lines.append(
                    "> [!TIP]\n> Logging continues until the program exits."
                )

            out_lines.append("\n**Parameters**\n")
            out_lines.append("| Name | Type | Description |")
            out_lines.append("|------|------|-------------|")
            for name, ann in [
                p.split(":") if ":" in p else (p, "Any")
                for p in func["sig"].split(", ")
                if p
            ]:
                name = name.strip()
                ann = ann.strip()
                desc = doc.get("params", {}).get(name, {}).get("desc", "")
                out_lines.append(f"| {name} | `{ann}` | {desc} |")

            out_lines.append("\n**Returns**\n")
            ret_desc = doc.get("returns", "")
            out_lines.append(f"`{func['returns']}` – {ret_desc}\n")

            out_lines.append("\n**Raises**\n")
            if doc.get("raises"):
                for exc, exc_desc in doc["raises"]:
                    out_lines.append(f"- `{exc}` – {exc_desc}")
            else:
                out_lines.append("- None")

            out_lines.append("\n**Examples**\n")
            out_lines.append("```python")
            if doc.get("examples"):
                out_lines.extend(doc["examples"])
            else:
                out_lines.append(f"result = {func['name']}(...)")
            out_lines.append("```")
    Path("docs/API_Docs.md").write_text("\n".join(out_lines))
