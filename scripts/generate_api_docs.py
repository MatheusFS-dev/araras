import ast
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "araras"
DOC = ROOT / "docs" / "API_Docs.md"


def parse_function(node: ast.FunctionDef, is_method: bool = False):
    params = []
    args = node.args
    arg_list = args.args
    if is_method and arg_list and arg_list[0].arg == "self":
        arg_list = arg_list[1:]
    for arg in arg_list:
        annotation = ast.unparse(arg.annotation) if arg.annotation else "Any"
        params.append((arg.arg, annotation))
    if args.vararg:
        ann = ast.unparse(args.vararg.annotation) if args.vararg.annotation else "Any"
        params.append(("*" + args.vararg.arg, ann))
    for arg in args.kwonlyargs:
        ann = ast.unparse(arg.annotation) if arg.annotation else "Any"
        params.append((arg.arg, ann))
    if args.kwarg:
        ann = ast.unparse(args.kwarg.annotation) if args.kwarg.annotation else "Any"
        params.append(("**" + args.kwarg.arg, ann))
    returns = ast.unparse(node.returns) if node.returns else "None"
    doc = ast.get_docstring(node) or ""
    return node.name, params, returns, doc


def parse_class(node: ast.ClassDef):
    class_doc = ast.get_docstring(node) or ""
    methods = []
    for child in node.body:
        if isinstance(child, ast.FunctionDef) and not child.name.startswith("_"):
            methods.append(parse_function(child, is_method=True))
    return node.name, class_doc, methods


def parse_file(path: Path):
    module_name = "araras." + ".".join(path.relative_to(SRC).with_suffix("").parts)
    tree = ast.parse(path.read_text())
    module_doc = ast.get_docstring(tree) or ""
    functions = []
    classes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            functions.append(parse_function(node))
        elif isinstance(node, ast.ClassDef):
            classes.append(parse_class(node))
    return module_name, module_doc, sorted(functions), sorted(classes)


def generate_docs():
    entries = []
    for path in sorted(SRC.rglob("*.py")):
        entries.append(parse_file(path))

    with open(DOC, "w") as f:
        f.write("# API Documentation\n\n")
        for mod, mod_doc, funcs, classes in entries:
            f.write(f"## {mod}\n\n")
            if mod_doc:
                f.write(textwrap.dedent(mod_doc).strip() + "\n\n")
            for name, params, ret, doc in funcs:
                sig = ", ".join(f"{n}: {t}" for n, t in params)
                f.write(f"### {name}\n\n")
                f.write("```python\n")
                f.write(f"def {name}({sig}) -> {ret}\n")
                f.write("```\n\n")
                if doc:
                    f.write(textwrap.dedent(doc).strip() + "\n\n")
            for cls_name, cls_doc, methods in classes:
                f.write(f"### class {cls_name}\n\n")
                if cls_doc:
                    f.write(textwrap.dedent(cls_doc).strip() + "\n\n")
                for mname, params, ret, mdoc in methods:
                    msig = ", ".join(f"{n}: {t}" for n, t in params)
                    f.write(f"#### {mname}\n\n")
                    f.write("```python\n")
                    f.write(f"def {mname}({msig}) -> {ret}\n")
                    f.write("```\n\n")
                    if mdoc:
                        f.write(textwrap.dedent(mdoc).strip() + "\n\n")


if __name__ == "__main__":
    generate_docs()
