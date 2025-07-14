import araras
import importlib.metadata
import pkgutil
import inspect
import importlib


def print_package_info(package_name):
    # Print basic metadata from importlib.metadata
    print(f'Metadata for distribution "{package_name}":')
    try:
        dist = importlib.metadata.distribution(package_name)
        metadata = dist.metadata
        for field in ("Name", "Version", "Summary", "Author", "Author-email", "License", "Home-page"):
            if field in metadata:
                print(f"{field}: {metadata[field]}")
        requires = dist.requires or []
        if requires:
            print("Requires:")
            for req in requires:
                print(f"  - {req}")
    except importlib.metadata.PackageNotFoundError:
        print(f'No metadata found for "{package_name}"')
    print()

    # List top-level packages and their contents recursively
    if hasattr(araras, "__path__"):
        print("Package structure and functions:")
        _explore_package_recursive(araras, "araras", indent="")
    else:
        print("No __path__ attribute on araras; cannot list submodules")
    print()

    # Inspect package namespace attributes
    print("Public attributes in araras namespace:")
    for name, value in inspect.getmembers(araras):
        if not name.startswith("_"):
            print(f"  - {name}")


def _explore_package_recursive(package, package_name, indent="", max_depth=3, current_depth=0):
    """Recursively explore package structure and list functions."""
    if current_depth > max_depth:
        print(f"{indent}  [Max depth reached]")
        return

    if hasattr(package, "__path__"):
        # It's a package, list its modules and subpackages
        for finder, name, is_pkg in pkgutil.iter_modules(package.__path__):
            full_name = f"{package_name}.{name}"
            kind = "submodule" if is_pkg else "module"
            print(f"{indent} {name} ({kind})")

            try:
                # Import the module/package
                submodule = importlib.import_module(full_name)

                # List functions in this module
                functions = [
                    func_name
                    for func_name, func_obj in inspect.getmembers(submodule, inspect.isfunction)
                    if not func_name.startswith("_") and func_obj.__module__ == full_name
                ]

                if functions:
                    print(f"{indent}  Functions: {', '.join(functions)}")

                # If it's a package, recurse into it
                if is_pkg:
                    _explore_package_recursive(
                        submodule, full_name, indent + "  ", max_depth, current_depth + 1
                    )

            except (ImportError, AttributeError) as e:
                print(f"{indent}  [Error importing {full_name}: {e}]")
    else:
        # It's a module, just list its functions
        functions = [
            func_name
            for func_name, func_obj in inspect.getmembers(package, inspect.isfunction)
            if not func_name.startswith("_")
        ]
        if functions:
            print(f"{indent}Functions: {', '.join(functions)}")


if __name__ == "__main__":
    print_package_info("araras")
