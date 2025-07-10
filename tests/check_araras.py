from araras.commons import *
import araras
import importlib.metadata
import pkgutil
import inspect


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

    # List top-level modules and subpackages
    if hasattr(araras, "__path__"):
        print("Top-level modules and subpackages:")
        for finder, name, is_pkg in pkgutil.iter_modules(araras.__path__):
            kind = "package" if is_pkg else "module"
            print(f"  - {name} ({kind})")
    else:
        print("No __path__ attribute on araras; cannot list submodules")
    print()

    # Inspect package namespace attributes
    print("Public attributes in araras namespace:")
    for name, value in inspect.getmembers(araras):
        if not name.startswith("_"):
            print(f"  - {name}")


if __name__ == "__main__":
    print_package_info("araras")
