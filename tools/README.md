This directory contains command line utilities intended for use by spec editors and contributors when authoring or reviewing changes.

- `reformat-js.py` - applies [clang-format](https://clang.llvm.org/docs/ClangFormat.html) to JavaScript blocks in the spec.
- `lint.mjs` - analyses the spec Bikeshed source and generated HTML to look for common errors like duplicate words or unclosed links, and helps enforce the [coding conventions](../docs/SpecCodingConventions.md).

The tools assume a POSIX-like command line environment, and have dependencies on various languages and libraries. Read the sources for more details.

Spec editors can use a flow like the following:

```
# Edit the spec, save changes:
$EDITOR index.bs

# Reformat JS blocks:
tools/reformat-js.py

# Build the spec:
bikeshed --die-on=fatal spec index.bs

# Check for common errors:
tools/lint.mjs

# If no issues were reported, commit the changes:
git commit -a -m "made it more awesome"
```
