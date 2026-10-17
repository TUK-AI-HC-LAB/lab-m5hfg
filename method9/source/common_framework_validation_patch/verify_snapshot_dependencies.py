"""Read-only syntax/local-import audit; no model imports, downloads or training."""

import argparse
import ast
import json
from pathlib import Path


def module_paths(root, name):
    parts = name.split('.')
    stem = root.joinpath(*parts)
    return (stem.with_suffix('.py'), stem / '__init__.py')


def exists(root, name):
    return any(path.is_file() for path in module_paths(root, name))


def audit(snapshot, reference=None):
    missing = []
    errors = []
    inspected = 0
    external = set()
    for path in sorted(snapshot.rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        inspected += 1
        relative = path.relative_to(snapshot)
        # Bundled AnomalyCLIP is imported with its own root on sys.path.
        root = snapshot
        external_root = snapshot / 'external' / 'AnomalyCLIP'
        if path.is_relative_to(external_root):
            root = external_root
        module = path.relative_to(root).with_suffix('').parts
        package = module[:-1]
        try:
            tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(relative))
        except (SyntaxError, UnicodeError) as exc:
            errors.append({'file': str(relative), 'error': str(exc)})
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ''
                if node.level:
                    prefix = '.'.join(package[:len(package) - node.level + 1]) + ('.' + prefix if prefix else '')
                if prefix:
                    names.append(prefix)
                    names.extend(prefix + '.' + alias.name for alias in node.names if alias.name != '*')
            for name in names:
                top = name.split('.')[0]
                if exists(root, name) or root.joinpath(*name.split('.')).is_dir():
                    continue
                # Only report absent local modules when the reference contains them.
                if reference and root == snapshot and exists(reference, name):
                    missing.append({'file': str(relative), 'module': name})
                elif not node.__class__ is ast.ImportFrom or not node.level:
                    if not (root / top).is_dir() and not (root / (top + '.py')).is_file():
                        external.add(top)
    return {
        'python_files_checked': inspected,
        'syntax_errors': errors,
        'missing_reference_local_imports': missing,
        'external_import_names_not_install_check': sorted(external),
        'limitations': 'Static inspection only; dynamic imports, data, weights and third-party runtime compatibility are not execution-verified.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, default=Path(__file__).parent / 'framework_snapshot')
    parser.add_argument('--reference', type=Path)
    arguments = parser.parse_args()
    result = audit(arguments.snapshot.resolve(), arguments.reference.resolve() if arguments.reference else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(result['syntax_errors'] or result['missing_reference_local_imports']))
