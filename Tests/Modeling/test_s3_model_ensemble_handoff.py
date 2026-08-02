import ast
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "Scripts"
    / "Modeling"
    / "s3_Model_Ensemble.py"
)


def _script_tree():
    return ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))


def test_ensemble_keeps_canonical_avg_adp_publication():
    tree = _script_tree()

    publisher_imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "Scripts.V2.production_handoff"
        and any(
            alias.name == "publish_current_avg_adps"
            for alias in node.names
        )
        for node in ast.walk(tree)
    )
    publisher_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "publish_current_avg_adps"
    ]

    assert publisher_imported
    assert len(publisher_calls) == 1
    assert any(
        keyword.arg == "year"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "set_year"
        for keyword in publisher_calls[0].keywords
    )


def test_ensemble_does_not_copy_simulation_database_to_apps():
    tree = _script_tree()
    normalized_literals = {
        node.value.replace("\\", "/").lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    }

    forbidden_destinations = (
        "fantasy_football_app/app/simulation.sqlite3",
        "fantasy_football_snake/app/simulation.sqlite3",
    )
    assert not any(
        destination in literal
        for destination in forbidden_destinations
        for literal in normalized_literals
    )

    assert not any(
        isinstance(node, ast.Import)
        and any(alias.name == "shutil" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "shutil"
        for node in ast.walk(tree)
    )

    forbidden_copy_calls = {"copyfile", "copy2", "copytree"}
    assert not any(
        isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "shutil"
                and node.func.attr in forbidden_copy_calls | {"copy"}
            )
            or (
                isinstance(node.func, ast.Name)
                and node.func.id in forbidden_copy_calls
            )
        )
        for node in ast.walk(tree)
    )
