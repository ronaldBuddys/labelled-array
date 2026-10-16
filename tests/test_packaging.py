"""Packaging checks."""

from importlib.resources import files


def test_py_typed_ships():
    """The PEP 561 marker is present so type checkers use the annotations."""
    assert files("labelled_array").joinpath("py.typed").is_file()
