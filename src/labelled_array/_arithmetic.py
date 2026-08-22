"""Element-wise arithmetic for :class:`~labelled_array.core.LabelledArray`.

Kept in a mixin so that a subclass whose semantics make arithmetic meaningless
can leave it out. The operators are written out rather than generated, so that
type checkers, editors and :mod:`pydoc` can see them.
"""

from __future__ import annotations

import operator
from typing import Any, Callable

__all__ = ["ArithmeticMixin"]


def _combine_masks(left, right):
    """Return the element-wise conjunction of two optional masks."""
    if left is None:
        return right
    if right is None:
        return left
    return left & right


class ArithmeticMixin:
    """Element-wise arithmetic applied to every variable.

    The other operand is either a scalar or array broadcastable against the
    variables, or another labelled array with identical dimensions and variable
    names. Masks are combined so a cell is populated only where both operands
    are. Coordinates and attributes are taken from the left operand.
    """

    def _binary_op(self, other: Any, op: Callable, reflected: bool = False) -> Any:
        """Apply ``op`` to every variable against ``other``."""
        from labelled_array.core import LabelledArray

        if isinstance(other, LabelledArray):
            if other.dims != self.dims:
                raise ValueError("operands have different dimensions")
            if other.variables != self.variables:
                raise ValueError(
                    f"operands have different variables: {self.variables} vs {other.variables}"
                )
            operands = {name: other[name] for name in self.variables}
            mask = _combine_masks(self.mask, other.mask)
        else:
            operands = dict.fromkeys(self.variables, other)
            mask = self.mask

        data = {
            name: op(operands[name], self[name]) if reflected else op(self[name], operands[name])
            for name in self.variables
        }
        return self._replace(data=data, mask=mask)

    def _unary_op(self, op: Callable) -> Any:
        """Apply ``op`` to every variable."""
        return self._replace(data={name: op(self[name]) for name in self.variables})

    def __add__(self, other: Any) -> Any:
        return self._binary_op(other, operator.add)

    def __radd__(self, other: Any) -> Any:
        return self._binary_op(other, operator.add, reflected=True)

    def __sub__(self, other: Any) -> Any:
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other: Any) -> Any:
        return self._binary_op(other, operator.sub, reflected=True)

    def __mul__(self, other: Any) -> Any:
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: Any) -> Any:
        return self._binary_op(other, operator.mul, reflected=True)

    def __truediv__(self, other: Any) -> Any:
        return self._binary_op(other, operator.truediv)

    def __rtruediv__(self, other: Any) -> Any:
        return self._binary_op(other, operator.truediv, reflected=True)

    def __floordiv__(self, other: Any) -> Any:
        return self._binary_op(other, operator.floordiv)

    def __rfloordiv__(self, other: Any) -> Any:
        return self._binary_op(other, operator.floordiv, reflected=True)

    def __mod__(self, other: Any) -> Any:
        return self._binary_op(other, operator.mod)

    def __rmod__(self, other: Any) -> Any:
        return self._binary_op(other, operator.mod, reflected=True)

    def __pow__(self, other: Any) -> Any:
        return self._binary_op(other, operator.pow)

    def __rpow__(self, other: Any) -> Any:
        return self._binary_op(other, operator.pow, reflected=True)

    def __neg__(self) -> Any:
        return self._unary_op(operator.neg)

    def __pos__(self) -> Any:
        return self._unary_op(operator.pos)

    def __abs__(self) -> Any:
        return self._unary_op(operator.abs)
