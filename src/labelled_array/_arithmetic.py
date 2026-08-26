"""Element-wise arithmetic for :class:`~labelled_array.core.LabelledArray`.

Kept in a mixin so that a subclass whose semantics make arithmetic meaningless
can leave it out. The operators are written out rather than generated, so that
type checkers, editors and :mod:`pydoc` can see them.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

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
    variables, or another labelled array with the same dimension and variable
    names. Two arrays whose labels differ are aligned first, so arithmetic is
    always between the cells that mean the same thing; the dimension *names*
    must match, because nothing says how to broadcast one dimension onto
    another. Masks are combined so a cell is populated only where both operands
    are, which includes the cells alignment invented. Coordinates and
    attributes are taken from the left operand.
    """

    #: How arithmetic resolves a dimension whose labels differ between operands.
    _arithmetic_join: str = "inner"

    def _align_operands(self, other: Any) -> tuple[Any, Any]:
        """Return this array and ``other`` on one set of dimensions."""
        if other.dims == self.dims:
            return self, other
        if set(other.dims.names) != set(self.dims.names):
            raise ValueError(
                f"operands have different dimensions: {list(self.dims.names)} "
                f"vs {list(other.dims.names)}"
            )
        from labelled_array._align import align

        if other.dims.names != self.dims.names:
            other = other.transpose(self.dims.names)
        return align([self, other], join=self._arithmetic_join)

    def _binary_op(self, other: Any, op: Callable, reflected: bool = False) -> Any:
        """Apply ``op`` to every variable against ``other``."""
        from labelled_array.core import LabelledArray

        left = self
        if isinstance(other, LabelledArray):
            if other.variables != self.variables:
                raise ValueError(
                    f"operands have different variables: {self.variables} vs {other.variables}"
                )
            left, other = self._align_operands(other)
            operands = {name: other[name] for name in left.variables}
            mask = _combine_masks(left.mask, other.mask)
        else:
            operands = dict.fromkeys(self.variables, other)
            mask = self.mask

        data = {
            name: op(operands[name], left[name])
            if reflected
            else op(left[name], operands[name])
            for name in left.variables
        }
        return left._replace(data=data, mask=mask)

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
