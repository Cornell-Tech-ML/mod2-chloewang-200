from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Apply a map function to the given tensor."""


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Apply a map function element-wise to a tensor."""
        ...

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Apply a zip function to combine two tensors element-wise."""
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Apply a reduce function along a specific dimension of a tensor."""
        ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Perform matrix multiplication on two tensors."""
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Initialize a tensor backend that uses tensor operations.

        Args:
            ops: Tensor operations object implementing map, zip, and reduce.

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Apply a function element-wise to a tensor, handling broadcasting."""
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Apply a function element-wise to two tensors, handling broadcasting."""
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Apply a reduce function over a specific dimension of a tensor."""
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start
            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Not yet implemented: Matrix multiplication for two tensors."""
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level implementation of a map function for tensors with different strides.

    Args:
        fn: The function to apply element-wise to the tensor.

    Returns:
        A callable that applies the map function.

    """
    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        for out_index in range(len(out)):
            in_pos = [0] * len(in_shape)
            out_pos = [0] * len(out_shape)
            to_index(out_index, out_shape, out_pos)

            for i in range(len(in_shape)):
                in_pos[i] = 0 if in_shape[i] == 1 else out_pos[i]

            in_index = index_to_position(in_pos, in_strides)
            out[out_index] = fn(in_storage[in_index])

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level implementation of a zip function for tensors with different strides.

    Args:
        fn: The function to apply element-wise to the two tensors.

    Returns:
        A callable that applies the zip function.

    """
    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        for out_index in range(len(out)):
            out_pos = [0] * len(out_shape)
            a_pos = [0] * len(a_shape)
            b_pos = [0] * len(b_shape)

            to_index(out_index, out_shape, out_pos)

            for i in range(len(a_shape)):
                a_pos[i] = out_pos[i] if a_shape[i] != 1 else 0

            for i in range(len(b_shape)):
                b_pos[i] = out_pos[i] if b_shape[i] != 1 else 0

            a_index = index_to_position(a_pos, a_strides)
            b_index = index_to_position(b_pos, b_strides)

            out[out_index] = fn(a_storage[a_index], b_storage[b_index])

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level implementation of a reduce function for tensors.

    Args:
        fn: The reduction function to apply.

    Returns:
        A callable that applies the reduce function.

    """
    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        for out_index in range(len(out)):
            out_pos = [0] * len(out_shape)
            in_pos = [0] * len(a_shape)

            to_index(out_index, out_shape, out_pos)

            for i in range(len(out_pos)):
                in_pos[i] = out_pos[i]

            in_pos[reduce_dim] = 0
            result = a_storage[index_to_position(in_pos, a_strides)]

            for j in range(1, a_shape[reduce_dim]):
                in_pos[reduce_dim] = j
                result = fn(result, a_storage[index_to_position(in_pos, a_strides)])

            out[out_index] = result

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
