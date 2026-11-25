"""Small scalar reverse-mode autodiff engine; no NumPy/PyTorch dependency."""

import math


class Value:
    """A scalar, its parents, and a local vector-Jacobian product rule.

    Repeated backward calls accumulate leaf gradients, as in PyTorch. Internal
    adjoints are cleared on each call. Clear parameter .grad before each step.
    Only first derivatives are supported; data must not be edited mid-pass.
    """

    def __init__(self, data, parents=(), op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self.parents = tuple(parents)
        self.op = op
        self.label = label
        self._backward = lambda: None

    @staticmethod
    def coerce(other):
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self.coerce(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = self.coerce(other)
        left, right = self.data, other.data
        out = Value(left * right, (self, other), "*")

        def backward():
            self.grad += right * out.grad
            other.grad += left * out.grad

        out._backward = backward
        return out

    __rmul__ = __mul__

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("Only constant numeric exponents are supported")
        base = self.data
        out = Value(base ** exponent, (self,), f"**{exponent}")

        def backward():
            # x**0 is constant, including when x == 0.
            if exponent != 0:
                self.grad += exponent * base ** (exponent - 1) * out.grad

        out._backward = backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -self.coerce(other)

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __truediv__(self, other):
        return self * self.coerce(other) ** -1

    def __rtruediv__(self, other):
        return self.coerce(other) / self

    def relu(self):
        active = self.data > 0
        out = Value(self.data if active else 0, (self,), "ReLU")

        def backward():
            self.grad += float(active) * out.grad

        out._backward = backward
        return out

    def exp(self):
        result = math.exp(self.data)
        out = Value(result, (self,), "exp")

        def backward():
            self.grad += result * out.grad

        out._backward = backward
        return out

    def log(self):
        base = self.data
        out = Value(math.log(base), (self,), "log")

        def backward():
            self.grad += out.grad / base

        out._backward = backward
        return out

    def sigmoid(self):
        # Branching avoids overflow for large negative inputs.
        if self.data >= 0:
            result = 1 / (1 + math.exp(-self.data))
        else:
            e = math.exp(self.data)
            result = e / (1 + e)
        out = Value(result, (self,), "sigmoid")

        def backward():
            self.grad += result * (1 - result) * out.grad

        out._backward = backward
        return out

    def softplus(self):
        # Stable log(1 + exp(x)); used for BCE with logits.
        x = self.data
        result = max(x, 0) + math.log1p(math.exp(-abs(x)))
        if x >= 0:
            derivative = 1 / (1 + math.exp(-x))
        else:
            e = math.exp(x)
            derivative = e / (1 + e)
        out = Value(result, (self,), "softplus")

        def backward():
            self.grad += derivative * out.grad

        out._backward = backward
        return out

    def sum(self):
        """A scalar has one element, so its sum is itself."""
        return self

    def topological(self):
        """Parents before children; iterative DFS also supports deep graphs."""
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                stack.extend((parent, False) for parent in reversed(node.parents))
        return order

    def backward(self):
        order = self.topological()
        if not self.parents:
            self.grad += 1.0
            return
        for node in order:
            if node.parents:
                node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"
