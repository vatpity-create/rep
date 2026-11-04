from __future__ import annotations

import math


class Value:
    def __init__(self, data, parents=(), operation=""):
        self.data = float(data)
        self.grad = 0.0
        self.parents = tuple(parents)
        self.operation = operation
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += output.grad
            other.grad += output.grad

        output._backward = backward
        return output

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = backward
        return output

    __rmul__ = __mul__

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("The exponent must be a constant number.")
        output = Value(self.data ** exponent, (self,), f"**{exponent}")

        def backward():
            if exponent != 0:
                self.grad += exponent * self.data ** (exponent - 1) * output.grad

        output._backward = backward
        return output

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def relu(self):
        output = Value(max(0.0, self.data), (self,), "relu")

        def backward():
            self.grad += (self.data > 0) * output.grad

        output._backward = backward
        return output

    def sigmoid(self):
        if self.data >= 0:
            probability = 1 / (1 + math.exp(-self.data))
        else:
            exponential = math.exp(self.data)
            probability = exponential / (1 + exponential)
        output = Value(probability, (self,), "sigmoid")

        def backward():
            self.grad += probability * (1 - probability) * output.grad

        output._backward = backward
        return output

    def log(self):
        output = Value(math.log(self.data), (self,), "log")

        def backward():
            self.grad += output.grad / self.data

        output._backward = backward
        return output

    def topological_order(self):
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
        order = self.topological_order()
        for node in order:
            if node.parents:
                node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            node._backward()
