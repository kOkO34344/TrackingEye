"""The One Euro filter — adaptive smoothing for pointing.

A fixed smoother forces a bad trade: enough smoothing to stop the cursor
shivering while your hand rests also adds lag to every deliberate movement.
This one varies its cutoff with hand speed — heavy smoothing when you're
nearly still, light when you're moving — so the cursor is steady at rest and
still keeps up with a sweep.

Casiez, Roussel & Vogel (2012).
"""

import math


class _LowPass:
    def __init__(self):
        self.value = None

    def __call__(self, sample, alpha):
        if self.value is None:
            self.value = sample
        else:
            self.value = alpha * sample + (1.0 - alpha) * self.value
        return self.value

    def reset(self, value=None):
        self.value = value


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return dt / (dt + tau)


class OneEuro2D:
    """Filters an (x, y) point. Distances are in normalized frame units."""

    def __init__(self, min_cutoff, speed_coefficient, derivative_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.speed_coefficient = speed_coefficient
        self.derivative_cutoff = derivative_cutoff
        self._x = (_LowPass(), _LowPass())
        self._dx = (_LowPass(), _LowPass())
        self._previous = None

    def reset(self, point=None):
        for axis in self._x:
            axis.reset(None if point is None else 0.0)
        for axis in self._dx:
            axis.reset(0.0)
        if point is not None:
            self._x[0].reset(point[0])
            self._x[1].reset(point[1])
        self._previous = point

    def __call__(self, point, dt):
        if dt <= 0.0:
            dt = 1e-3
        if self._previous is None:
            self.reset(point)
            return point

        smoothed = []
        for axis in (0, 1):
            speed = (point[axis] - self._previous[axis]) / dt
            speed = self._dx[axis](speed, _alpha(self.derivative_cutoff, dt))
            # Faster movement -> higher cutoff -> less smoothing -> less lag.
            cutoff = self.min_cutoff + self.speed_coefficient * abs(speed)
            smoothed.append(self._x[axis](point[axis], _alpha(cutoff, dt)))

        self._previous = point
        return tuple(smoothed)
