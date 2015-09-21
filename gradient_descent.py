import unittest
import numpy as np
from numpy.testing.utils import assert_almost_equal


def cost_function(x, theta, y):
    h = x * theta
    delta = h - y
    delta_sum = delta.T * delta
    return delta_sum / len(y) / 2


def gradient_descent_step(x, theta, y, alpha=.01):
    h = x * theta
    delta = h - y
    new_theta = theta - (alpha * (x.T * delta) / len(y))
    assert new_theta.shape == theta.shape
    return new_theta


def differ_insignificantly(theta, new_theta, threshold=.000001):
    delta = abs((theta - new_theta) / theta)
    delta = delta.sum() / len(theta)
    return abs(delta) < threshold


def get_theta_using_gradient_descent(x, y):
    theta = np.matrix(np.zeros(x.shape[1])).T
    while True:
        new_theta = gradient_descent_step(x, theta, y)
        if differ_insignificantly(theta, new_theta):
            return new_theta
        theta = new_theta


def get_approximated_function(training_x, y):
    theta = get_theta_using_gradient_descent(training_x, y)
    return lambda x: x * theta


class Tests(unittest.TestCase):
    def test_cost_function(self):
        x = np.matrix("1 2 3; 4 5 6")
        y = np.matrix("99; 66")
        theta = np.matrix("8; -3; 5")
        self.failUnless(cost_function(x, theta, y) == 1771.25)

    def test_gradient_descent(self):
        x = np.matrix("1  1; 1 0")
        y = np.matrix("3; 1")
        expected_theta = np.matrix("1; 2")
        actual_theta = get_theta_using_gradient_descent(x, y)
        self.failUnless(differ_insignificantly(expected_theta, actual_theta, .01))

    def test_get_approximated_function(self):
        expected_theta = np.matrix("1; 2; 3")
        training_x = np.matrix("1 0 0; 1 1 0; 1 1 1")
        expected_y = training_x * expected_theta
        approximated_function = get_approximated_function(training_x, expected_y)
        assert_almost_equal(expected_y, approximated_function(training_x), 3)
        test_x = np.matrix("8 9 -1; 3 7 -8; 10 66 888")
        self.failUnless(differ_insignificantly(test_x * expected_theta, approximated_function(test_x), 1), .01)








def main():
    unittest.main()


if __name__ == '__main__':
    main()
