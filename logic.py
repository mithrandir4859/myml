import unittest
import numpy as np


def cost_function(x, theta, y):
    h = x * theta.T
    delta = h - y
    delta_sum = delta.T * delta
    return delta_sum / len(y) / 2


def gradient_descent_step(x, theta, y, alpha=.05):
    h = x * theta.T
    delta = h - y
    new_theta = theta - (alpha * (x.T * delta) / len(y)).T
    assert new_theta.shape == theta.shape
    return new_theta


def differ_insignificantly(theta, new_theta, threshold=.000001):
    delta = abs((theta - new_theta) / theta)
    delta = delta.sum() / len(theta)
    return abs(delta) < threshold


def gradient_descent(x, y):
    theta = np.matrix(np.zeros(x.shape[1]))
    while True:
        new_theta = gradient_descent_step(x, theta, y)
        if differ_insignificantly(theta, new_theta):
            return new_theta
        print(new_theta)
        theta = new_theta


# def get_approximated_function(x, y):
#     theta = gradient_descent(x, y)
#     return lambda x:


class CostFunctionTest(unittest.TestCase):
    def test_cost_function(self):
        x = np.matrix("1 2 3; 4 5 6")
        y = np.matrix("99; 66")
        theta = np.matrix("8 -3 5")
        actual_cost = cost_function(x, theta, y)
        print("Actual cost is: " + str(actual_cost))
        self.failUnless(actual_cost == 1771.25)

    def test_gradient_descent(self):
        x = np.matrix("1  1; 1 0")
        y = np.matrix("3; 1")
        expected_theta = np.matrix("1, 2")
        actual_theta = gradient_descent(x, y)
        self.failUnless((expected_theta == actual_theta).all())


def main():
    unittest.main()


if __name__ == '__main__':
    main()
