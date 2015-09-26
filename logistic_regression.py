from numpy.testing.utils import assert_almost_equal
import math
import unittest
import numpy as np

__author__ = 'Y.Andrieiev'


def sigmoid(z):
    e_z = np.power(math.e, -z)
    return 1 / (1 + e_z)


def hypothesis(x, theta):
    return sigmoid(x * theta)


def gradient(x, theta, y, llambda=.5):
    h = hypothesis(x, theta)
    usual_gradient = x.T * (h - y)
    regularization_component = llambda * theta
    m = len(y)
    return (usual_gradient + regularization_component) / m


def differ_insignificantly(theta, new_theta, threshold=.000001):
    delta = abs((theta - new_theta) / theta)
    delta = delta.sum() / len(theta)
    return abs(delta) < threshold


def gradient_descent(x, y, llambda=0, alpha=.1):
    theta = np.matrix(np.zeros(x.shape[1])).T
    while True:
        gradient1 = gradient(x, theta, y, llambda)
        new_theta = theta - alpha * gradient1
        if differ_insignificantly(theta, new_theta):
            return new_theta
        theta = new_theta


def gradient_descent_function(x, y, llambda=0, alpha=.1):
    theta = gradient_descent(x, y, llambda, alpha)

    def logistic(x, theta):
        h = sigmoid(x * theta)
        h[h < .5] = 0
        h[h >= .5] = 1
        return h

    return lambda x: logistic(x, theta)


class Tests(unittest.TestCase):
    def testSigmoid(self):
        self.failUnless(sigmoid(100400) > .9)
        self.failUnless(sigmoid(-100400) < .1)
        self.failUnless(sigmoid(0) == .5)
        assert_almost_equal(sigmoid(-math.log(2)), 1 / 3)

    def test_gradient_descent(self):
        x = np.matrix("1 1; 1 2; 1 3; 1 4; 1 5")
        y = np.matrix("0 0 1 1 1").T
        f = gradient_descent_function(x, y, alpha=300)
        actual = f(x)
        self.failUnless(actual.all() == y.all())


def main():
    unittest.main()


if __name__ == '__main__':
    main()
