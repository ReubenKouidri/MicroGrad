import unittest
from src.value import Value


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.v1 = Value(1.0)
        self.v2 = Value(2.0)
        self.v3 = self.v1 + self.v2
        self.v3.backward()

    def test_addition(self):
        self.assertTrue(isinstance(self.v3, Value))
        self.assertEqual(self.v3.data, self.v1.data + self.v2.data)
        self.assertEqual(self.v1.grad, 1.0)
        self.assertEqual(self.v2.grad, 1.0)
        self.assertEqual(self.v3.grad, 1.0)
        self.assertEqual(self.v3.prev, {self.v1, self.v2})
        self.assertEqual(self.v3.op, "+")


if __name__ == '__main__':
    unittest.main()
