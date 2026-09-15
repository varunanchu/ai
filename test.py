"""Sample test suite using Python's built-in unittest module."""

import unittest


def add(a: int | float, b: int | float) -> int | float:
    """Return the sum of a and b."""
    return a + b


def divide(a: int | float, b: int | float) -> float:
    """Return the division of a by b. Raises ValueError if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def is_palindrome(text: str) -> bool:
    """Check whether a string is a palindrome, ignoring case and non-alphanumeric characters."""
    cleaned = "".join(ch.lower() for ch in text if ch.isalnum())
    return cleaned == cleaned[::-1]


class TestSampleFunctions(unittest.TestCase):
    """Test cases for sample functions."""

    def test_add_positive_numbers(self):
        self.assertEqual(add(2, 3), 5)

    def test_add_negative_numbers(self):
        self.assertEqual(add(-1, -1), -2)

    def test_divide_valid(self):
        self.assertEqual(divide(10, 2), 5.0)

    def test_divide_by_zero_raises_error(self):
        with self.assertRaises(ValueError):
            divide(10, 0)

    def test_is_palindrome_true(self):
        self.assertTrue(is_palindrome("racecar"))
        self.assertTrue(is_palindrome("A man, a plan, a canal: Panama"))

    def test_is_palindrome_false(self):
        self.assertFalse(is_palindrome("hello"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
