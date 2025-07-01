import unittest
from unittest.mock import patch
from rate_limiter import RateLimiter


class TestRateLimiterInitialization(unittest.TestCase):
    def test_valid_instantiation(self):
        limiter = RateLimiter(pk="user1", fill_rpm=60, max_burst=100)
        self.assertIsInstance(limiter, RateLimiter)

    def test_invalid_fill_rpm(self):
        with self.assertRaises(ValueError):
            RateLimiter(pk="user1", fill_rpm=-10, max_burst=100)

    def test_invalid_max_burst(self):
        with self.assertRaises(ValueError):
            RateLimiter(pk="user1", fill_rpm=60, max_burst=-1)

    def test_empty_pk(self):
        with self.assertRaises(ValueError):
            RateLimiter(pk="", fill_rpm=60, max_burst=100)


class TestRateLimiterBasicFunctionality(unittest.TestCase):
    def test_burst_limit(self):
        limiter = RateLimiter(pk="user2", fill_rpm=60, max_burst=5)
        for _ in range(5):
            self.assertTrue(limiter.consume())
        self.assertFalse(limiter.consume())

    def test_consume_exceeding_limit(self):
        limiter = RateLimiter(pk="user3", fill_rpm=60, max_burst=1)
        self.assertTrue(limiter.consume())
        self.assertFalse(limiter.consume())

    def test_weird_fill_rates(self):
        limiter = RateLimiter(pk="user-weird", fill_rpm=0, max_burst=1)
        self.assertTrue(limiter.consume())
        self.assertFalse(limiter.consume())

    def test_concurrent_consumers(self):
        import threading

        limiter = RateLimiter(pk="concurrent-user", fill_rpm=1, max_burst=10)
        results = []

        def attempt():
            result = limiter.consume()
            results.append(result)

        threads = [threading.Thread(target=attempt) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertLessEqual(results.count(True), 10)
        self.assertGreaterEqual(results.count(False), 10)


class TestRateLimiterTimeRefill(unittest.TestCase):
    @patch("rate_limiter.time.monotonic_ns")
    def test_partial_refill(self, mock_monotonic_ns):
        t0: int = 100_000_000_000_000
        mock_monotonic_ns.return_value = t0
        limiter = RateLimiter(pk="user4", fill_rpm=60, max_burst=10)

        # Drain all tokens
        for _ in range(10):
            limiter.consume()
        self.assertFalse(limiter.consume())

        # Advance time by 30 seconds → should refill 30 tokens, capped at 10
        mock_monotonic_ns.return_value = t0 + 30 * 1_000_000_000
        self.assertTrue(limiter.consume())  # token bucket should be full again

        # Consume another one, leaving 9
        self.assertTrue(limiter.consume())

        # Advance time by 1 second → should refill 1 token, back to 10
        mock_monotonic_ns.return_value = t0 + 31 * 1_000_000_000
        self.assertTrue(limiter.consume())  # should succeed

    @patch("rate_limiter.time.monotonic_ns")
    def test_refill_caps_at_max_burst(self, mock_monotonic_ns):
        t0: int = 100_000_000_000_000
        mock_monotonic_ns.return_value = t0
        limiter = RateLimiter(pk="user5", fill_rpm=60, max_burst=10)
        for _ in range(5):
            limiter.consume()

        mock_monotonic_ns.return_value = t0 + 60 * 10 * 1_000_000_000
        for _ in range(10):
            self.assertTrue(limiter.consume())
        self.assertFalse(limiter.consume())

    @patch("rate_limiter.time.monotonic_ns")
    def test_long_time_gap(self, mock_monotonic_ns):
        t0: int = 100_000_000_000_000
        mock_monotonic_ns.return_value = t0
        limiter = RateLimiter(pk="user-gap", fill_rpm=10, max_burst=5)
        for _ in range(5):
            limiter.consume()
        self.assertFalse(limiter.consume())
        mock_monotonic_ns.return_value = t0 + 60 * 60 * 2 * 1_000_000_000
        self.assertTrue(limiter.consume())


class TestRateLimiterMultiUser(unittest.TestCase):
    def test_independent_limiters(self):
        limiter_a = RateLimiter(pk="userA", fill_rpm=60, max_burst=3)
        limiter_b = RateLimiter(pk="userB", fill_rpm=60, max_burst=3)

        for _ in range(3):
            self.assertTrue(limiter_a.consume())
            self.assertTrue(limiter_b.consume())

        self.assertFalse(limiter_a.consume())
        self.assertFalse(limiter_b.consume())


class TestRateLimiterEdgeCases(unittest.TestCase):
    def test_spam_consume(self):
        limiter = RateLimiter(pk="user6", fill_rpm=100, max_burst=3)
        results = [limiter.consume() for _ in range(5)]
        self.assertEqual(results, [True, True, True, False, False])

    @patch("rate_limiter.time.monotonic_ns")
    def test_slow_user(self, mock_monotonic_ns):
        t0: int = 100_000_000_000_000
        mock_monotonic_ns.return_value = t0
        limiter = RateLimiter(pk="user7", fill_rpm=60, max_burst=1)

        mock_monotonic_ns.return_value = t0
        self.assertTrue(limiter.consume())

        mock_monotonic_ns.return_value = t0 + 60 * 1_000_000_000
        self.assertTrue(limiter.consume())

        mock_monotonic_ns.return_value = t0 + 120 * 1_000_000_000
        self.assertTrue(limiter.consume())

    @patch("rate_limiter.time.monotonic_ns")
    def test_time_goes_backwards(self, mock_monotonic_ns):
        t0 = 1000000
        mock_monotonic_ns.return_value = t0
        limiter = RateLimiter(pk="user8", fill_rpm=60, max_burst=1)

        mock_monotonic_ns.return_value = t0
        self.assertTrue(limiter.consume())

        mock_monotonic_ns.return_value = t0 - 30 * 1_000_000_000
        self.assertFalse(limiter.consume())  # should handle gracefully


class TestRateLimiterStress(unittest.TestCase):
    def test_high_freq_within_limit(self):
        limiter = RateLimiter(pk="user9", fill_rpm=600, max_burst=100)
        for _ in range(100):
            self.assertTrue(limiter.consume())

    def test_high_freq_exceeding_limit(self):
        limiter = RateLimiter(pk="user10", fill_rpm=60, max_burst=5)
        results = [limiter.consume() for _ in range(10)]
        self.assertEqual(results.count(True), 5)
        self.assertEqual(results.count(False), 5)


if __name__ == "__main__":
    unittest.main()
