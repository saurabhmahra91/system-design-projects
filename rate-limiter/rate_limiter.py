from _thread import RLock
import json
import time
import threading


class Capacity:
    """
    Manages the token bucket for rate limiting with millisecond-level refill precision.

    Attributes:
        PRECISION_FACTOR (int): Scale factor for simulating millisecond-precision refill using integer math.
        max_burst (int): Maximum number of tokens the bucket can hold.
        fill_rpm (int): Token refill rate, in requests per minute.
        value_precise (int): Current token count scaled by PRECISION_FACTOR.
    """

    PRECISION_FACTOR = 60_000  # Used to simulate precise filling using integers

    def __init__(self, max_burst: int, fill_rpm: int, lock: RLock | None = None) -> None:
        """
        Initialize the Capacity with maximum burst size and fill rate.

        Args:
            max_burst (int): Maximum tokens in the bucket.
            fill_rpm (int): Token refill rate per minute.

        Raises:
            ValueError: If fill_rpm < 0 or max_burst <= 0
        """
        if fill_rpm < 0:
            raise ValueError("Fill rate can not be negative!")

        if max_burst <= 0:
            raise ValueError("Max burst must be a positive integer value")

        self.fill_rpm: int = fill_rpm
        self.max_burst: int = max_burst
        self.value_precise = self.max_burst * self.PRECISION_FACTOR
        self._lock: RLock = lock if lock is not None else threading.RLock()

    @property
    def value(self) -> int:
        """
        Current token count (rounded to integer from a precise value).

        Returns:
            int: Number of available tokens.
        """
        return self.value_precise // self.PRECISION_FACTOR

    def is_available(self) -> bool:
        """
        Check if there's at least one token available.

        Returns:
            bool: True if tokens are available.
        """
        with self._lock:
            # check for value_precize > 0 here would be inaccurate because
            # in concurent requests case slight leak of value_precise would be evident.
            return self.value_precise >= self.PRECISION_FACTOR

    def deduct_one_request(self) -> None:
        """
        Consume one token from the bucket.
        """
        with self._lock:
            self.value_precise = max(self.value_precise - self.PRECISION_FACTOR, 0)

    def _clip_to_max_milliseconds(self, milliseconds: int) -> int:
        """
        Cap the time delta to prevent overfilling the bucket.

        Args:
            milliseconds (int): Time delta since last request.

        Returns:
            int: Capped time value.
        """

        if self.fill_rpm == 0:
            return milliseconds

        max_fill_time = int(self.PRECISION_FACTOR * self.max_burst / self.fill_rpm)
        return min(milliseconds, max_fill_time)

    def _clip_value_to_max(self) -> None:
        """
        Ensure token count doesn't exceed max burst capacity.
        """
        with self._lock:
            if self.value_precise > self.max_burst * self.PRECISION_FACTOR:
                self.value_precise = self.max_burst * self.PRECISION_FACTOR

    def add_delay(self, milliseconds: int) -> None:
        """
        Add tokens proportionally based on elapsed milliseconds since last request. Capped to avoid overfilling.

        Args:
            milliseconds (int): Milliseconds passed since last request.
        """
        milliseconds = self._clip_to_max_milliseconds(milliseconds)
        added_tokens_precise: int = milliseconds * self.fill_rpm
        with self._lock:
            self.value_precise += added_tokens_precise
        self._clip_value_to_max()

    def __str__(self) -> str:
        """
        Return a JSON string representation of the capacity state.
        """
        return json.dumps({"capacity": self.value, "fill_rpm": self.fill_rpm, "max_burst": self.max_burst})


class RateLimiter:
    """
    A token bucket rate limiter.

    Attributes:
        pk (str): Unique identifier for the client or source.
        fill_rpm (int): Token fill rate per minute.
        max_burst (int): Maximum number of tokens allowed at once.
        capacity (Capacity): Internal token bucket logic.
        _last_request_time (int): Timestamp of the last request.
    """

    def __init__(self, pk: str, fill_rpm: int, max_burst: int) -> None:
        """
        Initialize the rate limiter.

        Args:
            pk (str): Identifier for the client (IP, user ID, etc.)
            fill_rpm (int): Refill rate in requests per minute.
            max_burst (int): Max burst tokens allowed at once.

        Raises:
            ValueError: If pk is invalid, or fill/burst constraints violated.
        """
        if not isinstance(pk, str) or not pk.strip():
            raise ValueError("The request source id should be a non-empty string.")

        if fill_rpm < 0:
            raise ValueError("Fill rate can not be negative!")

        if max_burst <= 0:
            raise ValueError("Max burst must be a positive integer value")

        self.pk: str = pk
        self.fill_rpm: int = fill_rpm
        self.max_burst: int = max_burst
        self._last_request_time: int | None = None
        self._lock: RLock = threading.RLock()
        self.capacity: Capacity = Capacity(max_burst=self.max_burst, fill_rpm=self.fill_rpm, lock=self._lock)

    @property
    def last_request_time(self) -> int:
        """
        Return the last recorded request time.

        Returns:
            int: Timestamp of the last request. Returns 0 if unset.
        """
        return self._last_request_time or 0

    @last_request_time.setter
    def last_request_time(self, value: int) -> None:
        """
        Set the timestamp of the last request.

        Args:
            value (int): The current request time to be stored.
        """
        with self._lock:
            self._last_request_time = value

    def is_available(self) -> bool:
        """
        Check if a request can be made (i.e., tokens are available).

        Returns:
            bool: True if the rate limit allows it.
        """
        return self.capacity.is_available()

    def get_elapsed_time_from_last_request(self, current_time: int) -> int:
        """
        Compute milliseconds since the last request.

        Args:
            current_time: The current time.

        Returns:
            int: Milliseconds elapsed.
        """
        return current_time - self.last_request_time

    def consume(self) -> bool:
        """
        Attempt to process a request and consume a token.

        Returns:
            bool: True if the request is within rate limits, False otherwise.
        """

        with self._lock:
            current_time = time.monotonic_ns() // 1000_000
            elapsed_ms = self.get_elapsed_time_from_last_request(current_time)
            self.capacity.add_delay(elapsed_ms)

            if not self.capacity.is_available():
                return False

            self.capacity.deduct_one_request()
            self.last_request_time = current_time
            return True


if __name__ == "__main__":
    pass
