import time

from upstash_redis import Redis

from app.config import settings

_redis_client: Redis | None = None

def get_redis_client() -> Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis(
            url=settings.upstash_redis_url,
            token=settings.upstash_redis_token,
        )
    return _redis_client


'''Here is the code explained step-by-step in 5 numbered lines:

1. **`now = time.time()` & `window_start = 1000 - 60 = 940**`
Calculates the current timestamp (`1000`) and the cutoff point (`940`), setting a 60-second window where only requests between timestamp `940` and `1000` count.
2. **`pipe.zremrangebyscore(key, 0, 940)`**
Deletes all old requests from Redis that happened before timestamp `940` so they no longer count toward the quota.
3. **`pipe.zadd(key, {"1000": 1000})`**
Logs the current request timestamp (`1000`) into the user's Redis list.
4. **`pipe.zcard(key)` & `pipe.expire(key, 60)**`
Counts the total valid requests remaining in the list (e.g., `3` active requests) and resets the key's memory auto-delete timer to 60 seconds.
5. **`results = pipe.exec()`**
Executes all 4 commands in a single Redis call and returns the results list, where `results[2]` gives the final count of `3` requests to evaluate against your limit.'''

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def is_allowed(self, key: str) -> tuple[bool, int, int]:
        client = get_redis_client()
        now = time.time()
        window_start = now - self.window_seconds

        pipe = client.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, self.window_seconds)
        results = pipe.exec()

        request_count: int = results[2]  # type: ignore[assignment]
        remaining = max(0, self.max_requests - request_count)
        allowed = request_count <= self.max_requests

        return allowed, remaining, request_count

'''is_allowed_ip(ip="192.168.1.1", route="/auth/login", limit=5, window_seconds=60)'''    

def is_allowed_ip(ip: str, route: str, limit: int, window_seconds: int) -> tuple[bool, int, int]:
    limiter = RateLimiter(max_requests=limit, window_seconds=window_seconds)
    key = f"rate_limit:ip:{ip}:{route}"
    return limiter.is_allowed(key)

'''is_allowed_user(user_id="usr_998234")'''
def is_allowed_user(
    user_id: str, limit: int = 20, window_seconds: int = 60
) -> tuple[bool, int, int]:
    limiter = RateLimiter(max_requests=limit, window_seconds=window_seconds)
    key = f"rate_limit:user:{user_id}"
    return limiter.is_allowed(key)