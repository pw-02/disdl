from enum import Enum
class CacheStatus(Enum):
    CACHED = "CACHED"
    CACHING_IN_PROGRESS = "CACHING_IN_PROGRESS"
    NOT_CACHED = "NOT_CACHED"