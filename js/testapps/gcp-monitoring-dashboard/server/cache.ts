/**
 * Simple in-memory LRU cache with TTL support.
 *
 * Used to cache GCP API responses to minimize API calls and provide
 * responsive UX. Each entry has a configurable TTL after which it
 * is considered stale and will be re-fetched.
 */

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttlMs: number;
}

/** Default TTL values in milliseconds */
export const CacheTTL = {
  /** Metric time series data */
  METRICS: 60_000, // 60 seconds
  /** Trace listing */
  TRACE_LIST: 30_000, // 30 seconds
  /** Individual trace (immutable once written) */
  TRACE_DETAIL: 300_000, // 5 minutes
  /** Project list */
  PROJECTS: 600_000, // 10 minutes
} as const;

class LRUCache {
  private cache: Map<string, CacheEntry<unknown>>;
  private readonly maxSize: number;

  constructor(maxSize: number = 1000) {
    this.cache = new Map();
    this.maxSize = maxSize;
  }

  /**
   * Get a cached value if it exists and hasn't expired.
   * Accessing a key moves it to the "most recently used" position.
   */
  get<T>(key: string): T | undefined {
    const entry = this.cache.get(key);
    if (!entry) {
      return undefined;
    }

    // Check if entry has expired
    if (Date.now() - entry.timestamp > entry.ttlMs) {
      this.cache.delete(key);
      return undefined;
    }

    // Move to end (most recently used) by re-inserting
    this.cache.delete(key);
    this.cache.set(key, entry);

    return entry.data as T;
  }

  /**
   * Set a value in the cache with the given TTL.
   * If the cache is full, evicts the least recently used entry.
   */
  set<T>(key: string, data: T, ttlMs: number): void {
    // Delete first to ensure it goes to the end
    this.cache.delete(key);

    // Evict LRU entries if at capacity
    while (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        this.cache.delete(firstKey);
      }
    }

    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttlMs,
    });
  }

  /**
   * Get a value from cache, or compute it if missing/expired.
   * This is the primary method for cached API calls.
   */
  async getOrFetch<T>(
    key: string,
    ttlMs: number,
    fetchFn: () => Promise<T>
  ): Promise<T> {
    const cached = this.get<T>(key);
    if (cached !== undefined) {
      return cached;
    }

    const data = await fetchFn();
    this.set(key, data, ttlMs);
    return data;
  }

  /** Remove a specific entry */
  delete(key: string): boolean {
    return this.cache.delete(key);
  }

  /** Clear all entries */
  clear(): void {
    this.cache.clear();
  }

  /** Get the number of entries in the cache */
  size(): number {
    return this.cache.size;
  }

  /**
   * Create a cache key from a prefix and parameters.
   * This ensures consistent key generation across the app.
   */
  static key(prefix: string, params: Record<string, unknown>): string {
    const sortedParams = Object.keys(params)
      .sort()
      .map((k) => `${k}=${JSON.stringify(params[k])}`)
      .join('&');
    return `${prefix}:${sortedParams}`;
  }
}

/** Singleton cache instance */
export const cache = new LRUCache(1000);
