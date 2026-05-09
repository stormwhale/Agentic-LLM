import os
import json
from datetime import datetime, timedelta

class CacheManager:
    def __init__(self, cache_file='market_data_cache.json'):
        self.cache_file = cache_file
        self.cache = self._load()
    
    def _load(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)

    def get(self, key):
        entry = self.cache.get(key)
        if not entry:
            return None

        ttl_hours = entry.get('ttl_hours') #Time to Live - determines how long data is valid
        if ttl_hours is not None:
            cached_at = datetime.fromisoformat(entry['cached_at'])
            if datetime.now() - cached_at > timedelta(hours=ttl_hours):
                del self.cache[key]
                self._save()
                return None
        return entry['data']
    
    def set(self, key, data, ttl_hours=None):
        self.cache[key] = {
            'data': data,
            'cached_at': datetime.now().isoformat(),
            'ttl_hours':ttl_hours
        }

        self._save()

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]
            self._save()
    
    def clear(self): #Clears out all entries in disk or memory
        self.cache = {}
        self._save()
        print(f'[CACHE CLEARED] from {self.cache_file}')
    
    def stats(self): #For debugging - shows cache size and age of entries
        total = len(self.cache)
        print(f'\n[CACHE STATS] {self.cache_file}')
        print(f'Total entries: {total}')
        if total == 0:
            print('Cache is empty')
            return

        print(f"   {'Key':<45} {'Cached At': <25} {'TTL (hrs)'}")
        print(f"   {'-'*45} {'-'*25} {'-'*10}")

        for key, entry in self.cache.items():
            cached_at = entry.get('cached_at', 'unknown')
            ttl = entry.get('ttl_hours')
            ttl_str = 'Forever' if ttl is None else f'{ttl} hrs'

            display_key = key if len(key) <= 45 else key[:42] + '...'
            print(f"   {display_key:<45} {cached_at:<25} {ttl_str}")