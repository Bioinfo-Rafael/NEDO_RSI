# Experience Store simplification audit

| Check | Result |
|---|---:|
| tables | 3 |
| runs | 3 |
| experiences | 23 |
| raw events | 6873 |
| root nodes | 3 |
| missing-parent orphans | 0 |
| nodes participating in a cycle | 0 |
| node-bound events | 2781 |
| unbound events | 4092 |
| raw files checked | 1014 |
| raw source modified or missing | 0 |
| foreign-key violations | 0 |

## Score direction

| Direction | Experiences |
|---|---:|
| minimize | 12 |
| unknown | 11 |

`raw source modified or missing`は`ingest_manifest.json`に記録されたSHA-256と、
現在のraw fileを照合した件数です。
