# API Reference

## GET /weather/{city}

Returns current weather for `city`.

**Response**
```json
{
  "city": "Paris",
  "tempC": 14.2,
  "conditions": "Overcast"
}
```

Status codes:
- `200` — success
- `404` — unknown city
- `503` — upstream provider unavailable

## GET /health

Returns `{"status":"ok"}` if the service is running. No authentication.
