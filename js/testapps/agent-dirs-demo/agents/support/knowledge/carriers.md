---
type: Reference
title: Shipping carriers
description: Which carrier ACME uses per destination region, with tracking URL patterns.
status: stable
---

# Shipping carriers

| Region | Carrier | Tracking URL |
| ------ | ------- | ------------ |
| US domestic | UPS | https://ups.example/track/{id} |
| United Kingdom | DHL Express | https://dhl.example/track/{id} |
| EU (other) | DPD | https://dpd.example/track/{id} |
| Rest of world | ACME Post | https://acme.example/track/{id} |

Orders shipping to the United Kingdom always go via DHL Express; there is no
alternative carrier for that region.
