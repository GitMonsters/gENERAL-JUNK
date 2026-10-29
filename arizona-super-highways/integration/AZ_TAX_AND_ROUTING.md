# Arizona Tax + Dutchie Order Routing

Reference implementation:
- `integration/az-tax-engine.js` — tax math (integer cents)
- `integration/dutchie-order-routing.js` — distance, inventory filtering, payload build, submit

> Demo estimates only. Verify current rates with the Arizona Department of Revenue,
> ADHS, and each municipality before processing live orders. Medical cannabis is
> exempt from the 16% adult-use excise tax; local cannabis-specific taxes vary.

## Tax components

| Component | Adult-use | Medical | Notes |
|---|---|---|---|
| State TPT | 5.6% | 5.6% | Transaction privilege tax |
| County TPT | 0.5–0.8% | 0.5–0.8% | Mohave 0.8%, Maricopa 0.7%, Pima 0.5% |
| City TPT | 1.8–2.5% | 1.8–2.5% | Varies by municipality |
| Local cannabis tax | 0–2%+ | 0–2%+ | Set `cityCannabis` when applicable |
| AZ excise (adult-use) | 16% | **0%** | Applied to the retail subtotal |

Rates used in the demo (`AZ_RATES`):

| Jurisdiction | Combined TPT | Excise |
|---|---|---|
| Kingman (Mohave) | 8.40% | 16% |
| Phoenix (Maricopa) | 8.60% | 16% |
| Tucson (Pima) | 8.60% | 16% |
| Flagstaff (Coconino) | 8.70% | 16% |
| Mesa (Maricopa) | 8.10% | 16% |
| Yuma (Yuma) | 8.30% | 16% |
| Prescott (Yavapai) | 8.35% | 16% |

## Worked example

Cart: Blue Lobster 2g ($18.00) + Dime disposable ($33.75) + Baked Bros BOGO ($22.00 charged once) = **$73.75**
Gold passport tier = 10% discount. Delivery = $4.99.

| Line | Kingman | Phoenix | Tucson | Flagstaff | Yuma |
|---|---|---|---|---|---|
| Subtotal | 73.75 | 73.75 | 73.75 | 73.75 | 73.75 |
| Gold discount | −7.38 | −7.38 | −7.38 | −7.38 | −7.38 |
| Taxable subtotal | 66.37 | 66.37 | 66.37 | 66.37 | 66.37 |
| TPT | 5.58 | 5.71 | 5.71 | 5.77 | 5.51 |
| Excise 16% | 10.62 | 10.62 | 10.62 | 10.62 | 10.62 |
| Delivery | 4.99 | 4.99 | 4.99 | 4.99 | 4.99 |
| **Total** | **87.56** | **87.69** | **87.69** | **87.75** | **87.49** |

Medical, same cart, Kingman: excise drops to **$0.00** → total **$76.94**.

No discount, Kingman: total **$96.74** (the passport tier saves $9.18 on this cart).

## Order routing logic

1. **Eligible stores** — active, has `dutchieStoreId`.
2. **Service model** — delivery requires `delivery: true` and distance ≤ `deliveryRadiusMi`; pickup uses `pickupRadiusMi`.
3. **Inventory check** — every cart line must be in stock in the requested quantity. A store outage is skipped, never fatal.
4. **Rank** — full fills first, then nearest, then lowest subtotal.
5. **Tax** — calculated for the winning store's jurisdiction (not the user's).
6. **Submit** — order payload includes an `Idempotency-Key`; totals and a cents-level breakdown are attached for reconciliation.

If no single store can fill the cart, `placeRoutedOrder()` returns `NO_SINGLE_LOCATION`
plus candidate stores and their missing items, so the UI can offer a split order.

## Dutchie endpoint assumptions

```
POST /v2/retailers/{retailerId}/orders
Authorization: Bearer <server-side key>
Idempotency-Key: ash-...
```

- Keep the API key server-side only.
- Use Dutchie's sandbox for integration tests.
- Cannabis payments cannot be processed inside iOS/Google app wrappers — hand off
  to the retailer's compliant payment flow.
- Dutchie verifies age/ID at handoff; the app still enforces a 21+ gate.

## Wiring it into the demo app

`app-web/app-v2.html` mirrors the engine for the cart sheet (Kingman rates, Gold tier).
To go live:

```js
const { placeRoutedOrder } = require('./dutchie-order-routing');

const result = await placeRoutedOrder({
  cart: [{ productId: 'prod_123', qty: 2 }],
  userLocation: { lat: 35.1894, lng: -114.053 },
  fulfillment: 'delivery',
  customer,
  orderType: 'recreational',
  tier: 'gold',
  deliveryAddress,
  locations,
  getInventory,
  dutchie: { apiKey: process.env.DUTCHIE_API_KEY },
});
```

## Open items before production

- Confirm whether Dutchie calculates tax server-side for each retailer; if so, use
  our engine for display estimates and reconcile against Dutchie's final totals.
- Store per-retailer credentials (not one global key) if Dutchie issues them per store.
- Add tax-exempt medical patient verification flow.
- Add split-cart support and per-line substitution suggestions.
- Log every routing decision for partner analytics (distance, stock, chosen store).
