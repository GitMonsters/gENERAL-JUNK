/**
 * Arizona Super Highways — Dutchie order routing (demo / reference)
 * ------------------------------------------------------------------
 * Routes a cart to the closest participating location that can fill it,
 * calculates Arizona taxes, and builds a Dutchie-shaped order payload.
 *
 * Flow:
 *   1. Filter participating locations that have a Dutchie store id.
 *   2. Check live inventory for every cart line.
 *   3. Filter by service model (pickup vs delivery + delivery radius).
 *   4. Score by distance (then by stock confidence / price).
 *   5. Calculate tax for the winning jurisdiction.
 *   6. Build + submit the order with an idempotency key.
 *
 * The API shapes below are illustrative. Confirm endpoints, field names and
 * auth with Dutchie's partner team before going live.
 */

'use strict';

const { calculateOrder } = require('./az-tax-engine');

/* ------------------------------------------------------------------ */
/* GEO                                                                 */
/* ------------------------------------------------------------------ */

const EARTH_RADIUS_MI = 3958.8;

/** Haversine distance in miles. */
function distanceMiles(a, b) {
  if (!a || !b) return Infinity;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(b.lat - a.lat);
  const dLng = toRad(b.lng - a.lng);
  const lat1 = toRad(a.lat);
  const lat2 = toRad(b.lat);
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;
  return 2 * EARTH_RADIUS_MI * Math.asin(Math.min(1, Math.sqrt(h)));
}

/* ------------------------------------------------------------------ */
/* LOCATION / INVENTORY SHAPES                                         */
/* ------------------------------------------------------------------ */

/**
 * @typedef {Object} Location
 * @property {string} id
 * @property {string} name
 * @property {'kingman'|'phoenix'|'tucson'|'flagstaff'|'mesa'|'yuma'|'prescott'} jurisdiction
 * @property {{lat:number,lng:number}} geo
 * @property {string} [dutchieStoreId]
 * @property {boolean} delivery
 * @property {number} [deliveryRadiusMi=10]
 * @property {number} [pickupRadiusMi=40]
 * @property {boolean} active
 */

/**
 * Inventory entry keyed by Dutchie product id.
 * @typedef {{productId:string, inStock:boolean, quantity:number, price:number}} InventoryItem
 */

/* ------------------------------------------------------------------ */
/* CANDIDATE SELECTION                                                 */
/* ------------------------------------------------------------------ */

/**
 * @param {Object} params
 * @param {Array<{productId:string, qty:number}>} params.cart
 * @param {{lat:number,lng:number}} params.userLocation
 * @param {'pickup'|'delivery'} [params.fulfillment='pickup']
 * @param {Location[]} params.locations
 * @param {(location:Location) => Promise<InventoryItem[]>} params.getInventory
 *        Adapter that calls the Dutchie menu/inventory endpoint for a store.
 * @returns {Promise<{best:Object|null, candidates:Object[]}>}
 */
async function findFulfillmentLocation({ cart, userLocation, fulfillment = 'pickup', locations, getInventory }) {
  const candidates = [];

  for (const location of locations) {
    if (!location.active || !location.dutchieStoreId) continue;

    const distance = distanceMiles(userLocation, location.geo);
    const radius = fulfillment === 'delivery'
      ? (location.deliveryRadiusMi ?? 10)
      : (location.pickupRadiusMi ?? 40);

    if (fulfillment === 'delivery' && !location.delivery) continue;
    if (distance > radius) continue;

    let inventory;
    try {
      inventory = await getInventory(location);
    } catch (err) {
      // A single store outage should never block routing.
      continue;
    }

    const byId = new Map(inventory.map((i) => [i.productId, i]));
    const missing = [];
    let lineTotal = 0;

    for (const line of cart) {
      const stock = byId.get(line.productId);
      if (!stock || !stock.inStock || stock.quantity < line.qty) {
        missing.push(line.productId);
      } else {
        lineTotal += stock.price * line.qty;
      }
    }

    candidates.push({
      location,
      distance: Number(distance.toFixed(2)),
      fulfillment,
      canFulfillAll: missing.length === 0,
      missing,
      lineTotal: Number(lineTotal.toFixed(2)),
      etaMinutes: fulfillment === 'delivery'
        ? Math.round(15 + distance * 2.2)
        : 15,
    });
  }

  // Rank: full fills first, then distance, then lower subtotal.
  candidates.sort((a, b) => {
    if (a.canFulfillAll !== b.canFulfillAll) return a.canFulfillAll ? -1 : 1;
    if (a.distance !== b.distance) return a.distance - b.distance;
    return a.lineTotal - b.lineTotal;
  });

  return { best: candidates.find((c) => c.canFulfillAll) || null, candidates };
}

/* ------------------------------------------------------------------ */
/* ORDER BUILD                                                         */
/* ------------------------------------------------------------------ */

/**
 * @param {Object} params
 * @param {Object} params.route        result from findFulfillmentLocation().best
 * @param {Array} params.cart          [{productId,name,price,qty}]
 * @param {Object} params.customer     {firstName,lastName,email,phone,dob}
 * @param {'recreational'|'medical'} params.orderType
 * @param {'bronze'|'silver'|'gold'|'platinum'|'medusa'} params.tier
 * @param {number} [params.promoDiscount]
 * @param {number} [params.tip]
 * @param {{lat:number,lng:number,line1:string,city:string,state:string,zip:string}} [params.deliveryAddress]
 * @param {number} [params.deliveryFee=4.99]
 */
function buildDutchieOrder({ route, cart, customer, orderType, tier, promoDiscount = 0, tip = 0, deliveryAddress, deliveryFee = 4.99 }) {
  if (!route || !route.canFulfillAll) throw new Error('No location can fulfill this cart');

  const isDelivery = route.fulfillment === 'delivery';
  const taxes = calculateOrder({
    items: cart.map((i) => ({ price: i.price, qty: i.qty })),
    jurisdiction: route.location.jurisdiction,
    orderType,
    tier,
    promoDiscount,
    deliveryFee: isDelivery ? deliveryFee : 0,
    tip: isDelivery ? tip : 0,
  });

  const payload = {
    retailerId: route.location.dutchieStoreId,
    idempotencyKey: `ash-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
    source: 'arizona-super-highways',
    type: isDelivery ? 'delivery' : 'pickup',
    customer: {
      firstName: customer.firstName,
      lastName: customer.lastName,
      email: customer.email,
      phone: customer.phone,
      dateOfBirth: customer.dob, // Dutchie verifies 21+ on handoff
    },
    items: cart.map((i) => ({ productId: i.productId, quantity: i.qty })),
    totals: {
      subtotal: Number(taxes.formatted.subtotal),
      discounts: Number(taxes.formatted.discountTotal),
      tax: Number(taxes.formatted.taxTotal),
      deliveryFee: Number(taxes.formatted.deliveryFee),
      tip: Number(taxes.formatted.tip),
      total: Number(taxes.formatted.total),
      breakdown: taxes.cents, // cents — for reconciliation
    },
    ...(isDelivery ? {
      deliveryAddress,
      deliveryNotes: 'Arizona Super Highways passport order',
    } : {}),
    metadata: {
      passportTier: tier,
      loyaltyDiscount: taxes.cents.tierDiscount / 100,
      promoDiscount: taxes.cents.promoDiscount / 100,
      distanceMi: route.distance,
      etaMinutes: route.etaMinutes,
    },
  };

  return { payload, taxes };
}

/* ------------------------------------------------------------------ */
/* SUBMIT                                                              */
/* ------------------------------------------------------------------ */

/**
 * Submit an order to Dutchie. Replace ENDPOINT/auth with the credentials
 * issued through the Dutchie partner program. Never call this from the
 * browser — keep the API key server-side.
 */
async function submitDutchieOrder(payload, { baseUrl = 'https://api.dutchie.com', apiKey, fetchImpl = fetch } = {}) {
  if (!apiKey) throw new Error('Missing Dutchie API key');
  if (!payload?.retailerId) throw new Error('Missing retailerId');

  const res = await fetchImpl(`${baseUrl}/v2/retailers/${payload.retailerId}/orders`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
      'Idempotency-Key': payload.idempotencyKey,
    },
    body: JSON.stringify(payload),
  });

  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(`Dutchie order failed (${res.status}): ${body.message || res.statusText}`);
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return body;
}

/* ------------------------------------------------------------------ */
/* END-TO-END                                                          */
/* ------------------------------------------------------------------ */

/**
 * @param {Object} args  see findFulfillmentLocation + buildDutchieOrder
 */
async function placeRoutedOrder({ cart, userLocation, fulfillment, customer, orderType, tier, promoDiscount, tip, deliveryAddress, locations, getInventory, dutchie, deliveryFee }) {
  const { best, candidates } = await findFulfillmentLocation({
    cart, userLocation, fulfillment, locations, getInventory,
  });

  if (!best) {
    // Optional: allow splitting the cart across stores.
    return {
      ok: false,
      reason: 'NO_SINGLE_LOCATION',
      candidates,
      suggestions: candidates.filter((c) => c.missing.length < cart.length),
    };
  }

  const { payload, taxes } = buildDutchieOrder({
    route: best, cart, customer, orderType, tier, promoDiscount, tip, deliveryAddress, deliveryFee,
  });

  const order = await submitDutchieOrder(payload, dutchie);
  return { ok: true, route: best, payload, taxes, order };
}

module.exports = {
  distanceMiles,
  findFulfillmentLocation,
  buildDutchieOrder,
  submitDutchieOrder,
  placeRoutedOrder,
};
