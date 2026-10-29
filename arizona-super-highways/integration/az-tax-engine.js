/**
 * Arizona Super Highways — Arizona cannabis tax engine (demo)
 * ------------------------------------------------------------
 * Estimates the taxes applied to an adult-use (recreational) or medical
 * cannabis order in Arizona. Designed to sit between the app cart and the
 * Dutchie order payload.
 *
 * Included:
 *   - State transaction privilege tax (TPT) 5.6%
 *   - County + city TPT (varies by jurisdiction)
 *   - Adult-use excise tax 16% (medical is exempt)
 *   - Loyalty tier discount + promo discount (applied pre-tax)
 *   - Delivery fee and optional tip (never taxed)
 *
 * NOT legal or tax advice. Rates change; verify with AZDOR and each
 * municipality before production. Some cities also levy a cannabis-specific
 * local tax — add it to `cityCannabis` in the rate table.
 *
 * Amounts are handled in integer cents internally to avoid float drift.
 */

'use strict';

/** All rates are decimals (0.056 = 5.6%). */
const AZ_RATES = {
  kingman:   { label: 'Kingman (Mohave County)',  state: 0.056, county: 0.008,  city: 0.020,  cityCannabis: 0, excise: 0.16 },
  phoenix:   { label: 'Phoenix (Maricopa County)', state: 0.056, county: 0.007,  city: 0.023,  cityCannabis: 0, excise: 0.16 },
  tucson:    { label: 'Tucson (Pima County)',      state: 0.056, county: 0.005,  city: 0.025,  cityCannabis: 0, excise: 0.16 },
  flagstaff: { label: 'Flagstaff (Coconino County)', state: 0.056, county: 0.0075, city: 0.0235, cityCannabis: 0, excise: 0.16 },
  mesa:      { label: 'Mesa (Maricopa County)',    state: 0.056, county: 0.007,  city: 0.018,  cityCannabis: 0, excise: 0.16 },
  yuma:      { label: 'Yuma (Yuma County)',        state: 0.056, county: 0.007,  city: 0.020,  cityCannabis: 0, excise: 0.16 },
  prescott:  { label: 'Prescott (Yavapai County)', state: 0.056, county: 0.0075, city: 0.020,  cityCannabis: 0, excise: 0.16 },
};

/** Loyalty discount by tier — matches the passport tier ladder. */
const TIER_DISCOUNT = {
  bronze: 0,
  silver: 0.05,
  gold: 0.10,
  platinum: 0.15,
  medusa: 0.20,
};

const cents = (n) => Math.round(n * 100);
const dollars = (c) => c / 100;
const round2 = (n) => Math.round((n + Number.EPSILON) * 100) / 100;

/**
 * Calculate a full order breakdown.
 *
 * @param {Object} input
 * @param {Array<{price:number|string, qty:number, taxExempt?:boolean}>} input.items
 *        price is the unit price in dollars (string like "$18.00" is tolerated).
 * @param {keyof AZ_RATES} [input.jurisdiction='kingman']
 * @param {'recreational'|'medical'} [input.orderType='recreational']
 * @param {keyof TIER_DISCOUNT} [input.tier='bronze']
 * @param {number} [input.promoDiscount=0]      percent, e.g. 0.15 for 15%
 * @param {number} [input.deliveryFee=0]        dollars
 * @param {number} [input.tip=0]                dollars
 * @param {boolean} [input.taxDeliveryFee=false]
 * @returns {Object} breakdown with integer-cent fields plus formatted dollars
 */
function calculateOrder(input) {
  const {
    items = [],
    jurisdiction = 'kingman',
    orderType = 'recreational',
    tier = 'bronze',
    promoDiscount = 0,
    deliveryFee = 0,
    tip = 0,
    taxDeliveryFee = false,
  } = input || {};

  const rate = AZ_RATES[jurisdiction];
  if (!rate) throw new Error(`Unknown Arizona jurisdiction: ${jurisdiction}`);

  const itemSubtotalC = items.reduce((sum, item) => {
    const unit = typeof item.price === 'string'
      ? parseFloat(item.price.replace(/[^0-9.]/g, ''))
      : Number(item.price);
    if (!Number.isFinite(unit)) throw new Error('Invalid item price');
    return sum + cents(unit) * Math.max(1, Number(item.qty) || 1);
  }, 0);

  // Discounts apply to the cannabis product subtotal, before tax.
  const tierPct = TIER_DISCOUNT[tier] ?? 0;
  const tierDiscountC = Math.round(itemSubtotalC * tierPct);
  const promoDiscountC = Math.round(itemSubtotalC * Math.max(0, Math.min(1, promoDiscount)));

  // Discounts are capped so the order can never go negative.
  const discountC = Math.min(itemSubtotalC, tierDiscountC + promoDiscountC);
  const discountedSubtotalC = itemSubtotalC - discountC;

  const taxableBaseC = discountedSubtotalC + (taxDeliveryFee ? cents(deliveryFee) : 0);

  const stateC   = Math.round(taxableBaseC * rate.state);
  const countyC  = Math.round(taxableBaseC * rate.county);
  const cityC    = Math.round(taxableBaseC * rate.city);
  const localC   = Math.round(taxableBaseC * (rate.cityCannabis || 0));
  const tptC     = stateC + countyC + cityC + localC;

  const isMedical = orderType === 'medical';
  const exciseC   = isMedical ? 0 : Math.round(taxableBaseC * rate.excise);

  const deliveryC = cents(deliveryFee);
  const tipC      = cents(tip);

  const totalC = discountedSubtotalC + tptC + exciseC + deliveryC + tipC;
  const taxTotalC = tptC + exciseC;

  return {
    jurisdiction,
    jurisdictionLabel: rate.label,
    orderType,
    tier,
    rates: {
      state: rate.state,
      county: rate.county,
      city: rate.city,
      cityCannabis: rate.cityCannabis || 0,
      excise: isMedical ? 0 : rate.excise,
      tptCombined: round2((rate.state + rate.county + rate.city + (rate.cityCannabis || 0)) * 10000) / 10000,
    },
    cents: {
      itemSubtotal: itemSubtotalC,
      tierDiscount: tierDiscountC,
      promoDiscount: promoDiscountC,
      discountTotal: discountC,
      discountedSubtotal: discountedSubtotalC,
      stateTax: stateC,
      countyTax: countyC,
      cityTax: cityC,
      localCannabisTax: localC,
      tpt: tptC,
      excise: exciseC,
      taxTotal: taxTotalC,
      deliveryFee: deliveryC,
      tip: tipC,
      total: totalC,
    },
    formatted: {
      itemSubtotal: dollars(itemSubtotalC).toFixed(2),
      discountTotal: dollars(discountC).toFixed(2),
      subtotal: dollars(discountedSubtotalC).toFixed(2),
      tpt: dollars(tptC).toFixed(2),
      excise: dollars(exciseC).toFixed(2),
      taxTotal: dollars(taxTotalC).toFixed(2),
      deliveryFee: dollars(deliveryC).toFixed(2),
      tip: dollars(tipC).toFixed(2),
      total: dollars(totalC).toFixed(2),
    },
  };
}

/** Human-readable receipt lines for the cart sheet. */
function receiptLines(order) {
  const f = order.formatted;
  const lines = [
    ['Subtotal', `$${f.itemSubtotal}`],
  ];
  if (order.cents.discountTotal > 0) lines.push([`Discounts (${order.tier})`, `-$${f.discountTotal}`]);
  lines.push(
    [`TPT (${(order.rates.tptCombined * 100).toFixed(2)}%)`, `$${f.tpt}`],
    [`AZ excise (${(order.rates.excise * 100).toFixed(0)}%)`, `$${f.excise}`],
    ['Delivery', `$${f.deliveryFee}`],
  );
  if (order.cents.tip > 0) lines.push(['Tip', `$${f.tip}`]);
  lines.push(['Total', `$${f.total}`]);
  return lines;
}

/**
 * Split an order total across the tax buckets for accounting export.
 * Useful if Dutchie's POS calculates tax and you only need reconciliation.
 */
function taxBreakdown(order) {
  const c = order.cents;
  return {
    taxableSales: dollars(c.discountedSubtotal),
    stateTpt: dollars(c.stateTax),
    countyTpt: dollars(c.countyTax),
    cityTpt: dollars(c.cityTax),
    localCannabis: dollars(c.localCannabisTax),
    excise: dollars(c.excise),
    totalTax: dollars(c.taxTotal),
    grossReceipts: dollars(c.total),
  };
}

module.exports = { AZ_RATES, TIER_DISCOUNT, calculateOrder, receiptLines, taxBreakdown };

/* ------------------------------------------------------------------ *
 * Example
 * ------------------------------------------------------------------ *
 * calculateOrder({
 *   items: [
 *     { price: 18.00, qty: 1 },   // Blue Lobster 2g
 *     { price: 33.75, qty: 1 },   // Dime disposable
 *     { price: 22.00, qty: 2 },   // Baked Bros gummies (BOGO -> qty 2, charge 1 in caller)
 *   ],
 *   jurisdiction: 'kingman',
 *   orderType: 'recreational',
 *   tier: 'gold',            // 10% passport discount
 *   deliveryFee: 4.99,
 * });
 */
