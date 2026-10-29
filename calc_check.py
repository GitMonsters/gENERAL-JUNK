from math import floor

def calc(items, jur, order_type="recreational", tier_pct=0.10, promo=0.0, delivery=4.99, tip=0.0):
    rates = {
      "kingman":  (0.056,0.008,0.020,0.16),
      "phoenix":  (0.056,0.007,0.023,0.16),
      "tucson":   (0.056,0.005,0.025,0.16),
      "flagstaff":(0.056,0.0075,0.0235,0.16),
      "mesa":     (0.056,0.007,0.018,0.16),
      "yuma":     (0.056,0.007,0.020,0.16),
    }
    st,co,ci,ex = rates[jur]
    sub = sum(round(p*100)*q for p,q in items)
    tier_d = round(sub*tier_pct)
    promo_d = round(sub*promo)
    disc = min(sub, tier_d+promo_d)
    base = sub-disc
    tpt = round(base*(st+co+ci))
    exc = 0 if order_type=="medical" else round(base*ex)
    deliv = round(delivery*100)
    total = base+tpt+exc+deliv
    return dict(sub=sub/100, disc=disc/100, base=base/100, tpt=tpt/100, exc=exc/100,
                deliv=deliv/100, total=total/100, pct=(st+co+ci)*100)

items=[(18.00,1),(33.75,1),(22.00,1)]
for j in ["kingman","phoenix","tucson","flagstaff","yuma"]:
    r=calc(items,j); print(j, r)
print("medical kingman", calc(items,"kingman",order_type="medical"))
print("no discount kingman", calc(items,"kingman",tier_pct=0))
