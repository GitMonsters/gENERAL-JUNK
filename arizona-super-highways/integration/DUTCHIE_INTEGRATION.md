# Dutchie POS Integration Specification

## Overview
Integration with Dutchie (https://dutchie.com/) POS system to enable menu syncing, inventory checking, and order routing to the closest participating location.

## Objectives
- Sync real-time menus from participating Dutchie-powered dispensaries
- Route orders to optimal location based on proximity and inventory
- Enable in-app ordering with seamless handoff to Dutchie
- Track order status
- Support delivery where available

## Dutchie API Access
Dutchie provides APIs for retailers. Integration requires:
- Partnership agreement with Dutchie
- API credentials per retailer or via Dutchie partner program
- Store IDs for each participating location

**API Documentation:** https://developers.dutchie.com/

## Integration Methods

### Option 1: Dutchie Embed (Simpler)
- Use Dutchie's embeddable menu/checkout
- Redirect or iframe in app
- Less custom, faster to implement
- Limited control over routing

### Option 2: Dutchie API (Recommended)
- Full control over UX
- Custom order routing logic
- Unified cart across locations
- Better data/analytics
- More complex

## API Endpoints (Typical Dutchie Structure)

### Menu Sync
```http
GET https://api.dutchie.com/v2/retailers/{retailerId}/menu
Authorization: Bearer {apiKey}
```

**Response includes:**
- Products (flower, concentrates, edibles, etc.)
- Categories
- Pricing
- THC/CBD
- Images
- Inventory availability
- Brands

### Product Details
```http
GET https://api.dutchie.com/v2/retailers/{retailerId}/products/{productId}
```

### Create Order
```http
POST https://api.dutchie.com/v2/retailers/{retailerId}/orders
Authorization: Bearer {apiKey}
Content-Type: application/json

{
  "customer": {
    "firstName": "...",
    "lastName": "...",
    "email": "...",
    "phone": "...",
    "dateOfBirth": "YYYY-MM-DD"
  },
  "items": [
    {
      "productId": "...",
      "quantity": 1
    }
  ],
  "pickupOrDelivery": "pickup|delivery",
  "pickupTime": "ASAP|scheduled",
  "deliveryAddress": { ... }
}
```

### Order Status
```http
GET https://api.dutchie.com/v2/retailers/{retailerId}/orders/{orderId}
```

## Order Routing Algorithm

### Flow
1. User adds items to cart in Arizona Super Highways app
2. User taps "Checkout"
3. Get user location (geolocation)
4. Find all participating locations with Dutchie integration
5. For each location, check menu availability for all cart items
6. Filter to locations that have full inventory
7. Calculate distance from user
8. Sort by distance
9. If delivery available at location, consider delivery radius
10. Route to closest location with full inventory
11. If no location has full inventory, show split-order option or best match
12. Create order via Dutchie's API at selected location
13. Redirect to confirmation/tracking

### Implementation Logic
```javascript
async function routeOrder(cartItems, userLocation) {
  const participatingLocations = await getParticipatingDutchieLocations();
  
  const candidates = [];
  
  for (const location of participatingLocations) {
    const menu = await getMenu(location.dutchieStoreId);
    const availability = checkAvailability(menu, cartItems);
    
    if (availability.canFulfillAll) {
      const distance = calculateDistance(userLocation, location.geo);
      if (location.allowsDelivery && distance <= location.deliveryRadius) {
        candidates.push({ location, distance, type: 'delivery' });
      } else if (distance <= location.pickupRadius) {
        candidates.push({ location, distance, type: 'pickup' });
      }
    }
  }
  
  // Sort by distance
  candidates.sort((a, b) => a.distance - b.distance);
  
  return candidates[0] || null;
}
```

## Data Mapping
We need to store Dutchie references:
```typescript
interface Location {
  // ...
  dutchieEnabled: boolean;
  dutchieStoreId: string;
  dutchieRetailerId: string;
  dutchieApiKey?: string; // encrypted
  allowsDelivery: boolean;
  deliveryRadius: number; // miles
  pickupRadius: number;
}
```

## Security Considerations
- Store API keys encrypted at rest
- Use OAuth where possible
- Never expose keys to client
- All API calls go through backend
- Validate age on orders
- Comply with cannabis payment regulations

## Testing Plan
1. **Sandbox:** Use Dutchie's sandbox/test environment
2. **Mock:** Create mock Dutchie API for development
3. **Pilot:** Test with Medusa Farms (Kingman) once live
4. **Edge Cases:** Out of stock, delivery unavailable, split orders

## Implementation Phases
- **Phase 1:** Research Dutchie partner program, get API access
- **Phase 2:** Build menu sync service
- **Phase 3:** Implement order routing
- **Phase 4:** Test with Medusa Farms
- **Phase 5:** Scale to other locations

## Notes for Medusa Farms Demo
For the demo, we'll simulate Dutchie integration:
- Mock menu data based on their website
- Show routing UI
- Demonstrate the flow
- Note "Powered by Dutchie" in UI
- Show how orders would route

The actual integration would require Medusa Farms to be on Dutchie POS and have API access.
