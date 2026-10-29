# Technical Specification - Arizona Super Highways

## System Architecture

### Overview
Arizona Super Highways is a multi-platform system consisting of:
1. **Print Component** - Pocket passport magazine
2. **Web Application** - Public-facing site + member portal
3. **Mobile Applications** - iOS and Android native apps
4. **Backend API** - RESTful/GraphQL API
5. **Admin Dashboard** - For businesses, vendors, events
6. **POS Integration Layer** - Dutchie integration

### Architecture Diagram (Conceptual)
```
[Print QR Codes] -> [Mobile/Web App] -> [API Gateway] -> [Services]
                                                          ├── User/Auth
                                                          ├── Locations/Passports
                                                          ├── Rewards/Tier System
                                                          ├── Social/Reviews
                                                          ├── Messaging (E2E Encrypted)
                                                          ├── Live Video/Events
                                                          ├── E-commerce/Cart
                                                          └── Dutchie POS Integration
                                                                └── Order Routing
```

## Technology Stack

### Frontend - Web
- **Framework:** Next.js 14+ (React 18, TypeScript)
- **Styling:** Tailwind CSS
- **Maps:** Mapbox GL / Google Maps API
- **State:** Zustand / React Query
- **Auth:** NextAuth.js / Auth0
- **Real-time:** Socket.io client

### Mobile
- **Framework:** React Native (iOS/Android) or Flutter
- **Navigation:** React Navigation
- **Camera/QR:** react-native-camera / vision-camera
- **Maps:** react-native-maps
- **Push Notifications:** Firebase Cloud Messaging
- **Secure Storage:** react-native-keychain

### Backend
- **Runtime:** Node.js (Express/NestJS) or Python (FastAPI)
- **Database:** PostgreSQL (primary) + Redis (cache/sessions)
- **File Storage:** AWS S3 / Cloudflare R2
- **Real-time:** Socket.io / WebRTC for video
- **Queue:** BullMQ / Redis
- **Search:** Typesense/Meilisearch

### Security & Encryption
- **Auth:** JWT + Refresh tokens, OAuth 2.0
- **Encryption:** TLS 1.3, E2E encryption for messaging (Signal Protocol)
- **Data:** PII encryption at rest
- **Age Verification:** ID scanning / third-party verification
- **Compliance:** HIPAA considerations, state cannabis compliance

## Data Models

### User
```typescript
interface User {
  id: string;
  email: string;
  phone?: string;
  firstName: string;
  lastName: string;
  dob: Date; // age verification
  tier: 'bronze' | 'silver' | 'gold' | 'platinum' | 'medusa';
  points: number;
  passportStamps: Stamp[];
  avatar?: string;
  verified: boolean;
  createdAt: Date;
}
```

### Location (Dispensary/Business)
```typescript
interface Location {
  id: string;
  name: string;
  type: 'dispensary' | 'brand' | 'event' | 'treatment' | 'wellness';
  address: Address;
  geo: { lat: number; lng: number; };
  phone: string;
  email: string;
  hours: BusinessHours;
  menus: Menu[];
  dutchieStoreId?: string; // for integration
  offers: Offer[];
  photos: string[];
  social: SocialLinks;
  qrCodeId: string;
  featured: boolean;
}
```

### Passport/Check-in
```typescript
interface Stamp {
  id: string;
  userId: string;
  locationId: string;
  qrCodeId: string;
  checkedInAt: Date;
  pointsEarned: number;
  offerRedeemed?: string;
}
```

### Offer/Reward
```typescript
interface Offer {
  id: string;
  locationId: string;
  title: string;
  description: string;
  type: 'discount' | 'freebie' | 'drop' | 'swag' | 'exclusive';
  discountPercent?: number;
  discountAmount?: number;
  minPurchase?: number;
  tierRequirement: Tier[];
  pointsCost?: number;
  limitedQuantity?: number;
  expiresAt?: Date;
  hidden: boolean; // for exclusive drops
  code?: string;
}
```

### Social Post
```typescript
interface Post {
  id: string;
  userId: string;
  locationId?: string;
  content: string;
  images: string[];
  likes: string[];
  comments: Comment[];
  type: 'checkin' | 'review' | 'photo' | 'event';
  rating?: number; // for reviews
  createdAt: Date;
}
```

### Message (Encrypted)
```typescript
interface Message {
  id: string;
  conversationId: string;
  senderId: string;
  encryptedContent: string; // E2E encrypted
  iv: string;
  read: boolean;
  createdAt: Date;
}
```

## Features Implementation Details

### QR Code System
- Unique QR per location/page in passport
- QR encodes: `ash://checkin?loc={locationId}&qr={qrCodeId}&sig={signature}`
- Signature prevents forgery
- Scanning requires auth + location verification (optional)
- Awards points/stamps on successful check-in

### Tier System
| Tier | Points Required | Benefits |
|------|-----------------|----------|
| Bronze | 0-99 | Basic rewards, member pricing |
| Silver | 100-249 | 5% extra discount, early access |
| Gold | 250-499 | 10% extra discount, exclusive offers |
| Platinum | 500-999 | 15% extra discount, priority drops |
| Medusa | 1000+ | 20% extra discount, VIP drops, swag priority |

### Dutchie POS Integration
**API Endpoints Needed:**
- `GET /api/v2/retailers/{store_id}/menu` - Sync menus
- `GET /api/v2/retailers/{store_id}/products` - Product details
- `POST /api/v2/orders` - Create order
- `GET /api/v2/orders/{order_id}` - Order status

**Order Routing Logic:**
```javascript
function routeOrder(cart, userLocation) {
  // Find participating locations within delivery radius
  const nearby = findLocationsInRadius(userLocation, deliveryRadius);
  // Filter by inventory availability
  const available = filterByInventory(nearby, cart.items);
  // Sort by distance
  const sorted = sortByDistance(available, userLocation);
  // Route to closest with full inventory
  return sorted[0] || routeToBestMatch(sorted, cart.items);
}
```

### Live Video Feeds
- WebRTC for peer-to-peer or use services like Agora/Daily.co
- Room-based access (tier-gated)
- Recording with consent
- Moderation tools

### Encrypted Messaging
- Implement Signal Protocol (libsignal)
- Key exchange on first message
- Forward secrecy
- Disappearing messages (optional)
- Report/block functionality

## API Design (REST)

### Auth
- `POST /api/auth/register` - Register with age verification
- `POST /api/auth/login` - Login
- `POST /api/auth/verify` - Age/identity verification
- `POST /api/auth/refresh` - Refresh token

### Locations
- `GET /api/locations` - List with filters (type, geo)
- `GET /api/locations/:id` - Location details
- `GET /api/locations/nearby` - Geo search
- `POST /api/locations/:id/checkin` - QR check-in

### Rewards
- `GET /api/rewards` - Available offers
- `POST /api/rewards/redeem/:offerId` - Redeem offer
- `GET /api/passport` - User passport/stamps

### Social
- `GET /api/feed` - Social feed
- `POST /api/posts` - Create post
- `POST /api/posts/:id/like` - Like post
- `POST /api/posts/:id/comment` - Comment

### Messaging
- `GET /api/conversations` - List conversations
- `POST /api/conversations` - Start conversation
- `GET /api/conversations/:id/messages` - Get messages
- `POST /api/conversations/:id/messages` - Send encrypted message

### Commerce
- `GET /api/menus/:locationId` - Get menu
- `POST /api/cart` - Manage cart
- `POST /api/orders/route` - Route order to location
- `POST /api/orders` - Create order

## Deployment

### Environments
- **Dev:** Local development
- **Staging:** staging.arizona-super-highways.com
- **Prod:** arizona-super-highways.com

### Infrastructure
- **Hosting:** Vercel (web) + AWS Amplify/Expo EAS (mobile)
- **API:** Railway/Render/AWS ECS
- **Database:** Supabase/Neon (Postgres)
- **CDN:** Cloudflare
- **Monitoring:** Sentry, LogRocket

## Compliance Considerations
- **Age Gate:** 21+ verification required
- **Cannabis Regulations:** Comply with AZ state laws
- **Privacy:** GDPR/CCPA considerations
- **Content:** No medical claims, follow advertising restrictions
- **Payments:** Age-restricted e-commerce considerations
- **Data:** Secure storage of verification documents
