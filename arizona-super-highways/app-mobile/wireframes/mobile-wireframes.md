# Mobile App Wireframes - Arizona Super Highways

## Overview
Native mobile app for iOS and Android with QR scanning, maps, passport, social features, and ordering.

## Screen Structure

### 1. Splash/Onboarding
**Screen:** `splash.png` concept
- Logo: Arizona Super Highways
- Tagline: "The Road to Rewards"
- Loading animation
- Age gate on first launch (21+)

### 2. Age Verification
**Screen:** `age-gate.png`
- "Are you 21 or older?"
- Yes/No buttons
- Disclaimer text
- Must verify to proceed

### 3. Authentication
**Screen:** `auth.png`
- Sign Up / Log In tabs
- Email/Phone
- Password
- Social login (optional)
- Terms & Privacy

### 4. Home Feed
**Screen:** `home.png`
- Header: User tier badge, points
- "Scan to Check In" CTA button (prominent)
- Featured location (Medusa Farms)
- Nearby deals
- Recent activity from feed
- Bottom navigation

**Bottom Nav:** Home | Map | Scan | Passport | Profile

### 5. QR Scanner
**Screen:** `scanner.png`
- Full screen camera view
- QR code frame overlay
- "Scan location QR code" text
- Flash toggle
- Manual entry option
- Cancel button

**After Scan:** Success modal with points earned, stamp animation

### 6. Map View
**Screen:** `map.png`
- Full screen map
- User location button
- Filter chips (All, Dispensaries, Events, etc.)
- Location cards at bottom (swipeable)
- Cluster markers
- "Route" button on cards

**Card:** Photo, name, distance, "Open Now" badge, quick actions

### 7. Location Detail
**Screen:** `location-detail.png`
- Hero image
- Name, type, rating
- Address, hours, phone
- Tabs: Overview | Menu | Reviews | Photos
- "Check In" button
- "View Menu & Order" button
- "Get Directions" button
- Passport perk badge

### 8. Menu & Ordering (Dutchie)
**Screen:** `menu.png`
- Category tabs
- Product cards (image, name, price, THC%)
- Add to cart button
- Cart floating button with count
- Filter/sort
- "Route to nearest location" notice

**Cart:** Items, subtotal, tax estimate, "Proceed to Checkout"

### 9. Passport
**Screen:** `passport.png`
- Header: Avatar, name, tier (with crown icon)
- Points total
- Progress bar to next tier
- "My Stamps" grid (collected vs empty)
- "Rewards Wallet" section
- Stamp detail modal on tap

### 10. Rewards/Offers
**Screen:** `rewards.png`
- Tabs: Available | Claimed | Expired
- Offer cards with badges (Exclusive, Limited)
- Tier requirement shown
- "Claim" or "Redeem" button
- QR code for redemption in wallet

### 11. Social Feed
**Screen:** `feed.png`
- Create post button (FAB)
- Filter: All | Following | Nearby
- Post cards: avatar, location tag, photos, caption
- Like, comment, share buttons
- Verified purchase badge
- "Check In & Post" prompt

### 12. Create Post
**Screen:** `create-post.png`
- Photo/video picker
- Caption input
- Tag location
- Rating stars (for reviews)
- Privacy settings
- Post button

### 13. Messages (Encrypted)
**Screen:** `messages.png`
- Conversation list
- Unread badge
- Search
- "New Message" button
- E2E encrypted indicator

**Chat:** Messages, encrypted lock icon, typing indicator

### 14. Live Events
**Screen:** `events.png`
- Upcoming / Live Now tabs
- Event cards with live badge
- "Join Live" button
- Event details modal
- Tier gating shown

**Live Stream:** Video player, reactions, chat (moderated)

### 15. Profile
**Screen:** `profile.png`
- Cover/header
- Avatar, stats (Stamps, Points, Reviews)
- Menu items: My Wallet, Messages, Settings, Help
- "Become a Partner" link
- Logout

## UI Components

### Colors
```css
Primary: #D35400 (Arizona Orange)
Secondary: #27AE60 (Green)
Accent: #F39C12 (Gold)
Purple: #6C3483 (Medusa)
Dark: #1C1C1C
Light: #FDF6E3
Gold: #FFD700 (Tier)
```

### Typography
- **Font:** Inter, SF Pro, Roboto
- **Headings:** 24-32px, Bold/ExtraBold
- **Body:** 14-16px, Regular/Medium
- **Caption:** 12px

### Icons
- Font Awesome or SF Symbols
- Cannabis leaf, map pin, QR code, stamp, crown

### Animations
- Stamp collection animation
- Check-in success confetti
- Pull to refresh
- Skeleton loaders
- Tab transitions

## Native Features
- **Camera:** QR scanning, photo capture
- **Location:** Geolocation, geofencing
- **Push Notifications:** Drops, messages, events
- **Biometrics:** Face ID/Touch ID for sensitive actions
- **Deep Linking:** `ash://` scheme
- **Share:** Native share sheet
- **Offline:** Cached passport, basic offline access

## Wireframe Notes
- Prioritize thumb-friendly navigation (bottom nav)
- Large tap targets (44x44 minimum)
- Dark mode consideration for later
- Accessibility: VoiceOver/TalkBack support
- Age gate cannot be bypassed
- QR scanner needs camera permission
- Location permission for maps/nearby
