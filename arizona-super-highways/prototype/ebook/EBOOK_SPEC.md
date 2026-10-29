# Digital Ebook/Flipbook Specification

## Overview
Digital version of the Arizona Super Highways pocket passport, optimized for web and mobile viewing.

## Format Options
1. **HTML Flipbook** - Interactive, works in browser
2. **PDF** - Universal, printable
3. **ePub** - For e-readers
4. **Web App Flipbook** - Turn.js, FlipHTML5 style

## Recommended: Interactive Web Flipbook
Use libraries like:
- **Turn.js** - jQuery-based page flip
- **StPageFlip** - Modern, no jQuery
- **Flipbook.js** - Lightweight
- **PDF.js + custom UI** - If PDF-based

## Specifications
- **Dimensions:** 375px x 550px (mobile optimized) or responsive
- **Pages:** 32 pages
- **Format:** Vertical flip or spread view
- **Features:** Zoom, fullscreen, thumbnails, page navigation
- **QR Codes:** Clickable, deep link to app
- **Links:** Internal page links, external links

## Page Structure (Digital)
Same content as print but optimized for digital:
- Interactive elements
- Clickable locations -> app
- Clickable QR codes -> scanner/check-in
- Embedded videos (optional)
- Social sharing

## Demo Implementation
For this prototype, we'll create:
1. **HTML-based flipbook** (`flipbook.html`) using CSS/JS
2. **PDF version** concept (`Arizona-Super-Highways-Guide.pdf` - describe creation)
3. **Responsive web version** for easy viewing

## Tools to Create PDF
- Canva (export as PDF)
- Adobe InDesign
- Affinity Publisher
- Microsoft Publisher
- LibreOffice Draw
- Figma (export PDF)

## Features
- **Page Turn Animation** - Realistic flip effect
- **Table of Contents** - Clickable
- **Search** - Find locations
- **Bookmarking** - Save favorites
- **Share** - Share pages/locations
- **Download** - Save for offline
- **QR Integration** - Scan from digital guide

## File Structure
```
ebook/
├── EBOOK_SPEC.md
├── flipbook.html (interactive demo)
├── css/flipbook.css
├── js/flipbook.js
├── images/ (page images)
└── pdf/ (PDF version)
```
