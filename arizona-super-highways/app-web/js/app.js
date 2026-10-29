// Arizona Super Highways - Web App Demo JavaScript

// Navigation
const navToggle = document.getElementById('navToggle');
const navMenu = document.getElementById('navMenu');

if (navToggle) {
    navToggle.addEventListener('click', () => {
        navMenu.classList.toggle('active');
    });
}

// Section Navigation
const navLinks = document.querySelectorAll('.nav-link');
navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href').substring(1);
        showSection(targetId);
        
        // Update active nav link
        navLinks.forEach(l => l.classList.remove('active'));
        link.classList.add('active');
        
        // Close mobile menu
        if (navMenu) navMenu.classList.remove('active');
    });
});

function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Show target section
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.add('active');
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    
    // Initialize map if showing map section
    if (sectionId === 'map' && !window.mapInitialized) {
        initMap();
    }
}

// QR Scanner Modal
function openScanner() {
    const modal = document.getElementById('scannerModal');
    if (modal) {
        modal.classList.add('active');
    }
}

function closeScanner() {
    const modal = document.getElementById('scannerModal');
    if (modal) {
        modal.classList.remove('active');
    }
}

// Location Modal
function viewLocation(locId) {
    const modal = document.getElementById('locationModal');
    if (modal) {
        modal.classList.add('active');
    }
}

function closeLocationModal() {
    const modal = document.getElementById('locationModal');
    if (modal) {
        modal.classList.remove('active');
    }
}

// Check In Demo
function checkInDemo(locId) {
    // Close modals
    closeScanner();
    closeLocationModal();
    
    // Show success message
    alert('✅ Check-In Successful!\n\nLocation: Medusa Farms\nPoints Earned: +50\nStamp Added to Passport\n\nYou earned the "First Check-In" bonus!');
    
    // Update passport (demo)
    updatePassportDemo();
    
    // Navigate to passport
    showSection('passport');
    navLinks.forEach(l => {
        l.classList.remove('active');
        if (l.getAttribute('href') === '#passport') {
            l.classList.add('active');
        }
    });
}

function simulateCheckIn() {
    checkInDemo('medusa');
}

function updatePassportDemo() {
    // This would update the UI in a real app
    console.log('Passport updated with new stamp');
}

// Close modals when clicking outside
window.addEventListener('click', (e) => {
    const scannerModal = document.getElementById('scannerModal');
    const locationModal = document.getElementById('locationModal');
    
    if (e.target === scannerModal) {
        closeScanner();
    }
    if (e.target === locationModal) {
        closeLocationModal();
    }
});

// Map Initialization
let map;
window.mapInitialized = false;

function initMap() {
    // Center on Kingman, AZ (where Medusa Farms is located)
    map = L.map('mapContainer').setView([35.1894, -114.053], 10);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);
    
    // Add Medusa Farms marker
    const medusaIcon = L.divIcon({
        className: 'custom-marker',
        html: '<div style="background: #D35400; width: 30px; height: 30px; border-radius: 50% 50% 50% 0; transform: rotate(-45deg); border: 3px solid white; display: flex; align-items: center; justify-content: center;"><i class="fas fa-cannabis" style="color: white; transform: rotate(45deg);"></i></div>',
        iconSize: [30, 30],
        iconAnchor: [15, 30]
    });
    
    const medusaMarker = L.marker([35.1894, -114.053], {icon: medusaIcon}).addTo(map);
    medusaMarker.bindPopup(`
        <div style="text-align: center; min-width: 200px;">
            <strong style="color: #D35400; font-size: 1.1rem;">Medusa Farms</strong><br>
            <span style="color: #6C3483;">Recreational & Medical</span><br>
            <small>3490 N Bank St, Kingman, AZ</small><br><br>
            <button onclick="viewLocation('medusa'); map.closePopup();" style="background: #D35400; color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px; cursor: pointer;">View Details</button>
        </div>
    `);
    
    // Add some demo locations
    const demoLocations = [
        { lat: 34.0522, lng: -118.2437, name: 'Phoenix Demo Dispensary', type: 'dispensary' },
        { lat: 32.2226, lng: -110.9747, name: 'Tucson Wellness Co.', type: 'wellness' },
        { lat: 35.1983, lng: -111.6513, name: 'Flagstaff Events Hub', type: 'event' },
    ];
    
    demoLocations.forEach(loc => {
        const icon = L.divIcon({
            className: 'custom-marker',
            html: '<div style="background: #27AE60; width: 25px; height: 25px; border-radius: 50%; border: 2px solid white; display: flex; align-items: center; justify-content: center;"><i class="fas fa-map-pin" style="color: white; font-size: 0.75rem;"></i></div>',
            iconSize: [25, 25],
            iconAnchor: [12, 12]
        });
        L.marker([loc.lat, loc.lng], {icon: icon}).addTo(map).bindPopup(`<strong>${loc.name}</strong><br><small>${loc.type}</small>`);
    });
    
    window.mapInitialized = true;
}

// Map filters
const mapFilters = document.querySelectorAll('.map-filter');
mapFilters.forEach(filter => {
    filter.addEventListener('click', () => {
        mapFilters.forEach(f => f.classList.remove('active'));
        filter.classList.add('active');
        // In a real app, this would filter markers
        console.log('Filter:', filter.dataset.filter);
    });
});

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Arizona Super Highways Demo Loaded');
    console.log('Demo Location: Medusa Farms - https://www.medusafarms.com/');
});
