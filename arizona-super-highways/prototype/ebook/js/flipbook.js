// Arizona Super Highways Flipbook JavaScript

let currentPage = 0;
const pages = document.querySelectorAll('.page');
const totalPages = pages.length;
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const firstBtn = document.getElementById('firstBtn');
const lastBtn = document.getElementById('lastBtn');
const pageNum = document.getElementById('pageNum');

function showPage(n) {
    // Hide all pages
    pages.forEach(page => {
        page.classList.remove('active');
    });
    
    // Show current page
    if (n >= 0 && n < totalPages) {
        pages[n].classList.add('active');
        currentPage = n;
    }
    
    // Update page number
    pageNum.textContent = `Page ${currentPage + 1} of ${totalPages}`;
    
    // Update button states
    prevBtn.disabled = currentPage === 0;
    nextBtn.disabled = currentPage === totalPages - 1;
}

function nextPage() {
    if (currentPage < totalPages - 1) {
        showPage(currentPage + 1);
    }
}

function prevPage() {
    if (currentPage > 0) {
        showPage(currentPage - 1);
    }
}

function firstPage() {
    showPage(0);
}

function lastPage() {
    showPage(totalPages - 1);
}

// Event listeners
if (nextBtn) nextBtn.addEventListener('click', nextPage);
if (prevBtn) prevBtn.addEventListener('click', prevPage);
if (firstBtn) firstBtn.addEventListener('click', firstPage);
if (lastBtn) lastBtn.addEventListener('click', lastPage);

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') nextPage();
    if (e.key === 'ArrowLeft') prevPage();
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    showPage(0);
    console.log('Arizona Super Highways Flipbook Demo Loaded');
    console.log(`Total pages: ${totalPages}`);
});
