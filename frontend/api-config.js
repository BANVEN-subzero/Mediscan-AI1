// api-config.js
// MediScan AI - Global API Base Configuration

const API_BASE = (() => {
    // 1. Check for manual override in localStorage (great for testing or dynamic configuration)
    const savedUrl = localStorage.getItem('MEDISCAN_API_URL');
    if (savedUrl && savedUrl !== 'undefined') return savedUrl;
    const { protocol, hostname, origin } = window.location;

    // Use same origin since frontend and backend are now served together
    return origin;
})();

console.log('MediScan API Base configured to:', API_BASE);
