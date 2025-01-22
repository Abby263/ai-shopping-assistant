document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('searchForm');
    const searchInput = document.getElementById('searchQuery');
    const searchProgress = document.getElementById('searchProgress');
    const searchLogs = document.getElementById('searchLogs');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const error = document.getElementById('error');
    const exampleSearches = document.getElementById('exampleSearches');

    let logUpdateInterval;

    function addLogEntry(message, type = 'info') {
        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;
        entry.innerHTML = `
            <span class="timestamp">[${new Date().toLocaleTimeString()}]</span>
            <span class="message">${message}</span>
        `;
        searchLogs.appendChild(entry);
        searchLogs.scrollTop = searchLogs.scrollHeight;
    }

    async function startLogUpdates() {
        let lastTimestamp = null;
        
        logUpdateInterval = setInterval(async () => {
            try {
                const response = await fetch('/logs', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ last_timestamp: lastTimestamp })
                });
                
                const data = await response.json();
                if (data.logs && data.logs.length > 0) {
                    data.logs.forEach(log => {
                        addLogEntry(log.message, log.type);
                        lastTimestamp = log.timestamp;
                    });
                }
            } catch (error) {
                console.error('Error fetching logs:', error);
            }
        }, 500);
    }

    function stopLogUpdates() {
        if (logUpdateInterval) {
            clearInterval(logUpdateInterval);
        }
    }

    // Show example searches after a short delay
    setTimeout(() => {
        exampleSearches.classList.remove('d-none');
    }, 500);

    // Handle example search clicks
    document.querySelectorAll('.example-search').forEach(button => {
        button.addEventListener('click', () => {
            searchInput.value = button.textContent;
            searchForm.dispatchEvent(new Event('submit'));
        });
    });

    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const query = searchInput.value.trim();
        if (!query) {
            showError('Please enter a search query');
            return;
        }

        // Reset and show progress
        searchLogs.innerHTML = '';
        searchProgress.classList.remove('d-none');
        results.classList.add('d-none');
        error.classList.add('d-none');
        exampleSearches.classList.add('d-none');
        
        addLogEntry('Starting search...', 'info');
        addLogEntry(`Searching for: ${query}`, 'search');
        startLogUpdates();
        
        try {
            const response = await fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `query=${encodeURIComponent(query)}`
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to fetch results');
            }
            
            // Stop log updates and show final status
            stopLogUpdates();
            addLogEntry(`Found ${data.total_results} products`, 'success');
            
            if (data.error) {
                error.textContent = data.error;
                error.classList.remove('d-none');
                return;
            }
            
            // Display results
            displayResults(data);
            results.classList.remove('d-none');
            
        } catch (err) {
            console.error('Search error:', err);
            stopLogUpdates();
            addLogEntry(`Error: ${err.message}`, 'error');
            error.textContent = err.message;
            error.classList.remove('d-none');
        }
    });

    function showError(message) {
        loading.classList.add('d-none');
        error.textContent = message;
        error.classList.remove('d-none');
    }

    function displayResults(data) {
        const productsContent = document.getElementById('productsContent');
        const bestDealContent = document.getElementById('bestDealContent');
        
        // Clear previous results
        productsContent.innerHTML = '';
        bestDealContent.innerHTML = '';
        
        if (data.products && data.products.length > 0) {
            // Display all products
            productsContent.innerHTML = data.products.map(product => `
                <div class="col-md-4 mb-4">
                    <div class="card product-card h-100">
                        ${product.image_url ? `<img src="${product.image_url}" class="card-img-top" alt="${product.title}">` : ''}
                        <div class="card-body">
                            <h5 class="card-title">${product.title}</h5>
                            <p class="price mb-2">$${parseFloat(product.price).toFixed(2)}</p>
                            <p class="mb-3">${product.description || ''}</p>
                            <a href="${product.url}" class="btn btn-primary" target="_blank">View Product</a>
                        </div>
                    </div>
                </div>
            `).join('');
            
            // Display best deal
            const bestDeal = data.products[0];
            bestDealContent.innerHTML = `
                <div class="row align-items-center">
                    <div class="col-md-8">
                        <h4 class="mb-3">${bestDeal.title}</h4>
                        <p class="price mb-3">$${parseFloat(bestDeal.price).toFixed(2)}</p>
                        <p class="mb-3">${bestDeal.description || ''}</p>
                    </div>
                    <div class="col-md-4 text-end">
                        <a href="${bestDeal.url}" class="btn btn-success btn-lg" target="_blank">
                            <i class="fas fa-shopping-cart me-2"></i>View Best Deal
                        </a>
                    </div>
                </div>
            `;
        } else {
            productsContent.innerHTML = '<div class="col-12"><p class="text-center">No products found</p></div>';
        }
    }
}); 
