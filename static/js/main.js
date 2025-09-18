/**
 * Main JavaScript file for eConsult Analysis Platform
 */

document.addEventListener('DOMContentLoaded', function() {
    // File upload handling
    setupFileUploads();
    
    // Form submission
    setupFormSubmission();
    
    // Initialize console logs
    initConsoleLog();
});

/**
 * Sets up file upload handling for PDF and CSV files
 */
function setupFileUploads() {
    // PDF file input handler
    const pdfInput = document.getElementById('pdf-file');
    const pdfCard = document.getElementById('pdf-upload-card');
    const pdfStatus = document.getElementById('pdf-status');
    
    if (pdfInput) {
        pdfInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                pdfCard.classList.add('active');
                pdfStatus.classList.remove('hidden');
                pdfStatus.textContent = `Selected: ${file.name}`;
                logToConsole(`PDF file selected: ${file.name} (${formatFileSize(file.size)})`, 'info');
            } else {
                pdfCard.classList.remove('active');
                pdfStatus.classList.add('hidden');
            }
        });
    }
    
    // CSV file input handler
    const csvInput = document.getElementById('csv-file');
    const csvCard = document.getElementById('csv-upload-card');
    const csvStatus = document.getElementById('csv-status');
    
    if (csvInput) {
        csvInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                csvCard.classList.add('active');
                csvStatus.classList.remove('hidden');
                csvStatus.textContent = `Selected: ${file.name}`;
                logToConsole(`CSV file selected: ${file.name} (${formatFileSize(file.size)})`, 'info');
                
                // Preview CSV columns if possible
                if (window.FileReader) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const firstLine = e.target.result.split('\n')[0];
                        logToConsole(`CSV columns detected: ${firstLine}`, 'info');
                    };
                    reader.readAsText(file);
                }
            } else {
                csvCard.classList.remove('active');
                csvStatus.classList.add('hidden');
            }
        });
    }
}

/**
 * Sets up form submission handling with loading state
 */
function setupFormSubmission() {
    const form = document.querySelector('form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultsSection = document.getElementById('results-section');
    const uploadSection = document.querySelector('.upload-grid').parentNode;
    const loadingEl = document.createElement('div');
    
    // Create loading element
    loadingEl.className = 'loading-container';
    loadingEl.innerHTML = `
        <div class="spinner"></div>
        <p>Analyzing stakeholder feedback. This may take a minute...</p>
    `;
    
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const pdfFile = document.getElementById('pdf-file').files[0];
            const csvFile = document.getElementById('csv-file').files[0];
            
            // Validate files
            if (!pdfFile || !csvFile) {
                logToConsole('Please select both PDF and CSV files', 'error');
                return;
            }
            
            // Create FormData
            const formData = new FormData();
            formData.append('pdf_file', pdfFile);
            formData.append('csv_file', csvFile);
            
            // Show loading state
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = 'Analyzing...';
            uploadSection.style.display = 'none';
            form.parentNode.insertBefore(loadingEl, form);
            
            logToConsole('Starting analysis process...', 'info');
            
            // Send request to server
            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server responded with status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // Handle successful response
                logToConsole('Analysis completed successfully!', 'info');
                
                // Remove loading state
                loadingEl.remove();
                
                if (data.success) {
                    // Hide upload form
                    form.style.display = 'none';
                    
                    // Show results and populate
                    displayResults(data);
                    resultsSection.style.display = 'block';
                    
                    // Scroll to results
                    resultsSection.scrollIntoView({behavior: 'smooth'});
                } else {
                    // Show error
                    logToConsole(`Error: ${data.error}`, 'error');
                    uploadSection.style.display = 'block';
                    analyzeBtn.disabled = false;
                    analyzeBtn.innerHTML = 'Analyze Feedback';
                }
            })
            .catch(error => {
                // Handle error
                console.error('Error:', error);
                logToConsole(`Error: ${error.message}`, 'error');
                
                // Remove loading state
                loadingEl.remove();
                uploadSection.style.display = 'block';
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = 'Analyze Feedback';
            });
        });
    }
}

/**
 * Displays analysis results in the UI
 */
function displayResults(data) {
    // Display wordcloud
    const wordcloudImg = document.getElementById('wordcloud-img');
    if (wordcloudImg) {
        wordcloudImg.src = data.wordcloud_path;
        logToConsole('Wordcloud generated', 'info');
    }
    
    // Display sentiment chart
    createSentimentChart(data.sentiment_counts);
    
    // Display comments
    displayComments(data.results);
    
    // Set up filter if exists
    const sentimentFilter = document.getElementById('sentiment-filter');
    if (sentimentFilter) {
        sentimentFilter.addEventListener('change', function() {
            const selectedSentiment = this.value;
            displayComments(data.results, selectedSentiment);
        });
    }
    
    // Log summary
    logToConsole(`Analysis complete: Found ${data.results.length} comments`, 'info');
    logToConsole(`Sentiment breakdown: ${JSON.stringify(data.sentiment_counts)}`, 'info');
}

/**
 * Creates a pie chart showing sentiment distribution
 */
function createSentimentChart(sentimentData) {
    const chartCanvas = document.getElementById('sentiment-chart');
    if (!chartCanvas) return;
    
    const labels = Object.keys(sentimentData);
    const data = Object.values(sentimentData);
    
    // Define colors for sentiment categories
    const colors = {
        'Supportive': 'rgba(46, 204, 113, 0.7)',
        'Critical': 'rgba(231, 76, 60, 0.7)',
        'Neutral': 'rgba(149, 165, 166, 0.7)',
        'Suggestive': 'rgba(52, 152, 219, 0.7)'
    };
    
    const backgroundColors = labels.map(label => colors[label] || 'rgba(52, 73, 94, 0.7)');
    
    const chart = new Chart(chartCanvas, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: backgroundColors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        font: {
                            size: 14
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw;
                            const total = context.chart.data.datasets[0].data.reduce((a, b) => a + b, 0);
                            const percentage = Math.round((value / total) * 100);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Displays comments in the UI, with optional filtering
 */
function displayComments(comments, sentimentFilter = 'all') {
    const commentsContainer = document.getElementById('comments-container');
    if (!commentsContainer) return;
    
    // Clear current content
    commentsContainer.innerHTML = '';
    
    // Filter comments if needed
    const filteredComments = sentimentFilter === 'all' 
        ? comments 
        : comments.filter(comment => comment.sentiment === sentimentFilter);
    
    // Check if we have comments to display
    if (filteredComments.length === 0) {
        commentsContainer.innerHTML = '<p class="text-center">No comments found for the selected filter.</p>';
        return;
    }
    
    // Create comment list
    const commentList = document.createElement('div');
    commentList.className = 'comment-list';
    
    // Add each comment
    filteredComments.forEach(comment => {
        const commentCard = document.createElement('div');
        commentCard.className = `comment-card ${comment.sentiment.toLowerCase()}`;
        
        commentCard.innerHTML = `
            <div class="comment-header">
                <div class="comment-id">Comment #${comment.comment_id}</div>
                <div class="sentiment-tag sentiment-${comment.sentiment.toLowerCase()}">${comment.sentiment}</div>
            </div>
            <div class="comment-text">${comment.comment_text}</div>
            <div class="comment-footer">
                <div class="comment-summary">${comment.summary}</div>
            </div>
        `;
        
        commentList.appendChild(commentCard);
    });
    
    commentsContainer.appendChild(commentList);
    logToConsole(`Displaying ${filteredComments.length} comments`, 'info');
}

/**
 * Console Log Functionality
 */
function initConsoleLog() {
    // Create initial log entry
    logToConsole('System ready to analyze feedback', 'info');
}

function logToConsole(message, type = 'info') {
    const consoleContainer = document.getElementById('log-container');
    if (!consoleContainer) return;
    
    const time = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    
    logEntry.innerHTML = `
        <span class="log-time">[${time}]</span>
        <span class="log-${type}">${message}</span>
    `;
    
    consoleContainer.appendChild(logEntry);
    consoleContainer.scrollTop = consoleContainer.scrollHeight;
}

/**
 * Utility Functions
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}