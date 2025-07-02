document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadProgress = document.getElementById('uploadProgress');
    const fileInput = document.getElementById('file');

    // File validation
    fileInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            // Check file size (50MB limit)
            const maxSize = 50 * 1024 * 1024; // 50MB in bytes
            if (file.size > maxSize) {
                alert('File size exceeds 50MB limit. Please choose a smaller file.');
                this.value = '';
                return;
            }

            // Check file type
            if (!file.name.toLowerCase().endsWith('.csv')) {
                alert('Please select a CSV file.');
                this.value = '';
                return;
            }

            console.log(`File selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
        }
    });

    // Form submission
    uploadForm.addEventListener('submit', function(e) {
        const file = fileInput.files[0];
        
        if (!file) {
            e.preventDefault();
            alert('Please select a file to upload.');
            return;
        }

        // Show progress and disable button
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Uploading...';
        uploadProgress.style.display = 'block';

        // Note: The form will submit normally, we're just providing visual feedback
        console.log('Uploading file:', file.name);
    });

    // Drag and drop functionality
    const cardBody = document.querySelector('.card-body');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        cardBody.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        cardBody.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        cardBody.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        cardBody.classList.add('bg-light');
    }

    function unhighlight(e) {
        cardBody.classList.remove('bg-light');
    }

    cardBody.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            fileInput.files = files;
            
            // Trigger the change event
            const event = new Event('change', { bubbles: true });
            fileInput.dispatchEvent(event);
        }
    }
});
