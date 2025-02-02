document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const resultOverlay = document.getElementById('result-overlay');
    const percentage = resultOverlay.querySelector('.percentage');
    const resetBtn = document.getElementById('reset-btn');
    const loading = document.getElementById('loading');
    const errorMessage = document.getElementById('error-message');

    // Handle drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    // Handle file drop
    dropZone.addEventListener('drop', handleDrop, false);

    // Handle browse button click
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    // Handle reset button click
    resetBtn.addEventListener('click', resetUI);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        dropZone.classList.add('dragover');
    }

    function unhighlight() {
        dropZone.classList.remove('dragover');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFileSelect(e) {
        const files = e.target.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        if (!file.type.startsWith('image/')) {
            showError('Please upload an image file');
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            uploadFile(file);
        };
        reader.readAsDataURL(file);
    }

    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        loading.classList.remove('hidden');
        errorMessage.classList.add('hidden');
        resultOverlay.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Failed to process image');
            }

            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            // Display result
            percentage.textContent = `${Math.round(data.probability)}% Mogging`;
            resultOverlay.classList.remove('hidden');
        } catch (error) {
            showError(error.message);
            resetUI();
        } finally {
            loading.classList.add('hidden');
        }
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
    }

    function resetUI() {
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        errorMessage.classList.add('hidden');
        fileInput.value = '';
        previewImage.src = '';
        resultOverlay.classList.add('hidden');
    }
});
