(function () {
    const modal = document.getElementById('atrModal');
    const openBtn = document.getElementById('openModalBtn');
    const closeBtn = document.getElementById('closeModalBtn');
    const backdrop = modal.querySelector('[data-close]');

    const tabs = Array.from(document.querySelectorAll('.tab'));
    const panels = Array.from(document.querySelectorAll('.tabpanel'));

    const fileInput = document.getElementById('audioFile');
    const fileInfo = document.getElementById('fileInfo');
    const uploadAudio = document.getElementById('uploadAudio');

    // Phase 3 elements
    const recStartBtn = document.getElementById('recStartBtn');
    const recStopBtn = document.getElementById('recStopBtn');
    const recStatus = document.getElementById('recStatus');
    const recordedPreview = document.getElementById('recordedPreview');
    const textPrompt = document.getElementById('textPrompt');
    const submitInteractBtn = document.getElementById('submitInteractBtn');
    const submitStatus = document.getElementById('submitStatus');
    const responseText = document.getElementById('responseText');
    const responseAudio = document.getElementById('responseAudio');

    // Training elements
    const startRealTrainBtn = document.getElementById('startRealTrainBtn');
    const trainStatus = document.getElementById('trainStatus');
    const trainProgressBar = document.getElementById('trainProgressBar');
    const trainProgressText = document.getElementById('trainProgressText');
    const trainArtifact = document.getElementById('trainArtifact');
    let trainPollTimer = null;

    let mediaRecorder = null;
    let recordedChunks = [];
    let recordedBlob = null;

    function openModal() {
        modal.classList.add('modal--open');
        modal.setAttribute('aria-hidden', 'false');
    }

    function closeModal() {
        modal.classList.remove('modal--open');
        modal.setAttribute('aria-hidden', 'true');
    }

    openBtn.addEventListener('click', openModal);
    closeBtn.addEventListener('click', closeModal);
    backdrop.addEventListener('click', closeModal);
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') closeModal();
    });

    // Tabs
    tabs.forEach(function (tab) {
        tab.addEventListener('click', function () {
            const target = this.getAttribute('data-tab');
            tabs.forEach(function (t) { t.classList.remove('tab--active'); t.setAttribute('aria-selected', 'false'); });
            panels.forEach(function (p) { p.classList.remove('tabpanel--active'); });
            this.classList.add('tab--active');
            this.setAttribute('aria-selected', 'true');
            const panel = document.querySelector('.tabpanel[data-panel="' + target + '"]');
            if (panel) panel.classList.add('tabpanel--active');
        });
    });

    // Upload logic
    fileInput.addEventListener('change', function () {
        const file = this.files && this.files[0];
        
        // Reset upload button state when new file is selected
        uploadBtn.textContent = 'Upload';
        uploadBtn.style.backgroundColor = '';
        uploadBtn.disabled = false;
        
        if (!file) {
            fileInfo.textContent = 'No file chosen.';
            uploadAudio.removeAttribute('src');
            uploadAudio.load();
            return;
        }

        const valid = ['audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/mp4', 'audio/m4a', 'audio/x-m4a'];
        if (file.type && !valid.includes(file.type)) {
            fileInfo.textContent = 'Unsupported file type: ' + file.type;
            uploadAudio.removeAttribute('src');
            uploadAudio.load();
            return;
        }

        fileInfo.textContent = 'Selected: ' + file.name;
        const objectUrl = URL.createObjectURL(file);
        uploadAudio.src = objectUrl;
        uploadAudio.load();
    });

    // Upload button functionality with retry logic
    const uploadBtn = document.getElementById('uploadBtn');
    uploadBtn.addEventListener('click', async function () {
        const file = fileInput.files && fileInput.files[0];
        if (!file) {
            fileInfo.textContent = 'Please select a file first.';
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        // Retry logic
        let retryCount = 0;
        const maxRetries = 3;
        
        const attemptUpload = async () => {
            try {
                fileInfo.textContent = `Uploading... (Attempt ${retryCount + 1}/${maxRetries})`;
                uploadBtn.disabled = true;
                uploadBtn.textContent = 'Uploading...';
                
                // First check if backend is available
                if (retryCount === 0) {
                    try {
                        const healthCheck = await fetch('http://127.0.0.1:8000/progress', {
                            method: 'GET',
                            signal: AbortSignal.timeout(5000) // 5 second timeout for health check
                        });
                        if (!healthCheck.ok) {
                            throw new Error('Backend not responding');
                        }
                    } catch (healthError) {
                        throw new Error('Backend server not available. Please ensure the backend is running.');
                    }
                }
                
                const response = await fetch('http://127.0.0.1:8000/upload', {
                    method: 'POST',
                    body: formData,
                    // Add timeout and retry options
                    signal: AbortSignal.timeout(30000) // 30 second timeout
                });

                if (response.ok) {
                    const data = await response.json();
                    
                    if (data.status === 'already_trained') {
                        fileInfo.textContent = data.message || 'File already trained!';
                        uploadBtn.textContent = 'Already Trained ✓';
                        uploadBtn.style.backgroundColor = '#17a2b8'; // Blue color for already trained
                    } else {
                        fileInfo.textContent = data.message || 'Upload successful!';
                        uploadBtn.textContent = 'Uploaded ✓';
                        uploadBtn.style.backgroundColor = '#28a745'; // Green color for new upload
                    }
                    return true; // Success
                } else {
                    const errorData = await response.json().catch(() => ({ error: response.statusText }));
                    throw new Error(errorData.error || response.statusText);
                }
            } catch (error) {
                retryCount++;
                console.log(`Upload attempt ${retryCount} failed:`, error.message);
                
                if (retryCount < maxRetries) {
                    // Wait before retry (exponential backoff)
                    const delay = Math.pow(2, retryCount) * 1000; // 2s, 4s, 8s
                    fileInfo.textContent = `Upload failed, retrying in ${delay/1000}s... (${retryCount}/${maxRetries})`;
                    await new Promise(resolve => setTimeout(resolve, delay));
                    return attemptUpload(); // Retry
                } else {
                    // All retries failed
                    fileInfo.textContent = 'Upload failed after ' + maxRetries + ' attempts: ' + error.message;
                    uploadBtn.textContent = 'Upload Failed';
                    uploadBtn.style.backgroundColor = '#dc3545';
                    return false;
                }
            } finally {
                if (retryCount >= maxRetries) {
                    uploadBtn.disabled = false;
                    // Reset button after 5 seconds
                    setTimeout(() => {
                        uploadBtn.textContent = 'Upload';
                        uploadBtn.style.backgroundColor = '';
                    }, 5000);
                }
            }
        };

        // Start upload attempt
        await attemptUpload();
    });

    // Recording logic
    async function startRecording() {
        recordedChunks = [];
        recordedBlob = null;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = function (e) {
                if (e.data && e.data.size > 0) recordedChunks.push(e.data);
            };
            mediaRecorder.onstop = function () {
                recordedBlob = new Blob(recordedChunks, { type: 'audio/webm' });
                const url = URL.createObjectURL(recordedBlob);
                recordedPreview.src = url;
                recordedPreview.load();
                recStatus.textContent = 'Recorded';
            };
            mediaRecorder.start();
            recStatus.textContent = 'Recording...';
            recStartBtn.disabled = true;
            recStopBtn.disabled = false;
        } catch (err) {
            recStatus.textContent = 'Mic access denied or unavailable';
            console.error(err);
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            recStartBtn.disabled = false;
            recStopBtn.disabled = true;
        }
    }

    recStartBtn && recStartBtn.addEventListener('click', startRecording);
    recStopBtn && recStopBtn.addEventListener('click', stopRecording);

    // Submit to backend /interact
    submitInteractBtn && submitInteractBtn.addEventListener('click', async function () {
        submitStatus.textContent = 'Sending...';
        responseText.textContent = '';
        responseAudio.removeAttribute('src');
        responseAudio.load();

        const form = new FormData();
        const text = (textPrompt && textPrompt.value || '').trim();
        if (text) form.append('text', text);
        if (!text && recordedBlob) {
            // File name with extension expected by backend, use webm; backend accepts any for demo
            form.append('audio', recordedBlob, 'recording.webm');
        }

        if (!text && !recordedBlob) {
            submitStatus.textContent = 'Provide text or record audio first.';
            return;
        }

        try {
            const res = await fetch('http://127.0.0.1:8000/interact', {
                method: 'POST',
                body: form
            });
            if (!res.ok) throw new Error('HTTP ' + res.status);
            const data = await res.json();
            responseText.textContent = data.text || '';
            if (data.audio_b64_wav) {
                const wavBlob = b64ToBlob(data.audio_b64_wav, 'audio/wav');
                const url = URL.createObjectURL(wavBlob);
                responseAudio.src = url;
                responseAudio.load();
                // switch to Response tab
                const responseTab = document.querySelector('.tab[data-tab="response"]');
                responseTab && responseTab.click();
            }
            submitStatus.textContent = 'Done';
        } catch (e) {
            console.error(e);
            submitStatus.textContent = 'Failed: ' + e.message;
        }
    });

    function b64ToBlob(b64, contentType) {
        const byteChars = atob(b64);
        const byteNumbers = new Array(byteChars.length);
        for (let i = 0; i < byteChars.length; i++) byteNumbers[i] = byteChars.charCodeAt(i);
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: contentType });
    }

    // Training logic
    async function startTraining() {
        trainStatus.textContent = 'Starting AI training...';
        try {
            const res = await fetch('http://127.0.0.1:8000/train-real', { method: 'POST' });
            if (!res.ok) throw new Error('HTTP ' + res.status);
            const data = await res.json();
            trainStatus.textContent = data.status || 'AI training started';
            pollTraining();
        } catch (e) {
            trainStatus.textContent = 'Failed: ' + e.message;
        }
    }

    async function pollTraining() {
        if (trainPollTimer) clearInterval(trainPollTimer);
        trainPollTimer = setInterval(async function () {
            try {
                const res = await fetch('http://127.0.0.1:8000/progress');
                if (!res.ok) return;
                const data = await res.json();
                const p = Math.max(0, Math.min(100, Number(data.progress || 0)));
                trainProgressBar.style.width = p + '%';
                trainProgressText.textContent = p + '%';
                
                // Show more detailed status
                if (data.is_training) {
                    trainStatus.textContent = `Training... ${data.current_stage || ''}`;
                } else if (data.trained) {
                    trainStatus.textContent = 'Training Complete!';
                } else {
                    trainStatus.textContent = 'Ready to train';
                }
                
                trainArtifact.textContent = data.artifact || '—';
                if (!data.is_training) clearInterval(trainPollTimer);
            } catch (e) {
                // ignore intermittent errors
            }
        }, 300); // Reduced polling interval for more responsive updates
    }

    startRealTrainBtn && startRealTrainBtn.addEventListener('click', startTraining);
})();


