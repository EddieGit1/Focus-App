// Configuratie
const VIDEO_WIDTH = 640;  
const VIDEO_HEIGHT = 480; 
const STORAGE_KEY = 'focus-samples'; 
const DISTRACTION_THRESHOLD = 5000; 

const state = {
    classifier: knnClassifier.create(), 
    video: null, 
    canvas: null, 
    ctx: null, 
    currentPose: null, 
    distractionTimer: null, 
    audioContext: null, 
    alarmSound: null, 
    pose: null, 
    isFocusModeActive: false,
    detectionInterval: null 
};

const elements = {
    feedback: document.getElementById('feedback'),
    statusIcon: document.getElementById('status-icon'),
    distractionAlert: document.getElementById('distraction-alert'),
    statusText: document.getElementById('status-text'),
    statusBox: document.getElementById('current-status'),
    trainBtn: document.getElementById('btn-train')
};

async function init() {
    try {
        state.video = document.getElementById('webcam');
        state.canvas = document.getElementById('canvas');
        state.ctx = state.canvas.getContext('2d');
        
        state.pose = new Pose({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
        });
        
        state.pose.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            enableSegmentation: false,
            smoothSegmentation: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        
        state.pose.onResults(onPoseResults);
        setupEventListeners();
        await setupCamera();
        startDetectionLoop();
        
        elements.feedback.textContent = "Klaar voor gebruik!";
    } catch (err) {
        elements.feedback.textContent = `Fout: ${err.message}`;
        console.error(err);
    }
}

function startDetectionLoop() {
    state.detectionInterval = setInterval(() => {
        if (state.video.readyState >= 2) {
            state.pose.send({ image: state.video });
        }
    }, 100);
}

function stopDetectionLoop() {
    if (state.detectionInterval) {
        clearInterval(state.detectionInterval);
        state.detectionInterval = null;
    }
}

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT, facingMode: 'user' },
        audio: false
    });
    state.video.srcObject = stream;
    
    return new Promise((resolve) => {
        state.video.onloadedmetadata = () => resolve();
    });
}

function onPoseResults(results) {
    if (!results.poseLandmarks) {
        state.currentPose = null;
        updateStatusUI(null);
        return;
    }
    
    state.currentPose = results.poseLandmarks;
    state.canvas.width = state.video.videoWidth;
    state.canvas.height = state.video.videoHeight;
    
    state.ctx.save();
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    state.ctx.drawImage(results.image, 0, 0);
    
    state.ctx.fillStyle = '#FF0000';
    state.currentPose.forEach(landmark => {
        state.ctx.beginPath();
        state.ctx.arc(landmark.x * state.canvas.width, landmark.y * state.canvas.height, 5, 0, 2 * Math.PI);
        state.ctx.fill();
    });
    state.ctx.restore();
    
    if (state.isFocusModeActive) predictPose();
}

function updateStatusUI(prediction) {
    if (!prediction) {
        elements.statusText.textContent = "Status: Niet actief";
        elements.statusBox.style.backgroundColor = '#f5f5f5';
        return;
    }
    
    if (prediction.label === 'concentrated') {
        elements.statusText.textContent = "Status: Geconcentreerd ✅";
        elements.statusBox.style.backgroundColor = '#2ecc71';
        elements.statusIcon.textContent = "✅";
        clearDistractionTimer();
    } else {
        elements.statusText.textContent = "Status: Afgeleid 🚨";
        elements.statusBox.style.backgroundColor = '#e74c3c';
        elements.statusIcon.textContent = "🚨";
        startDistractionTimer();
    }
}

async function predictPose() {
    if (!state.currentPose) return;
    
    const features = state.currentPose.map(p => [p.x, p.y, p.visibility]).flat();
    const tensor = tf.tensor1d(features);
    const prediction = await state.classifier.predictClass(tensor);
    updateStatusUI(prediction);
}

function trainModel() {
    const savedData = localStorage.getItem(STORAGE_KEY);
    if (!savedData || JSON.parse(savedData).length === 0) {
        elements.feedback.textContent = "Geen trainingsdata. Ga naar trainingsmodus om samples toe te voegen.";
        return false;
    }
    
    const samples = JSON.parse(savedData);
    const counts = samples.reduce((acc, sample) => {
        acc[sample.label] = (acc[sample.label] || 0) + 1;
        return acc;
    }, { concentrated: 0, distracted: 0 });
    
    if (counts.concentrated < 20 || counts.distracted < 20) {
        elements.feedback.textContent = `Minimaal 20 samples van elk type nodig (${counts.concentrated} geconcentreerd, ${counts.distracted} afgeleid)`;
        return false;
    }
    
    state.classifier.clearAllClasses();
    samples.forEach(sample => {
        const tensor = tf.tensor1d(sample.pose);
        state.classifier.addExample(tensor, sample.label);
    });
    
    elements.feedback.textContent = `Model getraind met ${samples.length} samples`;
    return true;
}

function startDistractionTimer() {
    if (!state.distractionTimer) {
        state.distractionTimer = setTimeout(playDistractionAlert, DISTRACTION_THRESHOLD);
    }
}

function clearDistractionTimer() {
    if (state.distractionTimer) {
        clearTimeout(state.distractionTimer);
        state.distractionTimer = null;
    }
}

function playDistractionAlert() {
    elements.distractionAlert.classList.add('show');
    setTimeout(() => elements.distractionAlert.classList.remove('show'), 5000);
    
    if (Notification.permission === 'granted') {
        new Notification('Focus Melding', { body: 'Je bent al 5 seconden afgeleid!' });
    }
    
    if (state.audioContext) playAlertSound();
}

function playAlertSound() {
    if (state.alarmSound) return;
    
    const oscillator = state.audioContext.createOscillator();
    const gainNode = state.audioContext.createGain();
    
    oscillator.type = 'sine';
    oscillator.frequency.value = 800;
    gainNode.gain.value = 0.5;
    
    oscillator.connect(gainNode);
    gainNode.connect(state.audioContext.destination);
    oscillator.start();
    state.alarmSound = { oscillator, gainNode };
    
    setTimeout(() => {
        gainNode.gain.exponentialRampToValueAtTime(0.001, state.audioContext.currentTime + 0.5);
        setTimeout(() => {
            oscillator.stop();
            state.alarmSound = null;
        }, 500);
    }, 2000);
}

function setupEventListeners() {
    elements.trainBtn.addEventListener('click', () => {
        if (state.isFocusModeActive) {
            state.isFocusModeActive = false;
            elements.trainBtn.innerHTML = '<span class="icon">🎓</span> Start Focus Mode';
            elements.feedback.textContent = "Focus mode gepauzeerd";
            elements.statusIcon.textContent = "🔴";
            clearDistractionTimer();
            updateStatusUI(null);
        } else if (trainModel()) {
            state.isFocusModeActive = true;
            elements.trainBtn.innerHTML = '<span class="icon">⏸️</span> Pauzeer';
            elements.feedback.textContent = "Focus mode actief!";
            elements.statusIcon.textContent = "🟢";
        }
    });

    document.addEventListener('click', initAudioContext, { once: true });
}

function initAudioContext() {
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    if (Notification.permission !== 'granted') Notification.requestPermission();
}

document.addEventListener('DOMContentLoaded', init);