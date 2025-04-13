// Configuratie voor de applicatie
const VIDEO_WIDTH = 640;  
const VIDEO_HEIGHT = 480; 
const STORAGE_KEY = 'focus-samples'; 
const DISTRACTION_THRESHOLD = 5000; 

// Globale applicatiestate die alle belangrijke variabelen bevat
const state = {
    classifier: knnClassifier.create(), // KNN-classifier instantie
    video: null, // Referentie naar het video element
    canvas: null, // Referentie naar het canvas element
    ctx: null, // 2D context van het canvas
    currentPose: null, // Huidige gedetecteerde pose landmarks
    distractionTimer: null, // Timer voor afleidingsdetectie
    audioContext: null, // Web Audio API context
    alarmSound: null, // Referentie naar actief alarmsignaal
    pose: null, // MediaPipe Pose instantie
    isFocusModeActive: false, // Status van focusmodus
    detectionInterval: null // Interval ID voor posedetectie
};

// Cache van belangrijke DOM-elementen voor snelle toegang
const elements = {
    feedback: document.getElementById('feedback'), // Feedback tekst element
    statusIcon: document.getElementById('status-icon'), // Status icoon container
    distractionAlert: document.getElementById('distraction-alert'), // Afleidingsmelding
    statusText: document.getElementById('status-text'), // Status tekst
    statusBox: document.getElementById('current-status'), // Status box container
    trainBtn: document.getElementById('btn-train') // Train/focus mode knop
};

/**
 * Initialiseert de applicatie wanneer de DOM geladen is
 * @async
 */
async function init() {
    try {
        // Initialiseer camera en canvas elementen
        state.video = document.getElementById('webcam');
        state.canvas = document.getElementById('canvas');
        state.ctx = state.canvas.getContext('2d');
        
        // Initialiseer MediaPipe Pose met CDN URL
        state.pose = new Pose({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
        });
        
        // Configureer pose detectie opties
        state.pose.setOptions({
            modelComplexity: 1, // Balans tussen snelheid en nauwkeurigheid
            smoothLandmarks: true, // Maakt bewegingen vloeiender
            enableSegmentation: false, // Geen achtergrondsegmentatie nodig
            smoothSegmentation: true,
            minDetectionConfidence: 0.5, // Minimum confidence voor detectie
            minTrackingConfidence: 0.5 // Minimum confidence voor tracking
        });
        
        // Stel callback in voor pose resultaten
        state.pose.onResults(onPoseResults);
        
        // Initialiseer event listeners
        setupEventListeners();
        
        // Start camera en wacht tot deze klaar is
        await setupCamera();
        
        // Start de posedetectie lus
        startDetectionLoop();
        
        // Geef feedback aan gebruiker
        elements.feedback.textContent = "Klaar voor gebruik!";
    } catch (err) {
        // Toon foutmelding als initialisatie mislukt
        elements.feedback.textContent = `Fout: ${err.message}`;
        console.error(err);
    }
}

/**
 * Start de posedetectie lus die regelmatig poses detecteert
 */
function startDetectionLoop() {
    // Stel interval in om elke 100ms pose te detecteren
    state.detectionInterval = setInterval(() => {
        // Controleer of video klaar is voor verwerking
        if (state.video.readyState >= 2) {
            state.pose.send({ image: state.video });
        }
    }, 100);
}

/**
 * Stopt de posedetectie lus
 */
function stopDetectionLoop() {
    if (state.detectionInterval) {
        clearInterval(state.detectionInterval);
        state.detectionInterval = null;
    }
}

/**
 * Stelt de camera in en verkrijg videostream
 * @async
 * @returns {Promise} Resolved wanneer camera klaar is
 */
async function setupCamera() {
    // Vraag toegang tot gebruikerscamera
    const stream = await navigator.mediaDevices.getUserMedia({
        video: {
            width: VIDEO_WIDTH,
            height: VIDEO_HEIGHT,
            facingMode: 'user' // Gebruik frontcamera
        },
        audio: false // Geen audio nodig
    });
    
    // Stel videostream in
    state.video.srcObject = stream;
    
    // Return promise die resolved wanneer video klaar is
    return new Promise((resolve) => {
        state.video.onloadedmetadata = () => resolve();
    });
}

/**
 * Callback voor posedetectie resultaten
 * @param {Object} results - Pose detectie resultaten
 */
function onPoseResults(results) {
    // Als geen pose gedetecteerd, reset status
    if (!results.poseLandmarks) {
        state.currentPose = null;
        updateStatusUI(null);
        return;
    }
    
    // Sla huidige pose landmarks op
    state.currentPose = results.poseLandmarks;
    
    // Pas canvas grootte aan naar video grootte
    state.canvas.width = state.video.videoWidth;
    state.canvas.height = state.video.videoHeight;
    
    // Teken pose op canvas
    state.ctx.save();
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    state.ctx.drawImage(results.image, 0, 0); // Teken videoframe
    
    // Teken landmarks als rode stippen
    state.ctx.fillStyle = '#FF0000';
    state.currentPose.forEach(landmark => {
        state.ctx.beginPath();
        state.ctx.arc(
            landmark.x * state.canvas.width, // X positie
            landmark.y * state.canvas.height, // Y positie
            5, // Straal van de stippen
            0, 2 * Math.PI // Volledige cirkel
        );
        state.ctx.fill();
    });
    state.ctx.restore();
    
    // Als focusmodus actief is, voorspel pose status
    if (state.isFocusModeActive) predictPose();
}

/**
 * Update de UI op basis van de huidige voorspelling
 * @param {Object|null} prediction - Voorspelling van de classifier
 */
function updateStatusUI(prediction) {
    // Als geen voorspelling, toon inactieve status
    if (!prediction) {
        elements.statusText.textContent = "Status: Niet actief";
        elements.statusBox.style.backgroundColor = '#f5f5f5';
        elements.statusIcon.innerHTML = '<i class="fas fa-circle" style="color:red"></i>';
        return;
    }
    
    // Toon verschillende UI voor geconcentreerd/afgeleid
    if (prediction.label === 'concentrated') {
        elements.statusText.textContent = "Status: Geconcentreerd";
        elements.statusBox.style.backgroundColor = '#2ecc71'; // Groene achtergrond
        elements.statusIcon.innerHTML = '<i class="fas fa-check-circle" style="color:white"></i>';
        clearDistractionTimer(); // Reset afleidingstimer
    } else {
        elements.statusText.textContent = "Status: Afgeleid";
        elements.statusBox.style.backgroundColor = '#e74c3c'; // Rode achtergrond
        elements.statusIcon.innerHTML = '<i class="fas fa-exclamation-triangle" style="color:white"></i>';
        startDistractionTimer(); // Start afleidingstimer
    }
}

/**
 * Voorspel of huidige pose geconcentreerd of afgeleid is
 * @async
 */
async function predictPose() {
    if (!state.currentPose) return;
    
    // Converteer pose landmarks naar feature array
    const features = state.currentPose.map(p => [p.x, p.y, p.visibility]).flat();
    const tensor = tf.tensor1d(features); // Maak tensor van features
    
    // Vraag voorspelling aan classifier
    const prediction = await state.classifier.predictClass(tensor);
    updateStatusUI(prediction); // Update UI met voorspelling
}

/**
 * Train het model met opgeslagen samples
 * @returns {boolean} Of het trainen succesvol was
 */
function trainModel() {
    // Laad samples van localStorage
    const savedData = localStorage.getItem(STORAGE_KEY);
    if (!savedData || JSON.parse(savedData).length === 0) {
        elements.feedback.textContent = "Geen trainingsdata. Ga naar trainingsmodus om samples toe te voegen.";
        return false;
    }
    
    const samples = JSON.parse(savedData);
    
    // Tel aantal samples per klasse
    const counts = samples.reduce((acc, sample) => {
        acc[sample.label] = (acc[sample.label] || 0) + 1;
        return acc;
    }, { concentrated: 0, distracted: 0 });
    
    // Controleer of er voldoende samples zijn
    if (counts.concentrated < 20 || counts.distracted < 20) {
        elements.feedback.textContent = `Minimaal 20 samples van elk type nodig (${counts.concentrated} geconcentreerd, ${counts.distracted} afgeleid)`;
        return false;
    }
    
    // Wis vorig model
    state.classifier.clearAllClasses();
    
    // Voeg alle samples toe aan classifier
    samples.forEach(sample => {
        const tensor = tf.tensor1d(sample.pose);
        state.classifier.addExample(tensor, sample.label);
    });
    
    // Geef feedback
    elements.feedback.textContent = `Model getraind met ${samples.length} samples`;
    return true;
}

/**
 * Start timer voor afleiding detectie
 */
function startDistractionTimer() {
    if (!state.distractionTimer) {
        // Stel timer in die na threshold alarm activeert
        state.distractionTimer = setTimeout(playDistractionAlert, DISTRACTION_THRESHOLD);
    }
}

/**
 * Stop en reset de afleidingstimer
 */
function clearDistractionTimer() {
    if (state.distractionTimer) {
        clearTimeout(state.distractionTimer);
        state.distractionTimer = null;
    }
}

/**
 * Activeer afleidingsalarm (visueel, audio en notificatie)
 */
function playDistractionAlert() {
    // Visuele feedback (popup)
    elements.distractionAlert.classList.add('show');
    setTimeout(() => elements.distractionAlert.classList.remove('show'), 5000);
    
    // Browser notificatie (als toegestaan)
    if (Notification.permission === 'granted') {
        new Notification('Focus Melding', {
            body: 'Je bent al 5 seconden afgeleid!'
        });
    }
    
    // Audio feedback (als audio context beschikbaar)
    if (state.audioContext) playAlertSound();
}

/**
 * Speel alarmgeluid af via Web Audio API
 */
function playAlertSound() {
    if (state.alarmSound) return; // Vermijd dubbele geluiden
    
    // Maak oscillator voor geluid
    const oscillator = state.audioContext.createOscillator();
    const gainNode = state.audioContext.createGain();
    
    // Configureer geluid
    oscillator.type = 'sine'; // Sinus golf
    oscillator.frequency.value = 800; // Toonhoogte (800Hz)
    gainNode.gain.value = 0.5; // Volume (50%)
    
    // Verbind audio nodes
    oscillator.connect(gainNode);
    gainNode.connect(state.audioContext.destination);
    oscillator.start(); // Start geluid
    
    // Sla referentie op om later te stoppen
    state.alarmSound = { oscillator, gainNode };
    
    // Stel fade-out in (2 seconden spelen + fade)
    setTimeout(() => {
        gainNode.gain.exponentialRampToValueAtTime(0.001, state.audioContext.currentTime + 0.5);
        setTimeout(() => {
            oscillator.stop(); // Stop geluid
            state.alarmSound = null; // Reset referentie
        }, 500);
    }, 2000);
}

/**
 * Stel event listeners in voor UI interacties
 */
function setupEventListeners() {
    // Toggle focus mode knop
    elements.trainBtn.addEventListener('click', () => {
        if (state.isFocusModeActive) {
            // Pauzeer focus mode
            state.isFocusModeActive = false;
            elements.trainBtn.innerHTML = '<span class="icon"><i class="fas fa-graduation-cap"></i></span> Start Focus Mode';
            elements.feedback.textContent = "Focus mode gepauzeerd";
            elements.statusIcon.innerHTML = '<i class="fas fa-circle" style="color:red"></i>';
            clearDistractionTimer();
            updateStatusUI(null);
        } else if (trainModel()) {
            // Start focus mode als training succesvol
            state.isFocusModeActive = true;
            elements.trainBtn.innerHTML = '<span class="icon"><i class="fas fa-pause"></i></span> Pauzeer';
            elements.feedback.textContent = "Focus mode actief!";
            elements.statusIcon.innerHTML = '<i class="fas fa-circle" style="color:green"></i>';
        }
    });

    // Initialiseer audio context bij eerste klik (vereist voor Chrome)
    document.addEventListener('click', initAudioContext, { once: true });
}

/**
 * Initialiseer audio context en vraag notificatie permissie
 */
function initAudioContext() {
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    if (Notification.permission !== 'granted') Notification.requestPermission();
}

// Start applicatie wanneer DOM geladen is
document.addEventListener('DOMContentLoaded', init);