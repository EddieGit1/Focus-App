// Configuratie voor trainingsmodus
const VIDEO_WIDTH = 640;  
const VIDEO_HEIGHT = 480;
const STORAGE_KEY = 'focus-samples'; 
const MIN_SAMPLES = 3;

// Hoofd state object voor trainingsmodus
const state = {
    classifier: knnClassifier.create(), // KNN-classifier instance
    video: null, // Video element referentie
    canvas: null, // Canvas element referentie
    ctx: null, // Canvas 2D context
    samples: [], // Array van alle trainingssamples
    currentPose: null, // Huidige gedetecteerde pose
    pose: null, // MediaPipe Pose instance
    detectionInterval: null // Interval ID voor pose detectie
};

// DOM element referenties voor trainingsmodus
const elements = {
    feedback: document.getElementById('feedback'), // Feedback tekst
    statusIcon: document.getElementById('status-icon'), // Status icoon
    counters: {
        concentrated: document.getElementById('count-concentrated'), // Teller geconcentreerd
        distracted: document.getElementById('count-distracted') // Teller afgeleid
    },
    accuracyValue: document.getElementById('accuracy-value'), // Nauwkeurigheid waarde
    matrixContent: document.getElementById('matrix-content') // Confusion matrix inhoud
};

// Initialiseer trainingsmodus
async function init() {
    try {
        // Verkrijg DOM referenties
        state.video = document.getElementById('webcam');
        state.canvas = document.getElementById('canvas');
        state.ctx = state.canvas.getContext('2d');
        
        // Initialiseer MediaPipe Pose
        state.pose = new Pose({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
        });
        
        // Configureer pose detectie
        state.pose.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            enableSegmentation: false,
            smoothSegmentation: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        
        // Stel callback in voor pose resultaten
        state.pose.onResults(onPoseResults);
        
        // Laad opgeslagen samples
        loadSamples();
        
        // Stel event listeners in
        setupEventListeners();
        
        // Start camera
        await setupCamera();
        
        // Start detectie lus
        startDetectionLoop();
        
        // Geef feedback
        elements.feedback.textContent = "Klaar voor dataverzameling!";
    } catch (err) {
        elements.feedback.textContent = `Fout: ${err.message}`;
        console.error(err);
    }
}

// Start pose detectie lus
function startDetectionLoop() {
    state.detectionInterval = setInterval(() => {
        if (state.video.readyState >= 2) {
            state.pose.send({ image: state.video });
        }
    }, 100);
}

// Stop pose detectie lus
function stopDetectionLoop() {
    if (state.detectionInterval) {
        clearInterval(state.detectionInterval);
        state.detectionInterval = null;
    }
}

// Stel camera in
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT, facingMode: 'user' },
        audio: false
    });
    state.video.srcObject = stream;
    return new Promise((resolve) => state.video.onloadedmetadata = resolve);
}

// Verwerk pose detectie resultaten
function onPoseResults(results) {
    if (!results.poseLandmarks) {
        state.currentPose = null;
        return;
    }
    
    // Update huidige pose en canvas
    state.currentPose = results.poseLandmarks;
    state.canvas.width = state.video.videoWidth;
    state.canvas.height = state.video.videoHeight;
    
    // Teken pose op canvas
    state.ctx.save();
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    state.ctx.drawImage(results.image, 0, 0);
    
    // Teken landmarks
    state.ctx.fillStyle = '#FF0000';
    state.currentPose.forEach(landmark => {
        state.ctx.beginPath();
        state.ctx.arc(landmark.x * state.canvas.width, landmark.y * state.canvas.height, 5, 0, 2 * Math.PI);
        state.ctx.fill();
    });
    state.ctx.restore();
}

// Voeg sample toe aan trainingsset
function addSample(label) {
    if (!state.currentPose) return;
    
    // Extraheer features van huidige pose
    const features = state.currentPose.map(p => [p.x, p.y, p.visibility]).flat();
    state.samples.push({ pose: features, label });
    saveSamples(); // Sla op
    updateSampleCount(); // Update UI tellers
    elements.feedback.textContent = `Sample toegevoegd: ${label === 'concentrated' ? 'Geconcentreerd' : 'Afgeleid'}`;
}

// Sla samples op in localStorage
function saveSamples() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.samples));
    elements.feedback.textContent = `Opgeslagen: ${state.samples.length} samples`;
}

// Laad samples van localStorage
function loadSamples() {
    const savedData = localStorage.getItem(STORAGE_KEY);
    if (savedData) {
        state.samples = JSON.parse(savedData);
        updateSampleCount();
        elements.feedback.textContent = `Geladen: ${state.samples.length} samples`;
    }
}

// Exporteer samples naar JSON bestand
function exportToJSON() {
    if (state.samples.length === 0) {
        elements.feedback.textContent = "Geen samples om te exporteren!";
        return;
    }

    // Maak downloadbare JSON
    const dataStr = JSON.stringify(state.samples, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportName = `Focus-samples_${new Date().toISOString().slice(0,10)}.json`;
    
    // Maak tijdelijke download link
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportName);
    linkElement.click();
    
    elements.feedback.textContent = `Exported ${state.samples.length} samples naar JSON`;
}

// Update sample tellers in UI
function updateSampleCount() {
    const counts = state.samples.reduce((acc, sample) => {
        acc[sample.label] = (acc[sample.label] || 0) + 1;
        return acc;
    }, { concentrated: 0, distracted: 0 });
    
    elements.counters.concentrated.textContent = counts.concentrated;
    elements.counters.distracted.textContent = counts.distracted;
}

// Train het classificatiemodel
function trainModel() {
    // Tel samples per klasse
    const counts = state.samples.reduce((acc, sample) => {
        acc[sample.label] = (acc[sample.label] || 0) + 1;
        return acc;
    }, { concentrated: 0, distracted: 0 });
    
    // Controleer minimum samples
    if (counts.concentrated < MIN_SAMPLES || counts.distracted < MIN_SAMPLES) {
        elements.feedback.textContent = `Minimaal ${MIN_SAMPLES} samples van elk type nodig (${counts.concentrated} geconcentreerd, ${counts.distracted} afgeleid)`;
        return false;
    }
    
    // Wis vorig model
    state.classifier.clearAllClasses();
    
    // Train met alle samples
    state.samples.forEach(sample => {
        const tensor = tf.tensor1d(sample.pose);
        state.classifier.addExample(tensor, sample.label);
    });
    
    elements.feedback.textContent = `Model getraind met ${state.samples.length} samples`;
    elements.statusIcon.textContent = "🟢";
    return true;
}

// Bereken modelnauwkeurigheid met 80/20 train/test split
async function calculateAccuracy() {
    if (state.samples.length < 6) {
        elements.feedback.textContent = "Minimaal 6 samples nodig voor nauwkeurigheidstest";
        return;
    }

    // Split data in training en test sets
    const shuffled = [...state.samples].sort(() => 0.5 - Math.random());
    const splitIdx = Math.floor(shuffled.length * 0.8);
    const trainData = shuffled.slice(0, splitIdx);
    const testData = shuffled.slice(splitIdx);

    // Maak tijdelijke classifier voor testen
    const testClassifier = knnClassifier.create();
    
    try {
        // Train tijdelijke classifier
        for (const sample of trainData) {
            const tensor = tf.tensor1d(sample.pose);
            testClassifier.addExample(tensor, sample.label);
            tensor.dispose();
        }

        // Test nauwkeurigheid
        let correct = 0;
        const confusionMatrix = {
            trueConcentrated: 0,
            falseConcentrated: 0,
            trueDistracted: 0,
            falseDistracted: 0
        };

        for (const sample of testData) {
            const tensor = tf.tensor1d(sample.pose);
            const prediction = await testClassifier.predictClass(tensor);
            tensor.dispose();
            
            // Update confusion matrix
            if (prediction.label === sample.label) {
                correct++;
                if (sample.label === 'concentrated') confusionMatrix.trueConcentrated++;
                else confusionMatrix.trueDistracted++;
            } else {
                if (sample.label === 'concentrated') confusionMatrix.falseDistracted++;
                else confusionMatrix.falseConcentrated++;
            }
        }

        // Bereken en toon nauwkeurigheid
        const accuracy = (correct / testData.length) * 100;
        elements.accuracyValue.textContent = `${accuracy.toFixed(1)}%`;
        
        // Update confusion matrix UI
        elements.matrixContent.innerHTML = `
            <table>
                <tr>
                    <th></th>
                    <th>Voorspeld Concentrated</th>
                    <th>Voorspeld Distracted</th>
                </tr>
                <tr>
                    <td>Werkelijk Concentrated</td>
                    <td>${confusionMatrix.trueConcentrated}</td>
                    <td>${confusionMatrix.falseDistracted}</td>
                </tr>
                <tr>
                    <td>Werkelijk Distracted</td>
                    <td>${confusionMatrix.falseConcentrated}</td>
                    <td>${confusionMatrix.trueDistracted}</td>
                </tr>
            </table>
        `;

        elements.feedback.textContent = `Nauwkeurigheid: ${accuracy.toFixed(1)}% (${correct} van ${testData.length} correct)`;
    } catch (error) {
        console.error("Fout tijdens nauwkeurigheidstest:", error);
        elements.feedback.textContent = "Fout tijdens nauwkeurigheidstest";
    } finally {
        testClassifier.dispose(); // Opruimen
    }
}

// Stel event listeners in
function setupEventListeners() {
    // Sample toevoegen knoppen
    document.getElementById('btn-concentrated').addEventListener('click', () => addSample('concentrated'));
    document.getElementById('btn-distracted').addEventListener('click', () => addSample('distracted'));
    
    // Opslaan en export knoppen
    document.getElementById('btn-save').addEventListener('click', saveSamples);
    document.getElementById('btn-export').addEventListener('click', exportToJSON);
    
    // Train en test knoppen
    document.getElementById('btn-train').addEventListener('click', trainModel);
    document.getElementById('btn-test-accuracy').addEventListener('click', calculateAccuracy);
    
    // Reset knop
    document.getElementById('btn-reset').addEventListener('click', () => {
        if (confirm('Weet je zeker dat je alle data wilt resetten?')) {
            localStorage.removeItem(STORAGE_KEY);
            state.samples = [];
            state.classifier.clearAllClasses();
            updateSampleCount();
            elements.feedback.textContent = "Data gereset";
            elements.statusIcon.textContent = "🔴";
            elements.accuracyValue.textContent = "0%";
            elements.matrixContent.innerHTML = "Nog niet beschikbaar";
        }
    });
}

// Start trainingsmodus wanneer DOM geladen is
document.addEventListener('DOMContentLoaded', init);