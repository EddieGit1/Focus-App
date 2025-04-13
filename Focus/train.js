const VIDEO_WIDTH = 640;  
const VIDEO_HEIGHT = 480; 
const STORAGE_KEY = 'focus-samples'; 
const MIN_SAMPLES = 3; 

const state = {
    classifier: knnClassifier.create(), 
    video: null, 
    canvas: null, 
    ctx: null, 
    samples: [], 
    currentPose: null, 
    pose: null, 
    detectionInterval: null 
};

const elements = {
    feedback: document.getElementById('feedback'),
    statusIcon: document.getElementById('status-icon'),
    counters: {
        concentrated: document.getElementById('count-concentrated'),
        distracted: document.getElementById('count-distracted')
    },
    accuracyValue: document.getElementById('accuracy-value'),
    matrixContent: document.getElementById('matrix-content')
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
        loadSamples();
        setupEventListeners();
        await setupCamera();
        startDetectionLoop();
        
        elements.feedback.textContent = "Klaar voor dataverzameling!";
    } catch (err) {
        elements.feedback.textContent = `Fout: ${err.message}`;
        console.error(err);
    }
}

function startDetectionLoop() {
    state.detectionInterval = setInterval(() => {
        if (state.video.readyState >= 2) state.pose.send({ image: state.video });
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
    return new Promise((resolve) => state.video.onloadedmetadata = resolve);
}

function onPoseResults(results) {
    if (!results.poseLandmarks) {
        state.currentPose = null;
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
}

function addSample(label) {
    if (!state.currentPose) return;
    
    const features = state.currentPose.map(p => [p.x, p.y, p.visibility]).flat();
    state.samples.push({ pose: features, label });
    saveSamples();
    updateSampleCount();
    elements.feedback.textContent = `Sample toegevoegd: ${label === 'concentrated' ? 'Geconcentreerd' : 'Afgeleid'}`;
}

function saveSamples() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.samples));
    elements.feedback.textContent = `Opgeslagen: ${state.samples.length} samples`;
}

function loadSamples() {
    const savedData = localStorage.getItem(STORAGE_KEY);
    if (savedData) {
        state.samples = JSON.parse(savedData);
        updateSampleCount();
        elements.feedback.textContent = `Geladen: ${state.samples.length} samples`;
    }
}

function exportToJSON() {
    if (state.samples.length === 0) {
        elements.feedback.textContent = "Geen samples om te exporteren!";
        return;
    }

    const dataStr = JSON.stringify(state.samples, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportName = `Focus-samples_${new Date().toISOString().slice(0,10)}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportName);
    linkElement.click();
    
    elements.feedback.textContent = `Exported ${state.samples.length} samples naar JSON`;
}

function updateSampleCount() {
    const counts = state.samples.reduce((acc, sample) => {
        acc[sample.label] = (acc[sample.label] || 0) + 1;
        return acc;
    }, { concentrated: 0, distracted: 0 });
    
    elements.counters.concentrated.textContent = counts.concentrated;
    elements.counters.distracted.textContent = counts.distracted;
}

function trainModel() {
    const counts = state.samples.reduce((acc, sample) => {
        acc[sample.label] = (acc[sample.label] || 0) + 1;
        return acc;
    }, { concentrated: 0, distracted: 0 });
    
    if (counts.concentrated < MIN_SAMPLES || counts.distracted < MIN_SAMPLES) {
        elements.feedback.textContent = `Minimaal ${MIN_SAMPLES} samples van elk type nodig (${counts.concentrated} geconcentreerd, ${counts.distracted} afgeleid)`;
        return false;
    }
    
    state.classifier.clearAllClasses();
    state.samples.forEach(sample => {
        const tensor = tf.tensor1d(sample.pose);
        state.classifier.addExample(tensor, sample.label);
    });
    
    elements.feedback.textContent = `Model getraind met ${state.samples.length} samples`;
    elements.statusIcon.textContent = "🟢";
    return true;
}

async function calculateAccuracy() {
    if (state.samples.length < 6) {
        elements.feedback.textContent = "Minimaal 6 samples nodig voor nauwkeurigheidstest";
        return;
    }

    const shuffled = [...state.samples].sort(() => 0.5 - Math.random());
    const splitIdx = Math.floor(shuffled.length * 0.8);
    const trainData = shuffled.slice(0, splitIdx);
    const testData = shuffled.slice(splitIdx);

    const testClassifier = knnClassifier.create();
    
    try {
        for (const sample of trainData) {
            const tensor = tf.tensor1d(sample.pose);
            testClassifier.addExample(tensor, sample.label);
            tensor.dispose();
        }

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
            
            if (prediction.label === sample.label) {
                correct++;
                if (sample.label === 'concentrated') confusionMatrix.trueConcentrated++;
                else confusionMatrix.trueDistracted++;
            } else {
                if (sample.label === 'concentrated') confusionMatrix.falseDistracted++;
                else confusionMatrix.falseConcentrated++;
            }
        }

        const accuracy = (correct / testData.length) * 100;
        elements.accuracyValue.textContent = `${accuracy.toFixed(1)}%`;
        
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
        testClassifier.dispose();
    }
}

function setupEventListeners() {
    document.getElementById('btn-concentrated').addEventListener('click', () => addSample('concentrated'));
    document.getElementById('btn-distracted').addEventListener('click', () => addSample('distracted'));
    document.getElementById('btn-save').addEventListener('click', saveSamples);
    document.getElementById('btn-export').addEventListener('click', exportToJSON);
    document.getElementById('btn-train').addEventListener('click', trainModel);
    document.getElementById('btn-test-accuracy').addEventListener('click', calculateAccuracy);
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

document.addEventListener('DOMContentLoaded', init);