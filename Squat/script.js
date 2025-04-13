// the link to your model
const URL = "./model/";
let model, webcam, ctx, labelContainer, maxPredictions;

const waitingBox = document.querySelector(".waiting-box");
const countingBox = document.querySelector(".counting-box");
const tip = document.querySelector(".tip");
const canvas = document.getElementById("canvas");
const maxInput = document.querySelector(".max-input");
const countingWrapper = document.querySelector(".counting-wrapper");
const waitingWrapper = document.querySelector(".waiting-wrapper");
const starten = document.querySelector(".starten");

let status = "stand";
let count = "0"; // Amount of squats done
let time = 5; // Waiting time before cam turns on
let max = 5; // Amount of goal squat

function start() {
  max = maxInput.value;
    init();
}

function countdown() {
  return new Promise((resolve, reject) => {
    const timer = setInterval(() => {
      waitingBox.innerHTML = time;
      time--;
      if (time === -1) {
        clearInterval(timer);
        resolve(true);
      }
    }, 1000);
  });
}

async function init(max) {
  const modelURL = URL + "model.json";
  const metadataURL = URL + "metadata.json";

  // load the model and metadata
  // Refer to tmImage.loadFromFiles() in the API to support files from a file picker
  // Note: the pose library adds a tmPose object to your window (window.tmPose)
  model = await tmPose.load(modelURL, metadataURL);
  maxPredictions = model.getTotalClasses();

  // Convenience function to setup a webcam
  const size = 1800;
  const flip = true; // whether to flip the webcam
  webcam = new tmPose.Webcam(size, size, flip); // width, height, flip
  await webcam.setup(); // request access to the webcam

  await countdown();
  waitingWrapper.classList.add("hidden");
  countingWrapper.classList.remove("hidden");
  countingBox.classList.remove("hidden");
  canvas.parentElement.classList.remove("hidden");

  await webcam.play();
  window.requestAnimationFrame(loop);

  // append/get elements to the DOM
  // const canvas = document.getElementById("canvas");
  canvas.width = size;
  canvas.height = size;
  ctx = canvas.getContext("2d");
  labelContainer = document.getElementById("label-container");
  for (let i = 0; i < maxPredictions; i++) {
    // and class labels
    labelContainer.appendChild(document.createElement("div"));
  }
}

async function loop(timestamp) {
  webcam.update(); // update the webcam frame
  await predict();
  window.requestAnimationFrame(loop);
}

async function predict() {
  // Prediction #1: run input through posenet
  // estimatePose can take in an image, video or canvas html element
  const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
  // Prediction 2: run input through teachable machine classification model
  const prediction = await model.predict(posenetOutput);

  if (prediction[0].probability.toFixed(2) == 1.0) {
    if (status === "sit") {
      count++;
      if(count === 1){
        countingBox.innerHTML = `${count} squad`;
      } else {
        countingBox.innerHTML = `${count} squads`;
      }
      if (count >= max) {
        tip.innerHTML = "Goed gedaan!!";
        countingBox.classList.add("hidden")
        webcam.stop();
      }
    }
    var status = "stand";
  } else if (prediction[1].probability.toFixed(2) == 1.0) {
    var status = "sit";
  }

  for (let i = 0; i < maxPredictions; i++) {
    const classPrediction =
      prediction[i].className + ": " + prediction[i].probability.toFixed(2);
    labelContainer.childNodes[i].innerHTML = classPrediction;
  }

  // finally draw the poses
  drawPose(pose);
}

function drawPose(pose) {
  if (webcam.canvas) {
    ctx.drawImage(webcam.canvas, 0, 0);
    // draw the keypoints and skeleton
    if (pose) {
      const minPartConfidence = 0.5;
      tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
      tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
    }
  }
}
