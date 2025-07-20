// Update current time
function updateTime() {
  const now = new Date();
  document.getElementById("current-time").textContent = now
    .toTimeString()
    .substring(0, 8);
}
setInterval(updateTime, 1000);
updateTime();
// Audio player for ElevenLabs
let audioPlayer = null;
// DOM elements
const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("file-input");
const selectFileBtn = document.getElementById("select-file-btn");
const previewContainer = document.getElementById("preview-container");
const previewContainer1 = document.getElementById("preview-container1");

const imagePreview = document.getElementById("image-preview");
const imagePreview1 = document.getElementById("image-preview1");
const resultheadline = document.getElementById("resultheadline");
const processBtn = document.getElementById("process-btn");
const clearBtn = document.getElementById("clear-btn");
const extractionResult = document.getElementById("extraction-result");
const progressContainer = document.getElementById("progress-container");
const progressBar = document.getElementById("progress-bar");
const loader = document.getElementById("loader");
const statusMessage = document.getElementById("status-message");
const statusText = document.getElementById("status-text");
const playBtn = document.getElementById("play-btn");
const pauseBtn = document.getElementById("pause-btn");
const stopBtn = document.getElementById("stop-btn");
const voiceSelect = document.getElementById("voice-select");
const rateInput = document.getElementById("rate");
const rateValue = document.getElementById("rate-value");
// ElevenLabs configuration
const ELEVENLABS_API_KEY =
  "sk_03da7e86b9f9d32ba934fde325af04052a342d9b1e219071"; // Replace with your actual API key
const ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech";
// Update status
function updateStatus(message, isError = false) {
  statusText.textContent = message;
  if (isError) {
    statusText.style.color = "var(--alert-red)";
  } else {
    statusText.style.color = "var(--military-green)";
  }
}

// Drag and Drop handlers
["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
  dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

["dragenter", "dragover"].forEach((eventName) => {
  dropArea.addEventListener(eventName, highlight, false);
});

["dragleave", "drop"].forEach((eventName) => {
  dropArea.addEventListener(eventName, unhighlight, false);
});

function highlight() {
  dropArea.classList.add("active");
}

function unhighlight() {
  dropArea.classList.remove("active");
}

// Handle file drop
dropArea.addEventListener("drop", handleDrop, false);

function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;
  handleFiles(files);
}

// File selection button
selectFileBtn.addEventListener("click", () => {
  fileInput.click();
});

fileInput.addEventListener("change", function () {
  handleFiles(this.files);
});

function handleFiles(files) {
  if (files.length > 0) {
    const file = files[0];
    if (file.type.startsWith("image/")) {
      displayPreview(file);
      updateStatus("DOCUMENT RECEIVED");
    } else {
      updateStatus("ERROR: NOT AN IMAGE FILE", true);
      alert(
        "MISSION ABORT: File is not an image. Please select an image file."
      );
    }
  }
}

// Display image preview
function displayPreview(file) {
  previewContainer.style.display = "block";
  const reader = new FileReader();

  reader.onload = function (e) {
    imagePreview1.src = e.target.result;
    imagePreview.src = e.target.result;
  };

  reader.readAsDataURL(file);
}

// Process button handler - Modified to use POST request
processBtn.addEventListener("click", function () {
  if (imagePreview1.src) {
    sendImageToServer(imagePreview1.src);
  } else {
    alert("No document selected. Please upload an image first.");
  }
});

// New function to send image to server via POST
async function sendImageToServer(imageDataUrl) {
  progressContainer.style.display = "block";
  loader.style.display = "block";
  extractionResult.textContent = "INITIATING EXTRACTION PROTOCOL... STANDBY...";
  statusMessage.textContent = "INTELLIGENCE ACQUISITION IN PROGRESS";
  statusMessage.className = "status-msg status-success";
  updateStatus("PROCESSING DOCUMENT");

  try {
    // Convert data URL to blob
    const blob = await fetch(imageDataUrl).then((res) => res.blob());

    // Create FormData
    const formData = new FormData();
    formData.append("image", blob, "image.jpg");

    // Setup fetch options for POST request
    const fetchOptions = {
      method: "POST",
      body: formData,
    };

    // Setup streaming
    const response = await fetch("http://127.0.0.1:5000/upload", fetchOptions);

    if (!response.ok) {
      throw new Error(`Server responded with status: ${response.status}`);
    }

    // Handle streaming response
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let streamedText = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      // Decode and append to streamedText
      const chunk = decoder.decode(value, { stream: true });
      streamedText += chunk;

      // Update the extraction result with the streamed text
      extractionResult.textContent = streamedText;

      // Update progress bar - just for visual feedback
      const progressPercentage = Math.min(
        90,
        (streamedText.length / 200) * 100
      );
      progressBar.style.width = `${progressPercentage}%`;
    }

    // Once streaming is complete, update status
    updateStatus("EXTRACTION COMPLETE");
    statusMessage.textContent = "INTELLIGENCE ACQUISITION COMPLETE";
    statusMessage.className = "status-msg status-success";

    // Enable voice controls
    playBtn.disabled = false;
    pauseBtn.disabled = true;
    stopBtn.disabled = true;

    // Wait 5 seconds before fetching files
    setTimeout(() => {
      fetchFiles();
    }, 5000);
  } catch (error) {
    console.error("Error in image upload:", error);
    extractionResult.textContent = `ERROR: EXTRACTION FAILURE - ${error.message}`;
    progressContainer.style.display = "none";
    loader.style.display = "none";
    statusMessage.textContent = "MISSION ABORTED: EXTRACTION FAILED";
    statusMessage.className = "status-msg status-error";
    updateStatus("EXTRACTION FAILED", true);
  }
}

// Clear button handler
clearBtn.addEventListener("click", function () {
  previewContainer.style.display = "none";
  imagePreview1.src = "";
  imagePreview.src = "";
  fileInput.value = "";
  extractionResult.textContent =
    "NO INTELLIGENCE RECEIVED. AWAITING DATA UPLOAD.";
  progressContainer.style.display = "none";
  progressBar.style.width = "0%";
  playBtn.disabled = true;
  pauseBtn.disabled = true;
  stopBtn.disabled = true;
  statusMessage.textContent = "";
  statusMessage.className = "status-msg";
  updateStatus("AWAITING MISSION");
});

// Speech synthesis setup

// Populate voices dropdown
// Populate ElevenLabs voices dropdown
async function populateVoiceList() {
  try {
    voiceSelect.innerHTML = "<option value=''>Loading voices...</option>";
    const response = await fetch("https://api.elevenlabs.io/v1/voices", {
      headers: {
        "xi-api-key": ELEVENLABS_API_KEY,
      },
    });

    if (!response.ok) throw new Error("Failed to fetch voices");

    const data = await response.json();
    voiceSelect.innerHTML = "";

    data.voices.forEach((voice) => {
      const option = document.createElement("option");
      option.textContent = voice.name;
      option.value = voice.voice_id;
      voiceSelect.appendChild(option);
    });

    // Select a default voice if available
    if (data.voices.length > 0) {
      // Maybe select a deeper voice that sounds military
      const defaultVoice =
        data.voices.find(
          (voice) =>
            voice.name.toLowerCase().includes("deep") ||
            voice.name.toLowerCase().includes("male")
        ) || data.voices[0];

      voiceSelect.value = defaultVoice.voice_id;
    }
  } catch (error) {
    voiceSelect.innerHTML = "<option value=''>Default Voice</option>";
    console.error("Error loading ElevenLabs voices:", error);
    updateStatus("VOICE CATALOG UNAVAILABLE", true);
  }
}

// Initialize voice list

populateVoiceList();
// Fallback for browsers that don't support onvoiceschanged

// Handle speech rate change
rateInput.addEventListener("input", () => {
  rateValue.textContent = parseFloat(rateInput.value).toFixed(1);
});

// Text-to-speech function
// ElevenLabs text-to-speech function
async function speakText() {
  if (audioPlayer && !audioPlayer.paused) {
    audioPlayer.pause();
    audioPlayer = null;
  }

  // Update UI state
  playBtn.disabled = true;
  pauseBtn.disabled = true;
  stopBtn.disabled = true;
  statusMessage.textContent = "GENERATING AUDIO TRANSMISSION";
  statusMessage.className = "status-msg status-success";
  updateStatus("CONTACTING ELEVENLABS");

  try {
    const selectedVoiceId = voiceSelect.value;
    const textToSpeak = extractionResult.textContent;
    const stability = 0.5;
    const similarity_boost = 0.5;

    const response = await fetch(`${ELEVENLABS_API_URL}/${selectedVoiceId}`, {
      method: "POST",
      headers: {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: textToSpeak,
        model_id: "eleven_monolingual_v1",
        voice_settings: {
          stability,
          similarity_boost,
          speed: parseFloat(rateInput.value),
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`ElevenLabs API error: ${response.status}`);
    }

    // Get audio blob from response
    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);

    // Create audio player
    audioPlayer = new Audio(audioUrl);

    // Set up event handlers
    audioPlayer.onplay = () => {
      playBtn.disabled = true;
      pauseBtn.disabled = false;
      stopBtn.disabled = false;
      statusMessage.textContent = "BROADCAST IN PROGRESS";
      statusMessage.className = "status-msg status-success";
      updateStatus("BROADCASTING");
    };

    audioPlayer.onended = () => {
      playBtn.disabled = false;
      pauseBtn.disabled = true;
      stopBtn.disabled = true;
      statusMessage.textContent = "BROADCAST COMPLETE";
      statusMessage.className = "status-msg status-success";
      updateStatus("BROADCAST COMPLETE");
    };

    audioPlayer.onerror = (error) => {
      console.error("Audio playback error:", error);
      playBtn.disabled = false;
      pauseBtn.disabled = true;
      stopBtn.disabled = true;
      statusMessage.textContent = "BROADCAST ERROR";
      statusMessage.className = "status-msg status-error";
      updateStatus("BROADCAST ERROR", true);
    };

    // Start playing
    audioPlayer.play();
  } catch (error) {
    console.error("ElevenLabs API error:", error);
    playBtn.disabled = false;
    pauseBtn.disabled = true;
    stopBtn.disabled = true;
    statusMessage.textContent = `TRANSMISSION ERROR: ${error.message}`;
    statusMessage.className = "status-msg status-error";
    updateStatus("API ERROR", true);
  }
}
async function fetchFiles() {
  try {
    // Display loading status
    statusText.textContent = "FETCHING DATA";

    // Fetch Image
    const imageResponse = await fetch("http://localhost:5000/files/image");
    if (!imageResponse.ok) throw new Error("Image fetch failed");
    const imageBlob = await imageResponse.blob();
    const imageURL = URL.createObjectURL(imageBlob);
    imagePreview.src = imageURL;
    document.getElementById("preview-container").style.display = "block";

    // Fetch JSON data
    const jsonResponse = await fetch("http://127.0.0.1:5000/files/json");
    if (!jsonResponse.ok) {
      throw new Error("Failed to fetch JSON data.");
    }
    const jsonData = await jsonResponse.json();
    console.log(jsonData);

    // Create structured HTML for the extraction result
    let resultHTML = document.getElementById("");

    // Process each item in the JSON
    Object.keys(jsonData).forEach((key) => {
      const item = jsonData[key];
      if (item.headlines) {
        resultHTML += `<div><h3 class="intel-headline">${item.headlines}</h3>`;
      }

      if (item.subheadlines) {
        resultHTML += `<h4 class="intel-subheadline">${item.subheadlines}</h4>`;
      }

      if (item.content) {
        resultHTML += `<p class="intel-content">${item.content}</p>`;
      }

      resultHTML += '<hr class="intel-divider">';
    });

    resultHTML += "</div>";

    // Display the formatted JSON Data
    extractionResult.innerHTML = resultHTML;

    // Enable audio controls
    playBtn.disabled = false;
    pauseBtn.disabled = true;
    stopBtn.disabled = true;

    // Success message
    statusText.textContent = "MISSION SUCCESSFUL";
    document.getElementById("current-status").classList.remove("status-error");
    document.getElementById("current-status").classList.add("status-success");
  } catch (error) {
    console.error("Error fetching files:", error);

    extractionResult.innerHTML = `<p class="error-message">ERROR: Failed to retrieve intelligence data.<br>Details: ${error.message}</p>`;
    statusText.textContent = "MISSION FAILED";
    document
      .getElementById("current-status")
      .classList.remove("status-success");
    document.getElementById("current-status").classList.add("status-error");
  } finally {
    console.log("Fetched");
  }
}

// Voice control button handlers
playBtn.addEventListener("click", speakText);

pauseBtn.addEventListener("click", function () {
  if (audioPlayer) {
    if (audioPlayer.paused) {
      audioPlayer.play();
      pauseBtn.textContent = "PAUSE TRANSMISSION";
      statusMessage.textContent = "BROADCAST RESUMED";
    } else {
      audioPlayer.pause();
      pauseBtn.textContent = "RESUME TRANSMISSION";
      statusMessage.textContent = "BROADCAST PAUSED";
    }
  }
});

stopBtn.addEventListener("click", function () {
  if (audioPlayer) {
    audioPlayer.pause();
    audioPlayer.currentTime = 0;
    audioPlayer = null;
    playBtn.disabled = false;
    pauseBtn.disabled = true;
    stopBtn.disabled = true;
    pauseBtn.textContent = "PAUSE TRANSMISSION";
    statusMessage.textContent = "BROADCAST TERMINATED";
    statusMessage.className = "status-msg status-error";
    updateStatus("BROADCAST ABORTED");
  }
});
// Add keyboard shortcuts for control
// Add keyboard shortcuts for control
document.addEventListener("keydown", function (event) {
  // Space bar to toggle play/pause
  if (
    event.code === "Space" &&
    document.activeElement.tagName !== "BUTTON" &&
    document.activeElement.tagName !== "INPUT"
  ) {
    event.preventDefault();
    if (audioPlayer) {
      if (audioPlayer.paused) {
        audioPlayer.play();
        pauseBtn.textContent = "PAUSE TRANSMISSION";
      } else {
        audioPlayer.pause();
        pauseBtn.textContent = "RESUME TRANSMISSION";
      }
    } else if (!playBtn.disabled) {
      speakText();
    }
  }

  // Escape key to stop
  if (event.code === "Escape" && audioPlayer) {
    audioPlayer.pause();
    audioPlayer.currentTime = 0;
    audioPlayer = null;
    playBtn.disabled = false;
    pauseBtn.disabled = true;
    stopBtn.disabled = true;
    pauseBtn.textContent = "PAUSE TRANSMISSION";
  }
});

// Add easter egg - Konami code to toggle "ultra classified" mode
let konamiIndex = 0;
const konamiCode = [
  "ArrowUp",
  "ArrowUp",
  "ArrowDown",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
  "ArrowLeft",
  "ArrowRight",
  "b",
  "a",
];
document.addEventListener("keydown", function (event) {
  if (event.key === konamiCode[konamiIndex]) {
    konamiIndex++;
    if (konamiIndex === konamiCode.length) {
      document.body.style.color = "#ff3333";
      document
        .querySelectorAll(
          ".btn, .upload-area, .mission-briefing, .intelligence-panel, .content-display, .control-panel"
        )
        .forEach((el) => {
          el.style.borderColor = "#ff3333";
        });

      statusMessage.textContent = "ULTRAVIOLET CLEARANCE ACTIVATED";
      statusMessage.className = "status-msg status-error";

      // Reset index
      konamiIndex = 0;

      // Return to normal after 5 seconds
      setTimeout(() => {
        document.body.style.color = "var(--military-green)";
        document
          .querySelectorAll(
            ".btn, .upload-area, .mission-briefing, .intelligence-panel, .content-display, .control-panel"
          )
          .forEach((el) => {
            el.style.borderColor = "var(--military-green)";
          });
        statusMessage.textContent = "RETURNING TO STANDARD CLEARANCE";
        statusMessage.className = "status-msg status-success";
      }, 5000);
    }
  } else {
    konamiIndex = 0;
  }
});
