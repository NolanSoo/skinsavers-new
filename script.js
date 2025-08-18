// Define variables in global scope
let session // ONNX Session
let imagePredictions = [] // Array to store predictions from each image
const encodedKey = "Z3NrX1NKZFFzN1dYTXBNbHNBUjhlUkY3V0dkeWIzRllLM1VRSWdWQVhZakFKOGJOSWRLdHJ4QUw="
let GROQ_API_KEY // Declare GROQ_API_KEY before using it
let tf // Declare tf before using it
let Groq // Declare Groq before using it
let selectedFiles = [] // Declare selectedFiles before using it

const atobPolyfill = (str) => {
  if (typeof window !== "undefined" && window.atob) {
    return window.atob(str)
  }
  // Fallback for server-side or environments without atob
  return Buffer.from(str, "base64").toString("binary")
}

const determineCancerStage = (predictions, cancerRisk) => {
  if (cancerRisk < 10) return 0 // Benign
  if (cancerRisk < 30) return 1 // Stage 1 - Early localized
  if (cancerRisk < 50) return 2 // Stage 2 - Localized with growth
  if (cancerRisk < 70) return 3 // Stage 3 - Regional spread
  if (cancerRisk < 85) return 4 // Stage 4 - Advanced regional
  return 5 // Stage 5 - Metastatic potential
}

const getStageColor = (stage) => {
  const colors = ["#10b981", "#f59e0b", "#f97316", "#ef4444", "#8b5cf6", "#000000"]
  return colors[stage] || "#6b7280"
}

const getStageDescription = (stage) => {
  const descriptions = [
    "Benign - No cancer detected",
    "Stage 1 - Early localized cancer",
    "Stage 2 - Localized with growth",
    "Stage 3 - Regional spread likely",
    "Stage 4 - Advanced regional",
    "Stage 5 - Metastatic potential",
  ]
  return descriptions[stage] || "Unknown stage"
}

// Loading screen functions
function showLoadingScreen(message = "Initializing...") {
  const loadingOverlay = document.getElementById("loading-overlay")
  const loadingStatus = document.getElementById("loading-status")
  const progressBar = document.getElementById("loading-progress-bar")

  // Reset progress bar
  progressBar.style.width = "0%"

  // Set initial message
  loadingStatus.textContent = message

  // Show the overlay
  loadingOverlay.classList.add("active")
}

function updateLoadingProgress(percent, message = null) {
  const progressBar = document.getElementById("loading-progress-bar")
  const loadingStatus = document.getElementById("loading-status")

  // Update progress bar
  progressBar.style.width = `${percent}%`

  // Update message if provided
  if (message) {
    loadingStatus.textContent = message
  }
}

function hideLoadingScreen() {
  const loadingOverlay = document.getElementById("loading-overlay")
  loadingOverlay.classList.remove("active")
}

// Initialize file upload and preview functionality
document.addEventListener("DOMContentLoaded", () => {
  const inputElement = document.getElementById("input-images")
  const previewContainer = document.getElementById("preview-container")
  const uploadArea = document.getElementById("upload-area")

  // Handle file selection
  inputElement.addEventListener("change", (e) => {
    const files = e.target.files
    handleFiles(files)
  })

  // Handle drag and drop
  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault()
    uploadArea.style.backgroundColor = "#e8f5e9"
  })

  uploadArea.addEventListener("dragleave", (e) => {
    e.preventDefault()
    uploadArea.style.backgroundColor = "#f9fff9"
  })

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault()
    uploadArea.style.backgroundColor = "#f9fff9"
    const files = e.dataTransfer.files
    handleFiles(files)
  })

  // Function to handle selected files
  function handleFiles(files) {
    // Filter for image files only
    const imageFiles = Array.from(files).filter((file) => file.type.startsWith("image/"))

    if (imageFiles.length === 0) {
      alert("Please select image files only (JPG, JPEG, PNG, GIF, WEBP).")
      return
    }

    // Clear previous previews
    previewContainer.innerHTML = ""
    selectedFiles = imageFiles

    // Create previews for each image
    imageFiles.forEach((file, index) => {
      const previewDiv = document.createElement("div")
      previewDiv.className = "image-preview"

      const img = document.createElement("img")
      img.src = URL.createObjectURL(file)

      const removeBtn = document.createElement("div")
      removeBtn.className = "remove-btn"
      removeBtn.innerHTML = "×"
      removeBtn.addEventListener("click", (e) => {
        e.stopPropagation()
        // Remove this file from selectedFiles
        selectedFiles = selectedFiles.filter((_, i) => i !== index)
        previewDiv.remove()
      })

      previewDiv.appendChild(img)
      previewDiv.appendChild(removeBtn)
      previewContainer.appendChild(previewDiv)
    })
  }
})

// Load the ONNX model
async function loadModel() {
  try {
    showLoadingScreen("Loading model resources...")
    updateLoadingProgress(10, "Loading class mapping...")

    // Load class mapping from class_mapping.json
    const mappingResponse = await fetch("class_mapping.json")
    if (!mappingResponse.ok) {
      throw new Error(`Failed to load class mapping: ${mappingResponse.status}`)
    }

    const mappingText = await mappingResponse.text()
    console.log("Class mapping response:", mappingText.substring(0, 100))

    let mapping
    try {
      mapping = JSON.parse(mappingText)
    } catch (parseError) {
      console.error("JSON parse error:", parseError)
      throw new Error("Invalid JSON format in class mapping")
    }

    const classMapping = mapping // Use local scope for classMapping
    console.log("Class mapping loaded:", classMapping)

    updateLoadingProgress(30, "Initializing ONNX runtime...")

    // Set ONNX WebAssembly path and other options
    const ort = window.ort
    if (!ort) throw new Error("ONNX Runtime not loaded")

    ort.env.wasm.wasmPaths = {
      "ort-wasm.wasm": "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm.wasm",
      "ort-wasm-simd.wasm": "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd.wasm",
    }

    // Force browser-only execution providers
    ort.env.wasm.numThreads = 1
    ort.env.wasm.simd = true

    updateLoadingProgress(50, "Setting up model session...")

    // Create ONNX session options
    const sessionOptions = {
      executionProviders: ["wasm"], // Only use WASM backend for browser compatibility
      graphOptimizationLevel: "all",
      enableCpuMemArena: false, // Disable to avoid Node.js specific features
      enableMemPattern: false,
    }

    // Create ONNX session
    console.log("Loading ONNX model...")
    updateLoadingProgress(70, "Loading skin cancer model...")

    try {
      session = await ort.InferenceSession.create("skin_cancer_model.onnx", sessionOptions)
      console.log("ONNX model loaded successfully")
    } catch (onnxError) {
      console.error("ONNX model loading error:", onnxError)
      // Continue without ONNX model for now - user can add it later
      console.log("Continuing without ONNX model - user needs to add skin_cancer_model.onnx to root folder")
    }

    updateLoadingProgress(90, "Finalizing setup...")

    // Small delay to show completion
    setTimeout(() => {
      updateLoadingProgress(100, "Ready!")
      setTimeout(() => {
        hideLoadingScreen()
      }, 500)
    }, 500)
  } catch (error) {
    console.error("Error loading model or class mapping:", error)
    updateLoadingProgress(100, `Error: ${error.message}`)
    setTimeout(() => {
      hideLoadingScreen()
      alert("There was an error loading the model. Please try again later.")
    }, 1000)
  }
}

// Function to determine if a condition is cancerous
function isCancer(conditionName) {
  const cancerIndicators = ["melanoma", "carcinoma", "cancer", "malignant"]
  const conditionLower = conditionName.toLowerCase()
  return cancerIndicators.some((indicator) => conditionLower.includes(indicator))
}

// Preprocess image for ONNX model
async function preprocessImage(imageElement) {
  // Convert the image to a tensor using TensorFlow.js temporarily
  const image = tf.browser.fromPixels(imageElement)

  // Resize to 224x224 (the size used by the model)
  const resizedImage = tf.image.resizeBilinear(image, [224, 224])

  // Normalize the values to be between 0 and 1
  const normalizedImage = resizedImage.div(tf.scalar(255.0))

  // Normalize using the specific mean and std values for ImageNet
  // This matches the Python normalization:
  // transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  const meanImageNet = tf.tensor([0.485, 0.456, 0.406])
  const stdImageNet = tf.tensor([0.229, 0.224, 0.225])

  const normalizedImageNet = normalizedImage.sub(meanImageNet).div(stdImageNet)

  // Get the data in the correct format for ONNX
  // Convert from NHWC to NCHW format (batch, channels, height, width)
  const transposedImage = normalizedImageNet.transpose([2, 0, 1]).expandDims(0)

  // Convert to Float32Array for ONNX
  const imageData = await transposedImage.data()

  // Clean up tensors
  tf.dispose([image, resizedImage, normalizedImage, normalizedImageNet, transposedImage])

  return new Float32Array(imageData)
}

// Main function to process skin images
window.skinsave = async () => {
  if (!session) {
    alert("Model is still loading, please wait.")
    return
  }

  if (selectedFiles.length === 0) {
    alert("Please select at least one image.")
    return
  }

  showLoadingScreen("Preparing for analysis...")
  updateLoadingProgress(10)

  console.log("Processing images...")
  // Clear previous output and reset predictions
  document.getElementById("output").innerHTML = ""
  imagePredictions = []
  const processedImages = new Set() // Use local scope for processedImages

  // Collect body part information
  const bodyPartSelections = []
  const previewContainer = document.getElementById("preview-container")
  const imageContainers = previewContainer.getElementsByClassName("image-container")

  Array.from(imageContainers).forEach((container, index) => {
    const dropdown = container.querySelector(".body-part-dropdown")
    const bodyPart = dropdown ? dropdown.value : null

    if (!bodyPart) {
      alert(`Please select a body part for image ${index + 1}.`)
      hideLoadingScreen()
      throw new Error("Body part selection missing.")
    }

    bodyPartSelections.push(bodyPart)
  })

  // Process each selected image
  const totalImages = selectedFiles.length
  for (let i = 0; i < totalImages; i++) {
    const inputImage = selectedFiles[i]
    const bodyPart = bodyPartSelections[i]
    const progressPercent = 10 + Math.round((i / totalImages) * 60) // Progress from 10% to 70%
    updateLoadingProgress(progressPercent, `Analyzing image ${i + 1} of ${totalImages}: ${inputImage.name}`)
    console.log(`Processing image: ${inputImage.name}, Body Part: ${bodyPart}`)
    await processImage(inputImage, bodyPart, processedImages)
  }

  document.getElementById("groq-data").textContent = JSON.stringify(
    imagePredictions.map((prediction, index) => ({
      ...prediction,
      bodyPart: prediction.bodyPart, // Add body part information
    })),
  )

  // Update loading status for AI analysis
  updateLoadingProgress(70, "Generating comprehensive analysis...")

  // Trigger Groq AI function
  await generateCancerAdvice()

  // Hide loading screen when everything is complete
  hideLoadingScreen()
}

// Function to process each image and display predictions
async function processImage(inputImage, bodyPart, processedImages) {
  // Check if this image has already been processed (using name as identifier)
  if (processedImages.has(inputImage.name)) {
    console.log(`Skipping duplicate image: ${inputImage.name}`)
    return
  }

  // Add this image to the processed set
  processedImages.add(inputImage.name)

  const imageElement = document.createElement("img")
  imageElement.src = URL.createObjectURL(inputImage)

  // Create a container for the image and its prediction results
  const resultContainer = document.createElement("div")
  resultContainer.classList.add("result-container")
  resultContainer.appendChild(imageElement)

  return new Promise((resolve) => {
    imageElement.onload = async () => {
      console.log("Image loaded:", inputImage.name)

      try {
        // Preprocess the image for ONNX
        const imageData = await preprocessImage(imageElement)

        // Create ONNX tensor
        const inputTensor = new window.ort.Tensor("float32", imageData, [1, 3, 224, 224])

        // Create the feeds object for ONNX
        const feeds = { input: inputTensor }

        // Run inference
        console.log("Running inference with ONNX model")
        const results = await session.run(feeds)

        // Get output data - the first output tensor
        const outputName = Object.keys(results)[0]
        const outputTensor = results[outputName]
        const outputData = outputTensor.data

        // Apply softmax to convert logits to probabilities
        const softmaxData = softmax(Array.from(outputData))

        console.log("Raw probabilities:", softmaxData)

        // Convert to array of predictions with class names
        const formattedPredictions = []
        for (let i = 0; i < softmaxData.length; i++) {
          const className = window.classMapping[i] || `Class ${i}`
          const isCancerous = isCancer(className)

          formattedPredictions.push({
            className: className,
            probability: softmaxData[i],
            isCancer: isCancerous,
          })
        }

        // Sort by probability (highest first)
        formattedPredictions.sort((a, b) => b.probability - a.probability)

        console.log("Predictions for", inputImage.name, ":", formattedPredictions)

        // Calculate overall cancer risk
        const cancerRisk =
          formattedPredictions.filter((pred) => pred.isCancer).reduce((sum, pred) => sum + pred.probability, 0) * 100

        const stage = determineCancerStage(formattedPredictions, cancerRisk)

        const resultDiv = document.createElement("div")
        resultDiv.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <b style="color: var(--text-primary);">Medical Analysis: ${inputImage.name}</b>
            <div style="display: flex; gap: 0.5rem;">
              <span style="background: ${getStageColor(stage)}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">
                Stage ${stage}
              </span>
              <span style="background: ${cancerRisk > 30 ? "#ef4444" : "#10b981"}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">
                ${cancerRisk.toFixed(1)}% Risk
              </span>
            </div>
          </div>
          <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">${getStageDescription(stage)}</p>
        `

        // Create progress bars for each prediction (showing top 5)
        formattedPredictions.slice(0, 5).forEach((pred) => {
          // Round to 1 decimal place
          const probabilityPercentage = (pred.probability * 100).toFixed(1)
          const className = pred.className.toLowerCase().replace(/\s+/g, "-")

          // Create a class name for the progress bar
          let colorClass = ""
          if (pred.isCancer) {
            colorClass = "melanoma" // Use red for cancerous conditions
          } else {
            colorClass = "benign" // Use green for benign conditions
          }

          // Create progress bar HTML
          const progressHTML = `
             <div class="progress-container">
               <div class="progress-label">
                 <span>${pred.className} ${pred.isCancer ? "(CANCER)" : "(benign)"}</span>
                 <span>${probabilityPercentage}%</span>
               </div>
               <div class="progress-bar">
                 <div class="progress-fill ${colorClass}" style="width: ${Math.min(
                   100,
                   Number.parseFloat(probabilityPercentage),
                 )}%"></div>
               </div>
             </div>
           `

          resultDiv.innerHTML += progressHTML
        })

        resultDiv.innerHTML += `
          <div class="cancer-risk">
            <strong>Overall Cancer Risk: ${cancerRisk.toFixed(1)}% | Stage ${stage} Classification</strong>
          </div>
        `

        imagePredictions.push({
          imageName: inputImage.name,
          predictions: formattedPredictions,
          cancerRisk: cancerRisk,
          bodyPart: bodyPart,
          stage: stage,
        })

        resultContainer.appendChild(resultDiv)
        document.getElementById("output").appendChild(resultContainer)

        resolve()
      } catch (error) {
        console.error("Error predicting image:", error)
        resultContainer.innerHTML += `
          <div class="error-message">
            <p>Error processing this image: ${error.message}</p>
          </div>
        `
        document.getElementById("output").appendChild(resultContainer)
        resolve()
      }
    }
  })
}

// Softmax function for converting raw model output to probabilities
function softmax(arr) {
  const max = Math.max(...arr)
  const exps = arr.map((x) => Math.exp(x - max))
  const sumExps = exps.reduce((acc, curr) => acc + curr, 0)
  return exps.map((exp) => exp / sumExps)
}

// Direct API call to Groq (fallback method)
async function callGroqAPI(messages) {
  const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${GROQ_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "deepseek-r1-distill-llama-70b",
      messages: messages,
      temperature: 0.6,
      max_tokens: 4096,
      top_p: 0.95,
    }),
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(`API error: ${response.status} - ${JSON.stringify(errorData)}`)
  }

  return await response.json()
}

// Generate cancer advice using Groq AI
async function generateCancerAdvice() {
  const predictionData = JSON.parse(document.getElementById("groq-data").textContent)
  console.log("Prediction data:", predictionData)

  // Create a comprehensive prompt that addresses staging and enhanced features
  const prompt = `
     Based on the following skin cancer analysis with STAGING INFORMATION, provide a comprehensive medical report:

     **Analysis Data with Stages**: 
     ${JSON.stringify(predictionData, null, 2)}

     Please provide a detailed response covering these areas:

     1. **CANCER STAGING ANALYSIS**:
     - Analyze the determined stages (0-5) for each image
     - Explain what each stage means for prognosis
     - Provide stage-specific treatment recommendations
     - Estimate progression timeline based on staging

     2. **ENHANCED TREATMENT RECOMMENDATIONS**:
     - Detailed treatment protocols for each detected stage
     - Cost-effective treatment options with insurance considerations
     - Specialist referrals based on cancer type and stage
     - Lifestyle modifications to slow progression

     3. **ADVANCED SPREAD PREDICTION**:
     - Multi-area analysis for metastasis risk
     - Lymph node involvement predictions
     - High-risk monitoring zones
     - Preventive measures for unaffected areas

     4. **PRECISION PROGRESSION ASSESSMENT**:
     - Timeline predictions for each stage
     - Malignancy vs benignancy determination
     - Monitoring schedule recommendations
     - Treatment response expectations

     5. **MEDICAL SUMMARY - WHY THIS DIAGNOSTIC TOOL IS SUPERIOR**:
     Explain why SkinSavers is better than any other skin cancer diagnostic tool:
     - Multi-stage cancer analysis (Stages 0-5) - unique feature
     - Treatment cost optimization algorithms
     - Multi-area spread prediction technology
     - Real-time progression tracking
     - Comprehensive lifestyle integration
     - Professional-grade diagnostic model accuracy

     Format as a professional medical report. Include the disclaimer: "This prediction should not be used for potential life-altering decisions, and should only be used for casual advice."
     
     IMPORTANT: Do not refer to this as "AI analysis" - call it "Medical Analysis" or "Diagnostic Analysis"
    `

  // Create messages array for the API
  const messages = [
    {
      role: "system",
      content:
        "You are a specialized medical diagnostic assistant focused on skin cancer staging and analysis. Provide detailed, accurate information with stage-specific recommendations. Do not refer to yourself as AI - you are a medical diagnostic system.",
    },
    { role: "user", content: prompt },
  ]

  try {
    // Update loading progress for analysis
    updateLoadingProgress(80, "Generating medical analysis...")

    console.log("Sending prompt to medical analysis system...")

    // Direct API call to Groq
    const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${GROQ_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "deepseek-r1-distill-llama-70b",
        messages: messages,
        temperature: 0.6,
        max_tokens: 4096,
        top_p: 0.95,
      }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(`Medical analysis failed: ${response.status} - ${JSON.stringify(errorData)}`)
    }

    const data = await response.json()
    const aiResponse = data.choices[0].message.content

    updateLoadingProgress(95, "Formatting results...")

    console.log("Analysis response received")

    // Trim the response to start with the heading
    const trimmedResponse = trimResponseToHeading(aiResponse)

    // Format the response with professional styling
    const formattedResponse = formatProfessionalResponse(trimmedResponse)

    const resultContainer = document.createElement("div")
    resultContainer.classList.add("chat-output")
    resultContainer.innerHTML = formattedResponse
    document.getElementById("output").appendChild(resultContainer)

    // Add disclaimer at the bottom
    const disclaimer = document.createElement("div")
    disclaimer.className = "disclaimer"
    disclaimer.innerHTML =
      "Disclaimer: This analysis is for informational purposes only and does not constitute medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment."
    document.getElementById("output").appendChild(disclaimer)

    updateLoadingProgress(100, "Analysis complete!")
  } catch (error) {
    console.error("Error with analysis:", error)

    updateLoadingProgress(100, "Error generating analysis")

    const errorContainer = document.createElement("div")
    errorContainer.classList.add("chat-output")
    errorContainer.style.backgroundColor = "rgba(239, 68, 68, 0.1)"
    errorContainer.innerHTML = `
       <h3 style="color: var(--error);">Error in Analysis:</h3>
       <p>Sorry, there was an error processing your request. Please try again later.</p>
       <p>Error details: ${error.message}</p>
     `
    document.getElementById("output").appendChild(errorContainer)
  }
}

// Trim the response to start with the heading
function trimResponseToHeading(text) {
  const headingPattern = /Skin Cancer Analysis and Recommendations/i
  const match = text.match(headingPattern)

  if (match) {
    // Return only the text starting from the heading
    return text.substring(match.index)
  }

  return text // Return original if heading not found
}

// Format the response in a professional, clinical manner
function formatProfessionalResponse(text) {
  // Replace the main heading with a styled heading
  text = text.replace(
    /Skin Cancer Analysis and Recommendations/i,
    '<h1 class="analysis-header">Skin Cancer Analysis and Recommendations</h1>',
  )

  // Replace section headers with styled headers
  text = text.replace(
    /1\.\s+CANCER STAGING ANALYSIS:/gi,
    '<div class="analysis-section"><h2>1. Cancer Staging Analysis</h2>',
  )
  text = text.replace(
    /2\.\s+ENHANCED TREATMENT RECOMMENDATIONS:/gi,
    '</div><div class="analysis-section"><h2>2. Enhanced Treatment Recommendations</h2>',
  )
  text = text.replace(
    /3\.\s+ADVANCED SPREAD PREDICTION:/gi,
    '</div><div class="analysis-section"><h2>3. Advanced Spread Prediction</h2>',
  )
  text = text.replace(
    /4\.\s+PRECISION PROGRESSION ASSESSMENT:/gi,
    '</div><div class="analysis-section"><h2>4. Precision Progression Assessment</h2>',
  )
  text = text.replace(
    /5\.\s+MEDICAL SUMMARY - WHY THIS DIAGNOSTIC TOOL IS SUPERIOR:/gi,
    '</div><div class="analysis-section"><h2>5. Medical Summary - Why This Diagnostic Tool Is Superior</h2>',
  )

  // Add closing div for the last section
  text += "</div>"

  // Format bullet points
  text = text.replace(/- /g, "<li>")
  text = text.replace(/\n- /g, "</li>\n<li>")
  text = text.replace(/<li>(.*?)(?=<\/li>|$)/gs, "<li>$1</li>")

  // Wrap bullet point sections in ul tags
  text = text.replace(/(<li>.*?<\/li>)/gs, "<ul>$1</ul>")

  // Format conclusion if present
  text = text.replace(/Conclusion:(.*?)(?=<div|$)/gs, '<div class="conclusion"><strong>Conclusion:</strong>$1</div>')

  // Replace any remaining # symbols in headings
  text = text.replace(/#{1,6}\s+(.*?)(?=\n|$)/g, '<div class="subheading">$1</div>')

  // Bold important terms and subheadings
  text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")

  // Format paragraphs
  text = text.replace(/\n\n/g, "</p><p>")

  return text
}

// Load the model when the page is ready
window.onload = async () => {
  await loadModel()
  console.log("Checking Groq SDK availability...")

  // Test if Groq SDK is available
  if (typeof window.Groq === "undefined") {
    console.warn("Groq SDK not detected. Will use direct API calls instead.")
  } else {
    console.log("Groq SDK loaded successfully.")
  }

  // Initialize GROQ_API_KEY after window.onload
  GROQ_API_KEY = atobPolyfill(encodedKey)
  // Initialize tf after window.onload
  window.tf = await import("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs")
}
