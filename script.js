// Define variables in global scope
let session; // ONNX Session
let imagePredictions = []; // Array to store predictions from each image
const GROQ_API_KEY = "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"; // will figure out env variables later
// Track processed images to prevent duplicates
const processedImages = new Set();
// Store selected files
let selectedFiles = [];
// Store class mappings
let classMapping = {};

// Loading screen functions
function showLoadingScreen(message = "Initializing...") {
  const loadingOverlay = document.getElementById("loading-overlay");
  const loadingStatus = document.getElementById("loading-status");
  const progressBar = document.getElementById("loading-progress-bar");
  
  // Reset progress bar
  progressBar.style.width = "0%";
  
  // Set initial message
  loadingStatus.textContent = message;
  
  // Show the overlay
  loadingOverlay.classList.add("active");
}

function updateLoadingProgress(percent, message = null) {
  const progressBar = document.getElementById("loading-progress-bar");
  const loadingStatus = document.getElementById("loading-status");
  
  // Update progress bar
  progressBar.style.width = `${percent}%`;
  
  // Update message if provided
  if (message) {
    loadingStatus.textContent = message;
  }
}

function hideLoadingScreen() {
  const loadingOverlay = document.getElementById("loading-overlay");
  loadingOverlay.classList.remove("active");
}

// Initialize file upload and preview functionality
document.addEventListener("DOMContentLoaded", function () {
  const inputElement = document.getElementById("input-images");
  const previewContainer = document.getElementById("preview-container");
  const uploadArea = document.getElementById("upload-area");

  // Handle file selection
  inputElement.addEventListener("change", function (e) {
    const files = e.target.files;
    handleFiles(files);
  });

  // Handle drag and drop
  uploadArea.addEventListener("dragover", function (e) {
    e.preventDefault();
    uploadArea.style.backgroundColor = "#e8f5e9";
  });

  uploadArea.addEventListener("dragleave", function (e) {
    e.preventDefault();
    uploadArea.style.backgroundColor = "#f9fff9";
  });

  uploadArea.addEventListener("drop", function (e) {
    e.preventDefault();
    uploadArea.style.backgroundColor = "#f9fff9";
    const files = e.dataTransfer.files;
    handleFiles(files);
  });

  // Function to handle selected files
  function handleFiles(files) {
    // Filter for image files only
    const imageFiles = Array.from(files).filter((file) =>
      file.type.startsWith("image/")
    );

    if (imageFiles.length === 0) {
      alert("Please select image files only (JPG, JPEG, PNG, GIF, WEBP).");
      return;
    }

    // Clear previous previews
    previewContainer.innerHTML = "";
    selectedFiles = imageFiles;

    // Create previews for each image
    imageFiles.forEach((file, index) => {
      const reader = new FileReader();

      reader.onload = function (e) {
        const previewDiv = document.createElement("div");
        previewDiv.className = "image-preview";

        const img = document.createElement("img");
        img.src = e.target.result;

        const removeBtn = document.createElement("div");
        removeBtn.className = "remove-btn";
        removeBtn.innerHTML = "×";
        removeBtn.addEventListener("click", function (e) {
          e.stopPropagation();
          // Remove this file from selectedFiles
          selectedFiles = selectedFiles.filter((_, i) => i !== index);
          previewDiv.remove();
        });

        previewDiv.appendChild(img);
        previewDiv.appendChild(removeBtn);
        previewContainer.appendChild(previewDiv);
      };

      reader.readAsDataURL(file);
    });
  }
});

// Load the ONNX model
async function loadModel() {
  try {
    showLoadingScreen("Loading model resources...");
    updateLoadingProgress(10, "Loading class mapping...");
    
    // Load class mapping from class_mapping.json
    const mappingResponse = await fetch("class_mapping.json");
    if (!mappingResponse.ok) {
      throw new Error(`Failed to load class mapping: ${mappingResponse.status}`);
    }
    classMapping = await mappingResponse.json();
    console.log("Class mapping loaded:", classMapping);
    
    updateLoadingProgress(30, "Initializing ONNX runtime...");
    
    // Set ONNX WebAssembly path and other options
    const ort = window.ort;
    ort.env.wasm.wasmPaths = {
      'ort-wasm.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm.wasm',
      'ort-wasm-simd.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd.wasm',
      'ort-wasm-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-threaded.wasm',
      'ort-wasm-simd-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm'
    };
    
    updateLoadingProgress(50, "Setting up model session...");
    
    // Create ONNX session options
    const sessionOptions = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    };
    
    // Create ONNX session
    console.log("Loading ONNX model...");
    updateLoadingProgress(70, "Loading skin cancer model...");
    session = await ort.InferenceSession.create('skin_cancer_model.onnx', sessionOptions);
    
    updateLoadingProgress(90, "Finalizing setup...");
    console.log("ONNX model loaded successfully");
    
    // Small delay to show completion
    setTimeout(() => {
      updateLoadingProgress(100, "Ready!");
      setTimeout(() => {
        hideLoadingScreen();
      }, 500);
    }, 500);
  } catch (error) {
    console.error("Error loading model or class mapping:", error);
    updateLoadingProgress(100, `Error: ${error.message}`);
    setTimeout(() => {
      hideLoadingScreen();
      alert("There was an error loading the model. Please try again later.");
    }, 1000);
  }
}

// Function to determine if a condition is cancerous
function isCancer(conditionName) {
  const cancerIndicators = ['melanoma', 'carcinoma', 'cancer', 'malignant'];
  const conditionLower = conditionName.toLowerCase();
  return cancerIndicators.some(indicator => conditionLower.includes(indicator));
}

// Preprocess image for ONNX model
async function preprocessImage(imageElement) {
  // Convert the image to a tensor using TensorFlow.js temporarily
  const image = tf.browser.fromPixels(imageElement);
  
  // Resize to 224x224 (the size used by the model)
  const resizedImage = tf.image.resizeBilinear(image, [224, 224]);
  
  // Normalize the values to be between 0 and 1
  const normalizedImage = resizedImage.div(tf.scalar(255.0));
  
  // Normalize using the specific mean and std values for ImageNet
  // This matches the Python normalization:
  // transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  const meanImageNet = tf.tensor([0.485, 0.456, 0.406]);
  const stdImageNet = tf.tensor([0.229, 0.224, 0.225]);
  
  const normalizedImageNet = normalizedImage.sub(meanImageNet).div(stdImageNet);
  
  // Get the data in the correct format for ONNX
  // Convert from NHWC to NCHW format (batch, channels, height, width)
  const transposedImage = normalizedImageNet.transpose([2, 0, 1]).expandDims(0);
  
  // Convert to Float32Array for ONNX
  const imageData = await transposedImage.data();
  
  // Clean up tensors
  tf.dispose([image, resizedImage, normalizedImage, normalizedImageNet, transposedImage]);
  
  return new Float32Array(imageData);
}

// Main function to process skin images
window.skinsave = async function () {
    if (!session) {
        alert("Model is still loading, please wait.");
        return;
    }

    if (selectedFiles.length === 0) {
        alert("Please select at least one image.");
        return;
    }

    showLoadingScreen("Preparing for analysis...");
    updateLoadingProgress(10);

    console.log("Processing images...");
    // Clear previous output and reset predictions
    document.getElementById("output").innerHTML = "";
    imagePredictions = [];
    processedImages.clear();

    // Collect body part information
    const bodyPartSelections = [];
    const previewContainer = document.getElementById("preview-container");
    const imageContainers = previewContainer.getElementsByClassName("image-container");

    Array.from(imageContainers).forEach((container, index) => {
        const dropdown = container.querySelector(".body-part-dropdown");
        const bodyPart = dropdown ? dropdown.value : null;

        if (!bodyPart) {
            alert(`Please select a body part for image ${index + 1}.`);
            hideLoadingScreen();
            throw new Error("Body part selection missing.");
        }

        bodyPartSelections.push(bodyPart);
    });

    // Process each selected image
    const totalImages = selectedFiles.length;
    for (let i = 0; i < totalImages; i++) {
        const inputImage = selectedFiles[i];
        const bodyPart = bodyPartSelections[i];
        const progressPercent = 10 + Math.round((i / totalImages) * 60); // Progress from 10% to 70%
        updateLoadingProgress(progressPercent, `Analyzing image ${i + 1} of ${totalImages}: ${inputImage.name}`);
        console.log(`Processing image: ${inputImage.name}, Body Part: ${bodyPart}`);
        await processImage(inputImage, bodyPart);
    }


    document.getElementById("groq-data").textContent = JSON.stringify(
        imagePredictions.map((prediction, index) => ({
            ...prediction,
            bodyPart: bodyPartSelections[index], // Add body part information
        }))
    );

    

    // Update loading status for AI analysis
    updateLoadingProgress(70, "Generating comprehensive analysis...");

    // Trigger Groq AI function
    await generateCancerAdvice();

    // Hide loading screen when everything is complete
    hideLoadingScreen();
};


// Function to process each image and display predictions
async function processImage(inputImage) {
  // Check if this image has already been processed (using name as identifier)
  if (processedImages.has(inputImage.name)) {
    console.log(`Skipping duplicate image: ${inputImage.name}`);
    return;
  }

  // Add this image to the processed set
  processedImages.add(inputImage.name);

  const imageElement = document.createElement("img");
  imageElement.src = URL.createObjectURL(inputImage);

  // Create a container for the image and its prediction results
  const resultContainer = document.createElement("div");
  resultContainer.classList.add("result-container");
  resultContainer.appendChild(imageElement);

  return new Promise((resolve) => {
    imageElement.onload = async () => {
      console.log("Image loaded:", inputImage.name);
      
      try {
        // Preprocess the image for ONNX
        const imageData = await preprocessImage(imageElement);
        
        // Create ONNX tensor
        const inputTensor = new window.ort.Tensor('float32', imageData, [1, 3, 224, 224]);
        
        // Create the feeds object for ONNX
        // Note: The input name might be different based on your model
        // Common names are 'input', 'input.1', or 'images'
        const feeds = { 'input': inputTensor };
        
        // Run inference
        console.log("Running inference with ONNX model");
        const results = await session.run(feeds);
        
        // Get output data - the first output tensor
        // The output name might be different based on your model
        // Common names are 'output', 'logits', or 'probabilities'
        const outputName = Object.keys(results)[0];
        const outputTensor = results[outputName];
        const outputData = outputTensor.data;
        
        // Apply softmax to convert logits to probabilities
        const softmaxData = softmax(Array.from(outputData));
        
        console.log("Raw probabilities:", softmaxData);
        
        // Convert to array of predictions with class names
        const formattedPredictions = [];
        for (let i = 0; i < softmaxData.length; i++) {
          const className = classMapping[i] || `Class ${i}`;
          const isCancerous = isCancer(className);
          
          formattedPredictions.push({
            className: className,
            probability: softmaxData[i],
            isCancer: isCancerous
          });
        }
        
        // Sort by probability (highest first)
        formattedPredictions.sort((a, b) => b.probability - a.probability);
        
        console.log("Predictions for", inputImage.name, ":", formattedPredictions);

        const resultDiv = document.createElement("div");
        resultDiv.innerHTML = `<b>Prediction for ${inputImage.name}:</b><br>`;

        // Create progress bars for each prediction (showing top 5)
        formattedPredictions.slice(0, 5).forEach((pred) => {
          // Round to 1 decimal place
          const probabilityPercentage = (pred.probability * 100).toFixed(1);
          const className = pred.className.toLowerCase().replace(/\s+/g, "-");

          // Create a class name for the progress bar
          let colorClass = "";
          if (pred.isCancer) {
            colorClass = "melanoma"; // Use red for cancerous conditions
          } else {
            colorClass = "benign"; // Use green for benign conditions
          }

          // Create progress bar HTML
          const progressHTML = `
             <div class="progress-container">
               <div class="progress-label">
                 <span>${pred.className} ${pred.isCancer ? '(CANCER)' : '(benign)'}</span>
                 <span>${probabilityPercentage}%</span>
               </div>
               <div class="progress-bar">
                 <div class="progress-fill ${colorClass}" style="width: ${Math.min(
            100,
            parseFloat(probabilityPercentage)
          )}%"></div>
               </div>
             </div>
           `;

          resultDiv.innerHTML += progressHTML;
        });

        // Calculate overall cancer risk
        const cancerRisk = formattedPredictions
          .filter(pred => pred.isCancer)
          .reduce((sum, pred) => sum + pred.probability, 0) * 100;
          
        resultDiv.innerHTML += `
          <div class="cancer-risk">
            <strong>Overall Cancer Risk: ${cancerRisk.toFixed(1)}%</strong>
          </div>
        `;

        imagePredictions.push({
          imageName: inputImage.name,
          predictions: formattedPredictions,
          cancerRisk: cancerRisk
        });
        
        resultContainer.appendChild(resultDiv);
        document.getElementById("output").appendChild(resultContainer);
        
        resolve();
      } catch (error) {
        console.error("Error predicting image:", error);
        resultContainer.innerHTML += `
          <div class="error-message">
            <p>Error processing this image: ${error.message}</p>
          </div>
        `;
        document.getElementById("output").appendChild(resultContainer);
        resolve();
      }
    };
  });
}

// Softmax function for converting raw model output to probabilities
function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sumExps = exps.reduce((acc, curr) => acc + curr, 0);
  return exps.map(exp => exp / sumExps);
}

// Direct API call to Groq (fallback method)
async function callGroqAPI(messages) {
  const response = await fetch(
    "https://api.groq.com/openai/v1/chat/completions",
    {
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
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      `API error: ${response.status} - ${JSON.stringify(errorData)}`
    );
  }

  return await response.json();
}

// Generate cancer advice using Groq AI
async function generateCancerAdvice() {
  const predictionData = JSON.parse(
    document.getElementById("groq-data").textContent
  );
  console.log("Prediction data:", predictionData);

  // Create a comprehensive prompt that addresses all three key features
  const prompt = `
     Based on the following predictions from multiple images of skin cancer, please provide a comprehensive analysis:

     **Predictions Data**: 
     ${JSON.stringify(predictionData, null, 2)}

     Please provide a detailed response covering these three key areas:

     1. TREATMENT OPTIONS AND ADVICE:
     - Provide detailed treatment options based on the detected skin conditions
     - Explain potential outcomes and their severity
     - Suggest lifestyle changes that could help slow progression
     - Recommend specific medical specialists to consult

     2. CANCER SPREAD PREDICTION:
     - Analyze the confidence levels across different skin areas
     - Predict potential spread patterns based on the current data
     - Identify high-risk areas that should be monitored closely
     - Explain how the spread might occur based on the type of cancer detected

     3. CANCER PROGRESSION ASSESSMENT:
     - Estimate the current stage of progression based on confidence levels
     - Determine if the cancer appears malignant or benign
     - Suggest a monitoring schedule for tracking progression
     - Explain how the progression might change over time

     Be specific, detailed, and provide actionable advice. Include both immediate next steps and long-term recommendations.
     Use these descriptions for each section:
     AI Chatbot Advice
     This app will use an AI chatbot in order to give advice. For instance, it could tell you which doctor to go to for your specific cancer and for the cheapest price depending on your insurance. It can also tell you which possible outcomes are the most dangerous, as well as lifestyle changes that will help decrease the progression rate of your cancer. 

     Prediction of where cancer will spread next
     By getting information on multiple parts of the skin, this app will allow for a world-class analysis of where the cancer might spread next. For instance, if one area is predicted with 100% confidence to be cancerous, and the other areas have 66% and 36%, for example, it can give you advice on where it will spread and where it may metastasize to. This advice will help you get the correct treatment the first time without a long sequence of expensive doctors' visits and appointments. 

     Exact progression of your cancer
     Although data is difficult to receive on the exact stages of melanoma and other types of cancer (Stages 1-5), this app will be able to see how far your cancer has progressed in various areas based on the AI model's confidence. If it is 100 percent sure your skin has cancer, then it has almost certainly progressed much more than if it is 10% sure. It will also be able to tell if the cancer is malignant or benign. You can also take pictures of your skin consistently, which will allow you to measure your progression more clearly and get an even better picture of your current position. 

    Rough Prediction of Stage/Progression of Cancer
    Attempt to use the data provided, including confidence percentages in order to predict the stage, type, and progression of the cancer overall. Try to predict a timeline of how the next few years may be like (including estimated time to progress further or become treated completely with different lifestyle/treatment decisions), and how treatment can change that timeline for the better for cheap. Add the exact disclaimer: "This prediction should not be used for potiential life altering decisions, and should only be used for casual advice."
 

** Predictions Data **: 
${ JSON.stringify(predictionData, null, 2) }

Each prediction includes the associated body part for better context.Use this information to provide more accurate advice, such as identifying high - risk areas or predicting cancer spread patterns.

Please use the location information for each image to provide more accurate advice, such as identifying high-risk areas or predicting cancer spread patterns.


    IMPORTANT INSTRUCTIONS:
     1. Start your response with the heading "Skin Cancer Analysis and Recommendations"
     2. Refer to yourself as "assistant" not "AI"
     3. Format your response in a professional, clinical manner with clear sections and bullet points
     4. Do not include any disclaimers about not being a medical professional
     5. Use the probability to represent APPROXIMATELY how far the cancer has progressed in that specific area
     6. Do not say things like "the user wants" or "based on your request" - just provide the direct analysis
     7. DO NOT use any # symbols in headings or subheadings (like "#### Long-Term Recommendations")
     8. For subheadings, just use bold text without any # symbols
   `;

  // Create messages array for the API
  const messages = [
    {
      role: "system",
      content:
        "You are a specialized medical assistant focused on skin cancer analysis. Provide detailed, accurate, and helpful information about skin cancer based on image analysis data. Include specific advice on treatment options, cancer spread predictions, and progression assessment. Refer to yourself as 'assistant' not 'AI'.",
    },
    { role: "user", content: prompt },
  ];

  try {
    // Update loading progress for AI analysis
    updateLoadingProgress(80, "Generating medical analysis...");

    console.log("Sending prompt to Groq...");

    let aiResponse;

    // Try using the SDK first, if available
    try {
      if (typeof Groq !== "undefined") {
        console.log("Using Groq SDK");
        const groqClient = new Groq({
          apiKey: GROQ_API_KEY,
        });

        const chatCompletion = await groqClient.chat.completions.create({
          messages: messages,
          model: "deepseek-r1-distill-llama-70b",
          temperature: 0.6,
          max_tokens: 4096,
          top_p: 0.95,
        });

        aiResponse = chatCompletion.choices[0].message.content;
      } else {
        throw new Error("Groq SDK not available");
      }
    } catch (sdkError) {
      console.warn(
        "Groq SDK failed, falling back to direct API call",
        sdkError
      );

      updateLoadingProgress(85, "Processing analysis results...");
      
      // Fallback to direct API call
      const apiResponse = await callGroqAPI(messages);
      aiResponse = apiResponse.choices[0].message.content;
    }

    updateLoadingProgress(95, "Formatting results...");

    console.log("Analysis response received");

    // Trim the response to start with the heading
    aiResponse = trimResponseToHeading(aiResponse);

    // Format the response with professional styling
    const formattedResponse = formatProfessionalResponse(aiResponse);

    const resultContainer = document.createElement("div");
    resultContainer.classList.add("chat-output");
    resultContainer.innerHTML = formattedResponse;
    document.getElementById("output").appendChild(resultContainer);

    // Add disclaimer at the bottom
    const disclaimer = document.createElement("div");
    disclaimer.className = "disclaimer";
    disclaimer.innerHTML =
      "Disclaimer: This analysis is for informational purposes only and does not constitute medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment.";
    document.getElementById("output").appendChild(disclaimer);
    
    updateLoadingProgress(100, "Analysis complete!");
  } catch (error) {
    console.error("Error with analysis:", error);

    updateLoadingProgress(100, "Error generating analysis");

    const errorContainer = document.createElement("div");
    errorContainer.classList.add("chat-output");
    errorContainer.style.backgroundColor = "#ffebee";
    errorContainer.innerHTML = `
       <h3>Error in Analysis:</h3>
       <p>Sorry, there was an error processing your request. Please try again later.</p>
       <p>Error details: ${error.message}</p>
     `;
    document.getElementById("output").appendChild(errorContainer);
  }
}

// Trim the response to start with the heading
function trimResponseToHeading(text) {
  const headingPattern = /Skin Cancer Analysis and Recommendations/i;
  const match = text.match(headingPattern);

  if (match) {
    // Return only the text starting from the heading
    return text.substring(match.index);
  }

  return text; // Return original if heading not found
}

// Format the response in a professional, clinical manner
function formatProfessionalResponse(text) {
  // Replace the main heading with a styled heading
  text = text.replace(
    /Skin Cancer Analysis and Recommendations/i,
    '<h1 class="analysis-header">Skin Cancer Analysis and Recommendations</h1>'
  );

  // Replace section headers with styled headers
  text = text.replace(
    /1\.\s+TREATMENT OPTIONS AND ADVICE:/gi,
    '<div class="analysis-section"><h2>1. Treatment Options and Advice</h2>'
  );
  text = text.replace(
    /2\.\s+CANCER SPREAD PREDICTION:/gi,
    '</div><div class="analysis-section"><h2>2. Cancer Spread Prediction</h2>'
  );
  text = text.replace(
    /3\.\s+CANCER PROGRESSION ASSESSMENT:/gi,
    '</div><div class="analysis-section"><h2>3. Cancer Progression Assessment</h2>'
  );

  // Add closing div for the last section
  text += "</div>";

  // Format bullet points
  text = text.replace(/- /g, "<li>");
  text = text.replace(/\n- /g, "</li>\n<li>");
  text = text.replace(/<li>(.*?)(?=<\/li>|$)/gs, "<li>$1</li>");

  // Wrap bullet point sections in ul tags
  text = text.replace(/(<li>.*?<\/li>)/gs, "<ul>$1</ul>");

  // Format conclusion if present
  text = text.replace(
    /Conclusion:(.*?)(?=<div|$)/gs,
    '<div class="conclusion"><strong>Conclusion:</strong>$1</div>'
  );

  // Replace any remaining # symbols in headings
  text = text.replace(
    /#{1,6}\s+(.*?)(?=\n|$)/g,
    '<div class="subheading">$1</div>'
  );

  // Bold important terms and subheadings
  text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

  // Format paragraphs
  text = text.replace(/\n\n/g, "</p><p>");

  return text;
}

// Load the model when the page is ready
window.onload = async () => {
  await loadModel();
  console.log("Checking Groq SDK availability...");

  // Test if Groq SDK is available
  if (typeof Groq === "undefined") {
    console.warn("Groq SDK not detected. Will use direct API calls instead.");
  } else {
    console.log("Groq SDK loaded successfully.");
  }
};
