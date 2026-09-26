import { useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const fileInputRef = useRef(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  const [chatMessages, setChatMessages] = useState([
    {
      role: "assistant",
      text:
        "Hello! Upload an image above and I'll help you understand the AI analysis.",
    },
  ]);

  // ================================
  // IMAGE SELECTION
  // ================================

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];

    if (!file) return;

    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError("");
  };

  // ================================
  // IMAGE ANALYSIS
  // ================================

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError("Please select a skin image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(
        `${API_URL}/api/v1/upload`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(
          `Server returned ${response.status}`
        );
      }

      const data = await response.json();

      setResult(data);

      // Add analysis context to the chat
      setChatMessages([
        {
          role: "assistant",
          text: data.ood
            ? "The image has been analyzed. The system marked it as Unsupported / Uncertain because it did not meet the supported-class criteria."
            : `The image has been analyzed. The system predicted ${formatClassName(
              data.prediction
            )} with ${data.confidence}% confidence.`,
        },
      ]);
    } catch (err) {
      console.error(err);

      setError(
        "Unable to connect to the AI backend. Make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  // ================================
  // RESET ANALYSIS
  // ================================

  const resetAnalysis = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError("");

    setChatMessages([
      {
        role: "assistant",
        text:
          "Hello! Upload an image above and I'll help you understand the AI analysis.",
      },
    ]);

    setChatInput("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // ================================
  // CLASS NAME FORMATTER
  // ================================

  const formatClassName = (name) => {
    const names = {
      ACK: "Actinic Keratosis",
      BCC: "Basal Cell Carcinoma",
      MEL: "Melanoma",
      NEV: "Nevus",
      SCC: "Squamous Cell Carcinoma",
      SEK: "Seborrheic Keratosis",
    };

    return names[name] || name;
  };

  // ================================
  // CHAT
  // ================================

  const handleChat = async () => {
    const message = chatInput.trim();

    if (!message || chatLoading) {
      return;
    }

    const userMessage = {
      role: "user",
      text: message,
    };

    setChatMessages((previous) => [
      ...previous,
      userMessage,
    ]);

    setChatInput("");
    setChatLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/api/v1/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: message,

            prediction:
              result?.prediction || null,

            confidence:
              result?.confidence ?? null,

            ood:
              result?.ood ?? null,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          `Server returned ${response.status}`
        );
      }

      const data = await response.json();

      setChatMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          text: data.response,
        },
      ]);
    } catch (err) {
      console.error(err);

      setChatMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          text:
            "Unable to connect to the AI Assistant. Please make sure the FastAPI backend is running.",
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  // ================================
  // ENTER KEY FOR CHAT
  // ================================

  const handleChatKeyDown = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      handleChat();
    }
  };

  // ================================
  // UI
  // ================================

  return (
    <div className="app">

      {/* ================= HEADER ================= */}

      <header className="header">

        <div className="brand">

          <div className="brand-icon">
            ✦
          </div>

          <div>
            <h1>
              Dermatology AI Agents
            </h1>

            <p>
              AI-assisted dermatology image analysis
            </p>
          </div>

        </div>

        <div className="status">

          <span className="status-dot"></span>

          AI System

        </div>

      </header>


      {/* ================= MAIN ================= */}

      <main>

        {/* ================= HERO ================= */}

        <section className="hero-section">

          <div className="hero-content">

            <div className="badge">
              AI-POWERED DERMATOLOGY
            </div>

            <h2>
              Intelligent skin image
              <span> analysis</span>
            </h2>

            <p className="hero-description">
              Upload a skin lesion image for AI-assisted
              preliminary analysis with prediction,
              confidence, uncertainty screening and
              explainable AI.
            </p>

          </div>


          {/* ================= UPLOAD CARD ================= */}

          <div className="upload-card">

            {!preview ? (

              <div
                className="drop-zone"
                onClick={() =>
                  fileInputRef.current?.click()
                }
              >

                <div className="upload-icon">
                  ↑
                </div>

                <h3>
                  Upload a skin image
                </h3>

                <p>
                  Select a JPG, JPEG or PNG image
                </p>

                <button
                  type="button"
                  className="primary-button"
                  onClick={(event) => {
                    event.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                >
                  Choose Image
                </button>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/png,image/jpeg,image/jpg"
                  onChange={handleFileChange}
                  hidden
                />

              </div>

            ) : (

              <div className="preview-area">

                <div className="image-wrapper">

                  <img
                    src={preview}
                    alt="Selected skin image"
                  />

                </div>

                <div className="file-name">
                  {selectedFile?.name}
                </div>

                <div className="action-row">

                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() =>
                      fileInputRef.current?.click()
                    }
                  >
                    Change Image
                  </button>

                  <button
                    type="button"
                    className="primary-button"
                    onClick={handleAnalyze}
                    disabled={loading}
                  >
                    {loading
                      ? "Analyzing..."
                      : "Analyze Image"}
                  </button>

                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/png,image/jpeg,image/jpg"
                  onChange={handleFileChange}
                  hidden
                />

              </div>

            )}

          </div>

        </section>


        {/* ================= ERROR ================= */}

        {error && (

          <div className="error-message">

            <strong>
              Connection Error
            </strong>

            <p>
              {error}
            </p>

          </div>

        )}


        {/* ================= RESULT ================= */}

        {result && (

          <section className="results-section">

            <div className="section-heading">

              <div>

                <div className="badge">
                  AI ANALYSIS
                </div>

                <h2>
                  Analysis Result
                </h2>

              </div>

              <button
                type="button"
                className="secondary-button"
                onClick={resetAnalysis}
              >
                New Analysis
              </button>

            </div>


            {/* ================= OOD RESULT ================= */}

            {result.ood ? (

              <div className="uncertain-card">

                <div className="uncertain-icon">
                  !
                </div>

                <div>

                  <h3>
                    Unsupported / Uncertain
                  </h3>

                  <p>
                    The uploaded image does not meet
                    the system's confidence and similarity
                    criteria for the supported dermatology
                    classes.
                  </p>

                  <p className="disclaimer">
                    This system supports preliminary
                    research analysis only and is not a
                    medical diagnosis.
                  </p>

                </div>

              </div>

            ) : (

              <div className="prediction-card">

                <div className="prediction-main">

                  <span className="result-label">
                    Predicted Class
                  </span>

                  <h3>
                    {formatClassName(
                      result.prediction
                    )}
                  </h3>

                  <span className="class-code">
                    {result.prediction}
                  </span>

                </div>


                <div className="confidence-box">

                  <span className="result-label">
                    Confidence
                  </span>

                  <strong>
                    {result.confidence}%
                  </strong>

                </div>

              </div>

            )}


            {/* ================= METRICS ================= */}

            <div className="metrics-grid">

              <div className="metric-card">

                <span>
                  Feature Similarity
                </span>

                <strong>
                  {result.similarity}
                </strong>

              </div>


              <div className="metric-card">

                <span>
                  Prediction Entropy
                </span>

                <strong>
                  {result.entropy}
                </strong>

              </div>


              <div className="metric-card">

                <span>
                  OOD Status
                </span>

                <strong
                  className={
                    result.ood
                      ? "ood-text"
                      : "supported-text"
                  }
                >
                  {result.ood
                    ? "OOD"
                    : "Supported"}
                </strong>

              </div>

            </div>


            {/* ================= PROBABILITIES ================= */}

            <div className="probability-card">

              <h3>
                Class Probabilities
              </h3>

              <div className="probability-list">

                {Object.entries(
                  result.probabilities || {}
                ).map(
                  ([className, probability]) => (

                    <div
                      className="probability-row"
                      key={className}
                    >

                      <div className="probability-label">

                        <span>
                          {className}
                        </span>

                        <span>
                          {probability}%
                        </span>

                      </div>

                      <div className="progress-bar">

                        <div
                          className="progress-fill"
                          style={{
                            width: `${probability}%`,
                          }}
                        ></div>

                      </div>

                    </div>

                  )
                )}

              </div>

            </div>

            {/* EXPLAINABLE AI / GRAD-CAM */}
            {result.gradcam_available && result.gradcam_url && (
              <div className="gradcam-card">

                <div className="gradcam-header">

                  <div>
                    <div className="badge">
                      EXPLAINABLE AI
                    </div>

                    <h3>
                      Grad-CAM Visualization
                    </h3>

                    <p>
                      The highlighted regions show areas that
                      contributed to the model's prediction.
                    </p>
                  </div>

                </div>

                <div className="gradcam-content">

                  {/* Original Image */}
                  <div className="gradcam-panel">

                    <span className="gradcam-label">
                      Original Image
                    </span>

                    <img
                      src={preview}
                      alt="Original dermatology image"
                      className="gradcam-image"
                    />

                  </div>


                  {/* Grad-CAM */}
                  <div className="gradcam-panel">

                    <span className="gradcam-label">
                      Grad-CAM Heatmap
                    </span>

                    <img
                      src={`${API_URL}${result.gradcam_url}`}
                      alt="Grad-CAM explanation"
                      className="gradcam-image"
                    />

                  </div>

                </div>

                <div className="gradcam-note">
                  <strong>How to interpret:</strong>{" "}
                  Warmer highlighted regions indicate areas that
                  contributed more strongly to the model's prediction.
                  This visualization explains model attention and does
                  not prove that the highlighted region is clinically
                  diagnostic.
                </div>

              </div>
            )}

            {/* ================= DISCLAIMER ================= */}

            <div className="disclaimer-card">

              <strong>
                ⚠ Research / Decision Support
              </strong>

              <p>
                This AI system provides preliminary
                image analysis and should not be used
                as a substitute for professional medical
                examination, diagnosis or treatment.
              </p>

            </div>

          </section>

        )}


        {/* ================= AI ASSISTANT ================= */}

        <section className="assistant-section">

          <div className="assistant-header">

            <div className="assistant-icon">
              ✦
            </div>

            <div>

              <div className="badge">
                AI ASSISTANT
              </div>

              <h2>
                Dermatology AI Assistant
              </h2>

              <p>
                Ask questions about your symptoms,
                image analysis or general skin concerns.
              </p>

            </div>

          </div>


          <div className="chat-placeholder">

            {/* ================= CHAT MESSAGES ================= */}

            <div className="chat-messages">

              {chatMessages.map(
                (message, index) => (

                  <div
                    className={`chat-message ${message.role === "user"
                      ? "user-message"
                      : ""
                      }`}
                    key={index}
                  >

                    <div className="assistant-avatar">

                      {message.role === "user"
                        ? "U"
                        : "✦"}

                    </div>

                    <div>

                      <strong>
                        {message.role === "user"
                          ? "You"
                          : "Dermatology AI Assistant"}
                      </strong>

                      <p>
                        {message.text}
                      </p>

                    </div>

                  </div>

                )
              )}


              {/* ================= THINKING ================= */}

              {chatLoading && (

                <div className="chat-message">

                  <div className="assistant-avatar">
                    ✦
                  </div>

                  <div>

                    <strong>
                      Dermatology AI Assistant
                    </strong>

                    <p>
                      Thinking...
                    </p>

                  </div>

                </div>

              )}

            </div>


            {/* ================= CHAT INPUT ================= */}

            <div className="chat-input-row">

              <input
                type="text"
                value={chatInput}
                onChange={(event) =>
                  setChatInput(event.target.value)
                }
                onKeyDown={handleChatKeyDown}
                placeholder="Ask about your skin symptoms..."
                disabled={chatLoading}
              />

              <button
                type="button"
                onClick={handleChat}
                disabled={
                  chatLoading ||
                  !chatInput.trim()
                }
              >
                {chatLoading
                  ? "Sending..."
                  : "Send"}
              </button>

            </div>


            <small>
              AI-assisted information only • Not a medical diagnosis.
            </small>

          </div>

        </section>

      </main>


      {/* ================= FOOTER ================= */}

      <footer>

        <p>
          Dermatology AI Agents • Research Prototype
        </p>

        <p>
          AI-assisted analysis • Not a medical diagnosis
        </p>

      </footer>

    </div>
  );
}

export default App;