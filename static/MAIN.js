document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const preview = document.getElementById("preview");
  const classifyBtn = document.getElementById("classify-btn");
  const modelSelect = document.getElementById("model-select");

  // Drag & Drop
  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleImage(fileInput.files[0]);
    }
  });

  // File input
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) handleImage(e.target.files[0]);
  });

  function handleImage(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload an image file.");
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      preview.style.display = "block";

      // Reset result
      document.getElementById("result-section").style.display = "none";
      document.getElementById("predicted-class").textContent = "-";
      document.getElementById("confidence-score").textContent = "-";
      document.getElementById("prediction-time").textContent = "-";
      document.getElementById("heatmap").src = "";
    };
    reader.readAsDataURL(file);
  }

  classifyBtn.addEventListener("click", () => {
    const file = fileInput.files[0];
    const modelChoice = modelSelect.value;

    if (!file) {
      alert("Please upload an image.");
      return;
    }
    if (!modelChoice) {
      alert("Please select a model.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_choice", modelChoice);

    fetch("/predict", { method: "POST", body: formData })  // ✅ relative path
      .then(res => {
        if (!res.ok) throw new Error("Server error: " + res.status);
        return res.json();
      })
      .then(data => {
        // Show results
        document.getElementById("result-section").style.display = "block";
        document.getElementById("predicted-class").textContent = data.predicted_class;
        document.getElementById("confidence-score").textContent = data.confidence_score + "%";
        document.getElementById("prediction-time").textContent = data.prediction_time + " s";

        // Labels as short indices (1, 2, 3…) but tooltip shows full class
        const shortLabels = data.class_names.map((_, idx) => (idx + 1).toString());

        const ctx = document.getElementById("probs-chart").getContext("2d");
        if (window.probChart) window.probChart.destroy();

        window.probChart = new Chart(ctx, {
          type: "bar",
          data: {
            labels: shortLabels,
            datasets: [{
              label: "Probability (%)",
              data: data.class_probs.map(p => (p * 100).toFixed(2)),
              backgroundColor: "rgba(33, 22, 192, 0.6)"
            }]
          },
          options: {
            scales: { y: { beginAtZero: true, max: 100 } },
            plugins: {
              tooltip: {
                callbacks: {
                  title: (items) => data.class_names[items[0].dataIndex],
                  label: (context) => context.raw + " %"
                }
              },
              legend: { display: false }
            }
          }
        });

        if (data.heatmap) {
          document.getElementById("heatmap").src = data.heatmap;
        }
      })
      .catch(err => {
        console.error("Fetch error:", err);
        alert("Prediction failed: " + err.message);
      });
  });
});
