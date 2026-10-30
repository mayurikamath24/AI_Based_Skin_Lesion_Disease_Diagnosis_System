const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.querySelector(".preview-placeholder");
const analyzeBtn = document.getElementById("analyzeBtn");

const imageStatus = document.getElementById("imageStatus");


/* ================= IMAGE UPLOAD ================= */

imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) {
        return;
    }

    const reader = new FileReader();

    reader.onload = function (event) {

        previewImage.src = event.target.result;
        previewImage.style.display = "block";

        if (previewPlaceholder) {
            previewPlaceholder.style.display = "none";
        }

        if (imageStatus) {
            imageStatus.textContent = "READY";
        }

        analyzeBtn.disabled = false;
        analyzeBtn.style.pointerEvents = "auto";
        analyzeBtn.style.opacity = "1";
        analyzeBtn.classList.add("active");
    };

    reader.readAsDataURL(file);
});


/* ================= EXPLANATION ================= */

let explainCounter = 0;

async function loadExplanation(result, token) {

    const box = document.getElementById("explanationText");

    if (!box) {
        return;
    }

    box.textContent =
        "Generating explanation... this can take up to a minute.";

    let httpStatus = "no response";

    try {

        const response = await fetch("/explain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                "class": result["class"],
                uncertain: result.prediction === "Uncertain",
                agree: result.agreement ? result.agreement.count : null
            })
        });

        httpStatus = response.status;

        const data = await response.json();

        /* Ignore if a newer analysis has started meanwhile */
        if (token !== explainCounter) {
            return;
        }

        if (!response.ok || !data.success) {
            throw new Error(data.message || "Explanation failed.");
        }

        box.textContent = data.explanation;

        if (data.fallback) {
            box.textContent +=
                "\n\n(Built from the curated knowledge graph only; " +
                "the language model was unavailable.)";
        }

    } catch (error) {

        console.error("Explanation error:", error);

        if (token === explainCounter) {
            box.textContent =
                "An explanation is not available right now.\n" +
                "[debug: HTTP " + httpStatus + " - " + error.message + "]";
        }
    }
}


/* ================= KNOWLEDGE GRAPH ================= */

const KG_COLORS = {
    condition: "#26e6a6",
    feature: "#38bdf8",
    risk: "#fbbf24",
    lookalike: "#f87171"
};

function kgWrap(text, max) {

    const words = text.split(" ");
    const lines = [];
    let line = "";

    for (const w of words) {
        if ((line + " " + w).trim().length > max && line) {
            lines.push(line);
            line = w;
        } else {
            line = (line + " " + w).trim();
        }
    }

    if (line) {
        lines.push(line);
    }

    return lines;
}

function kgEl(name, attrs) {

    const el = document.createElementNS("http://www.w3.org/2000/svg", name);

    for (const [k, v] of Object.entries(attrs || {})) {
        el.setAttribute(k, v);
    }

    return el;
}

function kgNode(svg, node, cx, cy, w, color, onClick) {

    const lines = kgWrap(node.label, Math.floor(w / 6.4));
    const h = Math.max(34, lines.length * 15 + 16);

    const isCenter = color === KG_COLORS.condition;

    const g = kgEl("g", onClick ? { "class": "kg-clickable" } : {});

    const rect = kgEl("rect", {
        x: cx - w / 2, y: cy - h / 2, width: w, height: h, rx: 10,
        fill: isCenter ? color : "#0f211e",
        stroke: color, "stroke-width": 1.5
    });
    g.appendChild(rect);

    lines.forEach((ln, i) => {
        const t = kgEl("text", {
            x: cx,
            y: cy - ((lines.length - 1) * 15) / 2 + i * 15 + 4,
            "text-anchor": "middle", "font-size": 11,
            "font-weight": isCenter ? 700 : 500,
            fill: isCenter ? "#04110e" : "#d7e3e0"
        });
        t.textContent = ln;
        g.appendChild(t);
    });

    if (node.shared_with && node.shared_with.length) {
        const title = kgEl("title", {});
        title.textContent = "Also a risk factor for: " + node.shared_with.join(", ");
        g.appendChild(title);
    }

    if (onClick) {
        g.addEventListener("click", onClick);
    }

    svg.appendChild(g);

    return { x: cx, y: cy, w: w, h: h };
}

function drawGraph(graph) {

    const box = document.getElementById("kgView");

    if (!box) {
        return;
    }

    box.innerHTML = "";

    const center = graph.nodes.find(n => n.id === graph.center);
    const features = graph.nodes.filter(n => n.kind === "feature");
    const risks = graph.nodes.filter(n => n.kind === "risk");
    const looks = graph.nodes.filter(
        n => n.kind === "condition" && n.id !== graph.center
    );

    const W = 760;
    const rows = Math.max(features.length, risks.length, 1);
    const topH = rows * 56 + 20;
    const H = topH + (looks.length ? 90 : 10);

    const svg = kgEl("svg", { viewBox: "0 0 " + W + " " + H });

    const cx = W / 2;
    const cy = topH / 2;

    /* place nodes first, then draw lines underneath */
    const edgesLayer = kgEl("g", {});
    svg.appendChild(edgesLayer);

    function line(a, b, color) {
        edgesLayer.appendChild(kgEl("line", {
            x1: a.x, y1: a.y, x2: b.x, y2: b.y,
            stroke: color, "stroke-opacity": 0.45, "stroke-width": 1.5
        }));
    }

    const c = kgNode(svg, center, cx, cy, 150, KG_COLORS.condition);

    const place = (list, x, color) => {
        list.forEach((n, i) => {
            const y = 10 + (topH - 20) * (i + 0.5) / list.length;
            const p = kgNode(svg, n, x, y, 210, color);
            line(c, p, color);
        });
    };

    place(features, 115, KG_COLORS.feature);
    place(risks, W - 115, KG_COLORS.risk);

    looks.forEach((n, i) => {
        const x = cx + (i - (looks.length - 1) / 2) * 190;
        const y = topH + 45;
        const p = kgNode(svg, n, x, y, 170, KG_COLORS.lookalike,
            () => loadGraph(n.code));
        line(c, p, KG_COLORS.lookalike);
    });

    box.appendChild(svg);
}

async function loadGraph(code) {

    const box = document.getElementById("kgView");

    if (!box || !code) {
        return;
    }

    try {

        const response = await fetch("/graph/" + encodeURIComponent(code));
        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.message || "Graph failed.");
        }

        drawGraph(data.graph);

    } catch (error) {

        console.error("Graph error:", error);

        box.innerHTML =
            "<p class='model-compare-empty'>The knowledge graph is not available right now.</p>";
    }
}


/* ================= ANALYZE ================= */

async function analyzeImage() {

    console.log("ANALYZE BUTTON CLICKED");

    if (!imageInput.files.length) {
        alert("Please select an image first.");
        return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "✦ Analyzing...";

    if (imageStatus) {
        imageStatus.textContent = "ANALYZING";
    }


    const prediction =
        document.getElementById("prediction");

    const confidence =
        document.getElementById("confidence");

    const confidenceFill =
        document.getElementById("confidenceFill");

    const responsibleModel =
        document.getElementById("responsibleModel");

    const resultMessage =
        document.getElementById("resultMessage");


    const comparisonBox =
        document.getElementById("modelComparison");

    const explanationBox =
        document.getElementById("explanationText");

    const explainToken = ++explainCounter;

    if (explanationBox) {
        explanationBox.textContent =
            "The explanation will appear after analysis.";
    }

    const agreementNote = document.getElementById("agreementNote");

    if (agreementNote) {
        agreementNote.style.display = "none";
        agreementNote.textContent = "";
        agreementNote.className = "agreement-note";
    }

    const kgBox = document.getElementById("kgView");

    if (kgBox) {
        kgBox.innerHTML =
            "<p class='model-compare-empty'>The related facts for the predicted condition will appear here.</p>";
    }

    prediction.textContent = "Analyzing...";
    confidence.textContent = "--%";
    confidenceFill.style.width = "0%";

    /* Clear stale results from any previous image */
    if (responsibleModel) {
        responsibleModel.textContent = "—";
    }

    if (comparisonBox) {
        comparisonBox.innerHTML =
            "<p class='model-compare-empty'>Analyzing with all models...</p>";
    }


    /* ================= SEND IMAGE TO FLASK ================= */

    const formData = new FormData();

    formData.append(
        "image",
        imageInput.files[0]
    );


    try {

        console.log("Sending image to Flask...");

        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        console.log("Flask response:", response.status);

        const result = await response.json();

        console.log("Result:", result);


        if (!response.ok || !result.success) {

            throw new Error(
                result.message || "Analysis failed."
            );
        }


        /* ================= DISPLAY RESULT ================= */

        prediction.textContent =
            result.prediction;

        confidence.textContent =
            result.confidence + "%";

        confidenceFill.style.width =
            result.confidence + "%";

        resultMessage.textContent =
            result.message;


        /* ===== RESPONSIBLE MODEL ===== */

        if (responsibleModel) {
            if (result.prediction === "Uncertain") {
                responsibleModel.textContent = "None (uncertain result)";
            } else if (result.responsible_model) {
                responsibleModel.textContent = result.responsible_model;
            } else {
                responsibleModel.textContent = "—";
            }
        }


        /* ===== LOW CONFIDENCE WARNING ===== */

        if (result.prediction === "Uncertain") {

            prediction.textContent =
                "Uncertain result";

            resultMessage.textContent =
                "Uncertain result. Please do not rely on this prediction. " +
                "The model's closest guess was " +
                (result.class_name || result.class) +
                " at " + result.confidence + "% confidence.";
        }


        /* ===== ALL-MODELS COMPARISON ===== */

        if (comparisonBox) {

            comparisonBox.innerHTML = "";

            if (result.all_models) {

                for (const [modelName, modelResult] of Object.entries(result.all_models)) {

                    const isWinner =
                        modelName === result.responsible_model &&
                        result.prediction !== "Uncertain";

                    const row = document.createElement("div");
                    row.className =
                        "model-compare-row" + (isWinner ? " winner" : "");

                    const nameEl = document.createElement("span");
                    nameEl.className = "model-compare-name";
                    nameEl.textContent = modelName + (isWinner ? " ★" : "");

                    const classEl = document.createElement("span");
                    classEl.className = "model-compare-class";
                    classEl.textContent = modelResult.class_name;

                    const confEl = document.createElement("span");
                    confEl.className = "model-compare-confidence";
                    confEl.textContent = modelResult.confidence + "%";

                    row.append(nameEl, classEl, confEl);
                    comparisonBox.appendChild(row);
                }

            } else {
                comparisonBox.innerHTML =
                    "<p class='model-compare-empty'>The server did not return per-model results.</p>";
            }
        }


        /* ===== MODEL AGREEMENT ===== */

        if (agreementNote && result.agreement) {

            const a = result.agreement;

            agreementNote.style.display = "block";

            if (a.count < a.warn_below) {
                agreementNote.className = "agreement-note warn";
                agreementNote.textContent =
                    "⚠ Only " + a.count + " of " + a.total +
                    " models agree on this class. Treat this result with extra caution.";
            } else {
                agreementNote.className = "agreement-note";
                agreementNote.textContent =
                    a.count + " of " + a.total + " models agree on this class.";
            }
        }


        /* Not awaited: show the prediction now, explanation follows */
        loadExplanation(result, explainToken);
        loadGraph(result["class"]);

        analyzeBtn.textContent =
            "✓ Analysis Complete";

        if (imageStatus) {
            imageStatus.textContent =
                "COMPLETE";
        }

    } catch (error) {

        console.error("Analysis error:", error);

        prediction.textContent =
            "Analysis Failed";

        confidence.textContent =
            "--%";

        confidenceFill.style.width =
            "0%";

        if (responsibleModel) {
            responsibleModel.textContent = "—";
        }

        if (comparisonBox) {
            comparisonBox.innerHTML =
                "<p class='model-compare-empty'>No results available.</p>";
        }

        if (explanationBox) {
            explanationBox.textContent = "No explanation available.";
        }

        if (agreementNote) {
            agreementNote.style.display = "none";
        }

        resultMessage.textContent =
            "Unable to analyze the image. Please check that the Flask server is running.";

        analyzeBtn.textContent =
            "✦ Analyze Image";

        if (imageStatus) {
            imageStatus.textContent =
                "ERROR";
        }

        alert(
            "Analysis failed: " +
            error.message
        );
    }

    analyzeBtn.disabled = false;
}


/* ================= NAVIGATION ================= */

function scrollToAnalysis() {

    const section =
        document.getElementById("analysis");

    if (section) {
        section.scrollIntoView({
            behavior: "smooth"
        });
    }
}