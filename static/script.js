/**
 * IPL Match Predictor — Client-side Logic
 * ========================================
 */

document.addEventListener("DOMContentLoaded", () => {
    initParticles();
    initFormLogic();
});


/* ────────────────────────────────────────────
   Background Particles
   ──────────────────────────────────────────── */
function initParticles() {
    const container = document.getElementById("bgParticles");
    const count = 40;

    for (let i = 0; i < count; i++) {
        const p = document.createElement("div");
        p.classList.add("particle");
        p.style.left = Math.random() * 100 + "%";
        p.style.animationDuration = 8 + Math.random() * 14 + "s";
        p.style.animationDelay = Math.random() * 10 + "s";
        p.style.width = p.style.height = (2 + Math.random() * 3) + "px";
        p.style.opacity = 0.15 + Math.random() * 0.25;
        container.appendChild(p);
    }
}


/* ────────────────────────────────────────────
   Form Logic
   ──────────────────────────────────────────── */
function initFormLogic() {
    const team1Select = document.getElementById("team1");
    const team2Select = document.getElementById("team2");
    const tossWinnerSelect = document.getElementById("tossWinner");
    const form = document.getElementById("matchForm");

    // Update toss winner options when teams change
    function updateTossOptions() {
        const t1 = team1Select.value;
        const t2 = team2Select.value;

        tossWinnerSelect.innerHTML = '<option value="" disabled selected>Select toss winner</option>';

        if (t1) {
            const opt = document.createElement("option");
            opt.value = t1;
            opt.textContent = t1;
            tossWinnerSelect.appendChild(opt);
        }
        if (t2 && t2 !== t1) {
            const opt = document.createElement("option");
            opt.value = t2;
            opt.textContent = t2;
            tossWinnerSelect.appendChild(opt);
        }
    }

    team1Select.addEventListener("change", updateTossOptions);
    team2Select.addEventListener("change", updateTossOptions);

    // Handle form submission
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        await handlePredict();
    });
}


/* ────────────────────────────────────────────
   Prediction Handler
   ──────────────────────────────────────────── */
async function handlePredict() {
    const btn = document.getElementById("predictBtn");
    const btnContent = btn.querySelector(".btn-content");
    const btnLoading = btn.querySelector(".btn-loading");
    const errorAlert = document.getElementById("errorAlert");
    const resultsSection = document.getElementById("resultsSection");

    // Gather form data
    const team1 = document.getElementById("team1").value;
    const team2 = document.getElementById("team2").value;
    const venue = document.getElementById("venue").value;
    const tossWinner = document.getElementById("tossWinner").value;
    const tossDecision = document.querySelector('input[name="toss_decision"]:checked');

    if (!tossDecision) {
        showError("Please select the toss decision (Bat or Field).");
        return;
    }

    // Validate
    if (team1 === team2) {
        showError("Team 1 and Team 2 cannot be the same team.");
        return;
    }

    // Retrieve and validate Playing XI selections
    let squad1 = [];
    let squad2 = [];
    if (window.getSelectedSquads) {
        const squads = window.getSelectedSquads();
        squad1 = squads.squad1 || [];
        squad2 = squads.squad2 || [];
    }

    if (squad1.length !== 11) {
        showError(`Please select exactly 11 players for ${team1 || 'Team 1'} (currently ${squad1.length} selected).`);
        return;
    }
    if (squad2.length !== 11) {
        showError(`Please select exactly 11 players for ${team2 || 'Team 2'} (currently ${squad2.length} selected).`);
        return;
    }

    // Show loading
    btnContent.style.display = "none";
    btnLoading.style.display = "flex";
    btn.disabled = true;
    errorAlert.style.display = "none";
    resultsSection.style.display = "none";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                team1,
                team2,
                venue,
                toss_winner: tossWinner,
                toss_decision: tossDecision.value,
                squad1,
                squad2
            }),
        });

        const data = await response.json();

        if (!response.ok) {
            showError(data.error || "Prediction failed. Please try again.");
            return;
        }

        displayResults(data);
    } catch (err) {
        showError("Network error. Please make sure the server is running.");
        console.error(err);
    } finally {
        btnContent.style.display = "flex";
        btnLoading.style.display = "none";
        btn.disabled = false;
    }
}


/* ────────────────────────────────────────────
   Display Results
   ──────────────────────────────────────────── */
function displayResults(data) {
    const section = document.getElementById("resultsSection");

    // Winner
    document.getElementById("winnerName").textContent = data.predicted_winner;

    // Confidence
    const confEl = document.getElementById("confidenceValue");
    confEl.textContent = data.confidence;
    confEl.className = "conf-value " + data.confidence.toLowerCase();

    // Win probabilities
    document.getElementById("probTeam1Name").textContent = data.team1;
    document.getElementById("probTeam2Name").textContent = data.team2;
    document.getElementById("probPct1").textContent = data.team1_win_prob.toFixed(1) + "%";
    document.getElementById("probPct2").textContent = data.team2_win_prob.toFixed(1) + "%";

    // Animate probability bars after a brief delay
    setTimeout(() => {
        document.getElementById("probBar1").style.width = data.team1_win_prob + "%";
        document.getElementById("probBar2").style.width = data.team2_win_prob + "%";
    }, 100);

    // Head-to-Head
    const h2h = data.h2h_record;
    document.getElementById("h2hT1Name").textContent = data.team1;
    document.getElementById("h2hT2Name").textContent = data.team2;
    document.getElementById("h2hTotal").textContent = h2h.total_matches + " matches";

    // Parse h2h wins (keys are dynamic)
    const h2hKeys = Object.keys(h2h);
    let t1Wins = 0, t2Wins = 0;
    for (const key of h2hKeys) {
        if (key.includes(data.team1) && key.includes("wins")) t1Wins = h2h[key];
        if (key.includes(data.team2) && key.includes("wins")) t2Wins = h2h[key];
    }
    document.getElementById("h2hT1Wins").textContent = t1Wins;
    document.getElementById("h2hT2Wins").textContent = t2Wins;

    // Scoreboard
    if (data.scoreboard) {
        document.getElementById("sbInn1Team").textContent = data.scoreboard.inn1.team;
        document.getElementById("sbInn1Score").textContent = data.scoreboard.inn1.runs + "/" + data.scoreboard.inn1.wickets;
        document.getElementById("sbInn2Team").textContent = data.scoreboard.inn2.team;
        document.getElementById("sbInn2Score").textContent = data.scoreboard.inn2.runs + "/" + data.scoreboard.inn2.wickets;

        // Dynamic Player Scores under predicted scorecard
        document.getElementById("sbInn1TopBat").textContent = "🏏 " + (data.scoreboard.inn1.top_bat || "");
        document.getElementById("sbInn1TopBowl").textContent = "🎳 " + (data.scoreboard.inn1.top_bowl || "");
        document.getElementById("sbInn2TopBat").textContent = "🏏 " + (data.scoreboard.inn2.top_bat || "");
        document.getElementById("sbInn2TopBowl").textContent = "🎳 " + (data.scoreboard.inn2.top_bowl || "");
    }

    // Player of the Match
    if (data.potm) {
        document.getElementById("potmName").textContent = data.potm;
    }

    // Player Analysis
    const playerContainer = document.getElementById("playerAnalysis");
    playerContainer.innerHTML = "";

    if (data.player_analysis) {
        for (const [team, players] of Object.entries(data.player_analysis)) {
            const section = document.createElement("div");
            section.className = "player-team-section";

            const label = document.createElement("div");
            label.className = "player-team-label";
            label.textContent = team;
            section.appendChild(label);

            // Top scorer
            const ts = players.likely_top_scorer;
            const scorerRow = createPlayerRow("🏏", ts.name, `avg ${ts.recent_avg_runs} runs`);
            section.appendChild(scorerRow);

            // Top wicket taker
            const tw = players.likely_top_wicket_taker;
            const bowlerRow = createPlayerRow("🎳", tw.name, `avg ${tw.recent_avg_wickets} wkts`);
            section.appendChild(bowlerRow);

            playerContainer.appendChild(section);
        }
    }

    // Match details
    const detailsGrid = document.getElementById("matchDetails");
    detailsGrid.innerHTML = "";

    const details = [
        { icon: "⚔️", label: "Match", value: `${data.team1} vs ${data.team2}` },
        { icon: "🏟️", label: "Venue", value: data.venue },
        { icon: "🪙", label: "Toss Winner", value: data.toss_winner },
        { icon: "🎯", label: "Chose to", value: data.toss_decision.charAt(0).toUpperCase() + data.toss_decision.slice(1) },
    ];

    for (const d of details) {
        const item = document.createElement("div");
        item.className = "detail-item";
        item.innerHTML = `
            <span class="detail-icon">${d.icon}</span>
            <div class="detail-content">
                <span class="detail-label">${d.label}</span>
                <span class="detail-value">${d.value}</span>
            </div>
        `;
        detailsGrid.appendChild(item);
    }

    // Show results with animation
    section.style.display = "block";
    section.scrollIntoView({ behavior: "smooth", block: "start" });

    // Trigger confetti
    createConfetti();
}


/* ────────────────────────────────────────────
   Helpers
   ──────────────────────────────────────────── */
function createPlayerRow(emoji, name, stat) {
    const row = document.createElement("div");
    row.className = "player-row";
    row.innerHTML = `
        <span class="player-emoji">${emoji}</span>
        <span class="player-name">${name}</span>
        <span class="player-stat">${stat}</span>
    `;
    return row;
}

function showError(message) {
    const alert = document.getElementById("errorAlert");
    document.getElementById("errorText").textContent = message;
    alert.style.display = "flex";
    alert.scrollIntoView({ behavior: "smooth", block: "center" });
}

function createConfetti() {
    const container = document.getElementById("confetti");
    container.innerHTML = "";

    const colors = ["#6366f1", "#8b5cf6", "#a855f7", "#06b6d4", "#f59e0b", "#10b981", "#ef4444", "#ec4899"];

    for (let i = 0; i < 50; i++) {
        const piece = document.createElement("div");
        piece.className = "confetti-piece";
        piece.style.left = Math.random() * 100 + "%";
        piece.style.animationDuration = (1.5 + Math.random() * 2) + "s";
        piece.style.animationDelay = Math.random() * 0.6 + "s";
        piece.style.background = colors[Math.floor(Math.random() * colors.length)];
        piece.style.borderRadius = Math.random() > 0.5 ? "50%" : "2px";
        piece.style.width = (5 + Math.random() * 8) + "px";
        piece.style.height = (5 + Math.random() * 8) + "px";
        container.appendChild(piece);
    }

    // Clean up confetti after animation
    setTimeout(() => {
        container.innerHTML = "";
    }, 4000);
}
