window.onload = function () {
    console.log("✅ Script.js loaded!");

    // DOM Elements
    const lstmBtn = document.getElementById("lstmBtn");
    const bilstmBtn = document.getElementById("bilstmBtn");
    const gruBtn = document.getElementById("gruBtn");
    const sendBtn = document.getElementById("send");
    const inputText = document.getElementById("send_query");
    const word1Btn = document.getElementById("word1Btn");
    const word2Btn = document.getElementById("word2Btn");
    const word3Btn = document.getElementById("word3Btn");

    let selectedModel = "lstm"; // Default model

    // Check if all elements exist
    if (!lstmBtn || !bilstmBtn || !gruBtn || !sendBtn || !inputText || !word1Btn || !word2Btn || !word3Btn) {
        console.error("❌ One or more elements are missing!");
        return;
    }

    // Model selection event listeners
    lstmBtn.addEventListener("click", function () {
        selectedModel = "lstm";
        console.log("🔘 LSTM model selected!");
        lstmBtn.classList.add("selected");
        bilstmBtn.classList.remove("selected");
        gruBtn.classList.remove("selected");
    });

    bilstmBtn.addEventListener("click", function () {
        selectedModel = "bilstm";
        console.log("🔘 BiLSTM model selected!");
        bilstmBtn.classList.add("selected");
        lstmBtn.classList.remove("selected");
        gruBtn.classList.remove("selected");
    });

    gruBtn.addEventListener("click", function () {
        selectedModel = "gru";
        console.log("🔘 GRU model selected!");
        gruBtn.classList.add("selected");
        lstmBtn.classList.remove("selected");
        bilstmBtn.classList.remove("selected");
    });

    // Submit button event listener
    sendBtn.addEventListener("click", function () {
        console.log("🔘 Send button clicked!");

        const text = inputText.value.trim();

        if (!text) {
            alert("Please enter some text.");
            return;
        }

        console.log("📡 Sending request to /predict:", { model: selectedModel, phrase: text });

        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: selectedModel, phrase: text })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                console.error("❌ Error in response:", data.error);
                alert("Error: " + data.error);
            } else {
                console.log("✅ Predictions:", data.predictions);
                // Update the predicted words in the buttons
                word1Btn.innerText = data.predictions[0] || "WORD 1";
                word2Btn.innerText = data.predictions[1] || "WORD 2";
                word3Btn.innerText = data.predictions[2] || "WORD 3";
            }
        })
        .catch(error => {
            console.error("❌ Fetch error:", error);
            alert("An error occurred while fetching the prediction.");
        });
    });

    // Function to add a word to the textarea and trigger a new prediction
 // Function to add a word to the textarea and trigger a new prediction
function addWordToTextarea(word) {
    const textarea = document.getElementById("send_query");
    const currentText = textarea.value.trim();

    // Append the clicked word to the textarea
    if (currentText) {
        textarea.value = `${currentText} ${word}`; // Add a space before the word
    } else {
        textarea.value = word; // If the textarea is empty, just add the word
    }

    // Trigger the auto-resize functionality
    autoResizeTextarea(textarea);

    // Trigger the Send button click to make a new prediction
    document.getElementById("send").click();
}

// Function to auto-resize the textarea
function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto'; // Reset height
    textarea.style.height = (textarea.scrollHeight > 200 ? 200 : textarea.scrollHeight) + 'px'; // Limit max height
}

// Add event listeners to the predicted word buttons
if (word1Btn && word2Btn && word3Btn) {
    word1Btn.addEventListener("click", function () {
        const word = word1Btn.innerText;
        if (word !== "WORD 1") { // Ensure the button has a valid word
            addWordToTextarea(word);
        }
    });

    word2Btn.addEventListener("click", function () {
        const word = word2Btn.innerText;
        if (word !== "WORD 2") { // Ensure the button has a valid word
            addWordToTextarea(word);
        }
    });

    word3Btn.addEventListener("click", function () {
        const word = word3Btn.innerText;
        if (word !== "WORD 3") { // Ensure the button has a valid word
            addWordToTextarea(word);
        }
    });
} else {
    console.error("❌ Predicted word buttons are missing!");
}

// Textarea auto-resize and Shift+Enter functionality
const textarea = document.getElementById("send_query");

textarea.addEventListener("input", function () {
    autoResizeTextarea(textarea);
});

textarea.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && event.shiftKey) {
        // Allow Shift+Enter to create a new line
        event.preventDefault(); // Prevent default behavior (e.g., form submission)
        const startPos = textarea.selectionStart;
        const endPos = textarea.selectionEnd;
        const value = textarea.value;

        // Insert a new line at the cursor position
        textarea.value = value.substring(0, startPos) + "\n" + value.substring(endPos);

        // Move the cursor to the new line
        textarea.selectionStart = textarea.selectionEnd = startPos + 1;

        // Trigger the input event to auto-resize
        const inputEvent = new Event("input", { bubbles: true });
        textarea.dispatchEvent(inputEvent);
    } else if (event.key === "Enter" && !event.shiftKey) {
        // Prevent default behavior for Enter key (e.g., form submission)
        event.preventDefault();

        // Trigger the Send button click
        document.getElementById("send").click();
    }
});

    // Sidebar functionality
    const sidebarOpenBtn = document.getElementById("sidebarbtnopen");
    const sidebarCloseBtn = document.getElementById("sidebarbtnclose");
    const sidebar = document.getElementById("sidebar");
    const sidebarBtnMobile = document.getElementById("sidebarbtnmobile");
    const sidebarCloseBtnMobile = document.getElementById("sidebarbtnclosemobile");

    sidebarOpenBtn.addEventListener("click", function () {
        sidebar.style.width = '260px';
        sidebar.classList.add('p-2');
        sidebarOpenBtn.classList.add('d-md-none');
        setTimeout(function () {
            sidebar.style.minWidth = '260px';
        }, 200);
    });

    sidebarCloseBtn.addEventListener("click", function () {
        sidebar.style.minWidth = '0';
        sidebar.style.width = '0';
        sidebar.classList.remove('p-2');
        sidebarOpenBtn.classList.remove('d-md-none');
        sidebarOpenBtn.classList.add('d-md-block');
    });

    sidebarBtnMobile.addEventListener("click", function () {
        sidebar.style.display = 'block';
        sidebar.style.width = '260px';
        sidebar.style.minWidth = '260px';
        sidebar.classList.add('p-2');
        sidebarCloseBtnMobile.classList.remove('d-none');
    });

    sidebarCloseBtnMobile.addEventListener("click", function () {
        sidebar.style.display = 'none';
        sidebar.style.width = '0';
        sidebar.style.minWidth = '0';
        sidebar.classList.remove('p-2');
        sidebarCloseBtnMobile.classList.add('d-none');
    });

    // Window resize event
    window.onresize = function () {
        if (window.innerWidth <= 768) {
            sidebar.style.display = 'none';
            sidebar.style.position = 'absolute';
            sidebar.style.zIndex = 9999999;
            sidebar.style.minWidth = '0';
            sidebar.style.width = '0';
            sidebar.classList.remove('p-2');
            sidebarCloseBtn.style.display = 'none';
        } else {
            sidebar.style.display = 'block';
            sidebar.style.position = 'static';
            sidebar.style.minWidth = '260px';
            sidebar.style.width = '260px';
            sidebar.style.zIndex = 0;
            sidebar.classList.add('p-2');
            sidebarCloseBtn.style.display = 'block';
        }
    };

    // Initial check for mobile view
    if (window.innerWidth <= 768) {
        sidebar.style.display = 'none';
        sidebar.style.position = 'absolute';
        sidebar.style.zIndex = 999999;
        sidebar.style.minWidth = '0';
        sidebar.style.width = '0';
        sidebar.classList.remove('p-2');
        sidebarCloseBtn.style.display = 'none';
    }

    // Try Now button functionality
    document.getElementById('tryNowBtn').addEventListener('click', function () {
        let inputField = document.getElementById('send_query');
        inputField.style.border = "2px solid white";
        inputField.style.boxShadow = "0 0 10px rgba(255, 255, 255, 0.5)";
        inputField.focus();
        setTimeout(() => {
            inputField.style.border = "";
            inputField.style.boxShadow = "";
        }, 2000);
    });
};

//   F L O W
//1. User types "I am going to".
//2. Selects GRU model.
//3. Clicks Send → Sends { model: "gru", phrase: "I am going to" } to /predict.
//4. Server responds with { predictions: ["school", "market", "office"] }.
//5. Buttons update:
//   word1Btn → "school"
//   word2Btn → "market"
//   word3Btn → "office"