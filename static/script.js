function checkText() {
    let textInput = document.getElementById("textInput").value;

    fetch("/check", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: textInput })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerHTML =
            `<h3>Result:</h3>
             <p><strong>Text:</strong> ${data.text}</p>
             <p><strong>Classification:</strong> ${data.classification}</p>
             <p><strong>Confidence:</strong> ${data.confidence}</p>`;
    })
    .catch(error => {
        console.error("Error:", error);
    });
}
