document.getElementById("searchBox").addEventListener("input", async function (e) {
    const query = e.target.value.trim();

    if (query.length === 0) {
        document.getElementById("suggestions").innerHTML = "";
        return;
    }

    // Call Backend API
    const response = await fetch(`http://127.0.0.1:8080/suggest?query=${encodeURIComponent(query)}`);
    const data = await response.json();

    // Display Suggestions
    const suggestionsDiv = document.getElementById("suggestions");
    suggestionsDiv.innerHTML = data.suggestions.map(title => `<div>${title}</div>`).join('');
});