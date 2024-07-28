document.getElementById('prediction-form').addEventListener('submit', function(e) {
    e.preventDefault(); // Prevent default form submission

    // Create FormData object
    var formData = new FormData(this);

    // Send data to the server
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Redirect to result page with prediction
        window.location.href = `/result?prediction=${data.prediction}`;
    })
    .catch(error => console.error('Error:', error));
});
