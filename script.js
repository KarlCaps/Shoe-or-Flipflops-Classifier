let model;

let imageElement =
    document.getElementById("imagePreview");

let imageUpload =
    document.getElementById("imageUpload");

let predictButton =
    document.getElementById("predictButton");

let predictionText =
    document.getElementById("prediction");

let loadingMessage =
    document.getElementById("loadingMessage");

let sneakersBar =
    document.getElementById("sneakersBar");

let flipflopsBar =
    document.getElementById("flipflopsBar");

let sneakersConfidence =
    document.getElementById("sneakersConfidence");

let flipflopsConfidence =
    document.getElementById("flipflopsConfidence");


// ------------------------------------
// 1. Load the TensorFlow.js model
// ------------------------------------

async function loadModel() {

    try {

        loadingMessage.textContent =
            "Loading machine learning model...";

        /*
         * model.json contains the model structure.
         * The .bin files contain the model weights.
         */

        model = await tf.loadLayersModel(
            "model/model.json"
        );


        loadingMessage.textContent =
            "Model loaded successfully!";

        predictButton.disabled = false;

    }

    catch (error) {

        console.error(error);

        loadingMessage.textContent =
            "Could not load the model.";

    }
}


// ------------------------------------
// 2. Select an image
// ------------------------------------

imageUpload.addEventListener(
    "change",
    function(event) {

        const file =
            event.target.files[0];


        if (!file) {

            return;

        }


        const reader =
            new FileReader();


        reader.onload =
            function(e) {

                imageElement.src =
                    e.target.result;

                imageElement.style.display =
                    "block";

                predictionText.textContent =
                    "Ready to predict.";

            };


        reader.readAsDataURL(file);

    }
);


// ------------------------------------
// 3. Predict
// ------------------------------------

predictButton.addEventListener(
    "click",
    async function() {

        if (!model) {

            alert("Model is not loaded yet.");

            return;

        }


        if (!imageElement.src) {

            alert("Please choose an image first.");

            return;

        }


        predictionText.textContent =
            "Analyzing image...";


        // --------------------------------
        // Convert image into a Tensor
        // --------------------------------

        let tensor =
            tf.browser.fromPixels(imageElement);


        // --------------------------------
        // Resize image
        // --------------------------------

        tensor =
            tf.image.resizeBilinear(
                tensor,
                [224, 224]
            );


        // --------------------------------
        // Normalize pixel values
        // --------------------------------

        tensor =
            tensor.toFloat().div(255.0);


        // --------------------------------
        // Add batch dimension
        // --------------------------------

        tensor =
            tensor.expandDims(0);


        // --------------------------------
        // Make prediction
        // --------------------------------

        let prediction =
            model.predict(tensor);


        let probabilities =
            await prediction.data();


        // --------------------------------
        // Get probabilities
        // --------------------------------

        let sneakers =
            probabilities[0];

        let flipflops =
            probabilities[1];


        // --------------------------------
        // Convert to percentage
        // --------------------------------

        let sneakersPercent =
            sneakers * 100;

        let flipflopsPercent =
            flipflops * 100;


        // --------------------------------
        // Determine highest prediction
        // --------------------------------

        if (sneakers > flipflops) {

            predictionText.textContent =
                "Sneakers";

        }

        else {

            predictionText.textContent =
                "Flipflops";

        }


        // --------------------------------
        // Display confidence
        // --------------------------------

        sneakersConfidence.textContent =
            sneakersPercent.toFixed(2) + "%";


        flipflopsConfidence.textContent =
            flipflopsPercent.toFixed(2) + "%";


        sneakersBar.style.width =
            sneakersPercent + "%";


        flipflopsBar.style.width =
            flipflopsPercent + "%";


        // --------------------------------
        // Free Tensor memory
        // --------------------------------

        tensor.dispose();

        prediction.dispose();

    }
);


// ------------------------------------
// 4. Start loading the model
// ------------------------------------

loadModel();