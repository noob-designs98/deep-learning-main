function generateData(Samples, noiseVariance) {
    const xValues = tf.linspace(-1, 1, Samples).dataSync();
    const yValues = xValues.map(x => (x + 0.8) * (x - 0.2) * (x - 0.3) * (x - 0.6) + (Math.random() - 0.5) * noiseVariance);

    return { x: xValues, y: yValues };
}

async function trainModel(data, optimizer, epochs, learningRate) {
    showOverlay();

    const model = tf.sequential({
        layers: [
            tf.layers.dense({ inputShape: [1], units: 10, activation: 'relu' }),
            tf.layers.dense({ units: 1 })
        ]
    });

    model.compile({
        optimizer: tf.train[optimizer](learningRate), // Set learning rate
        loss: 'meanSquaredError'
    });

    const xs = tf.tensor2d(data.x, [data.x.length, 1]);
    const ys = tf.tensor2d(data.y, [data.y.length, 1]);

    await model.fit(xs, ys, { epochs: epochs, shuffle: true });

    hideOverlay();

    return model;
}


async function predictAndPlot(model, testData, noiseData) {
    const predictions = Array.from(model.predict(tf.tensor2d(testData.x, [testData.x.length, 1])).dataSync());

    const trace1 = {
        type: 'scatter',
        mode: 'lines',
        x: testData.x,
        y: testData.y,
        line: { width: 4, color: '#f4c255' },
        name: 'Control Graph',

    };

    const trace2 = {
        type: 'scatter',
        mode: 'lines',
        x: testData.x,
        y: predictions,
        line: { width: 4, color: '#ee0a74' },
        name: 'Predictions',
    };

    const trace3 = {
        type: 'scatter',
        mode: 'markers',
        x: testData.x,
        y: noiseData.y,
        marker: { size: 6, color: '#0aee74' },
        name: 'Noised Data',
    };

    const layout = {
        title: 'Regression mit FFNN',
        xaxis: {
            title: 'x',
            range: [-1, 1],
            zeroline: true,
            zerolinecolor: 'white', // Color of the zero line
            zerolinewidth: 3, // Width of the zero line
            gridcolor: 'red', // Color of the grid lines
            tickvals: [-1, 1]
        },
        yaxis: {
            title: 'y',
            range: [-0.6, 0.6],
            zeroline: true,
            zerolinecolor: 'white', // Color of the zero line
            zerolinewidth: 3, // Width of the zero line
            gridcolor: 'red', // Color of the grid lines
            tickvals: [-0.6, 0.6]
        },
        paper_bgcolor: 'rgba(0,0,0,0)', // Transparent background
        plot_bgcolor: 'rgba(0,0,0,0)', // Transparent plot area
        font: { color: 'white' } // Set text color
    };

    Plotly.newPlot('plot', [trace1, trace2, trace3], layout);
}

async function generateAndTrain() {
    showOverlay();

    const Samples = parseInt(document.getElementById('inputN').value) || 10;
    const noiseVariance = parseFloat(document.getElementById('inputNoiseVariance').value) || 0.1;
    const optimizer = document.getElementById('inputOptimizer').value || 'adam';
    const epochs = parseInt(document.getElementById('inputEpochs').value) || 100;
    const learningRate = parseFloat(document.getElementById('inputLearningRate').value) || 0.001; // Read learning rate

    const trainingData = generateData(Samples, noiseVariance);
    const testData = generateData(100, 0);
    const noiseData = generateData(100, noiseVariance);

    const model = await trainModel(trainingData, optimizer, epochs, learningRate);
    await predictAndPlot(model, testData, noiseData);

    hideOverlay();
}


function showOverlay() {
    document.getElementById('overlay').style.display = 'flex';
}

function hideOverlay() {
    document.getElementById('overlay').style.display = 'none';
}

function Underfitting() {
    showOverlay();

    const Samples = 100;
    const noiseVariance = 0.1;
    const optimizer = 'adam';
    const epochs = 100;

    const hiddenLayers = '1 layer with 10 units (linear activation), 1 layer with 1 unit';

    updateInformation(Samples, optimizer, epochs, noiseVariance, hiddenLayers);

    // Set a fixed seed for noise generation
    Math.seedrandom('fixed_seed_for_underfitting');

    const trainingData = generateData(Samples, noiseVariance);
    const testData = generateData(100, 0);
    const noiseData = generateData(100, noiseVariance);

    const model = tf.sequential({
        layers: [
            tf.layers.dense({ inputShape: [1], units: 10, activation: 'linear' }),
            tf.layers.dense({ units: 1 })
        ]
    });

    model.compile({
        optimizer: optimizer,
        loss: 'meanSquaredError'
    });

    const xs = tf.tensor2d(trainingData.x, [trainingData.x.length, 1]);
    const ys = tf.tensor2d(trainingData.y, [trainingData.y.length, 1]);

    model.fit(xs, ys, { epochs: epochs, shuffle: true }).then(() => {

        const trace1 = {
            type: 'scatter',
            mode: 'lines',
            x: testData.x,
            y: testData.y,
            line: { width: 4, color: '#f4c255' },
            name: 'Control Graph',
        };

        const trace2 = {
            type: 'scatter',
            mode: 'lines',
            line: { width: 4, color: '#ee0a74' },
            x: testData.x,
            y: Array.from(model.predict(tf.tensor2d(testData.x, [testData.x.length, 1])).dataSync()),
            name: 'Predictions',
        };

        const trace3 = {
            type: 'scatter',
            mode: 'markers',
            x: testData.x,
            y: noiseData.y,
            marker: { size: 6, color: '#0aee74' },
            name: 'Noised Data',
        };

        const layout = {
            title: 'UNDERFITTING EXAMPLE',
            xaxis: {
                title: 'x',
                range: [-1, 1],
                zeroline: true,
                zerolinecolor: 'white', // Color of the zero line
                zerolinewidth: 3, // Width of the zero line
                gridcolor: 'lightgray', // Color of the grid lines
                tickvals: [-1, 1]
            },
            yaxis: {
                title: 'y',
                zeroline: true,
                zerolinecolor: 'white', // Color of the zero line
                zerolinewidth: 3, // Width of the zero line
                gridcolor: 'lightgray', // Color of the grid lines
                tickvals: [0]
            },
            paper_bgcolor: 'rgba(0,0,0,0)', // Transparent background
            plot_bgcolor: 'rgba(0,0,0,0)', // Transparent plot area
            font: { color: 'white' } // Set text color
        };

        Plotly.newPlot('plot2', [trace1, trace2, trace3], layout);
        hideOverlay(); // Hide the overlay when plot appears
    });
}

function Bestfit() {
    showOverlay();

    const Samples = 40;
    const noiseVariance = 0.1;
    const optimizer = 'adam';
    const epochs = 1500;

    const hiddenLayers = '1 layer with 1 unit (linear activation), 4 layers with 10 units (ReLU activation), 1 layer with 1 unit';

    updateInformation(Samples, optimizer, epochs, noiseVariance, hiddenLayers);
    // Set a fixed seed for noise generation
    Math.seedrandom('fixed_seed_for_underfitting');

    const trainingData = generateData(Samples, noiseVariance);
    const testData = generateData(100, 0); // Generate test data without noise for better visualization
    const noiseData = generateData(100, noiseVariance); // Generate noisy test data with fixed seed

    const model = tf.sequential({
        layers: [
            tf.layers.dense({ inputShape: [1], units: 1, activation: 'linear' }),
            tf.layers.dense({ inputShape: [1], units: 10, activation: 'relu' }),
            tf.layers.dense({ inputShape: [1], units: 10, activation: 'relu' }),
            tf.layers.dense({ inputShape: [1], units: 10, activation: 'relu' }),
            tf.layers.dense({ inputShape: [1], units: 10, activation: 'relu' }),
            tf.layers.dense({ units: 1 })
        ]
    });

    model.compile({
        optimizer: optimizer,
        loss: 'meanSquaredError'
    });

    const xs = tf.tensor2d(trainingData.x, [trainingData.x.length, 1]);
    const ys = tf.tensor2d(trainingData.y, [trainingData.y.length, 1]);

    model.fit(xs, ys, { epochs: epochs, shuffle: true }).then(() => {
        const trace1 = {
            type: 'scatter',
            mode: 'lines',
            x: testData.x,
            y: testData.y,
            line: { width: 4, color: '#f4c255' },
            name: 'Control Graph',
        };

        const trace2 = {
            type: 'scatter',
            mode: 'lines',
            line: { width: 4, color: '#ee0a74' },
            x: testData.x,
            y: Array.from(model.predict(tf.tensor2d(testData.x, [testData.x.length, 1])).dataSync()),
            name: 'Predictions',
        };

        const trace3 = {
            type: 'scatter',
            mode: 'markers',
            x: testData.x,
            y: noiseData.y,
            marker: { size: 6, color: '#0aee74' },
            name: 'Noised Data',
        };

        const layout = {
            title: 'BEST-FIT EXAMPLE',
            xaxis: {
                title: 'x',
                range: [-1, 1],
                zeroline: true,
                zerolinecolor: 'white', // Color of the zero line
                zerolinewidth: 3, // Width of the zero line
                gridcolor: 'lightgray', // Color of the grid lines
                tickvals: [-1, 1]
            },
            yaxis: {
                title: 'y',
                zeroline: true,
                zerolinecolor: 'white', // Color of the zero line
                zerolinewidth: 3, // Width of the zero line
                gridcolor: 'red', // Color of the grid lines
                tickvals: [0]
            },
            paper_bgcolor: 'rgba(0,0,0,0)', // Transparent background
            plot_bgcolor: 'rgba(0,0,0,0)', // Transparent plot area
            font: { color: 'white' } // Set text color
        };

        Plotly.newPlot('plot2', [trace1, trace2, trace3], layout);
        hideOverlay(); // Hide the overlay when plot appears
    });
}

function Overfitting() {
    showOverlay(); // Show the overlay when "Underfitting Example" is clicked

    const Samples = 20;
    const noiseVariance = 0.3;
    const optimizer = 'adam';
    const epochs = 2000;

    const hiddenLayers = '1 layer with 1 unit (linear activation), 4 layers with 10 units (ReLU activation),1 layer with 1 unit';

    updateInformation(Samples, optimizer, epochs, noiseVariance, hiddenLayers);

    Math.seedrandom('fixed_seed_for_overfitting');

    const trainingData = generateData(Samples, noiseVariance);
    const testData = generateData(100, 0);
    const noiseData = generateData(100, noiseVariance);

    const model = tf.sequential({
        layers: [
            tf.layers.dense({ inputShape: [1], units: 10, activation: 'linear' }),
            tf.layers.dense({ inputShape: [1], units: 14, activation: 'relu' }),
            tf.layers.dense({ inputShape: [1], units: 14, activation: 'relu' }),
            tf.layers.dense({ inputShape: [1], units: 14, activation: 'relu' }),
            tf.layers.dense({ inputShape: [1], units: 14, activation: 'relu' }),
            tf.layers.dense({ units: 1 })
        ]
    });

    model.compile({
        optimizer: optimizer,
        loss: 'meanSquaredError'
    });

    const xs = tf.tensor2d(trainingData.x, [trainingData.x.length, 1]);
    const ys = tf.tensor2d(trainingData.y, [trainingData.y.length, 1]);

    model.fit(xs, ys, { epochs: epochs, shuffle: true }).then(() => {
        const trace1 = {
            type: 'scatter',
            mode: 'lines',
            x: testData.x,
            y: testData.y,
            line: { width: 4, color: '#f4c255' },
            name: 'Control Graph',
        };

        const trace2 = {
            type: 'scatter',
            mode: 'lines',
            line: { width: 4, color: '#ee0a74' },
            x: testData.x,
            y: Array.from(model.predict(tf.tensor2d(testData.x, [testData.x.length, 1])).dataSync()),
            name: 'Predictions',
        };

        const trace3 = {
            type: 'scatter',
            mode: 'markers',
            x: testData.x,
            y: noiseData.y,
            marker: { size: 6, color: '#0aee74' },
            name: 'Noised Data',
        };

        const layout = {
            title: 'OVERFITTING EXAMPLE',
            xaxis: {
                title: 'x',
                range: [-1, 1],
                zeroline: true,
                zerolinecolor: 'white', // Color of the zero line
                zerolinewidth: 3, // Width of the zero line
                gridcolor: 'lightgray', // Color of the grid lines
                tickvals: [-1, 1]
            },
            yaxis: {
                title: 'y',
                zeroline: true,
                zerolinecolor: 'white', // Color of the zero line
                zerolinewidth: 3, // Width of the zero line
                gridcolor: 'red', // Color of the grid lines
                tickvals: [0]
            },
            paper_bgcolor: 'rgba(0,0,0,0)', // Transparent background
            plot_bgcolor: 'rgba(0,0,0,0)', // Transparent plot area
            font: { color: 'white' } // Set text color
        };

        Plotly.newPlot('plot2', [trace1, trace2, trace3], layout);
        hideOverlay(); // Hide the overlay when plot appears
    });
}

function updateInformation(Samples, optimizer, epochs, noiseVariance, hiddenLayers) {
    const information = document.getElementById('information');
    information.innerHTML = `<p>Samples: ${Samples}</p><p>Optimizer: ${optimizer}</p><p>Epochs: ${epochs}</p><p>Noise: ${noiseVariance}</p><p>Hidden Layers: ${hiddenLayers}</p>`;
}