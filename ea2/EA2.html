let a = tf.variable(tf.scalar((Math.random() * 2) - 1));
let b = tf.variable(tf.scalar((Math.random() * 2) - 1));
let c = tf.variable(tf.scalar((Math.random() * 2) - 1));
let d = tf.variable(tf.scalar((Math.random() * 2) - 1));
let e = tf.variable(tf.scalar((Math.random() * 2) - 1));




// Step 2. Create an optimizer, we will use this later. You can play
// with some of these values to see how the model performs.
var dataGraph;
var dataGen = 50;
var numIterations = 50;
var learningRate = 0.5;
var optimizer = tf.train.sgd(learningRate);

var noise = 0.04;

let optimierer = tf.train.sgd(learningRate);


// Get the Sidebar
var slider = document.getElementById("myRange");
var sliderNoise = document.getElementById("myNoise");
var output = document.getElementById("demo");
output.innerHTML = slider.value;

var slider3 = document.getElementById("myRange3");
var output3 = document.getElementById("demo3");
output3.innerHTML = slider3.value / 100;

var sliderNoise = document.getElementById("myNoise");
var output4 = document.getElementById("noise");
output4.innerHTML = sliderNoise.value / 100;

var mySidebar = document.getElementById("mySidebar");

// Get the DIV with overlay effect
var overlayBg = document.getElementById("myOverlay");

// Toggle between showing and hiding the sidebar, and add overlay effect


slider.oninput = function() {
    output.innerHTML = this.value;
    numIterations = this.value;

}

slider3.oninput = function() {
    output3.innerHTML = this.value / 100;
    learningRate = this.value / 100;
    optimizer = tf.train.sgd(learningRate);
    optimierer = tf.train.sgd(learningRate);

}

sliderNoise.oninput = function() {
    output4.innerHTML = this.value / 100;
    noise = this.value / 100;
    trainingData = generateData(dataGen, trueCoefficients);
    plotGraph(trainingData.xs.dataSync(), trainingData.ys.dataSync());



}

function fitting(x) {
    if (x == 'u') {
        sliderNoise.value = 10;
        noise = 0.1;
        output4.innerHTML = sliderNoise.value.toString() / 100;

        slider.value = 5;
        numIterations = 5;
        output.innerHTML = slider.value;

        slider3.value = 10;
        learningrate = slider3.value / 100;
        optimizer = tf.train.sgd(learningRate);
        output3.innerHTML = slider3.value / 100;



        trainingData = generateData(dataGen, trueCoefficients);
        plotGraph(trainingData.xs.dataSync(), trainingData.ys.dataSync());



    }
    if (x == 'p') {
        sliderNoise.value = 30;
        noise = 0.3;
        output4.innerHTML = sliderNoise.value.toString() / 100;

        slider.value = 150;
        numIterations = 150;
        output.innerHTML = slider.value;

        slider3.value = 100;
        learningrate = slider3.value / 100;
        optimizer = tf.train.sgd(learningRate);
        output3.innerHTML = slider3.value / 100;

        trainingData = generateData(dataGen, trueCoefficients);
        plotGraph(trainingData.xs.dataSync(), trainingData.ys.dataSync());
    }
    if (x == 'o') {
        sliderNoise.value = 2;
        noise = 0.02;
        output4.innerHTML = sliderNoise.value.toString() / 100;

        slider.value = 75;
        numIterations = 75;
        output.innerHTML = slider.value;

        slider3.value = 50;
        learningrate = slider3.value / 100;
        optimizer = tf.train.sgd(learningRate);
        output3.innerHTML = slider3.value / 100;

        trainingData = generateData(dataGen, trueCoefficients);
        plotGraph(trainingData.xs.dataSync(), trainingData.ys.dataSync());
    }
}

function w3_open() {
    if (mySidebar.style.display === 'block') {
        mySidebar.style.display = 'none';
        overlayBg.style.display = "none";
    } else {
        mySidebar.style.display = 'block';
        overlayBg.style.display = "block";
    }
}

function getComboB(optimizerWert) {

    if (optimizerWert.value == "SGD") {
        optimierer = tf.train.sgd(learnRateAnzahl);
    }

    if (optimizerWert.value == "Adam") {
        optimierer = tf.train.adam(learnRateAnzahl);
    }


}

function getComboA(selectObject) {
    var value = selectObject.value;
    dataGen = parseInt(value);

    trainingData = generateData(dataGen, trueCoefficients);

    plotBefore = trainingData;
    var x2 = Array.from(trainingData.xs.dataSync());
    var y2 = Array.from(trainingData.ys.dataSync());

    var trace2 = {
        x: x2,
        y: y2,
        type: 'scatter',
        mode: 'markers',
        opacity: 0.5,
        name: 'Data Points',

        marker: {
            color: 'blue'
        }

    };

    var data = [trace2];
    dataOut = [trace2];
    var layout = {
        title: "Polynomial Data Points and Predicted Graph",
        xaxis: { range: [-1.2, 1.2] },
        yaxis: { range: [-0.05, 1.2] }
    };

    Plotly.newPlot("myDiv", data, layout);
}



// Close the sidebar with the close button
function w3_close() {
    mySidebar.style.display = "none";
    overlayBg.style.display = "none";
}


function generateData(numPoints, coeff, sigma = 0.04) {
    return tf.tidy(() => {
        const [a, b, c, d, e] = [
            tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c),
            tf.scalar(coeff.d), tf.scalar(coeff.e)
        ];

        const xs = tf.randomUniform([numPoints], -1, 1);

        // Generate polynomial data
        const three = tf.scalar(3, 'int32');
        const ys =
            a.mul(xs.pow(tf.scalar(4, 'int32')))
            .add(b.mul(xs.pow(three)))
            .add(c.mul(xs.square()))
            .add(d.mul(xs))
            .add(e)
            // Add random noise to the generated data            
            .add(tf.randomNormal([numPoints], 0, noise));

        // Normalize the y values to the range 0 to 1.
        const ymin = ys.min();
        const ymax = ys.max();
        const yrange = ymax.sub(ymin);
        const ysNormalized = ys.sub(ymin).div(yrange);

        return {
            xs,
            ys: ysNormalized
        };
    })
}

/**
 * We want to learn the coefficients that give correct solutions to the
 * following cubic equation:
 *      y = a * x^3 + b * x^2 + c * x + d
 * In other words we want to learn values for:
 *      a
 *      b
 *      c
 *      d
 * Such that this function produces 'desired outputs' for y when provided
 * with x. We will provide some examples of 'xs' and 'ys' to allow this model
 * to learn what we mean by desired outputs and then use it to produce new
 * values of y that fit the curve implied by our example.
 */

// Step 1. Set up variables, these are the things we want the model
// to learn in order to do prediction accurately. We will initialize
// them with random values.


// Step 3. Write our training process functions.

/*
 * This function represents our 'model'. Given an input 'x' it will try and
 * predict the appropriate output 'y'.
 *
 * It is also sometimes referred to as the 'forward' step of our training
 * process. Though we will use the same function for predictions later.
 *
 * @return number predicted y value
 */



function generateData2(numPoints, coeff, sigma = 0.04) {
    return tf.tidy(() => {
        const [a, b, c, d, e] = [
            tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c),
            tf.scalar(coeff.d), tf.scalar(coeff.e)
        ];

        const xs = tf.randomUniform([500], -1, 1);

        // Generate polynomial data
        const three = tf.scalar(3, 'int32');
        const ys =
            a.mul(xs.pow(tf.scalar(4, 'int32')))
            .add(b.mul(xs.pow(three)))
            .add(c.mul(xs.square()))
            .add(d.mul(xs))
            .add(e)


        // Normalize the y values to the range 0 to 1.
        const ymin = ys.min();
        const ymax = ys.max();
        const yrange = ymax.sub(ymin);
        const ysNormalized = ys.sub(ymin).div(yrange);

        return {
            xs,
            ys: ysNormalized
        };
    })
}

function predict(x) {
    // y = a * x ^ 3 + b * x ^ 2 + c * x + d
    return tf.tidy(() => {
        return a.mul(x.pow(tf.scalar(4, 'int32')))
            .add(b.mul(x.pow(tf.scalar(3, 'int32'))))
            .add(c.mul(x.square()))
            .add(d.mul(x))
            .add(e);
    });
}

/*
 * This will tell us how good the 'prediction' is given what we actually
 * expected.
 *
 * prediction is a tensor with our predicted y values.
 * labels is a tensor with the y values the model should have predicted.
 */
function loss(prediction, labels) {
    // Having a good error function is key for training a machine learning model
    const error = prediction.sub(labels).square().mean();

    return error;
}
var dataOut;
/*
 * This will iteratively train our model.
 *
 * xs - training data x values
 * ys — training data y values
 */
async function train(xs, ys, numIterations) {
    for (let iter = 0; iter < numIterations; iter++) {
        // optimizer.minimize is where the training happens.

        // The function it takes must return a numerical estimate (i.e. loss)
        // of how well we are doing using the current state of
        // the variables we created at the start.

        // This optimizer does the 'backward' step of our training process
        // updating variables defined previously in order to minimize the
        // loss.

        var predictIteration = generateData2(dataGen, { a: a.dataSync()[0], b: b.dataSync()[0], c: c.dataSync()[0], d: d.dataSync()[0], e: e.dataSync()[0] })

        var x10 = Array.from(predictIteration.xs.dataSync());
        var y10 = Array.from(predictIteration.ys.dataSync());

        var trace11 = {
            x: x10,
            y: y10,
            type: 'scatter',
            mode: 'markers',
            name: 'Predicted',


            marker: {
                color: 'red',
                size: 2.5,
                line: {
                    color: 'red',
                    width: 1
                }
            }

        };

        var trace12 = {
            x: Array.from(trainingData.xs.dataSync()),
            y: Array.from(trainingData.ys.dataSync()),
            type: 'scatter',
            mode: 'markers',
            opacity: 0.5,
            name: 'Data Points',

            marker: {
                color: 'blue'
            }

        };


        var plotData = [trace11, trace12];


        var layout2 = {
            title: "Polynomial Data Points and Predicted Graph",

            xaxis: { range: [-1.2, 1.2] },
            yaxis: { range: [-0.05, 1.2] }
        };

        Plotly.newPlot("myDiv", plotData, layout2);


        optimizer.minimize(() => {
            // Feed the examples into the model
            const pred = predict(xs);

            return loss(pred, ys);


        });

        // Use tf.nextFrame to not block the browser.
        await tf.nextFrame();
    }
}


function plotGraph(x, y) {
    var trace = {
        x: x,
        y: y,
        type: 'scatter',
        mode: 'markers',
        opacity: 0.5,
        name: 'Data Points',

        marker: {
            color: 'blue'
        }
    };

    var data = [trace];
    dataOut = [trace];
    var layout = {
        title: "Polynomial Data Points",
        xaxis: { range: [-1.2, 1.2] },
        yaxis: { range: [-0.05, 1.2] }
    };

    Plotly.newPlot("myDiv", data, layout);
}

function plotData(container, xs, ys) { return; }

function renderCoefficients(container, coeff) { return; }

var trueCoefficients = { a: 1, b: -0.3, c: -.52, d: .252, e: -.00288 };

document.getElementById('aBefore').innerText = trueCoefficients.a;
document.getElementById('bBefore').innerText = trueCoefficients.b;
document.getElementById('cBefore').innerText = trueCoefficients.c;
document.getElementById('dBefore').innerText = trueCoefficients.d;
document.getElementById('eBefore').innerText = trueCoefficients.e;


var trainingData = generateData(dataGen, trueCoefficients);

var plotBefore = trainingData;

async function learnCoefficients() {
    //const trueCoefficients = { a: 1, b: -0.3, c: - .52, d: .252, e: - .00288 };
    //const trainingData = generateData(dataGen, trueCoefficients);

    //console.log(Array.from(trainingData.xs.dataSync()));

    //console.log(Array.from(trainingData.ys.dataSync()));

    var x2 = Array.from(trainingData.xs.dataSync());
    var y2 = Array.from(trainingData.ys.dataSync());

    var trace2 = {
        x: x2,
        y: y2,
        type: 'scatter',
        mode: 'markers',
        name: 'Data Points',

        opacity: 0.5,

        marker: {
            color: 'blue'
        }

    };

    var data = [trace2];
    dataOut = [trace2];
    var layout = {
        title: "Polynomial Data Points and Predicted Graph"
    };

    Plotly.newPlot("myDiv", data, layout);

    // Plot original data
    //renderCoefficients('#data .coeff', trueCoefficients);
    console.log("abcd-1:", a.dataSync()[0], b.dataSync()[0], c.dataSync()[0], d.dataSync()[0], e.dataSync()[0]);
    //await plotData('#data .plot', trainingData.xs, trainingData.ys)

    // See what the predictions look like with random coefficients
    //renderCoefficients('#random .coeff', {a: a.dataSync()[0],b: b.dataSync()[0],c: c.dataSync()[0],d: d.dataSync()[0],});
    console.log("abcd-2:", a.dataSync()[0], b.dataSync()[0], c.dataSync()[0], d.dataSync()[0], e.dataSync()[0]);
    document.getElementById('aInit').innerText = a.dataSync()[0].toFixed(2);
    document.getElementById('bInit').innerText = b.dataSync()[0].toFixed(2);
    document.getElementById('cInit').innerText = c.dataSync()[0].toFixed(2);
    document.getElementById('dInit').innerText = d.dataSync()[0].toFixed(2);
    document.getElementById('eInit').innerText = e.dataSync()[0].toFixed(2);


    const PredictedDataBefore = generateData2(300, { a: a.dataSync()[0], b: b.dataSync()[0], c: c.dataSync()[0], d: d.dataSync()[0], e: e.dataSync()[0] })
    const predictionsBefore = predict(trainingData.xs);




    //await plotDataAndPredictions('#random .plot', trainingData.xs, trainingData.ys, predictionsBefore);

    // Train the model!
    await train(trainingData.xs, trainingData.ys, numIterations);
    //console.log(predictionsBefore.dataSync());


    // See what the final results predictions are after training.
    //renderCoefficients('#trained .coeff', {a: a.dataSync()[0],b: b.dataSync()[0],c: c.dataSync()[0],d: d.dataSync()[0],});
    console.log("abcd-3:", a.dataSync()[0], b.dataSync()[0], c.dataSync()[0], d.dataSync()[0], e.dataSync()[0]);
    const predictionsAfter = predict(trainingData.xs);
    //await plotDataAndPredictions('#trained .plot', trainingData.xs, trainingData.ys, predictionsAfter);
    //console.log(predictionsAfter.dataSync());
    console.log("abcd-4:", a.dataSync()[0], b.dataSync()[0], c.dataSync()[0], d.dataSync()[0], e.dataSync()[0]);

    document.getElementById('aPred').innerText = a.dataSync()[0].toFixed(2);
    document.getElementById('bPred').innerText = b.dataSync()[0].toFixed(2);
    document.getElementById('cPred').innerText = c.dataSync()[0].toFixed(2);
    document.getElementById('dPred').innerText = d.dataSync()[0].toFixed(2);
    document.getElementById('ePred').innerText = e.dataSync()[0].toFixed(2);

    const PredictedDataAfter = generateData2(dataGen, { a: a.dataSync()[0], b: b.dataSync()[0], c: c.dataSync()[0], d: d.dataSync()[0], e: e.dataSync()[0] })
    var x3 = Array.from(PredictedDataAfter.xs.dataSync());
    var y2 = Array.from(PredictedDataAfter.ys.dataSync());

    var trace3 = {
        x: x3,
        y: y2,
        type: 'scatter',
        mode: 'markers',
        name: 'Predicted',

        marker: {
            color: '#FF0000'
        }

    };

    var data3 = [trace3, trace2];

    var layout2 = {
        title: "Polynomial Data Points and Predicted Graph",
        xaxis: { range: [-1.2, 1.2] },
        yaxis: { range: [-0.05, 1.2] }
    };

    Plotly.newPlot("myDiv", data3, layout2);

    predictionsBefore.print();
    predictionsAfter.print();

    predictionsBefore.dispose();
    predictionsAfter.dispose();
}


function createModel() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

    // Add an output layer
    model.add(tf.layers.dense({ units: 1, useBias: true }));

    return model;
}

async function run() {
    // Create the model


    // Convert the data to a form we can use for training.
    const tensorData = generateDataInputs(dataGen, { a: 1, b: -0.3, c: -.52, d: .252, e: -.00288 });
    const data = tensorData;
    const model = createModel();

    tfvis.show.modelSummary({ name: 'Model' }, model);

    const { inputs, labels } = tensorData;
    console.log(tensorData);



    await trainModel(model, inputs, labels);





}
//document.addEventListener('DOMContentLoaded', run);
function start() {
    a = tf.variable(tf.scalar((Math.random() * 2) - 1));
    b = tf.variable(tf.scalar((Math.random() * 2) - 1));
    c = tf.variable(tf.scalar((Math.random() * 2) - 1));
    d = tf.variable(tf.scalar((Math.random() * 2) - 1));
    e = tf.variable(tf.scalar((Math.random() * 2) - 1));
    run();
    learnCoefficients();
}




async function trainModel(model, inputs, labels) {
    // Prepare the model for training.
    model.compile({
        optimizer: optimierer,
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });

    const batchSize = 32;
    const epochs = numIterations;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks({ name: 'Training Performance' }, ['loss', 'mse'], { height: 200, callbacks: ['onEpochEnd'] })
    });
}

function generateDataInputs(numPoints, coeff, sigma = 0.04) {
    return tf.tidy(() => {
        const [a, b, c, d, e] = [
            tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c),
            tf.scalar(coeff.d), tf.scalar(coeff.e)
        ];

        const xs = tf.randomUniform([numPoints], -1, 1);

        // Generate polynomial data
        const three = tf.scalar(3, 'int32');
        const ys =
            a.mul(xs.pow(tf.scalar(4, 'int32')))
            .add(b.mul(xs.pow(three)))
            .add(c.mul(xs.square()))
            .add(d.mul(xs))
            .add(e)
            // Add random noise to the generated data
            // to make the problem a bit more interesting
            .add(tf.randomNormal([numPoints], 0, noise));

        // Normalize the y values to the range 0 to 1.
        const ymin = ys.min();
        const ymax = ys.max();
        const yrange = ymax.sub(ymin);
        const ysNormalized = ys.sub(ymin).div(yrange);
        const inputMax = xs.max();
        const inputMin = xs.min();
        const labelMax = ys.max();
        const labelMin = ys.min();

        return {
            inputs: xs,
            labels: ysNormalized,
            inputMax,
            inputMin,
            labelMax,
            labelMin,

        };
    })
}

plotGraph(Array.from(trainingData.xs.dataSync()), Array.from(trainingData.ys.dataSync()));

var realGraph = generateData2(500, trueCoefficients);
var x4 = Array.from(realGraph.xs.dataSync());
var y5 = Array.from(realGraph.ys.dataSync());

var trace6 = {
    x: x4,
    y: y5,
    type: 'scatter',
    mode: 'markers'
};

var data5 = [trace6];

var layout4 = {
    title: "Real Graph",


    xaxis: { range: [-1.2, 1.2] },
    yaxis: { range: [-0.05, 1.2] }
};

//Plotly.newPlot("myDiv3", data5, layout4);