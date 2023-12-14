function trainModel() {
    // Insert code to train model here
}

function changeSample() {
    // Insert code to change sample here
}

window.onload = function() {
    document.getElementById('learn-rate-value').innerText = 0.5;

};

function changeLearnRate() {

    var learnRate = document.getElementById('learn-rate').value;
    document.getElementById('learn-rate-value').innerText = learnRate;
    // make default value 0.5
}

// Insert code to change learning rate here


function changeGaussianNoise() {
    // Insert code to change Gaussian noise here
}



// Create a trace for the table
var trace = {
    type: 'table',
    header: {
        values: [
            ["Column 1"],
            ["Column 2"],
            ["Column 3"]
        ],
        align: "center",
        line: { width: 1, color: 'black' },
        fill: { color: "grey" },
        font: { family: "Arial", size: 12, color: "white" }
    },
    cells: {
        values: [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],
        align: "center",
        line: { color: "black", width: 1 },
        font: { family: "Arial", size: 11, color: ["black"] }
    }
}

// Create the data array for the plot
var data = [trace];

// Define the plot layout
var layout = {
    title: "My Table"
};

// Plot the chart to a div tag with id "myTable"
Plotly.newPlot('myTable', data, layout);