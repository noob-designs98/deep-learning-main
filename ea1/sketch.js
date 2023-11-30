var classifier = ml5.imageClassifier('MobileNet', modelReady);

let img;
let testImage;
// const reader = new FileReader();
// var id = null;



const initApp = () => {
    const droparea = document.querySelector('.droparea');

    const active = () => droparea.classList.add("green-border");

    const inactive = () => droparea.classList.remove("green-border");

    const prevents = (e) => e.preventDefault();



    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evtName => {
        droparea.addEventListener(evtName, prevents);

    });

    ['dragenter', 'dragover'].forEach(evtName => {
        droparea.addEventListener(evtName, active);
    });

    ['dragleave', 'drop'].forEach(evtName => {
        droparea.addEventListener(evtName, inactive);
    });

    droparea.addEventListener("drop", handleDrop);

}



document.addEventListener("DOMContentLoaded", initApp);

function testBanana() {
    document.getElementById('droparea_img').src = "images/banane.png";
    testImage = createImg('images/banane.png', imageReady)
    testImage.hide();
    background(0);
    classifier.classify(testImage, gotResult);

}

function testStift() {
    document.getElementById('droparea_img').src = "images/stift.png";
    testImage = createImg('images/stift.png', imageReady)
    testImage.hide();
    background(0);
    classifier.classify(testImage, gotResult);

}

function testHund2() {
    document.getElementById('droparea_img').src = "images/hund2.png";
    testImage = createImg('images/hund2.png', imageReady)
    testImage.hide();
    background(0);
    classifier.classify(testImage, gotResult);

}

function testWassermelone() {
    document.getElementById('droparea_img').src = "images/wassermelone.png";
    testImage = createImg('images/wassermelone.png', imageReady)
    testImage.hide();
    background(0);
    classifier.classify(testImage, gotResult);

}

function testErdbeere() {
    document.getElementById('droparea_img').src = "images/erdbeere.png";
    testImage = createImg('images/erdbeere.png', imageReady)
    testImage.hide();
    background(0);
    classifier.classify(testImage, gotResult);

}

function testMotorrad() {
    document.getElementById('droparea_img').src = "images/motorrad.png";
    testImage = createImg('images/motorrad.png', imageReady)
    testImage.hide();
    background(0);
    classifier.classify(testImage, gotResult);

}






const handleDrop = (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    const fileArray = [...files];
    console.log(files[0]); // FileList
    console.log(fileArray);
    console.log(dt);
    testImage.src = window.URL.createObjectURL(files[0]);
    document.getElementById('droparea_img').src = testImage.src;
    setTimeout(classifyImage(), 100);

}


function handleFiles() {
    const fileInput = document.getElementById('fileUploader');
    const curFiles = fileInput.files;

    if (curFiles.length === 0) {
        testImage.src = 'images/Bildleer.png';
        setTimeout(classifyImage, 100);

    } else {
        testImage.src = window.URL.createObjectURL(curFiles[0]);
        document.getElementById('droparea_img').src = testImage.src;
        setTimeout(classifyImage(), 100);
    }
}
// function imageUpload() {
//     document.getElementById('fileUploader').click();
//}

function classifyImage() {
    testImage = createImg(testImage.src, imageReady)
    testImage.hide();
}

function modelReady() {
    console.log("Model bereit");
}

function imageReady() {

    classifier.classify(testImage, gotResult);
    console.log("Bild bereit");
}

function setup() {
    testImage = createImg('images/Bildleer.png')
    testImage.hide();
    console.log("Setup durchgefuehrt");
}

function round(num) {
    var m = Number((Math.abs(num) * 100).toPrecision(15));
    return Math.round(m) / 100 * Math.sign(num);
}

// A function to run when we get any errors and the results
function gotResult(error, results) {
    // Display error in the console
    if (error) {
        console.error(error);
    } else {

        for (let i = 0; i < 3; i++) {
            var result1 = (results[i].confidence * 100).toFixed(1) + "%";
            document.getElementById("ergebnis" + i).innerHTML = result1;
            document.getElementById("ergebnis" + i).style.width = result1;
            document.getElementById("ergebnistext" + i).innerHTML = `${results[i].label}`;

        }

        for (let i = 0; i < 3; i++) {
            var result1 = (results[i].confidence * 100).toFixed(4) + "%";
            document.getElementById("ergebnis0" + i).innerHTML = result1;
            document.getElementById("ergebnistext0" + i).innerHTML = `${results[i].label}`;


        }




    }
}