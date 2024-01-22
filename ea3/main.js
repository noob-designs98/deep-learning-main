let model_feed_forward = null;
let model_lstm = null
let result_elem = null
async function init_models() {
    model_feed_forward = await tf.loadLayersModel('http://localhost:5500/ea3/output_feed_forward/model.json');
    model_lstm = await tf.loadLayersModel('http://localhost:5500/ea3/output_lstm/model.json')
    
    result_elem = document.getElementById("result");

    document.getElementById('textfield').addEventListener("keyup", function (evt) {
        if(this.value.slice(-1) == " "){
        let result = predict_lstm(this.value.toLowerCase());
        result.then(function(res){
            result_elem.innerHTML = res;
        });
    }
    }, false);
}

function get_index_token(token, dict) {
    for(const [key, value] of Object.entries(dict)){
        if(key === token){
            return value;
        }
    }
    return 0;
}

async function predict_lstm(input) {
    let split = input.split(" ");
    let input_index =  [];

    for (const token of split)  {
        input_index.push(get_index_token(token));
    }
    console.log(input_index);
}

function pad(toPad, padChar, length){
    return (String(toPad).length < length)
        ? new Array(length - String(toPad).length + 1).join(padChar) + String(toPad)
        : toPad;
}

function prepend_zeros(arr, target_length) {
    const current_length = arr.length;
  
    if (current_length >= target_length) {
      return arr; 
    }
  
    const zeros_to_prepend = target_length - current_length;
    const zeros_array = Array(zeros_to_prepend).fill(0);
  
    return zeros_array.concat(arr);
  }

async function predict_feed_forward(input) {
    let split = input.split(" ");
    let input_index =  [];

    for (const token of split)  {
        input_index.push(get_index_token(token, feed_forward_dict));
    }
    
    if(input_index.length < 9){
       input_index = prepend_zeros(input_index, 9);
    }
    else if(input_index.length > 9){
        input_index = input_index.slice(-9);
    }

    console.log(input_index);
    let result = model_feed_forward.predict(tf.tensor(input_index).reshape([-1, 9]));
    console.log(result.arraySync());
    let argmax = result.argMax(1).arraySync()[0];
    
    for(const [key, value] of Object.entries(feed_forward_dict)){
        if(value === argmax){
            return key;
        }
    }
    return undefined;
}

async function predict_lstm(input){
    let split = input.split(" ");
    let input_index =  [];

    for (const token of split)  {
        input_index.push(get_index_token(token, lstm_dict));
    }
    
    if(input_index.length < 28){
        input_index = prepend_zeros(input_index, 28);
    }

    let result = model_lstm.predict(tf.tensor(input_index).reshape([-1, 28]));
    console.log(result.arraySync());
    let argmax = result.argMax(1).arraySync()[0];
    
    for(const [key, value] of Object.entries(lstm_dict)){
        if(value === argmax){
            return key;
        }
    }
}