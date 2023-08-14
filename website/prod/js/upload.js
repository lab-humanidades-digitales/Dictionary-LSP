

	AWS.config.region = 'us-east-1'; // 1. Enter your region

AWS.config.credentials = new AWS.CognitoIdentityCredentials({
	IdentityPoolId: 'us-east-1:b5013574-2741-4e18-97be-9395b5929162' // 2. Enter your identity pool
});

AWS.config.credentials.get(function(err) {
	if (err) alert(err);
	console.log("estoy aqui")
	console.log(AWS.config.credentials);
});

var bucketName = 'user-video-test'; // Enter your bucket name+


var S3 = new AWS.S3();

var bucket = new AWS.S3({
	params: {
		Bucket: bucketName
	}
});

var fileChooser = document.getElementById('file-chooser');
var button = document.getElementById('upload-button');
var results = document.getElementById('results');
var percentage = document.getElementById('percentage');
var cancelUpload = document.getElementById('cancel-button');

function dataURLtoFile(dataurl, filename) {
        var arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1],
            bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
        while(n--){
            u8arr[n] = bstr.charCodeAt(n);
        }
        return new File([u8arr], filename, {type:mime});
}
// Store a reference of the preview video element and a global reference to the recorder instance
var video = document.getElementById('my-preview');
var countdown = document.getElementById("countdown");
var recordIcon = document.getElementById("record-icon");
var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
var recorder;
countdown.style.visibility = "hidden";

var modelParams = {
    flipHorizontal: true,
    maxNumBoxes: 3,
    iouThreshold: 0.2,
    scoreThreshold: 0.6,
    bboxLineWidth: "0",
    fontSize: 0,
};

var isVideo = false;
var model = null;
var toggleButton =  document.getElementById('btn-start-recording')


toggleButton.addEventListener("click", function (){
    document.getElementById('video').removeAttribute("hidden")
    toggleVideo();
});

function toggleVideo(){
    if(!isVideo){
        startVideo()
    }
    else{
        handtrack.stopVideo(video);
        isVideo = false;
    }
}

function startVideo(){
    handTrack.startVideo(video).then(function(status)
    {
        if (status) {
            isVideo = true
            document.getElementById('instructions').remove();
            video.style.height="500px"

            const screenHeight = window.screen.height;
            const screenWidth = window.screen.width;

            console.log(screenHeight)

            if (screenHeight < 800) {
                footer.style.position = 'relative';
                document.getElementById('video').style.marginTop = 0
            }


            if (screenWidth < 450) {
                footer.style.position = 'relative';
                document.getElementById('video').style.marginTop = 0
            }

            rundetection();
         
        }
    }
)};

function rundetection(){
    model.detect(video).then(predictions => {
        document.getElementById('face').style.visibility = "visible"
        document.getElementById('hand1').style.visibility = "visible"
        document.getElementById('hand2').style.visibility = "visible"
    model.renderPredictions(predictions, canvas, context, video);
    let fc = document.getElementById('face')
    let h1c = document.getElementById('hand1')
    let h2c = document.getElementById('hand2')

    if (predictions.length !== 0){
        let face_x = predictions[0].bbox[0];
        let face_y = predictions[0].bbox[1];

        if (face_x > 185 && face_x < 330 && face_y < 70){
            fc.style.borderColor = "green";
        }
        else{
            fc.style.borderColor = "red";
        }

        if (predictions.length > 1){
            let hand1_x = predictions[1].bbox[0];
            let hand1_y = predictions[1].bbox[1];

            if (hand1_x < 90 && hand1_y > 180){
                h1c.style.borderColor = "green";
            }
            else{
                h1c.style.borderColor = "red";
            }

        }

        if (predictions.length > 2){
            let hand2_x = predictions[2].bbox[0];
            let hand2_y = predictions[2].bbox[1];

            console.log(hand2_y)
           
            if (hand2_x > 215 && hand2_y > 200 && hand2_y < 400){
                h2c.style.borderColor = "green";
            }
            else{
                h2c.style.borderColor = "red";
            }

        }
    }

    if (fc.style.borderColor == 'green' && h1c.style.borderColor == 'green' && h2c.style.borderColor == 'green'){          
            isVideo = false
            document.getElementById('my-preview').removeAttribute("hidden")
            canvas.style.visibility = "hidden";
            document.getElementById('face').style.visibility = "hidden"
            document.getElementById('hand1').style.visibility = "hidden"
            document.getElementById('hand2').style.visibility = "hidden"
            countdown.style.visibility = "visible";
            let timeLeft = 2;
            var videoTimer = setInterval(function(){
                if(timeLeft < 1){

                    clearInterval(videoTimer);
                    countdown.style.visibility = "hidden";

                    startRecording();

                }else{
                    countdown.innerHTML = timeLeft;
                }
                timeLeft-=1;

            }, 1000)
        
    }

    if (isVideo){
        requestAnimationFrame(rundetection);
    }
})}

handTrack.load(modelParams).then(lmodel =>{
    model = lmodel;
})


// When the user clicks on start video recording
function startRecording(){


    // Request access to the media devices
    navigator.mediaDevices.getUserMedia({
        audio: false, 
        video: true
    }).then(function(stream) {

        handTrack.stopVideo(video);
        let timeLeft = 3;
        var videoTimer = setInterval(function(){
            recordIcon.style.visibility = "visible";
            if(timeLeft <= 0){
                clearInterval(videoTimer);
                recordIcon.style.visibility = "hidden";
                sendRecording();
            }else{
                countdown.innerHTML = timeLeft;
            }
            timeLeft-=1;
        },1000);
    
        // Display a live preview on the video element of the page
        setSrcObject(stream, video);

        // Start to display the preview on the video element
        // and mute the video to disable the echo issue !
        video.play();
        video.muted = true;

        // Initialize the recorder
        recorder = new RecordRTCPromisesHandler(stream, {
            mimeType: 'video/webm',
            bitsPerSecond: 12000000 //128000
        });

        // Start recording the video
        recorder.startRecording().then(function() {
            console.info('Recording video ...');
        }).catch(function(error) {
            console.log(error);
            console.error('Cannot start video recording: ', error);
        });

        // release stream on stopRecording
        recorder.stream = stream;

        // Enable stop recording button
        document.getElementById('btn-stop-recording').disabled = false;
    }).catch(function(error) {
               console.log(error);
        console.error("Cannot access media devices: ", error);
    });
};

// When the user clicks on Stop video recording
function sendRecording(){
    document.getElementById('video').remove();
    loader =  document.getElementById('loader')
    
    loader.removeAttribute("hidden")
    loader.style.marginTop = "10%"
    loader.style.marginBottom = "30%"
    loaderText = document.getElementById('loader-text')
    loaderText.innerText = "Obteniendo resultados..."
    this.disabled = true;
	video.style.display =  "none"; 
    recorder.stopRecording().then(function() {
        console.info('stopRecording success');

       var DataUrl = recorder.getDataURL();
       var random = Math.random( );

       DataUrl.then(function(result) {

        console.log(result) //borrar
            
            var url_file = dataURLtoFile(result, random +'.webm');

            console.log(url_file)
            console.log(url_file.name)


        //  var bucket = new AWS.S3({params: {Bucket: 'lsp-web/videos_test'}});
        var uploadParams = {Key: url_file.name, ContentType: url_file.type, Body: url_file};
        bucket.upload(uploadParams).on('httpUploadProgress', function(progress) {
            percentage.innerHTML = parseInt((progress.loaded * 100) / progress.total)+'%'; 
            console.log("Uploaded :: " + parseInt((progress.loaded * 100) / progress.total)+'%');
        }).send(function(err, data) {
            $('#upFile').val(null);
            $("#showMessage").show();

            var lambda = new AWS.Lambda();
            var params = {
                FunctionName: 'sagemaker-invoker',
                Payload: JSON.stringify({
                    'video': url_file.name
                })
            };
            console.log("antes de invoke")

            lambda.invoke(params, function(err, data) {
                if (err) {
                    console.log(err, err.stack);

                // If OK
                } else {
                    const glossListStr = data.Payload;
                    const glossList = JSON.parse(glossListStr)
                    console.log(glossList)
                    var getListElement = document.getElementById("results");
                    var ul = document.createElement("ul");
                    glossList.forEach(item => {
                        console.log(item.gloss)
                        const li = document.createElement("li");
                        li.textContent = item.gloss;
                        ul.appendChild(li);

                        /*
                        const queryString = li.textContent;
                        
                        const makeUrl = filename => `https://isolatedsigns.s3.amazonaws.com/${encodeURI(filename)}`;
                        const getUrlFromNode = node => node.url.split("/")[3];
                        const buildUrl = node => makeUrl(getUrlFromNode(node));
                        const buildVideosSearchUrl = query => `https://cklvhhyl66.execute-api.us-east-1.amazonaws.com/?word=${query}`
                        const getVideosFromServer = query => fetch(buildVideosSearchUrl(query)).then(response => response.json())
                        const makeUrlSentences = filename => `https://sentencesigns.s3.amazonaws.com/${encodeURI(filename)}`;
                        const getUrlFromNodeSentences = node => node.urlSentence.split("/")[3];
                        const buildUrlSentence = node => makeUrlSentences(getUrlFromNodeSentences(node));
                        const mapValues = nodes => nodes.map(node => ({label: node.sign_gloss, imageUrl: buildUrl(node)}));
                        const mapValuesSentences = nodes => nodes.map(node => ({label: node.text, imageUrl: buildUrlSentence(node)}));
                        const getListElement = () => document.getElementById("products-list");
                        const getListSentence = () => document.getElementById("sentences-list");
                        const buildVideoNode = ({ label, imageUrl}) =>
                        `
                            <div class="product-item" category="adjectives">
                                <video height="205px" width="205px" controls>
                                    <source src="${imageUrl}" type="video/mp4">
                                </video>
                                <a href="#">${label}</a>
                            </div>
                        `

                        const appendVideo = node => {
                            getListElement().insertAdjacentHTML('beforeend', buildVideoNode(node))
                        }
                        getListElement().innerHTML = "";
                        getVideosFromServer(queryString).then(mapValues).then(nodes => nodes.map(appendVideo));
                        */
                    });
                    getListElement.appendChild(ul);
                    //document.getElementById("products-list").appendChild(ul);


                        loader.style.visibility = "hidden"
                        loader.style.marginTop = "0%"
                        loader.style.marginBottom = "0%"
                }
            });


        });
        
        
        
          cancelUpload.addEventListener('click', function() {
                if(request.abort()){
                    percentage.innerHTML = "Uploading has been canceled.";
                }
            });



// here you can use the result of promiseB
});
       
        // Retrieve recorded video as blob and display in the preview element
       /* var videoBlob = recorder.getBlob();
        var blobUrl = URL.createObjectURL(videoBlob);
        console.log("Blob url" + blobUrl);
         video.srcObject = videoBlob*/
      //  video.src = URL.createObjectURL(videoBlob);
        video.play();

        // Unmute video on preview
        video.muted = false;

        // Stop the device streaming
        recorder.stream.stop();

        // Enable record button again !
        document.getElementById('btn-start-recording').disabled = false;
    }).catch(function(error) {
        console.error('stopRecording failure', error);
    });
};