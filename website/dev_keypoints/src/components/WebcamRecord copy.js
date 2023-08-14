import React, { useEffect, useRef, useState, useCallback} from "react";
import Webcam from "react-webcam";
import { Holistic } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";


const useCleanupHolistic = (holisticRef, camera) => {
  useEffect(() => {
    return () => {
      console.log("Closing holistic model");
      holisticRef.current.close();
      if (camera) {
        camera = null;
      }
    };
  }, []);
};

export default function WebcamRecord() {
  
  // CAMERA
  const webcamRef = useRef(null);
  var camera = null;
  let recordingRef = useRef(false);
  let inPreparationRef = useRef(false)

  // RECORDS
  const [capturing, setCapturing] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const isInitialMount = useRef(true);
  const mediaRecorderRef = useRef(null);
  const [accumulatedResults, setAccumulatedResults] = useState([]);
  let timeToRecordRef = useRef(2500); // time to record

  // MODEL  
  const holisticRef = useRef(null);
  let doHolisticRef = useRef(false);
  let detectingHandsRef = useRef(true);

  // CANVAS MESSAGE
  const canvasRef = useRef(null);
  let messageRef = useRef("");
  /*
  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
    } else {
      if (!capturing) {
        console.log('running handleDownload');
        //handleModelProcess();
      }
    }
  }, [capturing]);

  const handleStartCapture = React.useCallback(() => {
    setCapturing(true);
    mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
      mimeType: 'video/webm',
    });
    mediaRecorderRef.current.addEventListener(
      'dataavailable',
      handleDataAvailable
    );
    mediaRecorderRef.current.start();
    doHolisticRef.current = true
    setTimeout(handleStopCapture, timeToRecordRef.current); // Stop recording after 2 seconds
  }, [webcamRef, setCapturing, mediaRecorderRef]);

  const handleDataAvailable = React.useCallback(
    ({ data }) => {
      if (data.size > 0) {
        console.log(data.size)
        setRecordedChunks((prev) => prev.concat(data));
      }
    },
    [setRecordedChunks]
  );

  const handleStopCapture = React.useCallback(() => {
    console.log("deteniendo")
    mediaRecorderRef.current.stop()
    doHolisticRef.current = false
    setCapturing(false);

  }, [mediaRecorderRef, webcamRef, setCapturing]);

  function dataURLtoFile(dataurl, filename) {
      var arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1],
          bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
      while(n--){
          u8arr[n] = bstr.charCodeAt(n);
      }
      return new File([u8arr], filename, {type:mime});
  }
  */
  const handleStopCapture = React.useCallback(() => {
    console.log("deteniendo")
    doHolisticRef.current = false
    setCapturing(false);
  }, [webcamRef, setCapturing]);

  const handleStartCapture = React.useCallback(() => {
    setCapturing(true);
    doHolisticRef.current = true
    setTimeout(handleStopCapture, timeToRecordRef.current); // Stop recording after 2 seconds
  }, [webcamRef, setCapturing]);

  // GIVE THE USER WITH SOME SECONDS BEFORE RECORDING
  const preparation = useCallback(() =>{

    console.log("Preparation")
    let timeLeft = 3;
    var videoTimer = setInterval(function(){
      messageRef.current = "Prepárate...\n"+timeLeft
        if(timeLeft === 0){
          inPreparationRef.current = false
          console.log(recordingRef)
          recordingRef.current = true;
          console.log(recordingRef)
          clearInterval(videoTimer);
          console.log("grabando")
          handleStartCapture();
        }
        timeLeft-=1;
    }, 600)

  }, [])

  // MODEL HAND FACE PROCESSING
  function onResults(results) {

    if(!inPreparationRef.current){
      messageRef.current = "";
    }

    if (results) {

      // 2 HANDS and FACE SQUARE
      if (detectingHandsRef.current){
        if (results.poseLandmarks){
          const {
            poseLandmarks: {
              15: { x: x15, y: y15 },
              17: { x: x17, y: y17 },
              19: { x: x19, y: y19 },
              16: { x: x16, y: y16 },
              18: { x: x18, y: y18 },
              20: { x: x20, y: y20 },
              2: { x: x2, y: y2 },
              5: { x: x5, y: y5 },
            },
          } = results;
    
          const left_hand_x = (x15 + x17 + x19) / 3;
          const left_hand_y = (y15 + y17 + y19) / 3;
    
          const right_hand_x = (x16 + x18 + x20) / 3;
          const right_hand_y = (y16 + y18 + y20) / 3;
    
          const eye_x = (x2 + x5) / 2;
          const eye_y = (y2 + y5) / 2;
    
          if (
            left_hand_x > 0.6 &&
            right_hand_x <= 0.3 &&
            left_hand_x < 1.0 &&
            right_hand_x > 0.0 &&
            left_hand_y > 0.77 &&
            right_hand_y > 0.77 &&
            left_hand_y < 1.0 &&
            right_hand_y < 1.0 &&
            eye_y < 0.33 &&
            eye_y > 0.0 &&
            eye_x < 0.6 &&
            eye_x > 0.3
          ) {
            console.log("detected")
            detectingHandsRef.current =  false
            inPreparationRef.current = true
            preparation()
          }
        }
      }
      if(doHolisticRef.current){
        console.log(results)
        setAccumulatedResults(prevResults => [...prevResults, results]); // Accumulate the results
      }
    }
  }

  useEffect(() => {
    if (accumulatedResults.length > 0 && !doHolisticRef.current) {
      // Perform your desired process with the accumulated results
      console.log(accumulatedResults.length);
      console.log(accumulatedResults);
      // ...other code...
    }
  }, [accumulatedResults]);

  // INITIALIZE THE MODEL
  const initializeHolistic = async() => { 

    messageRef.current = "preparando modelos"

    const holistic = new Holistic({
      locateFile: (file) => {
        return `./mediapipe/holistic/${file}`;
      },
    });
    holistic.setOptions({
      modelComplexity: 1,
    });

    holisticRef.current = holistic;
    holistic.onResults(onResults);

    // CREATE WEB CAMERA
      
    if (typeof webcamRef.current !== "undefined" && webcamRef.current !== null) {
      const camera = new Camera(webcamRef.current.video, {
        onFrame: async () => {
          if (webcamRef.current &&
              webcamRef.current.video &&
              holisticRef.current &&
              (detectingHandsRef.current || true)){

            await holisticRef.current.send({ image: webcamRef.current.video });
          }
        },
      });
      await camera.start();  
      // Update the message to an empty string
    } 
  }

  const handleUserMedia = (stream) => {
    initializeHolistic();
  };

  useCleanupHolistic(holisticRef, camera);

  return (
    <div className="BusquedaSeña__video-preview">
      <Webcam
          className="BusquedaSeña_Webcam_component"
          audio={false}
          ref={webcamRef}
          onUserMedia={handleUserMedia}
          videoConstraints={{
            facingMode: "user",
          }}/>
      
      {capturing ? (
        <button className="btn btn-danger" onClick={handleStopCapture}>
          Stop Capture
        </button>
      ) : (
        <button className="btn btn-danger" onClick={handleStartCapture}>
          Start Capture
        </button>
      )}
    </div>
  );
}