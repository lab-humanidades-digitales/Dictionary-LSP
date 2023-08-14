import React, { useEffect, useRef, useState, useCallback } from "react";
import { Holistic } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";
import AWS from 'aws-sdk';

export default function WebcamRecord() {
  // CAMERA
  const webcamRef = useRef(null);
  const cameraRef = useRef(null);
  const recordingRef = useRef(false);
  const inPreparationRef = useRef(false);
  const streamRef = useRef(null);
  
  // RECORDS
  const [accumulatedResults, setAccumulatedResults] = useState([]);
  const [capturing, setCapturing] = useState(false);
  const isInitialMount = useRef(true);

  // MODEL
  const holisticRef = useRef(null);
  const doHolisticRef = useRef(false);
  const detectingHandsRef = useRef(true);

  // CHUNKS
  const recordedChunksRef = useRef([]);

  //SQUARES
  const [showSquares, setShowSquares] = useState(false)

  // CANVAS MESSAGE
  const messageRef = useRef("");
  // eslint-disable-next-line
  const [message, setMessage] = useState("");

  // AWS
  const bucketRef = useRef("");
  const [results, setResults] = useState([]);


  // Function to set the message value
  const updateMessage = useCallback((newMessage) => {
    setMessage(newMessage);
    messageRef.current = newMessage
  },[]);

  ///////////////////////////////
  // AWS
  ///////////

  useEffect(() => {
    AWS.config.region = 'us-east-1';

    AWS.config.update({
      region: 'us-east-1',
      credentials: new AWS.CognitoIdentityCredentials({
        IdentityPoolId: 'us-east-1:b5013574-2741-4e18-97be-9395b5929162',
      }),
    });
  
    AWS.config.credentials.get((err) => {
      if (err) {
        console.error('Error retrieving AWS credentials:', err);
      } else {
        console.log('AWS credentials successfully initialized');
    
        // You can now use the credentials to make authenticated requests to other AWS services
      }
    });
    
    var bucketName = 'user-video-test'; // Enter your bucket name+
    bucketRef.current = new AWS.S3({
      params: {
        Bucket: bucketName
      }
    });
  },[])

  // Enviar video a S3
  function uploadToS3(blob, filename) {

    console.log("subiendo a S3")
    var uploadParams = {Key: filename, ContentType: 'video/webm', Body: blob};
    bucketRef.current.upload(uploadParams, function(err, data) {
      if (err) {
        console.log('Error uploading file:', err);
      } else {
        console.log('File uploaded successfully:', data.Location);

        // Solamente se le envia el nombre del video a AWS Lambda 

        /**
         * El proceso es el siguiente:
         * 
         *  - 1) Se envía el video a S3
         *  - 2) Se envía el nombre del video a lambda
         *  - 3) lambda se lo envía a Sagemaker con otro formato
         *  - 4) Sagemaker recupera el video y lo procesa
         *  - 5) le devuelve el resultado a Lambda
         *  - 6) se devuelve el resultado a la web 
         * 
         */
        const lambda = new AWS.Lambda();
        const lambdaParams = {
          FunctionName: 'sagemaker-invoker',
          Payload: JSON.stringify({
            video: uploadParams.Key
          })
        };

        lambda.invoke(lambdaParams, function(err, data) {
          if (err) {
            console.log('Error invoking Lambda function:', err);
          } else {
            const glossListStr = data.Payload;
            const glossList = JSON.parse(glossListStr);
            const words = [];

            for (const key in glossList) {
              if (glossList.hasOwnProperty(key)) {
                const glossObj = glossList[key];
                const gloss = glossObj.gloss;
                words.push(gloss);
              }
            }
            console.log(words);
            setResults(words);
            detectingHandsRef.current = true;
            setShowSquares(true)
            updateMessage("¿Nueva predicción? ubíquese correctamente");
            // Perform further operations with the Lambda function's response
          }
        });
      }
    });
  }

  ///////////////////////////////
  // DURANTE LA GRABACIÓN
  ///////////

  // Function to start the capture
  function handleStartCapture() {
    updateMessage("grabando");
    setCapturing(true);

    try {
      const mediaRecorder = new MediaRecorder(streamRef.current, { mimeType: "video/webm" });

      mediaRecorder.ondataavailable = (event) => {
        recordedChunksRef.current.push(event.data);
      };
      mediaRecorder.start(100);
      setCapturing(true);
      setTimeout(() => {
        updateMessage("procesando...");
        mediaRecorder.stop();
        setCapturing(false);
        handleStopCapture();
      }, 2000);
    } catch (e) {
      console.error('Exception while creating MediaRecorder: ' + e);
      return;
    }
  }

  const handleStopCapture = useCallback(async () => {
    const recordedChunks = recordedChunksRef.current;
    const blob = new Blob(recordedChunks, { type: "video/webm" });
    console.log("Recorded chunks:", recordedChunks.length);
    console.log("Blob:", blob);

    try {
      const randomNumber = Math.random()
      const result = await uploadToS3(blob, randomNumber +'.webm');
      console.log('Upload successful:', result);
    } catch (error) {
      console.error('Upload failed:', error);
    }

    // Further processing or uploading of the recorded video blob
    // e.g., send it to the server or save it locally

    recordedChunksRef.current = [];
    
  }, []);

  ///////////////////////////////
  // CONTEO HACIA ATRAS (ANTES DE GRABAR)
  ///////////

  // Give the user some seconds before recording
  const preparation = () => {
    console.log("Preparation");
    let timeLeft = 3;
    const videoTimer = setInterval(() => {
      updateMessage(`Prepárate...\n${timeLeft}`);
      if (timeLeft < 0) {
        inPreparationRef.current = false;
        recordingRef.current = true;
        clearInterval(videoTimer);
        console.log("grabando");
        handleStartCapture();
      }
      timeLeft -= 1;
    }, 900);
  };

  ///////////////////////////////
  // FUNCIÓN DURANTE LA DETECCIÓN DE KEYPOINTS
  ///////////

  // Function to get keypoint landmarks from mediapipe
  function onResults(results) {

    /**
     * Mediapipe ya devuelve los resultados normalizado
     * Por lo que no es necesario calcular el porcentaje
     * con respecto al tamaño de la cámara.
     * 
     * el valor de los keypoints suelen estar entre 0 a 1
     * pero puede ser negativo o mayor a uno si el modelo
     * predice que el punto podria estar ligeramente
     * fuera del video
     * 
     * Más información del modelo en este repositorio:
     * https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md
     */

    if(messageRef.current === "Cargando modelos..."){
      updateMessage("Póngase en posición")
      setShowSquares(true)
    }

    if (results) {
      // 2 HANDS and FACE SQUARE
      if (detectingHandsRef.current) {
        if (results.poseLandmarks) {
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
            console.log("detected");
            detectingHandsRef.current = false;
            inPreparationRef.current = true;
            setShowSquares(false)
            preparation();
          }
        }
      }

      if (doHolisticRef.current) {
        setAccumulatedResults((prevResults) => [...prevResults, results]); // Accumulate the results
      }
    }
  }


  ///////////////////////////////
  // INICIARLIZAR LA CÁMARAEL MODELO Y LA CÁMARA
  ///////////
  const prepareCameraHolistic = () => {
    return new Promise((resolve, reject) => {
      // Create video element
      const videoElement = webcamRef.current;
  
      if (videoElement) {
        // CameraUtils is a separate library that provides the Camera class
        cameraRef.current = new Camera(videoElement, {
          onFrame: async () => {
            if (
              videoElement &&
              holisticRef.current &&
              (detectingHandsRef.current || doHolisticRef.current)
            ) {
              await holisticRef.current.send({ image: videoElement });
            }
          },
        });
        cameraRef.current.start();
  
        resolve();
      } else {
        reject(new Error('No se encontró referencia al elemento de video'));
      }
    });
  };

  // Initialize the Holistic model
  const initializeHolistic = () => {
    return new Promise((resolve, reject) => {
      const holistic = new Holistic({
        locateFile: (file) => {  //aquí se hace la importancion desde la carpeta local en public
          return `./mediapipe/holistic/${file}`;
        },
      });
      holistic.setOptions({
        modelComplexity: 1,
      });
  
      holisticRef.current = holistic;
      holistic.onResults(onResults);
  
      resolve(); 
    });
  };

  const initializeModel = () => {
    updateMessage("Cargando modelos...")

    //PROMISE
    initializeHolistic().then((response) => {
      //PROMISE
      prepareCameraHolistic().then(()=>{
        console.log("Camera Holistic inicializado")
      })
      .catch((error) => {
        console.log("Error en cameraHolistic", error);
        updateMessage("Error al cargar la camara holistic");
      });

    })
    .catch((error) => {
      console.log("Error en initializeHolistic", error);
      updateMessage("Error al cargar los modelos");
    });
  };
  
  ///////////////////////////////
  // TERMINAR LOS PROCESOS ABIERTOS AL CERRAR, ACTUALIZAR O CAMBIAR DE PESTAÑA
  ///////////
  useEffect(() => {
    return () => {
      console.log("Closing camera");
      if (cameraRef.current && cameraRef.current.close) {
        cameraRef.current.close();
      }
      if(streamRef.current){
        streamRef.current = null
      }
    };
  }, []);

  ///////////////////////////////
  // INICIARLIZAR LA CÁMARA
  ///////////
  useEffect(() => {
    const handleUserMedia = async () => {
      try {
        streamRef.current = await navigator.mediaDevices.getUserMedia({ video: true });
        console.log("Miraaaa ->", streamRef.current)
        webcamRef.current.srcObject = streamRef.current;

        // Se realiza esta acción para asegurarse que el video está completamente cargado
        webcamRef.current.onloadedmetadata = () => {
          console.log("Webcam stream loaded");
          setMessage("Webcam stream loaded");

          initializeModel()
        };
      } catch (error) {
        console.log("Error accessing webcam:", error);
        setMessage("Error accessing webcam");
      }
    };

    handleUserMedia();

    return () => {

      if (cameraRef.current) {
        cameraRef.current.stop();
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  // This line and comment ensure that the camera load once at the beginning
  // eslint-disable-next-line
  }, []);

  ///////////////////////////////
  // html render
  ///////////
  return (
    <div className="BusquedaSeña__webcam-container">
      <div className="BusquedaSeña__contenedor-video">
        <video ref={webcamRef} autoPlay playsInline className="BusquedaSeña__video" />
        {showSquares && <div className="BusquedaSeña__cuadrado_1"></div>}
        {showSquares && <div className="BusquedaSeña__cuadrado_2"></div>}
        {showSquares && <div className="BusquedaSeña__cuadrado_3"></div>}
      </div>
      <div className="BusquedaSeña__message">{messageRef.current}</div>
     
      {results.length !== 0 &&  <div className="BusquedaSeña__results">
        <div className="BusquedaSeña__resultsTitle">Resultados</div>
        {results.map((word, index) => (
          <div className="BusquedaSeña__resultsWord" key={index}>{word}</div>
        ))}
      </div>}

     </div>
  );
}
