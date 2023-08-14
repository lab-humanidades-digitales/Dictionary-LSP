/* eslint-disable no-debugger */
import { Holistic } from "@mediapipe/holistic";
import { Camera } from "utils/mediapipes-camera";
import { useRef, useState } from "react";
import { RECORDING } from 'constants'

export const useHolistic = () => {
    const [isInitializingHolistic, setIsInitializingHoolistic] = useState(false);
    const [isInitializingCamera, setIsInitializingCamera] = useState(false);

    const holisticRef = useRef(null);
    const cameraRef = useRef(null);
    const webcamRef = useRef(null);
    const mediaRecorderRef = useRef(null);

    const processFrame = useRef(false);
    const recordFrame = useRef(false);

    const onFrameResult = useRef(null);
    const recordedMedia = useRef([]);
    const recordedLandmarks = useRef([]);

    const reset = async () => {
        if (cameraRef.current)
            cameraRef.current.stop();
        processFrame.current = false;

        stopRecording();

        recordedMedia.current = [];
        recordedLandmarks.current = [];
    };

    const onHolisticResults = (results) => {
        if (!processFrame.current) return;
        if (!results) return;
        if (!results.poseLandmarks) return;

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

        /*const inStartPosition = {
            leftHand:
                left_hand_x > 0.560 &&
                left_hand_x < 0.710 &&
                left_hand_y > 0.554 &&
                left_hand_y < 0.821,
            rightHand:
                right_hand_x > 0.290 &&
                right_hand_x < 0.440 &&
                right_hand_y > 0.554 &&
                right_hand_y < 0.821,
            face:
                eye_x > 0.400 &&
                eye_x < 0.560 &&
                eye_y > 0.143 &&
                eye_y < 0.321,
            old: {
                leftHand:
                    left_hand_x > 0.6 &&
                    left_hand_x < 1.0 &&
                    left_hand_y > 0.77 &&
                    left_hand_y < 1.0,
                rightHand:
                    right_hand_x <= 0.3 &&
                    right_hand_x > 0.0 &&
                    right_hand_y > 0.77 &&
                    right_hand_y < 1.0,
                face:
                    eye_y < 0.33 &&
                    eye_y > 0.0 &&
                    eye_x < 0.6 &&
                    eye_x > 0.3,
            },
            all: false
        }*/

        const inStartPosition = {
            leftHand: false,
            rightHand: false,
            face: false,
            all: false
        }

        if (Math.abs(eye_x - 0.480) < 0.080 && Math.abs(eye_y - 0.232) < 0.089) {
            //console.log('faceTemplate = ' + JSON.stringify(results.faceLandmarks))
            inStartPosition.face = true;
        }

        if (Math.abs(left_hand_x - 0.635) < 0.075 && Math.abs(left_hand_y - 0.688) < 0.134) {
            //console.log('leftHandTemplate = ' + JSON.stringify(results.leftHandLandmarks))
            inStartPosition.leftHand = true;
        }

        if (Math.abs(right_hand_x - 0.365) < 0.075 && Math.abs(right_hand_y - 0.688) < 0.134) {
            //console.log('rightHandTemplate = ' + JSON.stringify(results.rightHandLandmarks))
            inStartPosition.rightHand = true;
        }

        inStartPosition.all = inStartPosition.leftHand && inStartPosition.rightHand && inStartPosition.face;

        results.inStartPosition = inStartPosition;

        if (onFrameResult.current)
            onFrameResult.current(results)

        if (recordFrame.current)
            recordedLandmarks.current.push(results)
    };

    const initHolistic = async (onFrameResultFunc) => {
        setIsInitializingHoolistic(true)
        onFrameResult.current = onFrameResultFunc;

        if (!holisticRef.current) {
            const holistic = new Holistic({
                locateFile: (file) => {
                    return `./mediapipe/holistic/${file}`;
                },
            });

            holistic.setOptions({
                modelComplexity: 0,
                useCpuInference: true,
                refineFaceLandmarks: false,
            });

            holisticRef.current = holistic;
        }

        holisticRef.current.onResults(onHolisticResults);
        setIsInitializingHoolistic(false)
    };

    const initCamera = async (deviceId) => {
        if (!webcamRef.current)
            throw new Error('La cámara no está disponible')

        if (cameraRef.current) {
            cameraRef.current.stop();
            cameraRef.current = null;
        }
        const r = Math.random();

        cameraRef.current = new Camera(webcamRef.current.video, {
            onFrame: async () => {
                if (!webcamRef.current) return;
                if (!holisticRef.current) return;
                if (!processFrame.current) return;

                try {
                    await holisticRef.current.send({ image: webcamRef.current.video });
                } catch (ex) {
                    console.log(ex)
                }
            },
        });
        await cameraRef.current.start(deviceId);
        processFrame.current = true;
        setIsInitializingCamera(true)
    };

    const startRecording = () => {

        recordedMedia.current = [];
        recordedLandmarks.current = [];

        if (webcamRef.current && webcamRef.current.stream) {
            mediaRecorderRef.current = new MediaRecorder(cameraRef.current.g, {
                mimeType: "video/webm"
            });

            mediaRecorderRef.current.ondataavailable = function ({ data }) {
                if (data.size > 0) {
                    recordedMedia.current.push(data);
                }
            }

            mediaRecorderRef.current.start(RECORDING.FPS);
        }

        recordFrame.current = true;
    }

    const stopRecording = () => {
        recordFrame.current = false;

        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stop();
            mediaRecorderRef.current.ondataavailable = undefined;
        }
        const blob = new Blob(recordedMedia.current, {
            type: "video/webm"
        });

        const url = URL.createObjectURL(blob);

        /* PARA DESCARGAR EL VIDEO*/
        /*const a = document.createElement("a");
        document.body.appendChild(a);
        a.style = "display: none";
        a.href = url;
        a.download = "react-webcam-stream-capture.webm";
        a.click();
        window.URL.revokeObjectURL(url);*/

        return {
            uuid: Date.now(),
            landmarks: recordedLandmarks.current,
            media: recordedMedia.current,
            mediaBlob: blob,
            mediaUrl: url,
        }
    }

    return {
        reset,
        webcamRef,

        isInitializingHolistic,
        initHolistic,
        holisticRef,

        isInitializingCamera,
        initCamera,
        cameraRef,
        processFrame,

        startRecording,
        stopRecording,
        recordFrame
    };
};
