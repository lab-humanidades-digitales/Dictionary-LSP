import PropTypes from "prop-types";

import { useState, forwardRef, useRef, useImperativeHandle, useCallback, useEffect } from "react";

import { drawConnectors, drawLandmarks, lerp } from "@mediapipe/drawing_utils";
import {
  FACEMESH_TESSELATION,
  POSE_CONNECTIONS,
  HAND_CONNECTIONS,
  FACE_GEOMETRY,
  FACEMESH_FACE_OVAL,
  FACEMESH_CONTOURS,
} from "@mediapipe/holistic";
// @mui material components
import { Modal, Divider, Slide, LinearProgress, MenuItem, Menu, Tooltip } from "@mui/material";

// @mui icons
import CloseIcon from "@mui/icons-material/Close";

import SwitchVideoIcon from "@mui/icons-material/SwitchVideo";
import CameraswitchIcon from "@mui/icons-material/Cameraswitch";
import RadioButtonCheckedIcon from "@mui/icons-material/RadioButtonChecked";
import StopIcon from "@mui/icons-material/Stop";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import SlowMotionVideoIcon from "@mui/icons-material/SlowMotionVideo";

// Material Kit 2 React components
import MKBox from "components/MKBox";
import MKButton from "components/MKButton";
import MKTypography from "components/MKTypography";

import Webcam from "react-webcam";
import { useHolistic } from "../hooks/useMediapipe";
import { usePersistentConfig } from "hooks/usePersistentConfig";
import { RECORDING } from "constants";
import { leftHandTemplate, rightHandTemplate, faceTemplate } from "utils/landmark-helper";

import InstructionsModal from "./InstructionsModal";

const RecordSignModal = forwardRef(({ onRecorded }, ref) => {
  const { showRecordingPlayStopButtons, storedDeviceId, setStoredDeviceId } = usePersistentConfig();

  const [selectCameraEl, setSelectCameraEl] = useState(null);
  const [mirrorCamera, setMirrorCamera] = useState(true);
  const [deviceId, setDeviceId] = useState(storedDeviceId);
  const [devices, setDevices] = useState([]);
  const [show, setShow] = useState(false);
  const toggleModal = () => setShow(!show);

  const instructionsModalRef = useRef(null);

  const {
    webcamRef,

    isInitializingHolistic,
    initHolistic,
    holisticRef,

    isInitializingCamera,
    initCamera,
    cameraRef,
    reset: mediaPipeReset,
    recordFrame: isRecording,

    startRecording: startRecordingHolistic,
    stopRecording: stopRecordingHolistic,
  } = useHolistic();

  useImperativeHandle(ref, () => ({
    showModal(visible = true) {
      setShow(visible);
    },
  }));

  // #region Camera UI
  const handleDevices = useCallback(
    (mediaDevices) =>
      setDevices(mediaDevices.filter(({ kind, deviceId }) => kind === "videoinput" && deviceId)),
    [setDevices]
  );

  const showCameraList = (event) => {
    setSelectCameraEl(event.currentTarget);
  };

  const closeCameraList = (selectedDeviceId) => {
    if (selectedDeviceId) {
      setDeviceId(selectedDeviceId);
      setStoredDeviceId(selectedDeviceId);
    }

    setSelectCameraEl(null);
  };

  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then(handleDevices);
  }, [handleDevices]);

  const onWebcamUserMedia = async () => {
    clearCanvas();
    await mediaPipeReset();
    await initHolistic(onHolisticResults);
    await initCamera(deviceId);
  };

  // #endregion Camera UI

  // #region Paint UI
  const clearCanvas = () => {
    const canvas = document.getElementById("canvas");
    const canvasCtx = canvas.getContext("2d");
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const paintTemplateLandmarks = (results) => {
    if (isRecording.current) return;

    const canvas = document.getElementById("canvas");
    const canvasCtx = canvas.getContext("2d");

    canvasCtx.save();
    canvasCtx.globalCompositeOperation = "source-over";

    /*
      FACEMESH_TESSELATION
      FACE_GEOMETRY,
      FACEMESH_FACE_OVAL,
      FACEMESH_CONTOURS,
*/

    //FACEMESH
    if (!results.inStartPosition.face) {
      drawConnectors(canvasCtx, faceTemplate, FACEMESH_CONTOURS, {
        color: "#ffffff80",
        lineWidth: 1,
      });
    }

    //LEFT HAND
    if (!results.inStartPosition.leftHand) {
      drawConnectors(canvasCtx, leftHandTemplate, HAND_CONNECTIONS, {
        color: "#ffffff80",
        lineWidth: 2,
      });
      drawLandmarks(canvasCtx, leftHandTemplate, {
        radius: (data) => lerp(data.from.z, -0.15, 0.1, 5, 1),
        color: "#ffffff40",
        fillColor: "transparent",
        lineWidth: 1,
      });
    }

    //RIGHT HAND
    if (!results.inStartPosition.rightHand) {
      drawConnectors(canvasCtx, rightHandTemplate, HAND_CONNECTIONS, {
        color: "#ffffff80",
        lineWidth: 2,
      });
      drawLandmarks(canvasCtx, rightHandTemplate, {
        radius: (data) => lerp(data.from.z, -0.15, 0.1, 5, 1),
        color: "#ffffff40",
        fillColor: "transparent",
        lineWidth: 1,
      });
    }
    canvasCtx.restore();
  };

  const paintLandmarks = (results) => {
    if (isRecording.current) return;

    const canvas = document.getElementById("canvas");
    const canvasCtx = canvas.getContext("2d");

    canvasCtx.save();
    canvasCtx.globalCompositeOperation = "source-over";

    //POSE
    /*
    drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {
      color: (data) => {
        // eslint-disable-next-line no-debugger
        debugger;

        if (
          landmarkPaintingOptions.hidePoseLandmarks[POSE_CONNECTIONS[data.index][0]] &&
          landmarkPaintingOptions.hidePoseLandmarks[POSE_CONNECTIONS[data.index][1]]
        )
          return "transparent";
        else return "#C0C0C070";
      },
      lineWidth: 1,
    });
    drawLandmarks(canvasCtx, results.poseLandmarks, {
      radius: (data) => lerp(data.from.z, -0.15, 0.1, 5, 1) / 2,
      fillColor: (data) => {
        // eslint-disable-next-line no-debugger

        return landmarkPaintingOptions.hidePoseLandmarks[data.index]
          ? "transparent"
          : "transparent";
      },
      color: (data) =>
        landmarkPaintingOptions.hidePoseLandmarks[data.index] ? "transparent" : "cyan",
      lineWidth: 1,
    });
    */

    //FACEMESH
    /*
      FACEMESH_TESSELATION
      FACE_GEOMETRY,
      FACEMESH_FACE_OVAL,
      FACEMESH_CONTOURS,
    */

    const faceColor = results.inStartPosition.face ? "#00CC0070" : "#CC000070";
    drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_FACE_OVAL, {
      color: faceColor,
      lineWidth: 1,
    });

    //LEFT HAND
    const leftHandColor = results.inStartPosition.leftHand ? "#00CC0070" : "#CC000070";
    drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, {
      color: leftHandColor,
      lineWidth: 2,
    });
    drawLandmarks(canvasCtx, results.leftHandLandmarks, {
      radius: (data) => lerp(data.from.z, -0.15, 0.1, 5, 1),
      fillColor: "transparent",
      lineWidth: 1,
    });

    //RIGHT HAND
    const rightHandColor = results.inStartPosition.rightHand ? "#00CC0070" : "#CC000070";
    drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, {
      color: rightHandColor,
      lineWidth: 2,
    });
    drawLandmarks(canvasCtx, results.rightHandLandmarks, {
      radius: (data) => lerp(data.from.z, -0.15, 0.1, 5, 1),
      fillColor: "transparent",
      lineWidth: 1,
    });

    canvasCtx.restore();
  };

  // #endregion Paint UI

  // #region Recording

  const startRecording = () => {
    startRecordingHolistic();
    clearCanvas();
  };

  const endRecording = () => {
    const data = stopRecordingHolistic();
    if (onRecorded) onRecorded(data);
    setShow(false);
  };

  // #endregion Recording

  // #region Countdown

  const [countdownLabel, setCountdownLabel] = useState(RECORDING.PREPARATION_TIME);
  const [showCountdown, setShowCountdown] = useState(false);
  const countdown = useRef(RECORDING.PREPARATION_TIME);
  const inCountdown = useRef(false);
  const timer = useRef(null);

  const startCountdown = () => {
    inCountdown.current = true;
    countdown.current = RECORDING.PREPARATION_TIME;
    setCountdownLabel(countdown.current);
    setShowCountdown(true);

    timer.current = setTimeout(() => countdownStep(), 1000);
  };

  const countdownStep = () => {
    if (countdown.current == 0) {
      startRecording();
      setShowCountdown(false);
      timer.current = setTimeout(() => endRecording(), RECORDING.RECORDING_TIME);
      inCountdown.current = false;
    } else {
      countdown.current = countdown.current - 1;
      setCountdownLabel(countdown.current);
      timer.current = setTimeout(() => countdownStep(), 1000);
    }
  };

  const resetCountdown = () => {
    if (timer.current) clearTimeout(timer.current);
    inCountdown.current = false;
    setShowCountdown(false);
  };

  // #endregion Countdown

  const onHolisticResults = (results) => {
    clearCanvas();

    paintTemplateLandmarks(results);
    paintLandmarks(results);

    if (results.inStartPosition.all && !isRecording.current && !inCountdown.current) {
      startCountdown();
    }
  };

  useEffect(() => {
    return () => {
      mediaPipeReset();
      resetCountdown();
    };
  }, [show]);

  return (
    <>
      <InstructionsModal ref={instructionsModalRef}></InstructionsModal>

      <Modal open={show} onClose={toggleModal} sx={{ display: "grid", placeItems: "center" }}>
        <Slide
          direction="down"
          in={show}
          timeout={500}
          sx={{ width: { xs: "auto", md: 500 }, margin: { xs: 1, md: 0 } }}
        >
          <MKBox
            position="relative"
            width="500px"
            display="flex"
            flexDirection="column"
            borderRadius="xl"
            bgColor="white"
            shadow="xl"
          >
            <MKBox display="flex" alignItems="center" justifyContent="space-between" p={2}>
              <MKTypography variant="h5">Grabar seña</MKTypography>
              <MKBox display="flex" alignItems="center" justifyContent="space-between">
                {inCountdown.current && (
                  <RadioButtonCheckedIcon
                    fontSize="medium"
                    className="countdown-pulse"
                    sx={{ mr: 3 }}
                  />
                )}

                {isRecording.current && (
                  <RadioButtonCheckedIcon
                    fontSize="medium"
                    className="recording-pulse"
                    sx={{ mr: 3, color: "#e7464c" }}
                  />
                )}

                {showRecordingPlayStopButtons && (
                  <Tooltip title="Iniciar grabación">
                    <SlowMotionVideoIcon
                      fontSize="medium"
                      sx={{ cursor: "pointer", mr: 3 }}
                      onClick={() => startCountdown()}
                    />
                  </Tooltip>
                )}

                {showRecordingPlayStopButtons && (
                  <PlayArrowIcon
                    fontSize="medium"
                    sx={{ cursor: "pointer", mr: 3 }}
                    onClick={() => startRecording()}
                  />
                )}
                {showRecordingPlayStopButtons && (
                  <StopIcon
                    fontSize="medium"
                    sx={{ cursor: "pointer", mr: 3 }}
                    onClick={() => endRecording()}
                  />
                )}
                {showRecordingPlayStopButtons && (
                  <Tooltip title="Invertir cámara">
                    <SwitchVideoIcon
                      fontSize="medium"
                      sx={{ cursor: "pointer", mr: 3 }}
                      onClick={() => setMirrorCamera(!mirrorCamera)}
                    />
                  </Tooltip>
                )}
                {showRecordingPlayStopButtons && devices && devices.length > 0 && (
                  <>
                    <Tooltip title="Cambiar cámara">
                      <CameraswitchIcon
                        fontSize="medium"
                        sx={{ cursor: "pointer", mr: 3 }}
                        onClick={showCameraList}
                      />
                    </Tooltip>
                    <Menu
                      id="lock-menu"
                      anchorEl={selectCameraEl}
                      open={selectCameraEl != null}
                      onClose={() => closeCameraList()}
                      MenuListProps={{
                        "aria-labelledby": "lock-button",
                        role: "listbox",
                      }}
                    >
                      {devices.map((device, index) => (
                        <MenuItem
                          onClick={() => closeCameraList(device.deviceId)}
                          key={device.deviceId}
                        >
                          {device.label}
                        </MenuItem>
                      ))}
                    </Menu>
                  </>
                )}
                <CloseIcon fontSize="medium" sx={{ cursor: "pointer" }} onClick={toggleModal} />
              </MKBox>
            </MKBox>
            <Divider sx={{ my: 0 }} />
            <MKBox p={2}>
              <div style={{ position: "relative" }}>
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  videoConstraints={{ deviceId: deviceId }}
                  mirrored={mirrorCamera}
                  width={"100%"}
                  style={{ borderRadius: 8, marginBottom: -10, width: "100%" }}
                  onUserMedia={(stream) => onWebcamUserMedia(stream)}
                />
                <canvas
                  id="canvas"
                  style={{
                    position: "absolute",
                    left: 0,
                    width: "100%",
                    top: 0,
                    height: "100%",
                    transform: mirrorCamera ? "scaleX(-1)" : "none",
                  }}
                ></canvas>
                {showCountdown && (
                  <div
                    style={{
                      position: "absolute",
                      left: 0,
                      width: "100%",
                      top: 0,
                      height: "100%",
                      display: "flex",
                      alignContent: "center",
                      justifyContent: "center",
                      flexWrap: "wrap",
                    }}
                  >
                    <div
                      style={{
                        position: "absolute",
                        left: 0,
                        width: "100%",
                        top: 0,
                        height: "100%",
                        backgroundColor: "black",
                        opacity: 0.7,
                        borderRadius: 8,
                      }}
                    ></div>
                    <div style={{ zIndex: 100, textAlign: "center" }}>
                      <div style={{ color: "#ffffffaa" }}>Grabando en</div>
                      <MKTypography variant="h1" color="white" style={{ fontSize: 70 }}>
                        {countdownLabel}
                      </MKTypography>
                    </div>
                  </div>
                )}
              </div>
            </MKBox>

            <Divider sx={{ my: 0 }} />
            {!show ? (
              <LinearProgress
                variant="indeterminate"
                value={true}
                sx={{ width: "100%", overflow: "hidden", height: 4 }}
              ></LinearProgress>
            ) : (
              <MKBox sx={{ width: "100%", overflow: "hidden", height: 4 }}></MKBox>
            )}
            <MKBox display="flex" justifyContent="space-between" p={1.5}>
              <MKButton
                variant="gradient"
                color="info"
                onClick={() => instructionsModalRef.current.showModal(true, true)}
              >
                Instrucciones
              </MKButton>

              <div></div>
              <MKButton variant="gradient" color="info" onClick={toggleModal}>
                Cerrar
              </MKButton>
            </MKBox>
          </MKBox>
        </Slide>
      </Modal>
    </>
  );
});

RecordSignModal.propTypes = {
  onRecorded: PropTypes.func,
};

export default RecordSignModal;
