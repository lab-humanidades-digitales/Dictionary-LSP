/* eslint-disable no-debugger */
import React, { useEffect, useState, useRef, useCallback } from "react";
import PropTypes from "prop-types";

import { Chip, IconButton } from "@mui/material";

import { drawConnectors, drawLandmarks, lerp } from "@mediapipe/drawing_utils";
import {
  FACEMESH_TESSELATION,
  POSE_CONNECTIONS,
  HAND_CONNECTIONS,
  FACE_GEOMETRY,
  FACEMESH_FACE_OVAL,
  FACEMESH_CONTOURS,
} from "@mediapipe/holistic";

import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";

import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import { ArrowLeft, ArrowRight, FastForward, Pause, PlayArrow } from "@mui/icons-material";

LandmarkVideo.propTypes = {
  frames: PropTypes.array,
  landmarks: PropTypes.array,
  style: PropTypes.object,
};

function LandmarkVideo({ frames, landmarks, style, ...rest }) {
  const [index, setIndex] = useState(0);
  const [play, setPlay] = useState(false);
  const [speed, setSpeed] = useState(1);
  const canvasRef = useRef(null);
  const timer = useRef(null);

  const renderFrame = () => {
    debugger;
    var destCtx = canvasRef.current.getContext("2d");
    destCtx.drawImage(frames[index], 0, 0, canvasRef.current.width, canvasRef.current.height);
    paintLandmarks(landmarks[index]);
  };
  const nextFrame = () => {
    setIndex((prev) => (prev + 1) % frames.length);
  };

  const prevFrame = () => {
    setIndex((prev) => (prev == 0 ? frames.length - 1 : prev - 1));
  };

  const paintLandmarks = (results) => {
    const canvasCtx = canvasRef.current.getContext("2d");

    canvasCtx.save();
    canvasCtx.globalCompositeOperation = "source-over";

    const forceGreen = true;

    const faceColor = forceGreen || results.inStartPosition.face ? "#00CC0070" : "#CC000070";
    drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_FACE_OVAL, {
      color: faceColor,
      lineWidth: 1,
    });

    //LEFT HAND
    const leftHandColor =
      forceGreen || results.inStartPosition.leftHand ? "#00CC0070" : "#CC000070";
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
    const rightHandColor =
      forceGreen || results.inStartPosition.rightHand ? "#00CC0070" : "#CC000070";
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

  useEffect(() => {
    if (canvasRef && canvasRef.current != null) {
      debugger;
      renderFrame();
    }
  }, [canvasRef, index]);

  const startPlaying = (s) => {
    setPlay(true);
    setSpeed(s);
    clearInterval(timer.current);
    timer.current = setInterval(() => nextFrame(), 1000 / 30 / s);
  };

  const stopPlaying = () => {
    setPlay(false);
    clearInterval(timer.current);
  };

  useEffect(() => {
    debugger;
    //timer.current = setInterval(() => onTick(), 1000 / 30);

    return () => {
      clearInterval(timer.current);
    };
  }, []);

  return (
    <>
      <div style={{ position: "relative", lineHeight: "1rem", aspectRatio: "16 / 9" }}>
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            left: 0,
            width: "100%",
            top: 0,
            height: "100%",
          }}
        ></canvas>
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-around",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <IconButton onClick={() => prevFrame()} color="secondary">
            <ArrowLeft></ArrowLeft>
          </IconButton>
          <small style={{ width: 50, textAlign: "center" }}>
            {index + 1}/{frames.length}
          </small>

          <IconButton onClick={() => nextFrame()} color="secondary">
            <ArrowRight></ArrowRight>
          </IconButton>
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <IconButton
            onClick={() => (play ? stopPlaying() : startPlaying(speed))}
            color="secondary"
          >
            {play ? <Pause></Pause> : <PlayArrow></PlayArrow>}
          </IconButton>
        </div>
        <div>
          <Chip
            size="small"
            label="2x"
            variant={speed == 2 ? "filled" : "outlined"}
            color="secondary"
            onClick={() => startPlaying(2)}
          />
          <Chip
            size="small"
            label="1x"
            variant={speed == 1 ? "filled" : "outlined"}
            color="secondary"
            onClick={() => startPlaying(1)}
            style={{ marginLeft: 5 }}
          />
          <Chip
            size="small"
            label="0.5x"
            variant={speed == 0.5 ? "filled" : "outlined"}
            onClick={() => startPlaying(0.5)}
            color="secondary"
            style={{ marginLeft: 5 }}
          />
          <Chip
            size="small"
            label="0.25x"
            variant={speed == 0.25 ? "filled" : "outlined"}
            onClick={() => startPlaying(0.25)}
            color="secondary"
            style={{ marginLeft: 5 }}
          />
        </div>
      </div>
    </>
  );
}

export default LandmarkVideo;
