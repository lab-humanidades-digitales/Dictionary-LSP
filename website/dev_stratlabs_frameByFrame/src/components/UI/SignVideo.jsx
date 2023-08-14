import React, { useEffect, useState, useRef } from "react";
import PropTypes from "prop-types";

import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";

import PlayArrowIcon from "@mui/icons-material/PlayArrow";

SignVideo.propTypes = {
  source: PropTypes.string,
  style: PropTypes.object,
};

function SignVideo({ source, style, ...rest }) {
  const videoRef = useRef(null);
  const [available, setAvailable] = useState(true);
  const [hover, setHover] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    if (videoRef && videoRef.current) {
      videoRef.current.addEventListener("play", () => {
        setIsPlaying(true);
      });
      videoRef.current.addEventListener("pause", () => {
        setIsPlaying(false);
      });
    }
  }, [videoRef]);

  const play = () => {
    if (available) videoRef.current.play();
  };

  return (
    <>
      <div style={{ position: "relative", lineHeight: "1rem", aspectRatio: "16 / 9" }}>
        <video
          ref={videoRef}
          onMouseEnter={(e) => setHover(true)}
          onMouseLeave={(e) => setHover(false)}
          onError={(e) => setAvailable(false)}
          onLoadedMetadata={(e) => setAvailable(true)}
          style={{ width: "100%", ...style }}
          src={source}
          loop={true}
          controls={hover && isPlaying}
          {...rest}
        ></video>
        {!isPlaying && (
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
              cursor: available ? "pointer" : "default",
            }}
            onMouseEnter={(e) => setHover(true)}
            onMouseLeave={(e) => setHover(false)}
            onClick={() => play()}
          >
            {available ? (
              <div
                style={{
                  zIndex: 100,
                  textAlign: "center",
                  backgroundColor: "#000000a0",
                  borderRadius: "50%",
                }}
              >
                <PlayArrowIcon fontSize="large" color="light"></PlayArrowIcon>
              </div>
            ) : (
              <MKTypography variant="body2" fontSize="small">
                Video no disponible
              </MKTypography>
            )}
          </div>
        )}
      </div>
    </>
  );
}

export default SignVideo;
