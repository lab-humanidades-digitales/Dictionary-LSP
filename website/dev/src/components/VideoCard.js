import React from 'react';

/* css */
import '../css/videoCard.css'

const VideoCard = ({ video }) => {
    return (
      <div className="video-card">
        <div className="video-container">
          <video className="video-item" controls>
            <source src={video.imageUrl} type="video/mp4" />
          </video>
        </div>
        <h5 className="video-title">{video.label}</h5>
      </div>
    );
  };
export default VideoCard;